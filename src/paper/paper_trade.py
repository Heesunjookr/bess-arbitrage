"""
paper_trade.py
--------------
Live paper trading of the forecast-tier ladder on the real DE-LU day-ahead
auction — the forward test of the backtest, and the definitive answer to the
SMARD snapshot-revision caveat: forecasts are fetched and *frozen to disk
before the D-1 12:00 gate closure*, so they are point-in-time by
construction. From here on, capture ratios accumulate out of sample.

Daily cycle (one cron/launchd run, ideally ~11:00-11:45 Europe/Berlin):

  1. Fetch the trailing ~110 days of DA prices (SMARD 4169) and the public
     TSO forecasts (411/125/123/3791). The price-only tiers are always
     eligible; fundamental tiers trade only when a complete delivery-day
     forecast is actually present before gate closure.
  2. Settle every frozen bid whose delivery-day prices are now published:
     schedule (frozen) x realized prices, vs. the perfect-foresight LP on
     the same day. Append to the ledger.
  3. Create tomorrow's bid: build the tier forecasts (persist_d1, blend,
     rl_quad, blend_rl), solve the dispatch LP on each, and freeze
     forecast + schedule + timestamp to a JSON bid file.

Gate-closure guard: a bid is only created if the current Europe/Berlin time
is before the 12:00 auction. If the machine was asleep and the job fires
late, the day is *skipped and logged*, never back-filled — a paper trade
created after the prices exist would be lookahead, the exact sin this
project measures.

GBM is deliberately not in the live ladder: it underperformed the linear
fundamental tiers in the backtest and needs a 365-day training window.

Artifacts (data/paper/):
  bids/YYYY-MM-DD.json   frozen forecasts + schedules per delivery day
  ledger.csv             one settled row per delivery day and tier

Run:  python -m src.paper.paper_trade
"""

import json
import os
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import ROOT, get_logger, load_config
from src.data.fetch_smard_forecasts import FILTERS, fetch_series
from src.data.health import run_health_checks
from src.forecast.residual_load import (
    build_blend_rl_forecast,
    build_rl_quad_forecast,
)
from src.optimize.lp_dispatch import BatteryParams, settle_schedule, solve_day
from src.optimize.realistic import build_blend_forecast, build_persistence_forecast

logger = get_logger("paper_trade")

PRICE_FILTER = 4169  # Marktpreis: Deutschland/Luxemburg
AUCTION_HOUR = 12    # EPEX SDAC day-ahead gate closure, Europe/Berlin
PAPER_DIR = "data/paper"


def _now_berlin() -> pd.Timestamp:
    return pd.Timestamp.now(tz="Europe/Berlin")


def _notify_telegram(text: str) -> None:
    """
    Best-effort Telegram notification (HTML parse_mode, workspace
    convention). Credentials: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID from the
    environment or ~/.whale_secrets (KEY=VALUE lines). Silent no-op when
    absent, warning-only on send failure — notifications must never break
    the trading run.
    """
    secrets = dict(os.environ)
    path = Path.home() / ".whale_secrets"
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                secrets.setdefault(k.strip(), v.strip().strip("'\""))
    token = secrets.get("TELEGRAM_BOT_TOKEN")
    chat = secrets.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        logger.info("telegram: no TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID — skipping")
        return
    try:
        req = urllib.request.Request(
            "https://api.telegram.org/bot%s/sendMessage" % token,
            data=json.dumps({"chat_id": chat, "text": text,
                             "parse_mode": "HTML"}).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=15)
        logger.info("telegram notification sent")
    except Exception as e:  # noqa: BLE001 - never break the run on notify
        logger.warning("telegram notification failed: %s", e)


def fetch_live_frame(cfg: Optional[dict] = None, days_back: int = 110,
                     now: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Trailing window of hourly DA prices + residual-load forecast, complete
    24h days only (same convention as the backtest). Future delivery days
    carry NaN prices and (once published) real forecast values.
    """
    cfg = cfg or load_config()
    tz = cfg["data"]["timezone"]
    now = now or _now_berlin()
    start = (now - pd.Timedelta(days=days_back)).normalize()
    end = (now + pd.Timedelta(days=2)).normalize()
    start_ms, end_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)

    cols = {"da_price": fetch_series(PRICE_FILTER, start_ms, end_ms)}
    for name, fid in FILTERS.items():
        cols[name] = fetch_series(fid, start_ms, end_ms)
    df = pd.DataFrame(cols)
    df.index = df.index.tz_convert(tz)
    # Materialise tomorrow's delivery hours even when the public fundamental
    # feeds have not published values yet. This lets price-only D-1/blend
    # tiers freeze a legitimate schedule while fundamental tiers remain
    # explicitly no_forecast.
    full_index = pd.date_range(start, end - pd.Timedelta(hours=1), freq="1h", tz=tz)
    df = df.reindex(full_index)
    df["residual_load_fc"] = (df["load_fc"] - df["pv_fc"]
                              - df["wind_on_fc"] - df["wind_off_fc"])
    df["date"] = df.index.date

    counts = df.groupby("date")["date"].transform("size")
    df = df[counts == 24]
    logger.info("live frame: %s .. %s, %d complete days, price known through %s",
                df.index.min(), df.index.max(), df["date"].nunique(),
                df["da_price"].dropna().index.max())
    return df


def build_tier_forecasts(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """The live ladder — identical builders and windows as the backtest."""
    price, rl = df["da_price"], df["residual_load_fc"]
    return {
        "persist_d1": build_persistence_forecast(price, lag_days=1),
        "blend": build_blend_forecast(price),
        "rl_quad": build_rl_quad_forecast(price, rl),
        "blend_rl": build_blend_rl_forecast(price, rl),
    }


def create_bid(df: pd.DataFrame, p: BatteryParams,
               now: Optional[pd.Timestamp] = None) -> Optional[Path]:
    """
    Freeze tomorrow's forecasts and LP schedules to a bid file — but only
    before gate closure, and never twice for the same delivery day.
    """
    now = now or _now_berlin()
    delivery = (now + pd.Timedelta(days=1)).date()

    bids_dir = ROOT / PAPER_DIR / "bids"
    bids_dir.mkdir(parents=True, exist_ok=True)
    path = bids_dir / ("%s.json" % delivery)
    if path.exists():
        logger.info("bid for %s already frozen — skipping", delivery)
        return path
    if now.hour >= AUCTION_HOUR:
        logger.warning("MISSED AUCTION for %s: now=%s is past the %02d:00 "
                       "gate closure; refusing to bid on known prices",
                       delivery, now, AUCTION_HOUR)
        return None

    day = df[df["date"] == delivery]
    if len(day) != 24:
        logger.warning("no complete hourly delivery-day frame for %s — skipping bid", delivery)
        return None

    tiers = {}
    for tier, fc in build_tier_forecasts(df).items():
        f = fc.loc[day.index].to_numpy(dtype=float)
        if not np.isfinite(f).all():
            tiers[tier] = {
                "status": "no_forecast",
                "reason": "required inputs were not published before gate closure",
            }
            continue
        plan = solve_day(f, p)
        tiers[tier] = {
            "status": plan.status,
            "forecast": [round(x, 4) for x in f],
            "charge": [round(x, 6) for x in plan.charge],
            "discharge": [round(x, 6) for x in plan.discharge],
        }
    bid = {
        "delivery_day": str(delivery),
        "created_at": now.isoformat(),
        "gate_closure": "%sT%02d:00+02:00" % (now.date(), AUCTION_HOUR),
        "tiers": tiers,
    }
    path.write_text(json.dumps(bid, indent=1))
    logger.info("froze bid for %s (%d tiers) at %s", delivery, len(tiers), now)
    return path


def settle_bids(df: pd.DataFrame, p: BatteryParams) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Settle every frozen bid whose delivery-day prices are available and not
    yet in the ledger. Returns (full ledger, rows settled in this run).
    """
    paper_dir = ROOT / PAPER_DIR
    ledger_path = paper_dir / "ledger.csv"
    ledger = (pd.read_csv(ledger_path) if ledger_path.exists()
              else pd.DataFrame(columns=["delivery_day"]))
    done = set(ledger["delivery_day"].astype(str))

    rows = []
    for path in sorted((paper_dir / "bids").glob("*.json")):
        bid = json.loads(path.read_text())
        day_str = bid["delivery_day"]
        if day_str in done:
            continue
        day = df[df["date"] == pd.Timestamp(day_str).date()]
        realized = day["da_price"].to_numpy(dtype=float)
        if len(realized) != 24 or not np.isfinite(realized).all():
            continue  # prices not published yet (or DST day)
        row = {
            "delivery_day": day_str,
            "created_at": bid["created_at"],
            "settled_at": _now_berlin().isoformat(),
            "rev_pf": solve_day(realized, p).revenue,
        }
        for tier, t in bid["tiers"].items():
            if t.get("status") in (None, "no_forecast"):
                row["rev_%s" % tier] = np.nan
                continue
            row["rev_%s" % tier] = settle_schedule(
                realized, np.array(t["charge"]), np.array(t["discharge"]), p)
        rows.append(row)
        logger.info("settled %s: pf=%.2f blend_rl=%.2f", day_str,
                    row["rev_pf"], row.get("rev_blend_rl", np.nan))

    new = pd.DataFrame(rows)
    if rows:
        ledger = pd.concat([ledger, new], ignore_index=True)
        ledger.to_csv(ledger_path, index=False)
    return ledger, new


def summarize(ledger: pd.DataFrame) -> None:
    if ledger.empty:
        print("ledger empty — no settled paper trades yet")
        return
    rev_cols = [c for c in ledger.columns if c.startswith("rev_") and c != "rev_pf"]
    print("\n=== paper-trading ledger: %d settled days ===" % len(ledger))
    pf = ledger["rev_pf"].sum()
    for c in rev_cols:
        tier = c[len("rev_"):]
        print("  %-11s rev %8.2f EUR   capture %6.1f%%"
              % (tier, ledger[c].sum(), 100.0 * ledger[c].sum() / pf if pf else np.nan))
    print("  %-11s rev %8.2f EUR   (ceiling)" % ("pf", pf))


def run_daily() -> None:
    cfg = load_config()
    p = BatteryParams.from_config(cfg)
    now = _now_berlin()
    df = fetch_live_frame(cfg, now=now)
    # health gate: refuse to settle or bid on data that fails the checks
    run_health_checks(df["da_price"], df)
    ledger, new = settle_bids(df, p)
    delivery = (now + pd.Timedelta(days=1)).date()
    existed = (ROOT / PAPER_DIR / "bids" / ("%s.json" % delivery)).exists()
    bid_path = create_bid(df, p, now=now)
    summarize(ledger)

    # notify only when this run did something: fresh bid, settlements, or a
    # definitively missed auction (post-gate run with no frozen bid)
    newly_created = bid_path is not None and not existed
    missed = bid_path is None and now.hour >= AUCTION_HOUR
    if not (newly_created or len(new) or missed):
        return
    lines = ["<b>BESS paper trade</b> %s" % now.strftime("%Y-%m-%d %H:%M")]
    if newly_created:
        lines.append("bid for %s: frozen ✅" % delivery)
    elif missed:
        lines.append("bid for %s: MISSED ❌" % delivery)
    for _, r in new.iterrows():
        cap = 100.0 * r["rev_blend_rl"] / r["rev_pf"] if r["rev_pf"] else float("nan")
        lines.append("settled %s: pf %.0f EUR, blend_rl %.0f EUR (%.1f%%)"
                     % (r["delivery_day"], r["rev_pf"], r["rev_blend_rl"], cap))
    if len(ledger):
        pf = ledger["rev_pf"].sum()
        cap = 100.0 * ledger["rev_blend_rl"].sum() / pf if pf else float("nan")
        lines.append("cumulative: %d days, blend_rl capture %.1f%%" % (len(ledger), cap))
    _notify_telegram("\n".join(lines))


if __name__ == "__main__":
    run_daily()
