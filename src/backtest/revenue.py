"""
revenue.py
----------
Loop the per-day LP dispatch over the whole sample and aggregate
per-year / per-tier revenue, cycle and capture-ratio metrics.

Tiers (all on the DE-LU day-ahead price — the only delivery-period price
series in this repo that passed validation, see src/data/validate.py):

  - pf_da      : perfect foresight — optimize against the realized DA curve.
                 The ceiling; assumes the auction outcome is known when bidding.
  - persist_d1 : persistence bidding — schedule on the D-1 curve (the latest
                 curve actually known at the D-1 12:00 auction), settle at
                 the realized D curve. Fully executable.
  - persist_d7 : same with a one-week lag (captures weekday seasonality).
  - blend      : rolling-OLS blend of the D-1/D-7 curves — the first, cheapest
                 "forecast model". Also fully executable (no lookahead).

Metrics: EUR/yr, EUR/MW/yr, EUR/MWh(capacity)/yr, equivalent full cycles,
capture ratio (executable / ceiling) and the forecast-value gap
(pf_da − persist), i.e. the money a perfect DA price forecast would add.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import get_logger, load_config
from src.data.load_prices import load_prices
from src.optimize.lp_dispatch import BatteryParams, solve_day
from src.optimize.realistic import (
    build_blend_forecast,
    build_persistence_forecast,
    persistence_day,
)

logger = get_logger("revenue")


def _iter_days(df: pd.DataFrame):
    """Yield (date, sub_df) for complete 24h days, hours sorted."""
    for date, sub in df.groupby("date"):
        sub = sub.sort_index()
        if len(sub) == 24:
            yield date, sub


def run_backtest(
    df: Optional[pd.DataFrame] = None,
    p: Optional[BatteryParams] = None,
    cfg: Optional[dict] = None,
    extra_forecasts: Optional[Dict[str, pd.Series]] = None,
) -> pd.DataFrame:
    """
    Full-sample daily backtest -> DataFrame indexed by date.

    Columns: year, rev_pf_da, rev_persist_d{k}..., tput_* per tier.

    extra_forecasts: {tier_name: hourly forecast Series} for additional
    executable tiers (e.g. the src/forecast/ fundamental and GBM tiers).
    Each series is aligned onto the price index; NaN days -> no-trade.
    """
    cfg = cfg or load_config()
    if df is None:
        df = load_prices(cfg)
    if p is None:
        p = BatteryParams.from_config(cfg)

    bt_cfg = cfg.get("backtest", {})
    lags = list(bt_cfg.get("persistence_lags_days", [1, 7]))
    blend_cfg = bt_cfg.get("blend", {})

    df = df.copy()
    fc_cols = {}
    for k in lags:
        col = "fc_d%d" % k
        df[col] = build_persistence_forecast(df["da_price"], lag_days=k)
        fc_cols["persist_d%d" % k] = col
    df["fc_blend"] = build_blend_forecast(
        df["da_price"],
        lags=tuple(blend_cfg.get("lags_days", lags)),
        window_days=int(blend_cfg.get("window_days", 90)),
        min_days=int(blend_cfg.get("min_days", 14)),
    )
    fc_cols["blend"] = "fc_blend"

    for tier, fc in (extra_forecasts or {}).items():
        col = "fc_%s" % tier
        df[col] = fc.reindex(df.index)
        fc_cols[tier] = col

    rows: List[Dict] = []
    for date, sub in _iter_days(df):
        da = sub["da_price"].to_numpy(dtype=float)
        r_pf = solve_day(da, p)
        row = {
            "date": pd.Timestamp(date),
            "year": date.year,
            "rev_pf_da": r_pf.revenue,
            "tput_pf_da": r_pf.throughput_mwh,
        }
        for tier, col in fc_cols.items():
            fc = sub[col].to_numpy(dtype=float)
            r = persistence_day(da, fc, p)
            valid = r.status != "no_forecast"
            row["rev_%s" % tier] = r.revenue if valid else np.nan
            row["tput_%s" % tier] = r.throughput_mwh if valid else np.nan
        rows.append(row)

    res = pd.DataFrame(rows).set_index("date")
    logger.info("backtest done: %d days, tiers=%s",
                len(res), [c for c in res.columns if c.startswith("rev_")])
    return res


def summarize_yearly(daily: pd.DataFrame, p: BatteryParams) -> pd.DataFrame:
    """
    Aggregate daily results per year; partial years annualized to 365 days.

    Per tier: rev_<tier> (EUR/yr), per_mw_<tier>, per_mwh_<tier>,
    cycles_<tier> (equivalent full cycles/yr), plus n_days.
    NaN days (no forecast yet) are excluded per tier via mean*365.
    """
    rev_cols = [c for c in daily.columns if c.startswith("rev_")]
    tput_cols = [c for c in daily.columns if c.startswith("tput_")]

    recs = []
    for year, g in daily.groupby("year"):
        rec = {"year": int(year), "n_days": len(g)}
        for c in rev_cols:
            tier = c[len("rev_"):]
            total = g[c].mean() * 365.0  # per-tier annualization, NaN-safe
            rec["rev_%s" % tier] = total
            rec["per_mw_%s" % tier] = total / p.p_max_mw
            rec["per_mwh_%s" % tier] = total / p.e_max_mwh
        for c in tput_cols:
            tier = c[len("tput_"):]
            # one equivalent full cycle = 2*E_max MWh of throughput (charge+discharge)
            rec["cycles_%s" % tier] = g[c].mean() * 365.0 / (2.0 * p.e_max_mwh)
        recs.append(rec)

    return pd.DataFrame(recs).set_index("year").sort_index()


def build_findings_table(yearly: pd.DataFrame) -> pd.DataFrame:
    """
    Headline table: per-tier EUR/MW/yr, capture ratios vs the PF ceiling,
    and the forecast-value gap (ceiling − executable).
    """
    t = pd.DataFrame(index=yearly.index)
    t["pf_da_per_mw"] = yearly["per_mw_pf_da"]
    persist_tiers = sorted(
        c[len("per_mw_"):] for c in yearly.columns
        if c.startswith("per_mw_") and c != "per_mw_pf_da"
    )
    for tier in persist_tiers:
        t["%s_per_mw" % tier] = yearly["per_mw_%s" % tier]
        t["capture_%s" % tier] = yearly["per_mw_%s" % tier] / yearly["per_mw_pf_da"]
        t["gap_%s" % tier] = yearly["per_mw_pf_da"] - yearly["per_mw_%s" % tier]
    t["cycles_pf_da"] = yearly["cycles_pf_da"]
    return t


if __name__ == "__main__":
    cfg = load_config()
    p = BatteryParams.from_config(cfg)
    daily = run_backtest(cfg=cfg, p=p)
    yearly = summarize_yearly(daily, p)
    findings = build_findings_table(yearly)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: "%.1f" % x)
    print("\n=== yearly metrics (annualized) ===")
    print(yearly[[c for c in yearly.columns if c.startswith(("per_mw_", "cycles_"))] + ["n_days"]])
    print("\n=== headline findings (EUR/MW/yr and ratios) ===")
    print(findings)
