"""
gap_analysis.py
---------------
Where does the forecast-value gap live in time?

Persistence forecasts day D with day D-1's curve, so it fails exactly when
the intra-day *shape* changes between D-1 and D — weather fronts, holiday
boundaries, extreme events. Before building a forecast model, this module
maps that failure surface:

  1. Concentration — how much of the total gap sits in the worst N% of days.
     If the gap is concentrated, a model only needs to be right on the few
     regime-change days, not every day.
  2. Shape-change attribution — bucket days by the rank correlation between
     the realized curves of D and D-1 (i.e. how wrong the persistence
     *shape* was) and show how the gap loads on the low-correlation buckets.
  3. Calendar breakdown — gap by year, month and weekday, to spot seasonal
     regimes (e.g. solar ramp-up months, holiday clusters).

All revenues come from the same per-day LP as the backtest; the gap is
rev_pf − rev_tier per day (non-negative by construction, since perfect
foresight is optimal against realized prices).

Run:  python -m src.backtest.gap_analysis   (~2 min, writes gap_analysis_*.csv)
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import ROOT, get_logger, load_config
from src.backtest.decompose import run_decomposition
from src.data.load_prices import load_prices
from src.optimize.lp_dispatch import BatteryParams

logger = get_logger("gap_analysis")


def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation via Pearson on ranks (no scipy dependency)."""
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if np.std(ra) == 0.0 or np.std(rb) == 0.0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def shape_change_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-day shape descriptors of the realized DA curve vs the previous day:

      rank_corr_d1 : Spearman correlation between day D and day D-1 curves —
                     the shape quality of the persistence forecast itself.
      spread       : max − min of day D's curve (EUR/MWh) — the size of the
                     arbitrage prize on that day.
      mean_price   : daily mean (level; should NOT matter, per decompose.py).
    """
    days = []
    prev: Optional[np.ndarray] = None
    prev_date = None
    for date, sub in df.groupby("date"):
        sub = sub.sort_index()
        if len(sub) != 24:
            prev, prev_date = None, None
            continue
        p = sub["da_price"].to_numpy(dtype=float)
        consecutive = prev is not None and (pd.Timestamp(date) - pd.Timestamp(prev_date)).days == 1
        days.append({
            "date": pd.Timestamp(date),
            "rank_corr_d1": _rank_corr(prev, p) if consecutive else np.nan,
            "spread": float(p.max() - p.min()),
            "mean_price": float(p.mean()),
        })
        prev, prev_date = p, date
    return pd.DataFrame(days).set_index("date")


def run_gap_analysis(
    df: Optional[pd.DataFrame] = None,
    p: Optional[BatteryParams] = None,
    cfg: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Daily frame: gap_<tier>, shape/level costs (from decompose) merged with
    the shape-change descriptors. Index: date.
    """
    cfg = cfg or load_config()
    if df is None:
        df = load_prices(cfg)
    if p is None:
        p = BatteryParams.from_config(cfg)
    daily = run_decomposition(df=df, p=p, cfg=cfg)
    shape = shape_change_metrics(df)
    out = daily.join(shape, how="left")
    out["month"] = out.index.month
    out["weekday"] = out.index.weekday  # 0=Mon
    return out


def concentration_table(daily: pd.DataFrame, tier: str = "persist_d1") -> pd.DataFrame:
    """
    Share of the total gap held by the worst N% of days (days sorted by gap,
    descending). A steep curve = the gap is an "event" phenomenon.
    """
    g = daily["gap_%s" % tier].dropna().sort_values(ascending=False)
    total = g.sum()
    cum = g.cumsum() / total
    rows = []
    for pct in (5, 10, 20, 30, 50):
        n = max(1, int(round(len(g) * pct / 100.0)))
        rows.append({
            "worst_days_pct": pct,
            "n_days": n,
            "gap_share": float(cum.iloc[n - 1]),
            "mean_gap_eur": float(g.iloc[:n].mean()),
        })
    return pd.DataFrame(rows).set_index("worst_days_pct")


def shape_bucket_table(daily: pd.DataFrame, tier: str = "persist_d1",
                       n_buckets: int = 5) -> pd.DataFrame:
    """
    Bucket days by rank_corr_d1 quantiles (how similar today's realized shape
    was to yesterday's) and aggregate the gap per bucket.
    """
    d = daily.dropna(subset=["gap_%s" % tier, "rank_corr_d1"]).copy()
    d["bucket"] = pd.qcut(d["rank_corr_d1"], n_buckets,
                          labels=["q%d" % i for i in range(1, n_buckets + 1)])
    total = d["gap_%s" % tier].sum()
    out = d.groupby("bucket", observed=True).agg(
        n_days=("gap_%s" % tier, "size"),
        rank_corr_lo=("rank_corr_d1", "min"),
        rank_corr_hi=("rank_corr_d1", "max"),
        mean_gap_eur=("gap_%s" % tier, "mean"),
        gap_share=("gap_%s" % tier, lambda s: s.sum() / total),
        mean_spread=("spread", "mean"),
    )
    return out


def calendar_table(daily: pd.DataFrame, tier: str = "persist_d1") -> Dict[str, pd.DataFrame]:
    """Gap by year / month / weekday (mean EUR/day and share of total)."""
    col = "gap_%s" % tier
    d = daily.dropna(subset=[col])
    total = d[col].sum()
    out = {}
    for key in ("year", "month", "weekday"):
        out[key] = d.groupby(key).agg(
            n_days=(col, "size"),
            mean_gap_eur=(col, "mean"),
            gap_share=(col, lambda s: s.sum() / total),
        )
    return out


if __name__ == "__main__":
    cfg = load_config()
    p = BatteryParams.from_config(cfg)
    daily = run_gap_analysis(cfg=cfg, p=p)

    outdir = ROOT / cfg["paths"]["processed"]["dir"]
    daily.to_csv(outdir / "gap_analysis_daily.csv")

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: "%.3f" % x)

    for tier in ("persist_d1", "blend"):
        conc = concentration_table(daily, tier)
        buck = shape_bucket_table(daily, tier)
        cal = calendar_table(daily, tier)
        conc.to_csv(outdir / ("gap_concentration_%s.csv" % tier))
        buck.to_csv(outdir / ("gap_shape_buckets_%s.csv" % tier))
        print("\n=== [%s] gap concentration (worst-N%% days) ===" % tier)
        print(conc)
        print("\n=== [%s] gap by D/D-1 rank-correlation bucket (q1 = biggest shape change) ===" % tier)
        print(buck)
        print("\n=== [%s] gap by month ===" % tier)
        print(cal["month"].T)
