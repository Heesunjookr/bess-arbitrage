"""
heatwave_study.py
-----------------
Event study: the late-June 2026 European heatwave in the DE-LU day-ahead
market, seen through the battery's eyes.

Heatwaves are a battery's best weeks — midday solar keeps the trough low
while air-conditioning load and thermal/nuclear derates (river cooling
limits) push the evening peak up, so the intra-day spread widens sharply.
They are also exactly the regime-change days where persistence bidding
fails (see gap_analysis.py). This study quantifies both effects on one
concrete event:

  * event window vs. a same-length pre-event baseline window,
  * daily spread, PF ceiling revenue, per-tier revenue and capture,
  * the residual-load forecast that a fundamental tier would have seen.

Run after forecast_tiers.py (reads forecast_tiers_daily.parquet):

    python -m src.backtest.heatwave_study
"""

from typing import Optional

import pandas as pd

from src.config import ROOT, get_logger, load_config
from src.data.load_prices import load_prices

logger = get_logger("heatwave_study")

# late-June 2026 European heatwave; the price sample ends 2026-06-29
EVENT_START = "2026-06-19"
EVENT_END = "2026-06-29"
BASELINE_START = "2026-06-01"
BASELINE_END = "2026-06-11"


def _window_stats(daily: pd.DataFrame, prices: pd.DataFrame,
                  start: str, end: str, label: str) -> dict:
    d = daily.loc[start:end]
    px = prices.loc[(prices["date"] >= pd.Timestamp(start).date())
                    & (prices["date"] <= pd.Timestamp(end).date()), "da_price"]
    by_day = px.groupby(prices.loc[px.index, "date"])
    rec = {
        "window": label,
        "days": len(d),
        "mean_price": px.mean(),
        "mean_daily_spread": (by_day.max() - by_day.min()).mean(),
        "pf_eur_day": d["rev_pf_da"].mean(),
    }
    for c in d.columns:
        if c.startswith("rev_") and c != "rev_pf_da":
            tier = c[len("rev_"):]
            rec["%s_eur_day" % tier] = d[c].mean()
            rec["capture_%s" % tier] = d[c].sum() / d["rev_pf_da"].sum()
    return rec


def run_heatwave_study(cfg: Optional[dict] = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    outdir = ROOT / cfg["paths"]["processed"]["dir"]
    daily = pd.read_parquet(outdir / "forecast_tiers_daily.parquet")
    prices = load_prices(cfg)

    rows = [
        _window_stats(daily, prices, BASELINE_START, BASELINE_END, "baseline_early_june"),
        _window_stats(daily, prices, EVENT_START, EVENT_END, "heatwave"),
    ]
    res = pd.DataFrame(rows).set_index("window")
    res.loc["ratio"] = res.loc["heatwave"] / res.loc["baseline_early_june"]
    return res


if __name__ == "__main__":
    cfg = load_config()
    res = run_heatwave_study(cfg)
    outdir = ROOT / cfg["paths"]["processed"]["dir"]
    res.to_csv(outdir / "heatwave_study.csv")
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda x: "%.3f" % x)
    print("\n=== June 2026 heatwave vs early-June baseline ===")
    print(res.T)
