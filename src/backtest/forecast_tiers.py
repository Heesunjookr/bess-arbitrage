"""
forecast_tiers.py
-----------------
Evaluate the src/forecast/ tiers through the same dispatch harness as the
persistence tiers, and report how much of the forecast-value gap each one
recovers.

Tier ladder (all executable at the D-1 12:00 auction, zero lookahead):

    persist_d1  ->  blend  ->  rl_quad / blend_rl  ->  gbm  ->  pf_da (ceiling)

Gap recovery is measured against the persist_d1 baseline:

    recovered(tier) = (rev_tier − rev_persist_d1) / (rev_pf_da − rev_persist_d1)

computed on the *common* set of days where every tier has a forecast, so no
tier gets credit for simply trading on more days.

Run:  python -m src.backtest.forecast_tiers   (~5-10 min, LP over ~1600 days
      x 6 tiers; writes forecast_tiers_daily.parquet + forecast_findings.csv)
"""

from typing import Optional

import pandas as pd

from src.config import ROOT, get_logger, load_config
from src.data.health import run_health_checks
from src.data.load_prices import load_prices
from src.forecast.gbm_shape import build_gbm_forecast
from src.forecast.residual_load import (
    FORECAST_PATH,
    build_blend_rl_forecast,
    build_rl_quad_forecast,
)
from src.optimize.lp_dispatch import BatteryParams
from src.backtest.revenue import build_findings_table, run_backtest, summarize_yearly

logger = get_logger("forecast_tiers")


def run_forecast_tiers(
    df: Optional[pd.DataFrame] = None,
    p: Optional[BatteryParams] = None,
    cfg: Optional[dict] = None,
) -> pd.DataFrame:
    """Daily backtest frame including the fundamental and GBM tiers."""
    cfg = cfg or load_config()
    if df is None:
        df = load_prices(cfg)
    if p is None:
        p = BatteryParams.from_config(cfg)

    price = df["da_price"]
    fc_frame = pd.read_parquet(ROOT / FORECAST_PATH)
    run_health_checks(price, fc_frame)  # refuse to backtest on sick data
    rl = fc_frame["residual_load_fc"].reindex(price.index)
    logger.info("residual-load forecast coverage on price index: %.1f%%",
                100.0 * rl.notna().mean())

    extra = {
        "rl_quad": build_rl_quad_forecast(price, rl),
        "blend_rl": build_blend_rl_forecast(price, rl),
        "gbm": build_gbm_forecast(price, rl),
        "gbm_shape": build_gbm_forecast(price, rl, target="shape"),
    }

    # diagnostic (not executable): rl_quad on REALIZED residual load prices
    # the cost of weather-forecast error in the fundamental tiers
    actuals_path = ROOT / "data/raw/smard_actuals.parquet"
    if actuals_path.exists():
        rl_act = pd.read_parquet(actuals_path)["residual_load_act"].reindex(price.index)
        extra["rl_quad_pw"] = build_rl_quad_forecast(price, rl_act)

    return run_backtest(df=df, p=p, cfg=cfg, extra_forecasts=extra)


def gap_recovery_table(daily: pd.DataFrame,
                       baseline: str = "persist_d1") -> pd.DataFrame:
    """
    Per-tier capture and share of the (pf − baseline) gap recovered, on the
    common subset of days where all tiers traded.
    """
    rev_cols = [c for c in daily.columns if c.startswith("rev_")]
    common = daily.dropna(subset=rev_cols)
    pf = common["rev_pf_da"].sum()
    base = common["rev_%s" % baseline].sum()
    rows = []
    for c in rev_cols:
        tier = c[len("rev_"):]
        if tier == "pf_da":
            continue
        rev = common[c].sum()
        rows.append({
            "tier": tier,
            "n_days": len(common),
            "capture": rev / pf,
            "gap_recovered_vs_%s" % baseline:
                (rev - base) / (pf - base) if pf > base else float("nan"),
        })
    return pd.DataFrame(rows).set_index("tier").sort_values("capture")


if __name__ == "__main__":
    cfg = load_config()
    p = BatteryParams.from_config(cfg)
    daily = run_forecast_tiers(cfg=cfg, p=p)

    outdir = ROOT / cfg["paths"]["processed"]["dir"]
    daily.to_parquet(outdir / "forecast_tiers_daily.parquet")

    yearly = summarize_yearly(daily, p)
    findings = build_findings_table(yearly)
    findings.to_csv(outdir / "forecast_findings.csv")
    recovery = gap_recovery_table(daily)
    recovery.to_csv(outdir / "forecast_gap_recovery.csv")

    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda x: "%.3f" % x)
    print("\n=== capture ratios per year (executable / ceiling) ===")
    print(findings[[c for c in findings.columns if c.startswith("capture_")]])
    print("\n=== gap recovery vs persist_d1 (common days, full sample) ===")
    print(recovery)
