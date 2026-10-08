"""
decompose.py
------------
Decompose the forecast-value gap (pf_da − executable tier) into a *level*
component and a *shape* component.

A battery schedule is driven almost entirely by the intra-day *shape* of the
price curve — which hours rank cheap and which rank expensive. The absolute
*level* of the curve barely matters (a constant shift only interacts with
revenue through the round-trip efficiency loss). Persistence forecasts get
the shape mostly right and the level wrong, so the interesting question is:
how much of the gap does each error actually cost?

Construction, per day and per tier:

  rev_fc     = optimize on the forecast f, settle at realized p (the tier)
  rev_shape  = optimize on f − mean(f) + mean(p), settle at p
               (forecast *shape* kept, daily *level* corrected to truth —
               a diagnostic counterfactual: "what if the bidder knew the
               true daily mean but kept the forecast's shape?")
  rev_pf     = optimize on p, settle at p (the ceiling)

  level_cost = rev_shape − rev_fc     (cost of getting the daily level wrong)
  shape_cost = rev_pf − rev_shape     (cost of getting the hour ranking wrong)
  gap        = rev_pf − rev_fc  ==  level_cost + shape_cost  (by construction)

Run:  python -m src.backtest.decompose   (~2 min, writes gap_decomposition.csv)
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import ROOT, get_logger, load_config
from src.data.load_prices import load_prices
from src.optimize.lp_dispatch import BatteryParams, solve_day
from src.optimize.realistic import (
    build_blend_forecast,
    build_persistence_forecast,
    scheduled_then_settle,
)

logger = get_logger("decompose")


def decompose_day(realized: np.ndarray, forecast: np.ndarray,
                  p: BatteryParams) -> Dict[str, float]:
    """Level/shape gap decomposition for one day. See module docstring."""
    f_shape = forecast - float(np.mean(forecast)) + float(np.mean(realized))
    rev_pf = solve_day(realized, p).revenue
    rev_fc = scheduled_then_settle(forecast, realized, p).revenue
    rev_shape = scheduled_then_settle(f_shape, realized, p).revenue
    return {
        "rev_pf": rev_pf,
        "rev_fc": rev_fc,
        "rev_shape": rev_shape,
        "gap": rev_pf - rev_fc,
        "level_cost": rev_shape - rev_fc,
        "shape_cost": rev_pf - rev_shape,
    }


def run_decomposition(
    df: Optional[pd.DataFrame] = None,
    p: Optional[BatteryParams] = None,
    cfg: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Daily gap decomposition for the persist_d1 and blend tiers.
    Returns a DataFrame indexed by date with <metric>_<tier> columns.
    """
    cfg = cfg or load_config()
    if df is None:
        df = load_prices(cfg)
    if p is None:
        p = BatteryParams.from_config(cfg)

    bt_cfg = cfg.get("backtest", {})
    blend_cfg = bt_cfg.get("blend", {})
    df = df.copy()
    df["fc_d1"] = build_persistence_forecast(df["da_price"], lag_days=1)
    df["fc_blend"] = build_blend_forecast(
        df["da_price"],
        lags=tuple(blend_cfg.get("lags_days", [1, 7])),
        window_days=int(blend_cfg.get("window_days", 90)),
        min_days=int(blend_cfg.get("min_days", 14)),
    )
    tiers = {"persist_d1": "fc_d1", "blend": "fc_blend"}

    rows: List[Dict] = []
    for date, sub in df.groupby("date"):
        sub = sub.sort_index()
        if len(sub) != 24:
            continue
        da = sub["da_price"].to_numpy(dtype=float)
        row = {"date": pd.Timestamp(date), "year": date.year}
        for tier, col in tiers.items():
            fc = sub[col].to_numpy(dtype=float)
            if not np.isfinite(fc).all():
                continue
            d = decompose_day(da, fc, p)
            row.update({"%s_%s" % (k, tier): v for k, v in d.items()})
        rows.append(row)

    res = pd.DataFrame(rows).set_index("date")
    logger.info("decomposition done: %d days", len(res))
    return res


def summarize_decomposition(daily: pd.DataFrame,
                            p: BatteryParams) -> pd.DataFrame:
    """
    Annualized EUR/MW/yr means per year (+ full-sample row 'all'), and the
    share of the gap explained by shape error.
    """
    metrics = ("gap", "level_cost", "shape_cost")
    tiers = sorted(c[len("gap_"):] for c in daily.columns
                   if c.startswith("gap_"))

    def _agg(g: pd.DataFrame, label) -> Dict:
        rec = {"year": label, "n_days": len(g)}
        for tier in tiers:
            for m in metrics:
                col = "%s_%s" % (m, tier)
                rec[col] = g[col].mean() * 365.0 / p.p_max_mw
            rec["shape_share_%s" % tier] = (
                rec["shape_cost_%s" % tier] / rec["gap_%s" % tier]
            )
        return rec

    recs = [_agg(g, int(y)) for y, g in daily.groupby("year")]
    recs.append(_agg(daily, "all"))
    return pd.DataFrame(recs).set_index("year")


if __name__ == "__main__":
    cfg = load_config()
    p = BatteryParams.from_config(cfg)
    daily = run_decomposition(cfg=cfg, p=p)
    summary = summarize_decomposition(daily, p)
    outdir = ROOT / cfg["paths"]["processed"]["dir"]
    summary.to_csv(outdir / "gap_decomposition.csv")
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: "%.3f" % x)
    print("\n=== forecast-value gap decomposition (EUR/MW/yr, annualized) ===")
    print(summary)
