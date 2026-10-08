"""
lp_artifact_check.py
--------------------
Quantify the LP's simultaneous charge+discharge artifact.

The per-day LP has no binary variable forbidding simultaneous charging and
discharging (that would make it a MILP). Physically a battery does one or
the other; in the LP the combination burns energy through the round-trip
loss — and burning energy is *paid* when prices are negative. Per MWh of
paired flow (c = d = q) the net effect is

    q * ( -price * (1 - eta_dis) - 2 * lambda )

so the artifact only turns profitable below

    price < -2 * lambda / (1 - eta_dis)

≈ -51 EUR/MWh with the default eta_rt = 0.85, lambda = 2 EUR/MWh. Milder
negative prices leave it at or below break-even, where the solver may still
return overlapping flows as one of many optimal (degenerate) solutions.

This script scans every negative-price day in the sample, reports where the
solver used simultaneous flows and what they contributed to revenue. On the
full 2022–2026 sample the result is: ~300 negative-price days, overlap on
~58 of them (~150 hours), net contribution ≈ -7 EUR over 4.5 years —
solver degeneracy, not a revenue exploit. Hence a plain LP is the right
tool for this study; a real dispatch system would add the binaries or net
out overlapping flows in post-processing.

Run:  python -m src.backtest.lp_artifact_check   (~40s)
"""

import json
from typing import Optional

import numpy as np
import pandas as pd

from src.config import ROOT, get_logger, load_config
from src.data.load_prices import load_prices
from src.optimize.lp_dispatch import BatteryParams, solve_day

logger = get_logger("lp_artifact_check")

TOL_MW = 1e-4  # flows below this are solver noise, not dispatch


def break_even_price(p: BatteryParams) -> float:
    """Price below which simultaneous charge+discharge becomes profitable."""
    return -2.0 * p.throughput_cost_eur_mwh / (1.0 - p.eta_dis)


def scan(df: Optional[pd.DataFrame] = None,
         p: Optional[BatteryParams] = None,
         cfg: Optional[dict] = None) -> dict:
    """
    Solve the PF LP on every negative-price day and measure simultaneous
    charge+discharge: where it occurs and its net revenue contribution
    (paired-flow revenue minus its share of the throughput penalty).
    """
    cfg = cfg or load_config()
    if df is None:
        df = load_prices(cfg)
    if p is None:
        p = BatteryParams.from_config(cfg)

    neg_days = overlap_days = overlap_hours = 0
    pair_revenue = 0.0
    for _, sub in df.groupby("date"):
        sub = sub.sort_index()
        if len(sub) != 24:
            continue
        prices = sub["da_price"].to_numpy(dtype=float)
        if prices.min() >= 0:
            continue
        neg_days += 1
        r = solve_day(prices, p)
        both = (r.charge > TOL_MW) & (r.discharge > TOL_MW)
        if both.any():
            overlap_days += 1
            overlap_hours += int(both.sum())
            pair = np.minimum(r.charge[both], r.discharge[both])
            pair_revenue += float(
                np.sum(prices[both] * pair * (p.eta_dis - 1.0))
                - p.throughput_cost_eur_mwh * 2.0 * pair.sum()
            )

    result = {
        "break_even_price_eur_mwh": round(break_even_price(p), 1),
        "negative_price_days": neg_days,
        "days_with_simultaneous_cd": overlap_days,
        "hours_with_simultaneous_cd": overlap_hours,
        "net_pair_revenue_eur": round(pair_revenue, 1),
    }
    logger.info("artifact scan: %s", result)
    return result


if __name__ == "__main__":
    cfg = load_config()
    result = scan(cfg=cfg)
    out = ROOT / cfg["paths"]["processed"]["dir"] / "lp_artifact_check.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(
        "\nInterpretation: simultaneous charge+discharge only pays below "
        "%.0f EUR/MWh;\nits net contribution above is ~0 -> degenerate "
        "solutions, not a revenue exploit." % result["break_even_price_eur_mwh"]
    )
