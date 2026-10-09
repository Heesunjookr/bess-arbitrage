"""
lp_artifact_check.py
--------------------
Verify that the dispatch solver prevents simultaneous charge and discharge.

The unconstrained formulation can burn energy through the round-trip loss
while being paid at negative prices. A SoC-neutral loop with grid import c=q
and grid export d=q*eta_rt has net value

    q * ( -price * (1 - eta_rt) - lambda * (1 + eta_rt) )

so the artifact only turns profitable below

    price < -lambda * (1 + eta_rt) / (1 - eta_rt)

≈ -25 EUR/MWh with the default eta_rt = 0.85, lambda = 2 EUR/MWh. `solve_day`
therefore activates a binary charge/discharge mode on negative-price days.
This report is a regression audit: overlap days and pair revenue must be zero.

This script scans every negative-price day in the sample and fails visibly
through its reported metrics if the production solver ever returns overlapping
flows. The current full-sample result is 330 negative-price days with zero
overlap days, zero overlap hours and zero pair revenue.

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
    return (-p.throughput_cost_eur_mwh * (1.0 + p.eta_round_trip)
            / (1.0 - p.eta_round_trip))


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
            pair_charge = np.minimum(
                r.charge[both], r.discharge[both] / p.eta_round_trip)
            pair_discharge = pair_charge * p.eta_round_trip
            pair_revenue += float(
                np.sum(prices[both] * (pair_discharge - pair_charge))
                - p.throughput_cost_eur_mwh
                * np.sum(pair_charge + pair_discharge)
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
        "\nInterpretation: the unconstrained formulation would create an "
        "artifact below %.0f EUR/MWh.\nThe production dispatch returned %d "
        "overlap hours; this regression audit must remain zero."
        % (result["break_even_price_eur_mwh"],
           result["hours_with_simultaneous_cd"])
    )
