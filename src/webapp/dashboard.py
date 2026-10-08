"""Pure data transformations used by the Streamlit dashboard."""

from dataclasses import asdict
from datetime import date
from typing import Dict

import numpy as np
import pandas as pd

from src.optimize.lp_dispatch import BatteryParams, settle_schedule, solve_day


TIER_LABELS: Dict[str, str] = {
    "persist_d1": "D-1 persistence",
    "persist_d7": "D-7 persistence",
    "blend": "D-1/D-7 rolling OLS",
    "gbm": "Gradient boosting",
    "rl_quad": "Residual-load OLS",
    "blend_rl": "Price + residual-load OLS",
}


def forecast_ladder(findings: pd.DataFrame) -> pd.DataFrame:
    """Return presentation-ready capture and gap-recovery percentages."""
    required = {"tier", "n_days", "capture", "gap_recovered_vs_persist_d1"}
    missing = required.difference(findings.columns)
    if missing:
        raise ValueError("forecast findings missing columns: %s" % sorted(missing))

    result = findings.loc[:, sorted(required)].copy()
    result["label"] = result["tier"].map(TIER_LABELS).fillna(result["tier"])
    result["capture_pct"] = result["capture"] * 100
    result["gap_recovered_pct"] = result["gap_recovered_vs_persist_d1"] * 100
    return result


def available_dates(prices: pd.DataFrame) -> list:
    """Dates that have both a complete delivery day and a complete D-1 curve."""
    counts = prices.groupby("date").size()
    complete = set(counts[counts == 24].index)
    return sorted(d for d in complete if d - pd.Timedelta(days=1) in complete)


def compare_dispatch_day(
    prices: pd.DataFrame,
    delivery_day: date,
    params: BatteryParams,
) -> tuple[pd.DataFrame, dict]:
    """Compare a realised-price ceiling with a D-1 executable schedule.

    The D-1 schedule is optimized on yesterday's curve, then settled against
    the selected delivery day's realised prices. Perfect foresight is shown
    only as a research ceiling.
    """
    current = prices.loc[prices["date"] == delivery_day, "da_price"]
    prior_day = delivery_day - pd.Timedelta(days=1)
    prior = prices.loc[prices["date"] == prior_day, "da_price"]
    if len(current) != 24 or len(prior) != 24:
        raise ValueError("selected date and D-1 must each contain 24 hourly prices")

    realised = current.to_numpy(dtype=float)
    forecast = prior.to_numpy(dtype=float)
    perfect = solve_day(realised, params)
    persistence = solve_day(forecast, params)
    persistence_revenue = settle_schedule(
        realised, persistence.charge, persistence.discharge, params
    )

    hours = np.arange(24)
    frame = pd.DataFrame(
        {
            "hour": hours,
            "realised_price": realised,
            "d1_price_curve": forecast,
            "perfect_net_mw": perfect.discharge - perfect.charge,
            "persistence_net_mw": persistence.discharge - persistence.charge,
            "perfect_soc_mwh": perfect.soc,
            "persistence_soc_mwh": persistence.soc,
        }
    )
    metrics = {
        "perfect_foresight_eur": perfect.revenue,
        "persistence_eur": persistence_revenue,
        "capture": (
            persistence_revenue / perfect.revenue if perfect.revenue > 0 else np.nan
        ),
        "spread_eur_mwh": float(realised.max() - realised.min()),
        "battery": asdict(params),
    }
    return frame, metrics
