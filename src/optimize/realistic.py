"""
realistic.py
------------
Executable dispatch tiers.

Perfect foresight (PF) optimizes against realized prices — a ceiling that
assumes the future is known. The executable tiers schedule using only
information available at decision time and settle at realized prices.

Decision timing in the German day-ahead market: bids for delivery day D are
due at the D-1 12:00 EPEX auction. At that moment the most recent *known*
full price curve is the one for delivery day D-1 (cleared at D-2 12:00).
So a persistence bidder uses the D-k curve as the forecast for day D and
submits fixed (price-inelastic) quantities; the schedule then settles at
day D's realized clearing prices. The PF-minus-persistence gap is the
revenue a better price forecast would buy.
"""

import warnings
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.config import get_logger
from src.optimize.lp_dispatch import BatteryParams, DispatchResult, solve_day, settle_schedule

logger = get_logger("realistic")


def perfect_foresight_day(realized: np.ndarray, p: BatteryParams) -> DispatchResult:
    """Optimize against realized prices (the ceiling). Alias of solve_day."""
    return solve_day(realized, p)


def scheduled_then_settle(
    forecast: np.ndarray,
    realized: np.ndarray,
    p: BatteryParams,
) -> DispatchResult:
    """
    Optimize the schedule on `forecast`, then settle it at `realized` prices.
    With forecast == realized this reduces to perfect foresight.
    """
    plan = solve_day(forecast, p)
    rev = settle_schedule(realized, plan.charge, plan.discharge, p)
    tput = float(np.sum(plan.charge + plan.discharge))
    return DispatchResult(plan.charge, plan.discharge, plan.soc, rev, tput, plan.status)


def build_persistence_forecast(price: pd.Series, lag_days: int = 1) -> pd.Series:
    """
    Forecast for hour h of day D = realized price at hour h of day D-lag_days.
    Same index as the input; the first `lag_days` days are NaN.
    """
    return price.shift(24 * lag_days)


def build_blend_forecast(
    price: pd.Series,
    lags: Sequence[int] = (1, 7),
    window_days: int = 90,
    min_days: int = 14,
) -> pd.Series:
    """
    Rolling-OLS blend of persistence curves — the first, cheapest "forecast
    model" on top of naive persistence.

    For delivery day D, regress the realized prices of the trailing
    `window_days` days on their own D-k persistence curves (plus intercept),
    then apply the fitted weights to day D's known D-k curves. Every target
    and regressor in the training window is a price already cleared before
    the D-1 12:00 auction, so there is no lookahead.

    Days with fewer than `min_days` complete training days (start of the
    sample) fall back to the equal-weight mean of the available lag curves;
    hours where no lag is available stay NaN (-> no-trade downstream).
    """
    lag_cols = ["x%d" % k for k in lags]
    frame = pd.DataFrame({"y": price})
    for k, col in zip(lags, lag_cols):
        frame[col] = price.shift(24 * k)
    frame["date"] = frame.index.date

    out = pd.Series(np.nan, index=price.index, name="fc_blend")
    day_frames = [g for _, g in frame.groupby("date", sort=True)]

    for i, day in enumerate(day_frames):
        features = day[lag_cols].to_numpy(dtype=float)
        with warnings.catch_warnings():
            # all-NaN rows (very first day) legitimately yield NaN
            warnings.simplefilter("ignore", category=RuntimeWarning)
            fallback = np.nanmean(features, axis=1)  # equal-weight blend

        train = pd.concat(day_frames[max(0, i - window_days):i]) if i else None
        if train is not None:
            train = train.dropna(subset=["y"] + lag_cols)
        if (
            train is None
            or len(train) < min_days * 24
            or not np.isfinite(features).all()
        ):
            out.loc[day.index] = fallback
            continue

        X = np.column_stack([np.ones(len(train))] +
                            [train[c].to_numpy(dtype=float) for c in lag_cols])
        y = train["y"].to_numpy(dtype=float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        Xd = np.column_stack([np.ones(len(day)), features])
        out.loc[day.index] = Xd @ beta

    return out


def persistence_day(
    realized: np.ndarray,
    forecast: Optional[np.ndarray],
    p: BatteryParams,
) -> DispatchResult:
    """
    One day of persistence bidding: schedule on the lagged price curve,
    settle at realized prices. Days without a full forecast (start of the
    sample) do nothing.
    """
    if forecast is None or np.any(~np.isfinite(forecast)):
        T = len(realized)
        z = np.zeros(T)
        return DispatchResult(z, z, np.full(T, p.soc_start_mwh), 0.0, 0.0, "no_forecast")
    return scheduled_then_settle(forecast, realized, p)
