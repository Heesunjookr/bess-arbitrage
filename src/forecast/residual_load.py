"""
residual_load.py
----------------
Fundamental forecast tiers built on the SMARD D-1 residual-load forecast.

Why residual load: the DA price shape in Germany is set almost entirely by
residual load (load − wind − solar) walking up and down the merit-order
curve. The TSO day-ahead forecasts of all three components are published
before the D-1 12:00 auction, so a residual-load curve for delivery day D is
a *known* feature at bidding time — unlike day D's prices.

Tiers (both rolling-OLS calibrated, same protocol as the blend tier:
trailing `window_days`, refit per delivery day, zero lookahead):

  rl_quad  : price ~ 1 + RL + RL²        — pure fundamental, no price history.
             The quadratic term captures the convexity of the merit order.
  blend_rl : price ~ 1 + P(D-1) + P(D-7) + RL + RL²
             — the blend tier plus fundamentals; the natural "next tier up".

Days whose regressors are incomplete yield NaN -> no-trade downstream,
mirroring the persistence tiers' no_forecast behavior.
"""

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.config import ROOT, get_logger, load_config

logger = get_logger("residual_load")

FORECAST_PATH = "data/raw/smard_forecasts.parquet"


def load_residual_load_fc(cfg: Optional[dict] = None,
                          price_index: Optional[pd.DatetimeIndex] = None) -> pd.Series:
    """
    Hourly residual-load forecast (MWh), tz-aware. If `price_index` is given,
    reindex onto it (missing hours stay NaN -> no-trade days downstream).
    """
    cfg = cfg or load_config()
    df = pd.read_parquet(ROOT / FORECAST_PATH)
    s = df["residual_load_fc"]
    if price_index is not None:
        s = s.reindex(price_index)
    s.name = "residual_load_fc"
    return s


def build_rolling_ols_forecast(
    price: pd.Series,
    features: pd.DataFrame,
    window_days: int = 90,
    min_days: int = 14,
    name: str = "fc_ols",
) -> pd.Series:
    """
    Generic per-day rolling-OLS forecast, the same walk-forward protocol as
    build_blend_forecast: for delivery day D, regress the realized prices of
    the trailing `window_days` days on the feature columns (plus intercept),
    then apply the fitted weights to day D's features.

    Every feature column must be known before the D-1 12:00 auction (lagged
    prices, D-1 published forecasts). The training window ends at day D-1,
    whose prices cleared at D-2 — no lookahead.

    Days with insufficient training data or non-finite features return NaN.
    """
    frame = features.copy()
    frame["y"] = price
    frame["date"] = frame.index.date
    feat_cols = [c for c in frame.columns if c not in ("y", "date")]

    out = pd.Series(np.nan, index=price.index, name=name)
    day_frames = [g for _, g in frame.groupby("date", sort=True)]

    for i, day in enumerate(day_frames):
        X_day = day[feat_cols].to_numpy(dtype=float)
        if not np.isfinite(X_day).all():
            continue
        train = pd.concat(day_frames[max(0, i - window_days):i]) if i else None
        if train is not None:
            train = train.dropna(subset=["y"] + feat_cols)
        if train is None or len(train) < min_days * 24:
            continue
        X = np.column_stack([np.ones(len(train)),
                             train[feat_cols].to_numpy(dtype=float)])
        y = train["y"].to_numpy(dtype=float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        out.loc[day.index] = np.column_stack([np.ones(len(day)), X_day]) @ beta

    return out


def _rl_features(rl: pd.Series) -> pd.DataFrame:
    """RL and RL² in GW-scale units (numerical conditioning of the OLS)."""
    rl_gw = rl / 1000.0
    return pd.DataFrame({"rl": rl_gw, "rl2": rl_gw ** 2})


def build_rl_quad_forecast(price: pd.Series, rl: pd.Series,
                           window_days: int = 90, min_days: int = 14) -> pd.Series:
    """Tier rl_quad: price ~ 1 + RL + RL² (no price history at all)."""
    return build_rolling_ols_forecast(
        price, _rl_features(rl), window_days, min_days, name="fc_rl_quad")


def build_blend_rl_forecast(price: pd.Series, rl: pd.Series,
                            lags: Sequence[int] = (1, 7),
                            window_days: int = 90, min_days: int = 14) -> pd.Series:
    """Tier blend_rl: price ~ 1 + lagged price curves + RL + RL²."""
    feats = _rl_features(rl)
    for k in lags:
        feats["p_d%d" % k] = price.shift(24 * k)
    return build_rolling_ols_forecast(
        price, feats, window_days, min_days, name="fc_blend_rl")
