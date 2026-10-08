"""
gbm_shape.py
------------
Gradient-boosted forecast tier (gbm): a nonlinear model on top of the same
D-1-known information set as the linear tiers.

Design choices, all driven by the gap decomposition (decompose.py):

  * The gap is ~100% intra-day *shape* (hour-ranking) error, so the model's
    job is ranking hours, not nailing absolute prices. We still regress on
    the raw price (the LP is invariant to the daily level anyway, except for
    the efficiency threshold), but features emphasize within-day structure:
    the residual-load forecast plus its within-day rank and distance to the
    daily mean, calendar encodings, and the lagged price curves.

  * Walk-forward protocol: refit every `refit_days` on a trailing
    `train_days` window, predict the next block. The training window ends
    strictly before the first predicted delivery day; every feature is known
    before the D-1 12:00 auction. No lookahead.

  * sklearn HistGradientBoostingRegressor: handles NaNs natively (missing
    lag curves at the sample start), no extra dependency weight.

Runtime: ~55 fits over the full sample, a few minutes end to end.
"""

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.config import get_logger

logger = get_logger("gbm_shape")


def build_gbm_features(price: pd.Series, rl: pd.Series,
                       lags: Sequence[int] = (1, 7)) -> pd.DataFrame:
    """
    Hourly feature frame, everything known at the D-1 12:00 auction:

      rl, rl_rank, rl_dev : residual-load forecast (GW), its within-day rank
                            (0..1) and deviation from the daily mean (GW) —
                            the shape carriers.
      hour_sin/cos, dow   : calendar encodings.
      p_d{k}              : lagged realized price curves (persistence info).
      p_d1_rank           : within-day rank of the D-1 curve — yesterday's
                            shape as an explicit ranking feature.
    """
    idx = price.index
    day = pd.Series(idx.date, index=idx)
    rl_gw = rl / 1000.0

    f = pd.DataFrame(index=idx)
    f["rl"] = rl_gw
    f["rl_rank"] = rl_gw.groupby(day).rank(pct=True)
    f["rl_dev"] = rl_gw - rl_gw.groupby(day).transform("mean")
    hour = pd.Series(idx.hour, index=idx, dtype=float)
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    f["dow"] = pd.Series(idx.dayofweek, index=idx, dtype=float)
    for k in lags:
        f["p_d%d" % k] = price.shift(24 * k)
    f["p_d1_rank"] = f["p_d1"].groupby(day).rank(pct=True)
    return f


def build_gbm_forecast(
    price: pd.Series,
    rl: pd.Series,
    lags: Sequence[int] = (1, 7),
    train_days: int = 365,
    min_train_days: int = 90,
    refit_days: int = 30,
    max_iter: int = 300,
    random_state: int = 0,
    target: str = "price",
) -> pd.Series:
    """
    Walk-forward GBM price forecast. Returns a Series aligned to `price`;
    days before `min_train_days` of history, or with incomplete features
    that the model cannot handle, stay NaN (-> no-trade downstream).

    target="price": regress on the raw price (levels + shape mixed).
    target="shape": tell the model the actual game — regress on the
        *demeaned* daily curve (price − realized daily mean), so squared
        error can only be reduced by getting the intra-day shape right.
        The realized daily mean is used in training targets only; at
        prediction time the level is restored from the D-1 curve's daily
        mean (known at the auction), which decompose.py shows is worth
        nothing to the LP anyway.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    if target not in ("price", "shape"):
        raise ValueError("target must be 'price' or 'shape'")

    feats = build_gbm_features(price, rl)
    feat_cols = list(feats.columns)
    frame = feats.copy()
    day_key = pd.Series(price.index.date, index=price.index)
    if target == "shape":
        frame["y"] = price - price.groupby(day_key).transform("mean")
        # D-1-known level to add back at prediction time
        frame["level"] = frame["p_d1"].groupby(day_key).transform("mean")
    else:
        frame["y"] = price
    frame["date"] = frame.index.date

    day_frames = [g for _, g in frame.groupby("date", sort=True)]
    out = pd.Series(np.nan, index=price.index, name="fc_gbm")

    n_fits = 0
    for block_start in range(min_train_days, len(day_frames), refit_days):
        train = pd.concat(day_frames[max(0, block_start - train_days):block_start])
        train = train.dropna(subset=["y", "rl"])  # NaN lags are fine (native NaN support)
        if len(train) < min_train_days * 24:
            continue
        model = HistGradientBoostingRegressor(
            max_iter=max_iter, random_state=random_state)
        model.fit(train[feat_cols].to_numpy(dtype=float),
                  train["y"].to_numpy(dtype=float))
        n_fits += 1

        block = day_frames[block_start:block_start + refit_days]
        for day in block:
            X = day[feat_cols].to_numpy(dtype=float)
            if not np.isfinite(X[:, 0]).all():  # rl missing -> no-trade
                continue
            pred = model.predict(X)
            if target == "shape":
                level = day["level"].to_numpy(dtype=float)
                if not np.isfinite(level).all():  # no D-1 level yet
                    continue
                pred = pred + level
            out.loc[day.index] = pred
    logger.info("gbm walk-forward done: %d fits, %d forecast hours",
                n_fits, int(out.notna().sum()))
    return out
