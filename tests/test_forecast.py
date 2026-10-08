"""
test_forecast.py
----------------
Forecast-layer sanity checks: gap-analysis descriptors, the generic
rolling-OLS forecaster (fundamental tiers) and the GBM walk-forward tier —
above all that none of them can see the future.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.gap_analysis import _rank_corr, shape_change_metrics
from src.forecast.gbm_shape import build_gbm_features, build_gbm_forecast
from src.forecast.residual_load import (
    build_rl_quad_forecast,
    build_rolling_ols_forecast,
)


def _hourly_index(n_days, start="2024-01-01"):
    return pd.date_range(start, periods=n_days * 24, freq="1h",
                         tz="Europe/Berlin")


def _price_frame(n_days, seed=0):
    rng = np.random.default_rng(seed)
    idx = _hourly_index(n_days)
    df = pd.DataFrame({"da_price": rng.normal(100, 40, size=len(idx))}, index=idx)
    df["date"] = df.index.date
    return df


# ---------------------------------------------------------------- gap analysis

def test_rank_corr_bounds_and_signs():
    a = np.arange(24, dtype=float)
    assert _rank_corr(a, a) == pytest.approx(1.0)
    assert _rank_corr(a, -a) == pytest.approx(-1.0)
    assert np.isnan(_rank_corr(a, np.ones(24)))  # constant curve -> undefined


def test_shape_change_metrics_flags_shape_breaks():
    df = _price_frame(4, seed=1)
    base = 100 + 50 * np.sin(np.arange(24) / 24 * 2 * np.pi)
    df["da_price"] = np.concatenate([base, base, -base, base])
    m = shape_change_metrics(df)
    assert np.isnan(m["rank_corr_d1"].iloc[0])       # no previous day
    assert m["rank_corr_d1"].iloc[1] == pytest.approx(1.0)   # identical shape
    assert m["rank_corr_d1"].iloc[2] == pytest.approx(-1.0)  # inverted shape
    assert (m["spread"] > 0).all()


def test_shape_change_metrics_skips_non_consecutive_days():
    df = _price_frame(3, seed=2)
    df = df[df["date"] != df["date"].unique()[1]]    # drop the middle day
    m = shape_change_metrics(df)
    assert np.isnan(m["rank_corr_d1"]).all()         # day 3 vs day 1: not D-1


# ---------------------------------------------------------------- rolling OLS

def test_rolling_ols_recovers_linear_relation():
    # price = 2*x + 10 exactly -> after the window fills, forecast == price
    idx = _hourly_index(30)
    rng = np.random.default_rng(3)
    x = pd.Series(rng.normal(50, 10, size=len(idx)), index=idx)
    price = 2.0 * x + 10.0
    fc = build_rolling_ols_forecast(price, pd.DataFrame({"x": x}),
                                    window_days=10, min_days=5)
    tail = slice(10 * 24, None)
    np.testing.assert_allclose(fc.iloc[tail].values, price.iloc[tail].values,
                               rtol=1e-6)


def test_rolling_ols_has_no_lookahead():
    idx = _hourly_index(40)
    rng = np.random.default_rng(4)
    x = pd.Series(rng.normal(50, 10, size=len(idx)), index=idx)
    price = 2.0 * x + rng.normal(0, 5, size=len(idx))
    fc_a = build_rolling_ols_forecast(price, pd.DataFrame({"x": x}),
                                      window_days=20, min_days=5)
    price2 = price.copy()
    price2.iloc[30 * 24:] += 500.0
    fc_b = build_rolling_ols_forecast(price2, pd.DataFrame({"x": x}),
                                      window_days=20, min_days=5)
    pd.testing.assert_series_equal(fc_a.iloc[:30 * 24], fc_b.iloc[:30 * 24])


def test_rolling_ols_nan_features_yield_no_forecast_day():
    idx = _hourly_index(20)
    rng = np.random.default_rng(5)
    x = pd.Series(rng.normal(50, 10, size=len(idx)), index=idx)
    x.iloc[15 * 24 + 3] = np.nan                     # one bad hour on day 16
    price = 2.0 * x.fillna(50.0) + 10.0
    fc = build_rolling_ols_forecast(price, pd.DataFrame({"x": x}),
                                    window_days=10, min_days=5)
    day16 = slice(15 * 24, 16 * 24)
    assert fc.iloc[day16].isna().all()               # whole day -> no-trade
    assert fc.iloc[16 * 24:].notna().all()           # later days unaffected


def test_rl_quad_start_of_sample_is_nan():
    idx = _hourly_index(20)
    rng = np.random.default_rng(6)
    rl = pd.Series(rng.normal(50_000, 10_000, size=len(idx)), index=idx)
    price = pd.Series(rng.normal(100, 40, size=len(idx)), index=idx)
    fc = build_rl_quad_forecast(price, rl, window_days=10, min_days=5)
    assert fc.iloc[:5 * 24].isna().all()             # window not filled yet
    assert fc.iloc[10 * 24:].notna().all()


# ------------------------------------------------------------------------ GBM

def test_gbm_features_are_within_day_relative():
    idx = _hourly_index(3)
    rng = np.random.default_rng(7)
    price = pd.Series(rng.normal(100, 40, size=len(idx)), index=idx)
    rl = pd.Series(rng.normal(50_000, 10_000, size=len(idx)), index=idx)
    f = build_gbm_features(price, rl)
    day = pd.Series(idx.date, index=idx)
    # rl_dev sums to ~0 within each day; rank spans (0, 1]
    assert np.allclose(f["rl_dev"].groupby(day).sum(), 0.0, atol=1e-9)
    assert f["rl_rank"].max() <= 1.0 and f["rl_rank"].min() > 0.0
    assert f["p_d1"].iloc[:24].isna().all()


def test_gbm_shape_target_has_no_lookahead_and_restores_level():
    n = 40
    idx = _hourly_index(n)
    rng = np.random.default_rng(9)
    rl = pd.Series(rng.normal(50_000, 10_000, size=len(idx)), index=idx)
    price = pd.Series(0.002 * rl + rng.normal(0, 5, size=len(idx)), index=idx)
    kw = dict(train_days=20, min_train_days=10, refit_days=5, max_iter=30,
              target="shape")
    fc_a = build_gbm_forecast(price, rl, **kw)
    price2 = price.copy()
    price2.iloc[35 * 24:] += 500.0
    fc_b = build_gbm_forecast(price2, rl, **kw)
    pd.testing.assert_series_equal(fc_a.iloc[:35 * 24], fc_b.iloc[:35 * 24])
    # level comes from the D-1 daily mean, so day-means track yesterday's
    day = pd.Series(idx.date, index=idx)
    fc_mean = fc_a.groupby(day).mean().dropna()
    d1_mean = price.groupby(day).mean().shift(1)
    resid = (fc_mean - d1_mean.loc[fc_mean.index]).abs()
    assert resid.mean() < price.std()  # bounded by shape scale, not level scale


def test_gbm_forecast_has_no_lookahead():
    n = 40
    idx = _hourly_index(n)
    rng = np.random.default_rng(8)
    rl = pd.Series(rng.normal(50_000, 10_000, size=len(idx)), index=idx)
    price = pd.Series(0.002 * rl + rng.normal(0, 5, size=len(idx)), index=idx)
    kw = dict(train_days=20, min_train_days=10, refit_days=5, max_iter=30)
    fc_a = build_gbm_forecast(price, rl, **kw)
    price2 = price.copy()
    price2.iloc[35 * 24:] += 500.0                   # perturb the last 5 days
    fc_b = build_gbm_forecast(price2, rl, **kw)
    pd.testing.assert_series_equal(fc_a.iloc[:35 * 24], fc_b.iloc[:35 * 24])
    assert fc_a.iloc[:10 * 24].isna().all()          # before min_train_days
