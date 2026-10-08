"""
test_realistic.py
-----------------
Executable-tier sanity checks: persistence forecast construction and
schedule-then-settle behavior.
"""

import numpy as np
import pandas as pd
import pytest

from src.optimize.lp_dispatch import BatteryParams
from src.optimize.realistic import (
    build_blend_forecast,
    build_persistence_forecast,
    persistence_day,
    scheduled_then_settle,
)


def _params():
    return BatteryParams(e_max_mwh=1.0, p_max_mw=1.0, eta_round_trip=1.0,
                         throughput_cost_eur_mwh=0.0)


def test_persistence_forecast_is_lagged_by_days():
    idx = pd.date_range("2024-01-01", periods=72, freq="1h", tz="Europe/Berlin")
    s = pd.Series(np.arange(72, dtype=float), index=idx)
    fc = build_persistence_forecast(s, lag_days=1)
    assert fc.iloc[:24].isna().all()
    assert (fc.iloc[24:] == s.iloc[:48].values).all()


def test_persistence_day_without_forecast_does_nothing():
    realized = np.array([10.0, 50.0])
    r = persistence_day(realized, np.array([np.nan, 20.0]), _params())
    assert r.status == "no_forecast"
    assert r.revenue == 0.0
    assert r.throughput_mwh == 0.0


def test_perfect_forecast_equals_perfect_foresight():
    realized = np.array([10.0, 50.0])
    r = scheduled_then_settle(realized, realized, _params())
    assert r.revenue == pytest.approx(40.0, abs=1e-3)


def test_wrong_forecast_never_beats_perfect_foresight():
    rng = np.random.default_rng(7)
    p = _params()
    for _ in range(5):
        realized = rng.normal(100, 40, size=24)
        forecast = rng.normal(100, 40, size=24)
        r_pf = scheduled_then_settle(realized, realized, p)
        r_fc = scheduled_then_settle(forecast, realized, p)
        assert r_fc.revenue <= r_pf.revenue + 1e-6


def _random_price_series(n_days, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n_days * 24, freq="1h",
                        tz="Europe/Berlin")
    return pd.Series(rng.normal(100, 40, size=n_days * 24), index=idx)


def test_blend_has_no_lookahead():
    s = _random_price_series(40)
    fc_a = build_blend_forecast(s, window_days=20, min_days=5)
    s2 = s.copy()
    s2.iloc[30 * 24:] += 500.0  # perturb only the last 10 days
    fc_b = build_blend_forecast(s2, window_days=20, min_days=5)
    # forecasts for days that precede the perturbation must be identical
    pd.testing.assert_series_equal(fc_a.iloc[:30 * 24], fc_b.iloc[:30 * 24])


def test_blend_falls_back_to_equal_weight_before_window_fills():
    s = _random_price_series(10)
    fc = build_blend_forecast(s, lags=(1, 7), window_days=90, min_days=14)
    # day 9 (0-based index 8): both lags known but training too short
    day = slice(8 * 24, 9 * 24)
    expected = (s.shift(24) + s.shift(24 * 7)).iloc[day] / 2.0
    np.testing.assert_allclose(fc.iloc[day].values, expected.values)
    # day 2: only the D-1 lag exists -> fallback degenerates to persistence
    np.testing.assert_allclose(fc.iloc[24:48].values, s.iloc[:24].values)


def test_blend_learns_a_perfectly_periodic_series():
    # daily-periodic prices: the D-1 curve IS the realized curve, so the
    # rolling OLS should reproduce realized prices almost exactly
    idx = pd.date_range("2024-01-01", periods=30 * 24, freq="1h",
                        tz="Europe/Berlin")
    shape = np.tile(100 + 50 * np.sin(np.arange(24) / 24 * 2 * np.pi), 30)
    s = pd.Series(shape, index=idx)
    fc = build_blend_forecast(s, window_days=20, min_days=5)
    tail = slice(20 * 24, None)
    np.testing.assert_allclose(fc.iloc[tail].values, s.iloc[tail].values,
                               atol=1e-6)
