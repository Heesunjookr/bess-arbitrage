"""
test_decompose.py
-----------------
Level/shape gap-decomposition sanity checks.
"""

import numpy as np
import pytest

from src.backtest.decompose import decompose_day
from src.optimize.lp_dispatch import BatteryParams


def _params(**kw):
    base = dict(e_max_mwh=2.0, p_max_mw=1.0, eta_round_trip=0.85,
                throughput_cost_eur_mwh=2.0)
    base.update(kw)
    return BatteryParams(**base)


def test_decomposition_identity_holds():
    rng = np.random.default_rng(11)
    p = _params()
    for _ in range(5):
        realized = rng.normal(100, 40, size=24)
        forecast = rng.normal(100, 40, size=24)
        d = decompose_day(realized, forecast, p)
        assert d["gap"] == pytest.approx(d["level_cost"] + d["shape_cost"],
                                         abs=1e-6)


def test_pure_level_error_has_zero_shape_cost():
    # forecast = realized + constant: shape is perfect, only the level is off,
    # so the level-corrected forecast equals realized -> shape_cost == 0
    rng = np.random.default_rng(3)
    realized = rng.normal(100, 40, size=24)
    forecast = realized + 100.0
    d = decompose_day(realized, forecast, _params())
    assert d["shape_cost"] == pytest.approx(0.0, abs=1e-4)


def test_pure_level_error_is_free_for_lossless_battery():
    # with eta=1 and no throughput cost, a constant price shift does not
    # change the optimal schedule at all -> the whole gap vanishes
    rng = np.random.default_rng(5)
    realized = rng.normal(100, 40, size=24)
    forecast = realized + 100.0
    p = _params(eta_round_trip=1.0, throughput_cost_eur_mwh=0.0)
    d = decompose_day(realized, forecast, p)
    assert d["gap"] == pytest.approx(0.0, abs=1e-4)


def test_pure_shape_error_has_zero_level_cost():
    # forecast with the correct daily mean: level correction is a no-op,
    # the entire gap is shape error
    rng = np.random.default_rng(9)
    realized = rng.normal(100, 40, size=24)
    forecast = rng.normal(100, 40, size=24)
    forecast = forecast - forecast.mean() + realized.mean()
    d = decompose_day(realized, forecast, _params())
    assert d["level_cost"] == pytest.approx(0.0, abs=1e-6)
