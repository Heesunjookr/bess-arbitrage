"""
test_health.py
--------------
The pre-flight health gate must catch exactly the failure classes the
project's post-mortems produced: mislabeled series (no duck curve, solar at
night), broken identities, dead series — and must pass on healthy data.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.health import DataHealthError, run_health_checks


def _healthy_data(n_days=30, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n_days * 24, freq="1h",
                        tz="Europe/Berlin")
    hour = idx.hour.to_numpy()
    duck = 40 * np.sin((hour - 3) / 24 * 2 * np.pi)  # pronounced daily shape
    price = pd.Series(100 + duck + rng.normal(0, 10, len(idx)), index=idx)

    load = pd.Series(rng.normal(55_000, 5_000, len(idx)), index=idx)
    pv = pd.Series(np.where((hour >= 8) & (hour <= 18),
                            rng.uniform(0, 30_000, len(idx)), 0.0), index=idx)
    won = pd.Series(rng.uniform(0, 30_000, len(idx)), index=idx)
    woff = pd.Series(rng.uniform(0, 7_000, len(idx)), index=idx)
    fc = pd.DataFrame({"load_fc": load, "pv_fc": pv, "wind_on_fc": won,
                       "wind_off_fc": woff})
    fc["residual_load_fc"] = load - pv - won - woff
    return price, fc


def test_healthy_data_passes():
    price, fc = _healthy_data()
    report = run_health_checks(price, fc)
    assert report["passed"].all()


def test_flat_price_fails_duck_curve_check():
    price, fc = _healthy_data()
    flat = pd.Series(100.0, index=price.index)  # the Norway-NO2 signature
    with pytest.raises(DataHealthError, match="duck_curve"):
        run_health_checks(flat, fc)


def test_broken_residual_identity_fails():
    price, fc = _healthy_data()
    fc = fc.copy()
    fc.loc[fc.index[100:110], "residual_load_fc"] += 500.0
    with pytest.raises(DataHealthError, match="residual_identity"):
        run_health_checks(price, fc)


def test_solar_at_night_fails():
    price, fc = _healthy_data()
    fc = fc.copy()
    fc.loc[fc.index.hour == 2, "pv_fc"] = 5_000.0  # not a PV series
    with pytest.raises(DataHealthError, match="pv_dark_at_night"):
        run_health_checks(price, fc)


def test_dead_series_fails():
    price, fc = _healthy_data()
    fc = fc.copy()
    fc["wind_on_fc"] = 12_000.0
    fc["residual_load_fc"] = (fc["load_fc"] - fc["pv_fc"]
                              - fc["wind_on_fc"] - fc["wind_off_fc"])
    with pytest.raises(DataHealthError, match="not_degenerate"):
        run_health_checks(price, fc)


def test_non_strict_mode_reports_without_raising():
    price, fc = _healthy_data()
    flat = pd.Series(100.0, index=price.index)
    report = run_health_checks(flat, fc, strict=False)
    assert not report.loc["price_duck_curve", "passed"]
