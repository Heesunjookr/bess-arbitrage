from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.optimize.lp_dispatch import BatteryParams
from src.webapp.dashboard import artifact_status, available_dates, compare_dispatch_day, forecast_ladder


def _two_days() -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=48, freq="h", tz="Europe/Berlin")
    prices = np.tile(np.arange(24, dtype=float), 2)
    return pd.DataFrame({"da_price": prices, "date": index.date}, index=index)


def test_forecast_ladder_formats_percentages():
    raw = pd.DataFrame(
        {"tier": ["persist_d1", "blend_rl"], "n_days": [100, 100],
         "capture": [.77, .92], "gap_recovered_vs_persist_d1": [0, .65]}
    )
    result = forecast_ladder(raw)
    assert result["capture_pct"].tolist() == [77, 92]
    assert result.iloc[1]["label"] == "Price + residual-load OLS"
    assert result.iloc[1]["information_set"] == "D-1, D-7 and residual-load forecast"


def test_artifact_status_reports_missing_and_latest(tmp_path):
    ready = tmp_path / "ready.csv"
    ready.write_text("ok\n")
    result = artifact_status([ready, tmp_path / "missing.json"])
    assert result["status"] == "incomplete"
    assert result["artifact_count"] == 1
    assert result["missing"] == ["missing.json"]


def test_available_dates_requires_complete_previous_day():
    assert available_dates(_two_days()) == [date(2026, 1, 2)]


def test_day_comparison_is_auditable():
    frame, metrics = compare_dispatch_day(
        _two_days(), date(2026, 1, 2), BatteryParams()
    )
    assert len(frame) == 24
    assert metrics["perfect_foresight_eur"] >= metrics["persistence_eur"]
    assert frame["d1_price_curve"].tolist() == list(np.arange(24, dtype=float))


def test_day_comparison_rejects_missing_curve():
    with pytest.raises(ValueError):
        compare_dispatch_day(_two_days().iloc[:-1], date(2026, 1, 2), BatteryParams())
