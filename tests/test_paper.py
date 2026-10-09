"""
test_paper.py
-------------
Paper-trading guards: the bid creator must refuse to bid after the D-1
12:00 gate closure (that would be trading on known prices), must not
double-bid a delivery day, and must freeze a valid bid before the gate.
"""

import json

import numpy as np
import pandas as pd
import pytest

import src.paper.paper_trade as pt
from src.optimize.lp_dispatch import BatteryParams


def _params():
    return BatteryParams(e_max_mwh=1.0, p_max_mw=1.0, eta_round_trip=1.0,
                         throughput_cost_eur_mwh=0.0)


def _live_frame(now):
    """Two complete days: today (price known) + tomorrow (price NaN)."""
    idx = pd.date_range(now.normalize(), periods=48, freq="1h",
                        tz="Europe/Berlin")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "da_price": np.concatenate([rng.normal(100, 40, 24),
                                    np.full(24, np.nan)]),
        "residual_load_fc": rng.normal(50_000, 10_000, 48),
    }, index=idx)
    df["date"] = df.index.date
    return df


def _price_only_live_frame(now):
    df = _live_frame(now)
    df.loc[df.index[-24:], "residual_load_fc"] = np.nan
    return df


def test_bid_refused_after_gate_closure(tmp_path, monkeypatch):
    monkeypatch.setattr(pt, "ROOT", tmp_path)
    now = pd.Timestamp("2026-07-04 13:00", tz="Europe/Berlin")
    assert pt.create_bid(_live_frame(now), _params(), now=now) is None
    assert not list((tmp_path / pt.PAPER_DIR / "bids").glob("*.json"))


def test_bid_frozen_before_gate_closure(tmp_path, monkeypatch):
    monkeypatch.setattr(pt, "ROOT", tmp_path)
    now = pd.Timestamp("2026-07-04 10:30", tz="Europe/Berlin")
    path = pt.create_bid(_live_frame(now), _params(), now=now)
    assert path is not None and path.exists()
    bid = json.loads(path.read_text())
    assert bid["delivery_day"] == "2026-07-05"
    # persistence tier must be live (D-1 curve known); its schedule is 24h
    t = bid["tiers"]["persist_d1"]
    assert t["status"] == "optimal"
    assert len(t["charge"]) == 24 and len(t["forecast"]) == 24
    # frozen forecast == today's price curve (persistence, lag 1 day)
    frame = _live_frame(now)
    np.testing.assert_allclose(t["forecast"],
                               frame["da_price"].iloc[:24].round(4).values)


def test_bid_not_created_twice(tmp_path, monkeypatch):
    monkeypatch.setattr(pt, "ROOT", tmp_path)
    now = pd.Timestamp("2026-07-04 10:30", tz="Europe/Berlin")
    df = _live_frame(now)
    p1 = pt.create_bid(df, _params(), now=now)
    mtime = p1.stat().st_mtime_ns
    p2 = pt.create_bid(df, _params(), now=now + pd.Timedelta(minutes=30))
    assert p2 == p1 and p2.stat().st_mtime_ns == mtime


def test_price_tiers_are_frozen_when_public_fundamentals_are_late(tmp_path, monkeypatch):
    monkeypatch.setattr(pt, "ROOT", tmp_path)
    now = pd.Timestamp("2026-07-04 10:30", tz="Europe/Berlin")
    path = pt.create_bid(_price_only_live_frame(now), _params(), now=now)
    bid = json.loads(path.read_text())
    assert bid["tiers"]["persist_d1"]["status"] == "optimal"
    assert bid["tiers"]["blend"]["status"] == "optimal"
    assert bid["tiers"]["rl_quad"]["status"] == "no_forecast"
    assert bid["tiers"]["blend_rl"]["status"] == "no_forecast"
