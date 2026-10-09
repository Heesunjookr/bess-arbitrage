"""
test_lp.py
----------
LP dispatch sanity checks:
- charges low / discharges high on a toy price vector
- SoC stays within [0, E_max]
- perfect-foresight revenue is >= 0 (doing nothing is always feasible)
"""

import numpy as np
import pytest

from src.optimize.lp_dispatch import BatteryParams, solve_day, settle_schedule


def _toy_params(throughput_cost=0.0):
    # 1 MW / 1 MWh, 1-hour battery, lossless
    return BatteryParams(
        e_max_mwh=1.0,
        p_max_mw=1.0,
        eta_round_trip=1.0,
        throughput_cost_eur_mwh=throughput_cost,
        soc_start_mwh=0.0,
        soc_end_mwh=0.0,
    )


def test_buys_low_sells_high():
    # 2 hours: cheap (10) then expensive (50) -> charge, then discharge
    prices = np.array([10.0, 50.0])
    p = _toy_params()
    r = solve_day(prices, p)
    assert r.status in ("optimal", "optimal_inaccurate")
    # charge in hour 1, discharge in hour 2
    assert r.charge[0] > 0.99
    assert r.discharge[1] > 0.99
    # lossless: revenue ~= (50-10) * 1 MWh = 40
    assert r.revenue == pytest.approx(40.0, abs=1e-3)


def test_soc_bounds():
    rng = np.random.default_rng(0)
    prices = rng.normal(100, 50, size=24)
    p = BatteryParams(e_max_mwh=2.0, p_max_mw=1.0, eta_round_trip=0.85,
                      throughput_cost_eur_mwh=2.0)
    r = solve_day(prices, p)
    assert r.soc.min() >= -1e-6
    assert r.soc.max() <= p.e_max_mwh + 1e-6
    # end-of-day SoC = 0
    assert abs(r.soc[-1] - p.soc_end_mwh) < 1e-6


def test_revenue_nonnegative():
    # perfect-foresight revenue can never be negative (no-trade is feasible)
    rng = np.random.default_rng(1)
    for _ in range(5):
        prices = rng.normal(80, 40, size=24)
        p = _toy_params(throughput_cost=2.0)
        r = solve_day(prices, p)
        assert r.revenue >= -1e-6


def test_flat_prices_no_trade():
    # flat prices + throughput cost -> no trading (zero revenue)
    prices = np.full(24, 75.0)
    p = _toy_params(throughput_cost=1.0)
    r = solve_day(prices, p)
    assert r.revenue == pytest.approx(0.0, abs=1e-3)
    assert r.throughput_mwh == pytest.approx(0.0, abs=1e-3)


def test_efficiency_loss_reduces_revenue():
    # round-trip losses shrink revenue for the same price spread
    prices = np.array([10.0, 50.0])
    r_lossless = solve_day(prices, _toy_params())
    p_lossy = BatteryParams(e_max_mwh=1.0, p_max_mw=1.0, eta_round_trip=0.85,
                            throughput_cost_eur_mwh=0.0)
    r_lossy = solve_day(prices, p_lossy)
    assert r_lossy.revenue < r_lossless.revenue


def test_round_trip_efficiency_is_applied_exactly_once():
    # AC-side convention: buy 1/eta_ch MWh to fill a 1 MWh battery, then
    # export eta_dis MWh. With eta_rt=.81, eta_ch=eta_dis=.9.
    prices = np.array([10.0, 100.0])
    p = BatteryParams(e_max_mwh=1.0, p_max_mw=2.0, eta_round_trip=0.81,
                      throughput_cost_eur_mwh=0.0)
    r = solve_day(prices, p)
    expected = 100.0 * 0.9 - 10.0 / 0.9
    assert r.revenue == pytest.approx(expected, abs=1e-3)
    assert r.charge[0] == pytest.approx(1.0 / 0.9, abs=1e-3)
    assert r.discharge[1] == pytest.approx(0.9, abs=1e-3)


def test_settle_schedule_matches():
    # settle_schedule must reproduce solve_day's revenue
    prices = np.array([20.0, 30.0, 80.0])
    p = _toy_params()
    r = solve_day(prices, p)
    rev = settle_schedule(prices, r.charge, r.discharge, p)
    assert rev == pytest.approx(r.revenue, abs=1e-6)


def test_negative_prices_never_cause_simultaneous_charge_discharge():
    # An unconstrained LP would profit from a loss-making internal loop below
    # this threshold. The production solver activates charge/discharge mode
    # binaries on negative-price days and must remove that physical artifact.
    from src.backtest.lp_artifact_check import break_even_price

    p = BatteryParams(e_max_mwh=2.0, p_max_mw=1.0, eta_round_trip=0.85,
                      throughput_cost_eur_mwh=2.0)
    be = break_even_price(p)
    assert be == pytest.approx(-24.7, abs=0.5)

    # Deeply negative block, below the unconstrained break-even threshold.
    deep = np.array([-200.0] * 6 + [50.0] * 18)
    r = solve_day(deep, p)
    both = (r.charge > 1e-4) & (r.discharge > 1e-4)
    assert not both.any()

    # Mildly negative block must also remain mutually exclusive.
    mild = np.array([-10.0] * 6 + [50.0] * 18)
    r = solve_day(mild, p)
    both = (r.charge > 1e-4) & (r.discharge > 1e-4)
    assert not both.any()
