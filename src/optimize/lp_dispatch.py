"""
lp_dispatch.py
--------------
Battery arbitrage as a linear program.

One battery is dispatch-optimized over a single day (24h window).

Decision variables (per hour t):
  c_t >= 0 : charging power (MW)
  d_t >= 0 : discharging power (MW)
  s_t      : state of charge (MWh)

Objective (maximize revenue):
  max  sum_t [ price_t * d_t * eta_dis - price_t * c_t ]  -  lambda * sum_t (c_t + d_t)
       (discharge revenue)   (charging cost)                 (throughput/degradation penalty)

Constraints:
  SoC dynamics:  s_t = s_{t-1} + c_t * eta_ch - d_t / eta_dis
  0 <= c_t, d_t <= P_max
  0 <= s_t <= E_max
  s_{-1} = soc_start,  s_{T-1} = soc_end (end-of-day SoC fixed)

NOTE: dt = 1h, so power (MW) and energy (MWh) are numerically interchangeable.
The same price is used for buying and selling -> pure time arbitrage, not a
bid-ask spread play.
"""

from dataclasses import dataclass
from typing import Optional

import cvxpy as cp
import numpy as np


@dataclass
class BatteryParams:
    """Physical/cost parameters. Built from the `battery` block of settings.yaml."""
    e_max_mwh: float = 2.0
    p_max_mw: float = 1.0
    eta_round_trip: float = 0.85
    throughput_cost_eur_mwh: float = 2.0
    soc_start_mwh: float = 0.0
    soc_end_mwh: float = 0.0

    @property
    def eta_ch(self) -> float:
        """Charging efficiency = sqrt(round-trip), symmetric split."""
        return float(np.sqrt(self.eta_round_trip))

    @property
    def eta_dis(self) -> float:
        """Discharging efficiency = sqrt(round-trip)."""
        return float(np.sqrt(self.eta_round_trip))

    @classmethod
    def from_config(cls, cfg: dict) -> "BatteryParams":
        b = cfg["battery"]
        return cls(
            e_max_mwh=float(b["e_max_mwh"]),
            p_max_mw=float(b["p_max_mw"]),
            eta_round_trip=float(b["eta_round_trip"]),
            throughput_cost_eur_mwh=float(b["throughput_cost_eur_mwh"]),
            soc_start_mwh=float(b["soc_start_mwh"]),
            soc_end_mwh=float(b["soc_end_mwh"]),
        )


@dataclass
class DispatchResult:
    """Result of one day's dispatch optimization."""
    charge: np.ndarray       # c_t (MW), length T
    discharge: np.ndarray    # d_t (MW)
    soc: np.ndarray          # s_t (MWh), length T (end-of-hour SoC)
    revenue: float           # net revenue settled at realized prices (EUR)
    throughput_mwh: float    # sum(c_t + d_t), throughput (MWh)
    status: str              # solver status


def _settled_revenue(
    prices: np.ndarray,
    charge: np.ndarray,
    discharge: np.ndarray,
    p: BatteryParams,
) -> float:
    """Net revenue (EUR) of a given dispatch settled at `prices`."""
    gross = np.sum(prices * discharge * p.eta_dis - prices * charge)
    penalty = p.throughput_cost_eur_mwh * np.sum(charge + discharge)
    return float(gross - penalty)


def solve_day(
    prices: np.ndarray,
    p: BatteryParams,
    solver: Optional[str] = None,
) -> DispatchResult:
    """
    Solve the LP for one day's price vector (length T=24) and return the
    optimal dispatch.

    Optimizing against realized prices == perfect foresight. The executable
    tiers optimize on a forecast and settle at realized prices; that split
    lives in realistic.py — this function is pure single-vector optimization.
    """
    T = len(prices)
    c = cp.Variable(T, nonneg=True, name="charge")
    d = cp.Variable(T, nonneg=True, name="discharge")
    s = cp.Variable(T, name="soc")

    revenue = cp.sum(cp.multiply(prices, d) * p.eta_dis - cp.multiply(prices, c))
    penalty = p.throughput_cost_eur_mwh * cp.sum(c + d)
    objective = cp.Maximize(revenue - penalty)

    cons = [
        c <= p.p_max_mw,
        d <= p.p_max_mw,
        s >= 0,
        s <= p.e_max_mwh,
    ]
    # SoC dynamics (dt=1h): s_0 = soc_start + c_0*eta_ch - d_0/eta_dis
    cons.append(s[0] == p.soc_start_mwh + c[0] * p.eta_ch - d[0] / p.eta_dis)
    for t in range(1, T):
        cons.append(s[t] == s[t - 1] + c[t] * p.eta_ch - d[t] / p.eta_dis)
    # end-of-day SoC fixed
    cons.append(s[T - 1] == p.soc_end_mwh)

    prob = cp.Problem(objective, cons)
    prob.solve(solver=solver)

    if c.value is None:
        # infeasible/failed solve (rare): fall back to doing nothing
        zeros = np.zeros(T)
        return DispatchResult(zeros, zeros, np.full(T, p.soc_start_mwh), 0.0, 0.0, prob.status)

    charge = np.clip(np.asarray(c.value).ravel(), 0, None)
    discharge = np.clip(np.asarray(d.value).ravel(), 0, None)
    soc = np.asarray(s.value).ravel()
    rev = _settled_revenue(prices, charge, discharge, p)
    tput = float(np.sum(charge + discharge))
    return DispatchResult(charge, discharge, soc, rev, tput, prob.status)


def settle_schedule(
    realized_prices: np.ndarray,
    charge: np.ndarray,
    discharge: np.ndarray,
    p: BatteryParams,
) -> float:
    """Revenue of an externally fixed (charge, discharge) schedule settled at realized_prices."""
    return _settled_revenue(realized_prices, charge, discharge, p)
