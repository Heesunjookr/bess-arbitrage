"""
health.py
---------
Pre-flight data health gate — runs automatically before data is consumed.

Every data post-mortem in this repo ended the same way: the forensics became
a permanent check (mislabeled "intraday" series -> validate.py, forecast
labeling -> validate_forecasts.py, DST bucketing -> the SQL equivalence
test). This module is the operational layer on top: the *fast, local* subset
of those checks bundled into one gate that pipelines call before touching
the data — the backtest runner (src/backtest/forecast_tiers.py) and the
live paper trader (src/paper/paper_trade.py) both refuse to run on data
that fails it. Slow or networked forensics stay in their own modules.

Price checks (mislabeling + corruption guards):
  price_no_duplicates      duplicate timestamps
  price_plausible_range    within EPEX DA technical bounds (-500..4000)
  price_duck_curve         hour-of-day profile has a real daily shape —
                           the check that would have caught the NO2 series
  price_nan_share          bounded missing data

Forecast checks (labeling + integrity guards):
  fc_residual_identity     residual_load_fc == load - pv - wind_on - wind_off
  fc_plausible_magnitudes  load in 20..90 GW, generation non-negative
  fc_pv_dark_at_night      PV forecast ~0 at local 00-03h — a series that
                           produces solar at night is not a PV series
  fc_not_degenerate        no constant/dead series
  fc_coverage              covers enough of the price index
  fc_freshness             extends at least as far as the price series

Usage:
    report = run_health_checks(price, forecasts)   # raises DataHealthError
    run_health_checks(price, forecasts, strict=False)  # log-only

Run standalone:  python -m src.data.health
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import get_logger

logger = get_logger("health")

PRICE_MIN, PRICE_MAX = -500.0, 4000.0   # EPEX SDAC technical price bounds
DUCK_RANGE_MIN_EUR = 10.0               # mean hour-of-day profile range
PRICE_NAN_MAX = 0.05
LOAD_GW_MIN, LOAD_GW_MAX = 20.0, 90.0   # German grid load envelope
PV_NIGHT_MAX_MWH = 200.0                # PV at 00-03h local (should be ~0)
RL_IDENTITY_TOL = 1.0                   # MWh
FC_COVERAGE_MIN = 0.95


class DataHealthError(RuntimeError):
    """Raised when any strict health check fails."""


def _check(name: str, passed: bool, detail: str) -> Dict:
    return {"check": name, "passed": bool(passed), "detail": detail}


def check_price(price: pd.Series) -> List[Dict]:
    """Fast integrity + mislabeling checks on an hourly DA price series."""
    out = []
    dups = int(price.index.duplicated().sum())
    out.append(_check("price_no_duplicates", dups == 0, "%d duplicates" % dups))

    p = price.dropna()
    oob = int(((p < PRICE_MIN) | (p > PRICE_MAX)).sum())
    out.append(_check("price_plausible_range", oob == 0,
                      "%d hours outside [%.0f, %.0f]" % (oob, PRICE_MIN, PRICE_MAX)))

    profile = p.groupby(p.index.hour).mean()
    rng = float(profile.max() - profile.min())
    out.append(_check("price_duck_curve", rng >= DUCK_RANGE_MIN_EUR,
                      "hour-of-day profile range %.1f EUR (min %.0f)"
                      % (rng, DUCK_RANGE_MIN_EUR)))

    nan_share = float(price.isna().mean())
    out.append(_check("price_nan_share", nan_share <= PRICE_NAN_MAX,
                      "NaN share %.3f (max %.2f)" % (nan_share, PRICE_NAN_MAX)))
    return out


def check_forecasts(fc: pd.DataFrame,
                    price_index: Optional[pd.DatetimeIndex] = None) -> List[Dict]:
    """
    Integrity + labeling checks on the SMARD forecast frame. Expects columns
    load_fc, pv_fc, wind_on_fc, wind_off_fc, residual_load_fc.
    """
    out = []
    comp = ["load_fc", "pv_fc", "wind_on_fc", "wind_off_fc"]
    have = [c for c in comp + ["residual_load_fc"] if c in fc.columns]
    missing_cols = set(comp + ["residual_load_fc"]) - set(have)
    out.append(_check("fc_columns_present", not missing_cols,
                      "missing: %s" % sorted(missing_cols) if missing_cols else "all present"))
    if missing_cols:
        return out

    ident = (fc["residual_load_fc"]
             - (fc["load_fc"] - fc["pv_fc"] - fc["wind_on_fc"] - fc["wind_off_fc"]))
    bad = int((ident.abs() > RL_IDENTITY_TOL).sum())
    out.append(_check("fc_residual_identity", bad == 0,
                      "%d hours violate RL identity (tol %.1f MWh)"
                      % (bad, RL_IDENTITY_TOL)))

    load_gw = fc["load_fc"].dropna() / 1000.0
    oob = int(((load_gw < LOAD_GW_MIN) | (load_gw > LOAD_GW_MAX)).sum())
    neg = int((fc[["pv_fc", "wind_on_fc", "wind_off_fc"]].dropna() < 0).sum().sum())
    out.append(_check("fc_plausible_magnitudes", oob == 0 and neg == 0,
                      "%d load hours outside %.0f-%.0f GW, %d negative generation values"
                      % (oob, LOAD_GW_MIN, LOAD_GW_MAX, neg)))

    night = fc.loc[fc.index.hour <= 3, "pv_fc"].dropna()
    pv_night = float(night.max()) if len(night) else 0.0
    out.append(_check("fc_pv_dark_at_night", pv_night <= PV_NIGHT_MAX_MWH,
                      "max PV forecast at 00-03h local: %.0f MWh (max %.0f)"
                      % (pv_night, PV_NIGHT_MAX_MWH)))

    dead = [c for c in comp if fc[c].dropna().std() == 0.0]
    out.append(_check("fc_not_degenerate", not dead,
                      "constant series: %s" % dead if dead else "all series vary"))

    if price_index is not None and len(price_index):
        cov = float(fc["residual_load_fc"].reindex(price_index).notna().mean())
        out.append(_check("fc_coverage", cov >= FC_COVERAGE_MIN,
                          "covers %.1f%% of price index (min %.0f%%)"
                          % (100 * cov, 100 * FC_COVERAGE_MIN)))
        fresh = fc["residual_load_fc"].dropna().index.max() >= price_index.max()
        out.append(_check("fc_freshness", bool(fresh),
                          "forecasts end %s, prices end %s"
                          % (fc["residual_load_fc"].dropna().index.max(),
                             price_index.max())))
    return out


def run_health_checks(price: pd.Series,
                      forecasts: Optional[pd.DataFrame] = None,
                      strict: bool = True) -> pd.DataFrame:
    """
    Run all applicable checks. Returns the report frame; in strict mode
    raises DataHealthError listing every failed check.
    """
    results = check_price(price)
    if forecasts is not None:
        results += check_forecasts(forecasts, price_index=price.dropna().index)
    report = pd.DataFrame(results).set_index("check")

    failed = report[~report["passed"]]
    if len(failed):
        msg = "data health check FAILED:\n%s" % failed["detail"].to_string()
        if strict:
            raise DataHealthError(msg)
        logger.warning(msg)
    else:
        logger.info("data health: %d/%d checks passed", len(report), len(report))
    return report


if __name__ == "__main__":
    from src.config import ROOT, load_config
    from src.data.load_prices import load_prices
    from src.forecast.residual_load import FORECAST_PATH

    cfg = load_config()
    price = load_prices(cfg)["da_price"]
    forecasts = pd.read_parquet(ROOT / FORECAST_PATH)
    report = run_health_checks(price, forecasts, strict=False)
    pd.set_option("display.width", 200)
    print(report.to_string())
