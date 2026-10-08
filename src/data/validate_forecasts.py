"""
validate_forecasts.py
---------------------
Labeling check for the SMARD "Prognose" series, in the spirit of the
intraday-index post-mortem: prove that the forecast series used by the
fundamental tiers (411/125/123/3791) are genuinely *day-ahead forecasts*,
not mislabeled copies of the realized series (410/4068/4067/1225).

If a "forecast" were actually the realized series, the fundamental tiers
would be reading day D's true residual load at bid time — total lookahead —
and their capture numbers would be fiction. The fingerprint of a real D-1
forecast is a realistic TSO error profile and a near-zero exact-match rate:

    load     ~4%  MAPE   (TSO D-1 load forecasts: 2-4% typical)
    wind on  ~14% MAPE   (characteristic D-1 horizon error; intraday-updated
                          wind forecasts would be markedly better)
    wind off ~40%+ MAPE  (small fleet, lumpy; MAPE inflated by low base)
    exact matches ~0%    (a copied series would be ~100%)

Verified 2026-07-04 on the full 2022-2026 sample — all four series pass.
What this check cannot resolve: SMARD serves latest snapshots without
publication timestamps, so post-auction *revisions* of a genuine forecast
remain possible (documented as a limitation; ENTSO-E point-in-time data is
the definitive fix).

Run:  python -m src.data.validate_forecasts
      (~950 requests to fetch the realized series; writes
       data/processed/forecast_labeling_check.csv)
"""

from typing import Optional

import pandas as pd

from src.config import ROOT, get_logger, load_config
from src.data.fetch_smard_forecasts import FORECAST_PATH, fetch_series

logger = get_logger("validate_forecasts")

# forecast column in smard_forecasts.parquet -> realized SMARD filter id
REALIZED_FILTERS = {
    "load_fc": ("load", 410),
    "pv_fc": ("pv", 4068),
    "wind_on_fc": ("wind_on", 4067),
    "wind_off_fc": ("wind_off", 1225),
}

# a mislabeled copy would sit near 100%; real forecasts sit near 0
MAX_EXACT_MATCH_SHARE = 0.10
# genuine D-1 load forecasts run 2-4% MAPE; suspiciously below = copy/revision
MIN_LOAD_MAPE_PCT = 1.0


def check_forecast_labeling(cfg: Optional[dict] = None) -> pd.DataFrame:
    """
    Compare each forecast series against its realized counterpart.
    Returns one row per series with error stats and a pass flag; raises
    AssertionError if any series looks like a mislabeled copy.
    """
    cfg = cfg or load_config()
    tz = cfg["data"]["timezone"]
    start = pd.Timestamp(cfg["data"]["start_date"], tz=tz)
    end = pd.Timestamp(cfg["data"]["end_date"], tz=tz) + pd.Timedelta(days=1)
    start_ms, end_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)

    fc_df = pd.read_parquet(ROOT / FORECAST_PATH).tz_convert("UTC")

    rows = []
    for fc_col, (name, act_id) in REALIZED_FILTERS.items():
        logger.info("fetching realized %s (filter %d) ...", name, act_id)
        act = fetch_series(act_id, start_ms, end_ms)
        both = pd.DataFrame({"fc": fc_df[fc_col], "act": act}).dropna()
        err = both["fc"] - both["act"]
        rows.append({
            "series": name,
            "n_hours": len(both),
            "corr": both["fc"].corr(both["act"]),
            "mae_mw": err.abs().mean(),
            "mape_pct": (err.abs() / both["act"].abs().clip(lower=1.0)).mean() * 100,
            "exact_match_share": (err.abs() < 0.5).mean(),
        })
    res = pd.DataFrame(rows).set_index("series")

    assert (res["exact_match_share"] < MAX_EXACT_MATCH_SHARE).all(), (
        "forecast series matches realized series too often — mislabeled copy?\n%s"
        % res.to_string())
    assert res.loc["load", "mape_pct"] > MIN_LOAD_MAPE_PCT, (
        "load 'forecast' is implausibly accurate — copy or post-hoc revision?\n%s"
        % res.to_string())
    logger.info("labeling checks passed:\n%s", res.round(3).to_string())
    return res


if __name__ == "__main__":
    cfg = load_config()
    res = check_forecast_labeling(cfg)
    out = ROOT / cfg["paths"]["processed"]["dir"] / "forecast_labeling_check.csv"
    res.to_csv(out)
    pd.set_option("display.width", 200)
    print("\n=== SMARD forecast-vs-realized labeling check (all must look like real D-1 forecasts) ===")
    print(res.round(3))
