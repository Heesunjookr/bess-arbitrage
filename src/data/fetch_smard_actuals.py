"""
fetch_smard_actuals.py
----------------------
Fetch the *realized* counterparts of the day-ahead forecast series:

    410   Stromverbrauch: Gesamt (Netzlast)      (realized load)
    4068  Stromerzeugung: Photovoltaik
    4067  Stromerzeugung: Wind Onshore
    1225  Stromerzeugung: Wind Offshore

Two uses:
  * the forecast labeling check (src/data/validate_forecasts.py), and
  * the "perfect weather" counterfactual tier: run the rl_quad protocol on
    *realized* residual load to price how much of the remaining gap is
    weather-forecast error vs. price-mapping error. (Realized RL is not
    known at bid time — this is a diagnostic ceiling, not an executable
    tier, exactly like perfect foresight itself.)

Run:  python -m src.data.fetch_smard_actuals
      (~950 requests; writes data/raw/smard_actuals.parquet)
"""

from typing import Optional

import pandas as pd

from src.config import ROOT, get_logger, load_config
from src.data.fetch_smard_forecasts import fetch_series

logger = get_logger("fetch_smard_actuals")

ACTUAL_FILTERS = {
    "load_act": 410,
    "pv_act": 4068,
    "wind_on_act": 4067,
    "wind_off_act": 1225,
}
OUT_PATH = "data/raw/smard_actuals.parquet"


def fetch_actuals(cfg: Optional[dict] = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    tz = cfg["data"]["timezone"]
    start = pd.Timestamp(cfg["data"]["start_date"], tz=tz)
    end = pd.Timestamp(cfg["data"]["end_date"], tz=tz) + pd.Timedelta(days=1)
    start_ms, end_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)

    cols = {}
    for name, fid in ACTUAL_FILTERS.items():
        logger.info("fetching %s (filter %d) ...", name, fid)
        cols[name] = fetch_series(fid, start_ms, end_ms)
    df = pd.DataFrame(cols)
    df.index = df.index.tz_convert(tz)
    df.index.name = "ts"
    df["residual_load_act"] = (df["load_act"] - df["pv_act"]
                               - df["wind_on_act"] - df["wind_off_act"])
    return df


if __name__ == "__main__":
    cfg = load_config()
    df = fetch_actuals(cfg)
    out = ROOT / OUT_PATH
    df.to_parquet(out)
    logger.info("wrote %s: %d rows, NaN share:\n%s", out, len(df),
                df.isna().mean().round(4).to_string())
