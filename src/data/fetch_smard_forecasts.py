"""
fetch_smard_forecasts.py
------------------------
Fetch the SMARD *day-ahead forecast* series that drive the DA price shape
and build the residual-load forecast feature:

    residual_load_fc = load_fc − pv_fc − wind_onshore_fc − wind_offshore_fc

Filter IDs (chart_data API, region DE, resolution hour):

    411   Prognostizierter Stromverbrauch: Gesamt   (load forecast)
    125   Prognostizierte Erzeugung: Photovoltaik
    123   Prognostizierte Erzeugung: Wind Onshore
    3791  Prognostizierte Erzeugung: Wind Offshore

125/123/3791 are documented in the SMARD API spec (bundesAPI/smard-api);
411 is the load-forecast series served by the same endpoint (verified: same
magnitude as realized load 410, values differ -> a forecast, not a copy).

Point-in-time caveat (documented, not solved): these are the TSO day-ahead
forecasts, i.e. produced before the D-1 12:00 EPEX auction by publication
convention (ENTSO-E transparency mandates day-ahead publication). SMARD
serves only the latest snapshot without a publication timestamp, so revisions
cannot be excluded. Treated as D-1-known features; noted as a limitation.

API shape: index_hour.json lists week-start timestamps (ms, epoch); each
{filter}_DE_hour_{ts}.json holds one week of [timestamp_ms, value] pairs.

Run:  python -m src.data.fetch_smard_forecasts
      (~950 requests, a few minutes; writes data/raw/smard_forecasts.parquet)
"""

import json
import time
import urllib.request
from typing import List, Optional

import pandas as pd

from src.config import ROOT, get_logger, load_config

logger = get_logger("fetch_smard")

BASE = "https://www.smard.de/app/chart_data"
FILTERS = {
    "load_fc": 411,
    "pv_fc": 125,
    "wind_on_fc": 123,
    "wind_off_fc": 3791,
}
REGION = "DE"
RESOLUTION = "hour"
OUT_PATH = "data/raw/smard_forecasts.parquet"


def _get_json(url: str, retries: int = 3, backoff: float = 2.0) -> dict:
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001 - retry on any transport error
            last_err = e
            time.sleep(backoff * (attempt + 1))
    raise RuntimeError("SMARD request failed after %d tries: %s (%s)"
                       % (retries, url, last_err))


def fetch_series(filter_id: int, start_ms: int, end_ms: int) -> pd.Series:
    """One SMARD series as a tz-aware (UTC) hourly Series, NaNs kept."""
    index = _get_json("%s/%d/%s/index_%s.json" % (BASE, filter_id, REGION, RESOLUTION))
    weeks: List[int] = [t for t in index["timestamps"]
                        if start_ms - 7 * 86400_000 <= t <= end_ms]
    frames = []
    for i, ts in enumerate(weeks):
        url = "%s/%d/%s/%d_%s_%s_%d.json" % (
            BASE, filter_id, REGION, filter_id, REGION, RESOLUTION, ts)
        data = _get_json(url)
        frames.append(pd.DataFrame(data["series"], columns=["ts_ms", "value"]))
        if (i + 1) % 50 == 0:
            logger.info("filter %d: %d/%d weeks", filter_id, i + 1, len(weeks))
    s = pd.concat(frames, ignore_index=True)
    idx = pd.to_datetime(s["ts_ms"], unit="ms", utc=True)
    out = pd.Series(s["value"].to_numpy(dtype=float), index=idx)
    out = out[~out.index.duplicated()].sort_index()
    return out.loc[(out.index >= pd.Timestamp(start_ms, unit="ms", tz="UTC"))
                   & (out.index <= pd.Timestamp(end_ms, unit="ms", tz="UTC"))]


def fetch_forecasts(cfg: Optional[dict] = None) -> pd.DataFrame:
    """
    All forecast series on a common hourly index (Europe/Berlin), plus the
    residual-load forecast. Values are MWh per hour as served by SMARD.
    """
    cfg = cfg or load_config()
    tz = cfg["data"]["timezone"]
    start = pd.Timestamp(cfg["data"]["start_date"], tz=tz)
    end = pd.Timestamp(cfg["data"]["end_date"], tz=tz) + pd.Timedelta(days=1)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    cols = {}
    for name, fid in FILTERS.items():
        logger.info("fetching %s (filter %d) ...", name, fid)
        cols[name] = fetch_series(fid, start_ms, end_ms)
    df = pd.DataFrame(cols)
    df.index = df.index.tz_convert(tz)
    df.index.name = "ts"

    df["residual_load_fc"] = (df["load_fc"] - df["pv_fc"]
                              - df["wind_on_fc"] - df["wind_off_fc"])
    return df


if __name__ == "__main__":
    cfg = load_config()
    df = fetch_forecasts(cfg)
    out = ROOT / OUT_PATH
    df.to_parquet(out)
    logger.info("wrote %s: %d rows, %s .. %s, NaN share per col:\n%s",
                out, len(df), df.index.min(), df.index.max(),
                df.isna().mean().round(4).to_string())
