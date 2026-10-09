"""Refresh all public research inputs through the configured end date."""

from __future__ import annotations

import os
import gzip
import pandas as pd

from src.config import ROOT, get_logger, load_config
from src.data.fetch_smard_actuals import OUT_PATH as ACTUAL_PATH, fetch_actuals
from src.data.fetch_smard_forecasts import OUT_PATH as FORECAST_PATH, fetch_forecasts

logger = get_logger("refresh")


def export_public_prices(cfg: dict) -> None:
    """Export the cleaned hourly series consumed by the static web app."""
    from src.data.load_prices import load_prices

    public_path = ROOT / "data/public/da_prices.csv.gz"
    public_path.parent.mkdir(parents=True, exist_ok=True)
    prices = load_prices(cfg)[["da_price"]].copy()
    prices.index = prices.index.tz_convert("UTC")
    prices.index.name = "timestamp_utc"
    with gzip.open(public_path, "wt", newline="") as handle:
        prices.to_csv(handle)
    logger.info("exported %d public hourly prices -> %s", len(prices), public_path)


def _load_entsoe_key() -> str:
    """Load a local ENTSO-E token without ever logging its value."""
    key = os.environ.get("ENTSOE_API_KEY")
    candidates = [ROOT / ".env", ROOT.parent / "da-id-arbitrage" / ".env"]
    for path in candidates:
        if key or not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            name, sep, value = line.partition("=")
            if sep and name.strip() == "ENTSOE_API_KEY":
                key = value.strip().strip("'\"")
                break
    if not key:
        raise RuntimeError("ENTSOE_API_KEY is required to refresh day-ahead prices")
    return key


def fetch_entsoe_prices(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Fetch DE-LU DA prices month-wise, preserving the native MTU."""
    from entsoe import EntsoePandasClient

    client = EntsoePandasClient(api_key=_load_entsoe_key())
    # ENTSO-E allows at most one year per document request; annual chunks
    # avoid dozens of slow monthly round trips while respecting that limit.
    boundaries = list(pd.date_range(start.normalize(), end.normalize(), freq="YS"))
    if not boundaries or boundaries[0] != start.normalize():
        boundaries.insert(0, start.normalize())
    if boundaries[-1] != end:
        boundaries.append(end)
    frames = []
    for left, right in zip(boundaries, boundaries[1:]):
        logger.info("ENTSO-E prices: %s -> %s", left.date(), right.date())
        frames.append(client.query_day_ahead_prices("DE_LU", start=left, end=right))
    price = pd.concat(frames).sort_index()
    price = price[~price.index.duplicated(keep="last")]
    price.name = "da_price"
    price.index.name = "ts"
    return price


def refresh_all() -> None:
    cfg = load_config()
    tz = cfg["data"]["timezone"]
    start = pd.Timestamp(cfg["data"]["start_date"], tz=tz)
    end = pd.Timestamp(cfg["data"]["end_date"], tz=tz) + pd.Timedelta(days=1)
    logger.info("refreshing DE-LU day-ahead prices")
    price = fetch_entsoe_prices(start, end)
    price.to_frame().to_parquet(ROOT / cfg["paths"]["raw"]["da_prices"])

    logger.info("refreshing SMARD forecast fundamentals")
    forecasts = fetch_forecasts(cfg)
    forecasts.to_parquet(ROOT / FORECAST_PATH)

    logger.info("refreshing SMARD realised fundamentals")
    actuals = fetch_actuals(cfg)
    actuals.to_parquet(ROOT / ACTUAL_PATH)
    export_public_prices(cfg)
    logger.info(
        "refresh complete: prices=%s, forecasts=%s, actuals=%s",
        price.index.max(), forecasts.index.max(), actuals.index.max(),
    )


if __name__ == "__main__":
    refresh_all()
