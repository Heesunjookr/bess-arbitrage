"""
load_prices.py
--------------
Load and clean the DE-LU day-ahead price series.

Source parquet (copied from the da-id study; data only, no code dependency):
  - data/raw/da_prices.parquet : hourly day-ahead auction price (col da_price),
    ENTSO-E Transparency, EUR/MWh, tz-aware (Europe/Berlin).

Note: the file previously loaded here as the "intraday" series turned out to
be neighbor-country day-ahead prices (SMARD 4996 = Belgium, 4997 = Norway
NO2) — see src/data/validate.py for the checks that exposed this. It is kept
in data/raw/ for the validation module only and is not used in the backtest.
"""

from typing import Optional

import pandas as pd

from src.config import ROOT, get_logger, load_config

logger = get_logger("load_prices")


def load_da_prices(cfg: Optional[dict] = None) -> pd.Series:
    """Hourly DE-LU day-ahead price (EUR/MWh), deduplicated, tz-aware."""
    cfg = cfg or load_config()
    tz = cfg["data"]["timezone"]
    path = ROOT / cfg["paths"]["raw"]["da_prices"]
    s = pd.read_parquet(path)["da_price"]
    if s.index.tz is None:
        s.index = s.index.tz_localize(tz)
    s = s[~s.index.duplicated()]
    # DA is already hourly; resample-mean is a no-op except it normalizes
    # the index to exact hour boundaries.
    s = s.resample("1h").mean()
    s.name = "da_price"
    return s


def load_prices(cfg: Optional[dict] = None) -> pd.DataFrame:
    """
    Cleaned hourly DA price frame for the backtest.

    Columns: da_price, date (local delivery day).
    Cleaning:
      - clipped to settings start_date..end_date
      - NaN hours dropped
      - only complete 24-hour delivery days kept (the LP optimizes fixed
        24h windows; DST days with 23/25 hours are excluded)
    """
    cfg = cfg or load_config()
    da = load_da_prices(cfg)
    df = da.to_frame()

    start = pd.Timestamp(cfg["data"]["start_date"], tz=cfg["data"]["timezone"])
    end = pd.Timestamp(cfg["data"]["end_date"], tz=cfg["data"]["timezone"]) + pd.Timedelta(days=1)
    df = df.loc[(df.index >= start) & (df.index < end)].dropna()

    # local delivery-day column (robust to DST when grouping)
    df["date"] = df.index.tz_convert(cfg["data"]["timezone"]).date

    counts = df.groupby("date")["da_price"].transform("size")
    df = df[counts == 24]

    logger.info(
        "loaded %s .. %s: %d hours, %d complete days",
        df.index.min(), df.index.max(), len(df), df["date"].nunique(),
    )
    return df


if __name__ == "__main__":
    d = load_prices()
    print(d.head())
    print(d["da_price"].describe())
    print("days per year:")
    print(d.groupby(pd.Index(d["date"]).map(lambda x: x.year)).size() // 24)
