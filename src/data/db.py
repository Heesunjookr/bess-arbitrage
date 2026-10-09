"""
db.py
-----
SQL access layer for the price data (DuckDB).

The backtest reads parquet via pandas (load_prices.py); this module serves
the same cleaned series through SQL, moving the cleaning and data-quality
logic into queries:

  raw table  da_prices_raw     ingested 1:1 from the source parquet
  view       da_prices_hourly  dedup (ROW_NUMBER) + hour buckets (AVG)
  query      CLEAN_SQL         date window + complete-24h-day filter
                               (GROUP BY local delivery day HAVING COUNT=24)

DuckDB is the embedded engine so the repo stays reproducible without a
server; the SQL is kept portable and runs unchanged on PostgreSQL (window
functions, AT TIME ZONE, HAVING). `connect()` pins the session time zone
to the market time zone so TIMESTAMPTZ literals behave identically on
both engines.

A DST subtlety the equivalence test caught: hour bucketing must happen on
the *UTC instant* (`date_trunc('hour', ts AT TIME ZONE 'UTC')`), not on
local wall time — otherwise the two 02:00 hours of the autumn clock change
merge into one bucket, the 25-hour day masquerades as complete, and four
DST days that the study excludes leak back in. `tests/test_db.py` asserts
the SQL path reproduces the pandas path row-for-row.

Run:  python -m src.data.db   (ingest + data-quality report + sample)
"""

from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

from src.config import ROOT, get_logger, load_config

logger = get_logger("db")

RAW_TABLE = "da_prices_raw"
HOURLY_VIEW = "da_prices_hourly"

# dedup + normalize to hourly buckets. Bucketing is done on the UTC instant
# (see module docstring); the result is re-cast to TIMESTAMPTZ.
HOURLY_VIEW_SQL = """
CREATE OR REPLACE VIEW {view} AS
WITH dedup AS (
    SELECT ts, da_price
    FROM (
        SELECT ts, da_price,
               ROW_NUMBER() OVER (PARTITION BY ts ORDER BY ts) AS rn
        FROM {raw}
    ) t
    WHERE rn = 1
)
SELECT date_trunc('hour', ts AT TIME ZONE 'UTC') AT TIME ZONE 'UTC' AS ts,
       AVG(da_price) AS da_price
FROM dedup
WHERE da_price IS NOT NULL
GROUP BY 1
"""

# cleaned, complete-day hourly series — the SQL twin of load_prices()
CLEAN_SQL = """
WITH windowed AS (
    SELECT ts, da_price,
           CAST(ts AT TIME ZONE '{tz}' AS DATE) AS delivery_day
    FROM {view}
    WHERE ts >= TIMESTAMPTZ '{start}'
      AND ts <  TIMESTAMPTZ '{end}' + INTERVAL '1 day'
),
complete_days AS (
    SELECT delivery_day
    FROM windowed
    GROUP BY delivery_day
    HAVING COUNT(*) = 24
)
SELECT w.ts, w.da_price, w.delivery_day
FROM windowed w
JOIN complete_days USING (delivery_day)
ORDER BY w.ts
"""

# hours per local delivery day, post hour-normalization:
# 23/25-hour rows are the DST transition days the backtest excludes
DAY_LENGTH_SQL = """
SELECT hours_in_day, COUNT(*) AS n_days
FROM (
    SELECT CAST(ts AT TIME ZONE '{tz}' AS DATE) AS delivery_day,
           COUNT(*) AS hours_in_day
    FROM {view}
    GROUP BY 1
) d
GROUP BY hours_in_day
ORDER BY hours_in_day
"""

# raw rows per local delivery day: exposes the source granularity —
# 24 rows/day (hourly era) vs 96 rows/day (15-minute MTU era, from 2025-10)
RAW_GRANULARITY_SQL = """
SELECT rows_per_day, COUNT(*) AS n_days,
       MIN(delivery_day) AS first_day, MAX(delivery_day) AS last_day
FROM (
    SELECT CAST(ts AT TIME ZONE '{tz}' AS DATE) AS delivery_day,
           COUNT(*) AS rows_per_day
    FROM {raw}
    WHERE da_price IS NOT NULL
    GROUP BY 1
) d
GROUP BY rows_per_day
ORDER BY rows_per_day
"""

NEGATIVE_HOURS_SQL = """
SELECT EXTRACT(year FROM ts AT TIME ZONE '{tz}') AS year,
       COUNT(*) FILTER (WHERE da_price < 0) AS negative_hours,
       ROUND(MIN(da_price), 1) AS min_price,
       ROUND(AVG(da_price), 1) AS mean_price
FROM {view}
GROUP BY 1
ORDER BY 1
"""


def connect(cfg: Optional[dict] = None,
            db_path: Optional[Path] = None) -> duckdb.DuckDBPyConnection:
    """
    Open the DuckDB database (default data/bess.duckdb) with the session
    time zone pinned to the market time zone, so TIMESTAMPTZ literals and
    date casts behave identically here and on PostgreSQL.
    """
    cfg = cfg or load_config()
    path = db_path if db_path is not None else ROOT / "data" / "bess.duckdb"
    con = duckdb.connect(str(path))
    con.execute("SET TimeZone = '%s'" % cfg["data"]["timezone"])
    return con


def ingest_prices(con: duckdb.DuckDBPyConnection,
                  cfg: Optional[dict] = None) -> int:
    """
    Load the raw parquet into da_prices_raw and (re)create the hourly
    view. Returns the raw row count.
    """
    cfg = cfg or load_config()
    parquet = ROOT / cfg["paths"]["raw"]["da_prices"]
    con.execute(
        """
        CREATE OR REPLACE TABLE %s AS
        SELECT ts, da_price
        FROM read_parquet('%s')
        """ % (RAW_TABLE, parquet)
    )
    con.execute(HOURLY_VIEW_SQL.format(view=HOURLY_VIEW, raw=RAW_TABLE))
    n = con.execute("SELECT COUNT(*) FROM %s" % RAW_TABLE).fetchone()[0]
    logger.info("ingested %d rows into %s (+ view %s)", n, RAW_TABLE, HOURLY_VIEW)
    return int(n)


def data_quality_report(con: duckdb.DuckDBPyConnection,
                        cfg: Optional[dict] = None) -> dict:
    """
    SQL-side data-quality summary:
      day_lengths     - 23/25-hour local days = DST transitions (excluded)
      raw_granularity - 24 vs 96 rows/day = the 2025-10 15-minute MTU switch
      by_year         - negative-price hours and price stats per year
    """
    cfg = cfg or load_config()
    tz = cfg["data"]["timezone"]
    return {
        "day_lengths": con.execute(
            DAY_LENGTH_SQL.format(view=HOURLY_VIEW, tz=tz)).df(),
        "raw_granularity": con.execute(
            RAW_GRANULARITY_SQL.format(raw=RAW_TABLE, tz=tz)).df(),
        "by_year": con.execute(
            NEGATIVE_HOURS_SQL.format(view=HOURLY_VIEW, tz=tz)).df(),
    }


def load_prices_sql(cfg: Optional[dict] = None,
                    con: Optional[duckdb.DuckDBPyConnection] = None
                    ) -> pd.DataFrame:
    """
    SQL twin of load_prices(): cleaned hourly frame with columns
    (da_price, date), tz-aware index, complete 24h delivery days only.
    """
    cfg = cfg or load_config()
    own = con is None
    if own:
        con = connect(cfg)
        ingest_prices(con, cfg)
    try:
        sql = CLEAN_SQL.format(
            view=HOURLY_VIEW,
            tz=cfg["data"]["timezone"],
            start=cfg["data"]["start_date"],
            end=cfg["data"]["end_date"],
        )
        df = con.execute(sql).df()
    finally:
        if own:
            con.close()

    df = df.set_index("ts")
    df.index = df.index.tz_convert(cfg["data"]["timezone"]).as_unit("ns")
    df.index.name = None
    df["date"] = pd.to_datetime(df.pop("delivery_day")).dt.date
    logger.info("SQL load: %d hours, %d complete days",
                len(df), df["date"].nunique())
    return df


if __name__ == "__main__":
    cfg = load_config()
    con = connect(cfg)
    ingest_prices(con, cfg)
    report = data_quality_report(con, cfg)
    print("\n=== hours per local day (23h/25h = DST transitions) ===")
    print(report["day_lengths"].to_string(index=False))
    print("\n=== raw rows per day (96 = 15-min MTU era, from 2025-10) ===")
    print(report["raw_granularity"].to_string(index=False))
    print("\n=== per-year price stats ===")
    print(report["by_year"].to_string(index=False))
    df = load_prices_sql(cfg, con=con)
    con.close()
    print("\n=== cleaned series (via SQL) ===")
    print(df.head())
