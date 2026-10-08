"""
test_db.py
----------
The SQL access layer must reproduce the pandas loader row-for-row:
same index, same prices, same delivery days, same DST exclusions.
"""

import numpy as np
import pytest

duckdb = pytest.importorskip("duckdb")

from src.config import load_config
from src.data.db import connect, data_quality_report, ingest_prices, load_prices_sql
from src.data.load_prices import load_prices


@pytest.fixture(scope="module")
def con(tmp_path_factory):
    cfg = load_config()
    c = connect(cfg, db_path=tmp_path_factory.mktemp("db") / "test.duckdb")
    ingest_prices(c, cfg)
    yield c
    c.close()


def test_sql_loader_matches_pandas_loader(con):
    cfg = load_config()
    via_sql = load_prices_sql(cfg, con=con)
    via_pandas = load_prices(cfg)

    assert len(via_sql) == len(via_pandas)
    assert (via_sql.index == via_pandas.index).all()
    # prices agree to float-summation order (SQL AVG vs pandas mean over
    # the 15-minute rows); tolerance is far below any economic relevance
    assert np.allclose(via_sql["da_price"].values,
                       via_pandas["da_price"].values, rtol=0, atol=1e-9)
    assert (via_sql["date"].values == via_pandas["date"].values).all()


def test_sql_report_flags_dst_days_and_mtu_switch(con):
    cfg = load_config()
    report = data_quality_report(con, cfg)

    lengths = report["day_lengths"].set_index("hours_in_day")["n_days"]
    # DST transitions exist as 23h (spring) and 25h (autumn) local days
    assert 23 in lengths.index and 25 in lengths.index
    # the overwhelming majority of days are complete 24h days
    assert lengths.loc[24] > 100 * lengths.loc[23]

    gran = report["raw_granularity"].set_index("rows_per_day")["n_days"]
    # the raw file mixes hourly (24 rows/day) and 15-minute MTU rows
    # (96 rows/day, from 2025-10) — both eras must be visible
    assert 24 in gran.index and 96 in gran.index
