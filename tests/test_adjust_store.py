import datetime as dt
from pathlib import Path

import pandas as pd

from adjust.store import read_prices, write_prices


def _prices(symbol: str, dates: list[dt.date], closes: list[float]) -> pd.DataFrame:
    n = len(dates)
    return pd.DataFrame(
        {
            "symbol": [symbol] * n,
            "date": dates,
            "series": ["EQ"] * n,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "prev_close": closes,
            "volume": [1000] * n,
            "adj_multiplier": [1.0] * n,
            "adj_close": closes,
        }
    )


def test_write_and_read_roundtrip(tmp_path: Path):
    db_path = tmp_path / "test.duckdb"
    df = _prices("ACME", [dt.date(2024, 1, 1), dt.date(2024, 1, 2)], [100.0, 101.0])

    write_prices(df, db_path)
    result = read_prices(db_path, symbol="ACME")

    assert len(result) == 2
    assert result["close"].tolist() == [100.0, 101.0]


def test_write_prices_upserts_overlapping_dates(tmp_path: Path):
    db_path = tmp_path / "test.duckdb"
    first = _prices("ACME", [dt.date(2024, 1, 1), dt.date(2024, 1, 2)], [100.0, 101.0])
    write_prices(first, db_path)

    # Re-run over an overlapping range with a corrected close for 01-02 plus
    # a new day -- should replace, not duplicate, the overlapping row.
    second = _prices("ACME", [dt.date(2024, 1, 2), dt.date(2024, 1, 3)], [999.0, 102.0])
    write_prices(second, db_path)

    result = read_prices(db_path, symbol="ACME")
    assert len(result) == 3
    by_date = dict(zip(result["date"], result["close"]))
    assert by_date[dt.date(2024, 1, 1)] == 100.0
    assert by_date[dt.date(2024, 1, 2)] == 999.0
    assert by_date[dt.date(2024, 1, 3)] == 102.0


def test_read_prices_filters_by_symbol_and_date_range(tmp_path: Path):
    db_path = tmp_path / "test.duckdb"
    df = pd.concat(
        [
            _prices("ACME", [dt.date(2024, 1, 1), dt.date(2024, 1, 5)], [10.0, 11.0]),
            _prices("OTHER", [dt.date(2024, 1, 1)], [50.0]),
        ],
        ignore_index=True,
    )
    write_prices(df, db_path)

    only_acme = read_prices(db_path, symbol="ACME")
    assert set(only_acme["symbol"]) == {"ACME"}

    ranged = read_prices(db_path, symbol="ACME", start=dt.date(2024, 1, 3), end=dt.date(2024, 1, 31))
    assert ranged["date"].tolist() == [dt.date(2024, 1, 5)]


def test_read_prices_returns_empty_frame_when_db_missing(tmp_path: Path):
    result = read_prices(tmp_path / "does_not_exist.duckdb")
    assert result.empty
