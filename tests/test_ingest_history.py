import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from ingest.bhavcopy import BhavcopyNotAvailable
from ingest.history import build_history, load_history, save_history, trading_date_range


def test_trading_date_range_excludes_weekends():
    # 2025-09-05 is a Friday, 2025-09-08 is a Monday
    dates = trading_date_range(dt.date(2025, 9, 5), dt.date(2025, 9, 8))
    assert dates == [dt.date(2025, 9, 5), dt.date(2025, 9, 8)]


def _fake_df(date: dt.date) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [date],
            "symbol": ["ACME"],
            "series": ["EQ"],
            "isin": ["INE000A00000"],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "prev_close": [99.5],
            "volume": [1000],
            "turnover": [100500.0],
            "trades": [10],
        }
    )


def test_build_history_skips_holidays_and_concatenates(tmp_path: Path, monkeypatch):
    # Mon 2025-09-08, Tue 2025-09-09 (holiday), Wed 2025-09-10
    holiday = dt.date(2025, 9, 9)

    def fake_download(date, raw_dir, force=False):
        if date == holiday:
            raise BhavcopyNotAvailable(f"no bhavcopy {date}")
        return raw_dir / f"{date}.csv.zip"

    def fake_parse(path):
        # recover the date we encoded into the fake path's stem
        date = dt.date.fromisoformat(path.stem.split(".")[0])
        return _fake_df(date)

    monkeypatch.setattr("ingest.history.download_bhavcopy", fake_download)
    monkeypatch.setattr("ingest.history.parse_bhavcopy", fake_parse)

    history, holidays = build_history(
        dt.date(2025, 9, 8), dt.date(2025, 9, 10), tmp_path, sleep=0, progress=False
    )

    assert holidays == [holiday]
    assert list(history["date"]) == [dt.date(2025, 9, 8), dt.date(2025, 9, 10)]


def test_build_history_all_holidays_returns_empty_frame(tmp_path: Path, monkeypatch):
    def always_unavailable(date, raw_dir, force=False):
        raise BhavcopyNotAvailable("nope")

    monkeypatch.setattr("ingest.history.download_bhavcopy", always_unavailable)

    history, holidays = build_history(
        dt.date(2025, 9, 8), dt.date(2025, 9, 8), tmp_path, sleep=0, progress=False
    )

    assert history.empty
    assert holidays == [dt.date(2025, 9, 8)]


def test_save_and_load_history_roundtrip(tmp_path: Path):
    df = _fake_df(dt.date(2025, 9, 8))
    path = tmp_path / "history.parquet"

    save_history(df, path)
    loaded = load_history(path)

    assert loaded["symbol"].tolist() == ["ACME"]
    assert loaded["close"].iloc[0] == pytest.approx(100.5)
