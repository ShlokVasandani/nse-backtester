import datetime as dt
import json
from pathlib import Path

import pytest

from ingest.corporate_actions import download_corporate_actions, parse_corporate_actions, raw_path

SAMPLE_RECORDS = [
    {
        "bcEndDate": "-",
        "bcStartDate": "-",
        "caBroadcastDate": None,
        "comp": "Acme Ltd",
        "exDate": "10-Jan-2024",
        "faceVal": "10",
        "ind": "-",
        "isin": "INE000A00000",
        "ndEndDate": "-",
        "ndStartDate": "-",
        "recDate": "12-Jan-2024",
        "series": "EQ",
        "subject": "Bonus 3:1",
        "symbol": "ACME",
    },
    {
        "bcEndDate": "-",
        "bcStartDate": "-",
        "caBroadcastDate": None,
        "comp": "Widget Corp",
        "exDate": "-",
        "faceVal": "10",
        "ind": "-",
        "isin": "INE000B00000",
        "ndEndDate": "-",
        "ndStartDate": "-",
        "recDate": "-",
        "series": "EQ",
        "subject": "Annual General Meeting",
        "symbol": "WIDGET",
    },
]


@pytest.fixture
def sample_json(tmp_path: Path) -> Path:
    path = tmp_path / "sample.json"
    path.write_text(json.dumps(SAMPLE_RECORDS))
    return path


def test_parse_corporate_actions_drops_rows_without_ex_date(sample_json: Path):
    df = parse_corporate_actions(sample_json)
    # WIDGET's "-" exDate (AGM, not a corporate action with a settlement date)
    # should be dropped.
    assert list(df["symbol"]) == ["ACME"]


def test_parse_corporate_actions_types_dates_and_fields(sample_json: Path):
    df = parse_corporate_actions(sample_json)
    row = df.iloc[0]
    assert row["ex_date"] == dt.date(2024, 1, 10)
    assert row["record_date"] == dt.date(2024, 1, 12)
    assert row["subject"] == "Bonus 3:1"
    assert row["isin"] == "INE000A00000"


def test_download_corporate_actions_skips_existing_file(tmp_path: Path, monkeypatch):
    start, end = dt.date(2024, 1, 1), dt.date(2024, 12, 31)
    dest = raw_path(start, end, tmp_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"cached")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("should not hit the network when file already exists")

    monkeypatch.setattr("ingest.corporate_actions.requests.get", fail_if_called)

    result = download_corporate_actions(start, end, tmp_path)
    assert result == dest
    assert result.read_bytes() == b"cached"
