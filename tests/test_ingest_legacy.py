import datetime as dt
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from ingest.bhavcopy import BhavcopyNotAvailable
from ingest.bhavcopy_legacy import (
    download_legacy_bhavcopy,
    legacy_raw_path,
    parse_legacy_bhavcopy,
)
from ingest.fetch import parse_auto, raw_path_for, uses_modern_format

# Real headers/rows taken verbatim from NSE's legacy archive.
# 2012-onward variant: has TOTALTRADES and ISIN, and a trailing comma.
MODERN_LEGACY_CSV = """SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,TIMESTAMP,TOTALTRADES,ISIN,
20MICRONS,EQ,35.5,36.4,35.05,36.1,36.15,35.4,63091,2262158.85,28-DEC-2023,438,INE144J01027,
1018GS2026,GS,118.77,120.65,118.77,120.47,120.65,121,11,1325.27,28-DEC-2023,3,IN0020010081,
"""

# Pre-2012 variant: no TOTALTRADES, no ISIN, unpadded day in TIMESTAMP.
EARLY_LEGACY_CSV = """SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,TIMESTAMP,
21STCENMGM,EQ,20,24.6,20,24.55,24.6,19.65,54000,1271385,3-JAN-2000,
AARTIDRUGS,EQ,33.95,34.35,32.05,34.35,34.35,31.8,8800,299230,3-JAN-2000,
"""


def _zip_with(tmp_path: Path, name: str, content: str) -> Path:
    zip_path = tmp_path / f"{name}.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(f"{name}.csv", content)
    return zip_path


def test_parse_legacy_modern_variant(tmp_path: Path):
    path = _zip_with(tmp_path, "cm28DEC2023bhav", MODERN_LEGACY_CSV)
    df = parse_legacy_bhavcopy(path)

    # GS (government security) row dropped; only the EQ row survives.
    assert list(df["symbol"]) == ["20MICRONS"]
    row = df.iloc[0]
    assert row["date"] == dt.date(2023, 12, 28)
    assert row["close"] == pytest.approx(36.1)
    assert row["prev_close"] == pytest.approx(35.4)
    assert row["volume"] == 63091
    assert row["trades"] == 438
    assert row["isin"] == "INE144J01027"


def test_parse_legacy_early_variant_without_trades_or_isin(tmp_path: Path):
    path = _zip_with(tmp_path, "cm03JAN2000bhav", EARLY_LEGACY_CSV)
    df = parse_legacy_bhavcopy(path)

    assert set(df["symbol"]) == {"21STCENMGM", "AARTIDRUGS"}
    row = df[df["symbol"] == "21STCENMGM"].iloc[0]
    # Unpadded day ("3-JAN-2000") must still parse.
    assert row["date"] == dt.date(2000, 1, 3)
    assert row["close"] == pytest.approx(24.55)
    # Missing columns are null, not fabricated.
    assert pd.isna(row["trades"])
    assert pd.isna(row["isin"])


def test_legacy_and_modern_parsers_emit_the_same_columns(tmp_path: Path):
    from tests.test_ingest_bhavcopy import SAMPLE_CSV
    from ingest.bhavcopy import parse_bhavcopy

    legacy = parse_legacy_bhavcopy(_zip_with(tmp_path, "cm28DEC2023bhav", MODERN_LEGACY_CSV))
    modern_zip = _zip_with(tmp_path, "BhavCopy_NSE_CM_0_0_0_20250905_F_0000.csv", SAMPLE_CSV)
    modern = parse_bhavcopy(modern_zip)

    assert list(legacy.columns) == list(modern.columns)
    assert legacy["trades"].dtype == modern["trades"].dtype


def test_parse_auto_dispatches_on_header(tmp_path: Path):
    from tests.test_ingest_bhavcopy import SAMPLE_CSV

    legacy_path = _zip_with(tmp_path, "cm28DEC2023bhav", MODERN_LEGACY_CSV)
    modern_path = _zip_with(tmp_path, "BhavCopy_NSE_CM_0_0_0_20250905_F_0000.csv", SAMPLE_CSV)

    assert parse_auto(legacy_path)["date"].iloc[0] == dt.date(2023, 12, 28)
    assert parse_auto(modern_path)["date"].iloc[0] == dt.date(2025, 9, 5)


def test_parse_auto_rejects_unknown_header(tmp_path: Path):
    path = _zip_with(tmp_path, "junk", "A,B,C\n1,2,3\n")
    with pytest.raises(ValueError):
        parse_auto(path)


def test_router_picks_format_by_date():
    assert uses_modern_format(dt.date(2024, 1, 1)) is True
    assert uses_modern_format(dt.date(2025, 6, 1)) is True
    assert uses_modern_format(dt.date(2023, 12, 29)) is False
    assert uses_modern_format(dt.date(2005, 4, 4)) is False


def test_router_raw_paths_do_not_collide(tmp_path: Path):
    legacy = raw_path_for(dt.date(2023, 12, 28), tmp_path)
    modern = raw_path_for(dt.date(2024, 1, 1), tmp_path)
    assert legacy != modern
    assert legacy.name.startswith("cm")
    assert modern.name.startswith("BhavCopy_")


def test_download_legacy_skips_existing_file(tmp_path: Path, monkeypatch):
    date = dt.date(2023, 12, 28)
    dest = legacy_raw_path(date, tmp_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"cached")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("should not hit the network when file already exists")

    monkeypatch.setattr("ingest.bhavcopy_legacy.requests.get", fail_if_called)

    assert download_legacy_bhavcopy(date, tmp_path) == dest


def test_download_legacy_raises_on_holiday(tmp_path: Path, monkeypatch):
    class FakeResp:
        status_code = 404

    monkeypatch.setattr("ingest.bhavcopy_legacy.requests.get", lambda *a, **k: FakeResp())

    with pytest.raises(BhavcopyNotAvailable):
        download_legacy_bhavcopy(dt.date(2023, 12, 30), tmp_path)
