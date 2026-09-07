import datetime as dt
import zipfile
from pathlib import Path

import pytest

from ingest.bhavcopy import download_bhavcopy, parse_bhavcopy, raw_path

SAMPLE_CSV = """TradDt,BizDt,Sgmt,Src,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,SctySrs,XpryDt,FininstrmActlXpryDt,StrkPric,OptnTp,FinInstrmNm,OpnPric,HghPric,LwPric,ClsPric,LastPric,PrvsClsgPric,UndrlygPric,SttlmPric,OpnIntrst,ChngInOpnIntrst,TtlTradgVol,TtlTrfVal,TtlNbOfTxsExctd,SsnId,NewBrdLotQty,Rmks,Rsvd1,Rsvd2,Rsvd3,Rsvd4
2025-09-05,2025-09-05,CM,NSE,STK,13061,INE466L01038,360ONE,EQ,,,,,360 ONE WAM LIMITED,1035.00,1052.00,1028.90,1037.60,1039.00,1032.80,,1037.60,,,311880,323814300.30,12894,F1,1,,,,,
2025-09-05,2025-09-05,CM,NSE,STK,3742,IN0020210053,SGBMAY29I,GB,,,,,2.5%GOLDBONDS2029SR-I,11561.00,11900.00,11560.00,11831.18,11825.00,11549.79,,11834.72,,,536,6262477.78,128,F1,1,,,,,
2025-09-05,2025-09-05,CM,NSE,STK,4,INE253B01015,21STCENMGM,BE,,,,,21ST CENTURY MGMT SERVICE,52.37,54.23,52.36,52.57,53.78,53.37,,52.92,,,1595,84439.30,108,F1,1,,,,,
"""


@pytest.fixture
def sample_zip(tmp_path: Path) -> Path:
    zip_path = tmp_path / "BhavCopy_NSE_CM_0_0_0_20250905_F_0000.csv.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("BhavCopy_NSE_CM_0_0_0_20250905_F_0000.csv", SAMPLE_CSV)
    return zip_path


def test_parse_bhavcopy_filters_to_equity_series(sample_zip: Path):
    df = parse_bhavcopy(sample_zip)

    # GB (gold bond) row must be dropped; EQ and BE rows kept.
    assert set(df["symbol"]) == {"360ONE", "21STCENMGM"}
    assert set(df["series"]) == {"EQ", "BE"}


def test_parse_bhavcopy_columns_and_types(sample_zip: Path):
    df = parse_bhavcopy(sample_zip)
    row = df[df["symbol"] == "360ONE"].iloc[0]

    assert row["date"] == dt.date(2025, 9, 5)
    assert row["isin"] == "INE466L01038"
    assert row["open"] == pytest.approx(1035.00)
    assert row["close"] == pytest.approx(1037.60)
    assert row["volume"] == 311880
    assert row["trades"] == 12894


def test_download_bhavcopy_skips_existing_file(tmp_path: Path, monkeypatch):
    date = dt.date(2025, 9, 5)
    dest = raw_path(date, tmp_path)
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"already here")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("should not hit the network when file already exists")

    monkeypatch.setattr("ingest.bhavcopy.requests.get", fail_if_called)

    result = download_bhavcopy(date, tmp_path)
    assert result == dest
    assert result.read_bytes() == b"already here"


def test_download_bhavcopy_raises_on_holiday(tmp_path: Path, monkeypatch):
    class FakeResp:
        status_code = 404

    monkeypatch.setattr("ingest.bhavcopy.requests.get", lambda *a, **k: FakeResp())

    from ingest.bhavcopy import BhavcopyNotAvailable

    with pytest.raises(BhavcopyNotAvailable):
        download_bhavcopy(dt.date(2025, 9, 7), tmp_path)
