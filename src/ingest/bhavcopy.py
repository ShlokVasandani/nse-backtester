"""Download and parse NSE's daily equity bhavcopy (UDiFF full bhavcopy CSV).

Source: https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_<YYYYMMDD>_F_0000.csv.zip
This is a static archive (no auth/cookies needed), but NSE blocks requests
without a browser-like User-Agent.

Known limitation, confirmed by binary search: this archive only goes back
to EARLIEST_AVAILABLE_DATE below. Every weekday before that 404s exactly
like a holiday would, even though the market was open -- so
build_history()/BhavcopyNotAvailable will silently (and misleadingly)
report those as "non-trading days skipped" rather than "data not
available here." Requesting a range that starts earlier won't error; it
will just quietly return less history than asked for. Getting further
back would need NSE's older (pre-UDiFF) bhavcopy format, which has a
different URL and column schema -- not implemented here.
"""

from __future__ import annotations

import datetime as dt
import zipfile
from pathlib import Path

import pandas as pd
import requests

BHAVCOPY_URL = (
    "https://nsearchives.nseindia.com/content/cm/"
    "BhavCopy_NSE_CM_0_0_0_{date:%Y%m%d}_F_0000.csv.zip"
)

# 2023-12-29 -> 404, 2024-01-01 -> 200 (verified live). Earlier requests
# don't fail loudly; they just return no data for those dates.
EARLIEST_AVAILABLE_DATE = dt.date(2024, 1, 1)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}

# Equity delivery series worth keeping for a backtester. EQ is the vast
# majority of listed companies; BE/BZ are trade-to-trade (no intraday/margin)
# series some illiquid names get moved into temporarily.
EQUITY_SERIES = {"EQ", "BE", "BZ"}


class BhavcopyNotAvailable(Exception):
    """Raised when NSE has no bhavcopy for the given date (weekend/holiday)."""


def raw_path(date: dt.date, raw_dir: Path) -> Path:
    return raw_dir / f"{date:%Y}" / f"{date:%m}" / f"BhavCopy_NSE_CM_0_0_0_{date:%Y%m%d}_F_0000.csv.zip"


def download_bhavcopy(date: dt.date, raw_dir: Path, *, force: bool = False) -> Path:
    """Download one day's raw bhavcopy zip, saved untouched under raw_dir.

    Returns the local path. Skips the network call if the file is already
    on disk, unless force=True.
    """
    dest = raw_path(date, raw_dir)
    if dest.exists() and not force:
        return dest

    url = BHAVCOPY_URL.format(date=date)
    resp = requests.get(url, headers=HEADERS, timeout=30)
    if resp.status_code == 404:
        raise BhavcopyNotAvailable(f"No bhavcopy for {date} (weekend/holiday?)")
    resp.raise_for_status()

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)
    return dest


def parse_bhavcopy(path: Path) -> pd.DataFrame:
    """Parse a raw bhavcopy zip into a clean per-symbol OHLCV dataframe.

    Filters to cash-market equity series only (drops SGBs, derivatives-linked
    rows, etc. that ride along in the same file).
    """
    with zipfile.ZipFile(path) as zf:
        csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(csv_name) as f:
            df = pd.read_csv(f)

    df = df[df["SctySrs"].isin(EQUITY_SERIES)].copy()

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["TradDt"]).dt.date,
            "symbol": df["TckrSymb"],
            "series": df["SctySrs"],
            "isin": df["ISIN"],
            "open": df["OpnPric"].astype(float),
            "high": df["HghPric"].astype(float),
            "low": df["LwPric"].astype(float),
            "close": df["ClsPric"].astype(float),
            "prev_close": df["PrvsClsgPric"].astype(float),
            "volume": df["TtlTradgVol"].astype("int64"),
            "turnover": df["TtlTrfVal"].astype(float),
            # Nullable Int64, not int64: the legacy parser (pre-2012 files)
            # has no trade-count column, and both eras must share one
            # schema to land in the same DuckDB table.
            "trades": df["TtlNbOfTxsExctd"].astype("Int64"),
        }
    )
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)


def fetch_bhavcopy(date: dt.date, raw_dir: Path, *, force: bool = False) -> pd.DataFrame:
    """Download (if needed) and parse a single day's bhavcopy."""
    path = download_bhavcopy(date, raw_dir, force=force)
    return parse_bhavcopy(path)
