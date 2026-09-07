"""Download and parse NSE's LEGACY (pre-UDiFF) daily equity bhavcopy.

Source: https://nsearchives.nseindia.com/content/historical/EQUITIES/<YYYY>/<MON>/cm<DD><MON><YYYY>bhav.csv.zip

Why this exists: the modern UDiFF archive (ingest/bhavcopy.py) only goes
back to 2024-01-01, which is far too little history to tell a real edge
apart from a market regime. This legacy archive goes back to 2000 and runs
until roughly 2024-07-05, verified live by probing both boundaries.

Two schema variants, also verified by probing:
  - 2000..2011: SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,
                TOTTRDQTY,TOTTRDVAL,TIMESTAMP           (no trades/ISIN)
  - 2012..2024: the same, plus TOTALTRADES,ISIN
Both are parsed into the SAME canonical column set the modern parser
emits, with the missing early fields left null rather than faked.
"""

from __future__ import annotations

import datetime as dt
import re
import zipfile
from pathlib import Path

import pandas as pd
import requests

from ingest.bhavcopy import HEADERS, BhavcopyNotAvailable, EQUITY_SERIES

LEGACY_BHAVCOPY_URL = (
    "https://nsearchives.nseindia.com/content/historical/EQUITIES/"
    "{date:%Y}/{month}/cm{date:%d}{month}{date:%Y}bhav.csv.zip"
)

# Verified live: 2000-01-03 -> 200, and 2024-07-05 -> 200 while
# 2024-07-08 -> 404. The legacy archive was retired mid-2024, overlapping
# the modern one for Jan..Jul 2024.
LEGACY_EARLIEST_DATE = dt.date(2000, 1, 3)
LEGACY_LATEST_DATE = dt.date(2024, 7, 5)


# NSE's TIMESTAMP column is *mostly* "28-DEC-2023", but not always: of
# 1364 archived files checked, cm13JUL2020bhav uses "13-Jul-20" -- mixed
# case and a 2-digit year -- for every one of its 2001 rows. Rather than
# guess, try the known formats and cross-check the result against the date
# in the filename, which is authoritative. That check is what catches the
# next variant instead of silently landing wrong dates in the store.
_TIMESTAMP_FORMATS = ("%d-%b-%Y", "%d-%b-%y")
_LEGACY_FILENAME_RE = re.compile(r"cm(\d{1,2})([A-Za-z]{3})(\d{4})bhav", re.IGNORECASE)


def _month_token(date: dt.date) -> str:
    return date.strftime("%b").upper()


def _date_from_filename(path: Path) -> dt.date | None:
    m = _LEGACY_FILENAME_RE.search(path.name)
    if not m:
        return None
    try:
        return dt.datetime.strptime(
            f"{int(m.group(1)):02d}-{m.group(2).upper()}-{m.group(3)}", "%d-%b-%Y"
        ).date()
    except ValueError:
        return None


def _parse_timestamps(timestamps: pd.Series, expected: dt.date | None) -> pd.Series:
    failures = []
    for fmt in _TIMESTAMP_FORMATS:
        try:
            parsed = pd.to_datetime(timestamps, format=fmt).dt.date
        except (ValueError, TypeError) as exc:
            failures.append(f"{fmt}: {exc}")
            continue
        if expected is not None and not (parsed == expected).all():
            failures.append(f"{fmt}: parsed dates disagree with filename date {expected}")
            continue
        return parsed

    raise ValueError(
        "Could not parse legacy bhavcopy TIMESTAMP column "
        f"(sample {timestamps.iloc[0]!r}). Tried: " + "; ".join(failures)
    )


def legacy_raw_path(date: dt.date, raw_dir: Path) -> Path:
    month = _month_token(date)
    return raw_dir / f"{date:%Y}" / f"{date:%m}" / f"cm{date:%d}{month}{date:%Y}bhav.csv.zip"


def download_legacy_bhavcopy(date: dt.date, raw_dir: Path, *, force: bool = False) -> Path:
    """Download one day's legacy bhavcopy zip, saved untouched."""
    dest = legacy_raw_path(date, raw_dir)
    if dest.exists() and not force:
        return dest

    url = LEGACY_BHAVCOPY_URL.format(date=date, month=_month_token(date))
    resp = requests.get(url, headers=HEADERS, timeout=30)
    if resp.status_code == 404:
        raise BhavcopyNotAvailable(f"No legacy bhavcopy for {date} (weekend/holiday?)")
    resp.raise_for_status()

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)
    return dest


def parse_legacy_bhavcopy(path: Path) -> pd.DataFrame:
    """Parse a legacy bhavcopy zip into the canonical OHLCV schema."""
    with zipfile.ZipFile(path) as zf:
        csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        with zf.open(csv_name) as f:
            df = pd.read_csv(f)

    # The files carry a trailing comma, so pandas invents an empty column;
    # names also pick up stray whitespace in some years.
    df.columns = [c.strip() for c in df.columns]
    df = df[df["SERIES"].astype(str).str.strip().isin(EQUITY_SERIES)].copy()

    # Pre-2012 files have neither of these.
    isin = df["ISIN"] if "ISIN" in df.columns else pd.Series(pd.NA, index=df.index)
    trades = df["TOTALTRADES"] if "TOTALTRADES" in df.columns else pd.Series(pd.NA, index=df.index)

    out = pd.DataFrame(
        {
            "date": _parse_timestamps(df["TIMESTAMP"], _date_from_filename(path)),
            "symbol": df["SYMBOL"].astype(str).str.strip(),
            "series": df["SERIES"].astype(str).str.strip(),
            "isin": isin,
            "open": df["OPEN"].astype(float),
            "high": df["HIGH"].astype(float),
            "low": df["LOW"].astype(float),
            "close": df["CLOSE"].astype(float),
            "prev_close": df["PREVCLOSE"].astype(float),
            "volume": df["TOTTRDQTY"].astype("int64"),
            "turnover": df["TOTTRDVAL"].astype(float),
            "trades": pd.to_numeric(trades, errors="coerce").astype("Int64"),
        }
    )
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)
