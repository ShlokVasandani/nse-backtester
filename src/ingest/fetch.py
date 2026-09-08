"""Format router: picks the right bhavcopy archive for a given date.

NSE has two incompatible bhavcopy archives:
  - modern UDiFF (ingest/bhavcopy.py):  2024-01-01 -> present
  - legacy       (ingest/bhavcopy_legacy.py): 2000-01-03 -> ~2024-07-05

They overlap for Jan..Jul 2024. This routes to the modern format wherever
it exists (it's the one NSE still publishes) and falls back to legacy for
everything older, so callers can just ask for a date and get canonical
columns back either way.

Parsing dispatches on the file's own header rather than on the date, so a
cached file is always read with the parser that actually matches it.
"""

from __future__ import annotations

import datetime as dt
import zipfile
from pathlib import Path

import pandas as pd

from ingest.bhavcopy import (
    EARLIEST_AVAILABLE_DATE,
    BhavcopyNotAvailable,
    download_bhavcopy,
    parse_bhavcopy,
)
from ingest.bhavcopy import raw_path as modern_raw_path
from ingest.bhavcopy_legacy import (
    LEGACY_EARLIEST_DATE,
    download_legacy_bhavcopy,
    legacy_raw_path,
    parse_legacy_bhavcopy,
)

# Oldest date any archive covers.
EARLIEST_COVERED_DATE = LEGACY_EARLIEST_DATE


def uses_modern_format(date: dt.date) -> bool:
    return date >= EARLIEST_AVAILABLE_DATE


def raw_path_for(date: dt.date, raw_dir: Path) -> Path:
    """Where this date's raw archive lives on disk. The two formats use
    different filenames, so they coexist without collision."""
    if uses_modern_format(date):
        return modern_raw_path(date, raw_dir)
    return legacy_raw_path(date, raw_dir)


def download_for_date(date: dt.date, raw_dir: Path, *, force: bool = False) -> Path:
    if uses_modern_format(date):
        return download_bhavcopy(date, raw_dir, force=force)
    return download_legacy_bhavcopy(date, raw_dir, force=force)


def parse_auto(path: Path) -> pd.DataFrame:
    """Parse either format, detected from the CSV header inside the zip."""
    with zipfile.ZipFile(path) as zf:
        csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        with zf.open(csv_name) as f:
            header = f.readline().decode("utf-8", errors="replace")

    if "TradDt" in header:
        return parse_bhavcopy(path)
    if "TIMESTAMP" in header:
        return parse_legacy_bhavcopy(path)
    raise ValueError(f"Unrecognized bhavcopy header in {path}: {header[:120]!r}")


def fetch_for_date(date: dt.date, raw_dir: Path, *, force: bool = False) -> pd.DataFrame:
    """Download (if needed) and parse one day, whichever era it's from."""
    return parse_auto(download_for_date(date, raw_dir, force=force))


__all__ = [
    "EARLIEST_COVERED_DATE",
    "BhavcopyNotAvailable",
    "download_for_date",
    "fetch_for_date",
    "parse_auto",
    "raw_path_for",
    "uses_modern_format",
]
