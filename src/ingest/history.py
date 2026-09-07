"""Build a raw (unadjusted) historical store by looping the bhavcopy
downloader over a date range.

Output is a single concatenated parquet of every trading day's parsed
bhavcopy rows. It is intentionally *unadjusted* -- split/bonus adjustment
happens later, in src/adjust, and produces a separate store.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import time
from pathlib import Path

import pandas as pd

from ingest.bhavcopy import BhavcopyNotAvailable, download_bhavcopy, parse_bhavcopy, raw_path

HISTORY_COLUMNS = [
    "date",
    "symbol",
    "series",
    "isin",
    "open",
    "high",
    "low",
    "close",
    "prev_close",
    "volume",
    "turnover",
    "trades",
]


def trading_date_range(start: dt.date, end: dt.date) -> list[dt.date]:
    """Weekdays (Mon-Fri) in [start, end]. NSE holidays within this range
    still need to be skipped at fetch time -- there's no calendar here, just
    a cheap filter to avoid hitting the network for Saturdays/Sundays."""
    return [d.date() for d in pd.bdate_range(start, end)]


def build_history(
    start: dt.date,
    end: dt.date,
    raw_dir: Path,
    *,
    force: bool = False,
    sleep: float = 0.3,
    progress: bool = True,
) -> tuple[pd.DataFrame, list[dt.date]]:
    """Download + parse every trading day in [start, end].

    Returns (combined dataframe sorted by date/symbol, list of dates that
    turned out to have no bhavcopy -- i.e. holidays NSE doesn't publish for).
    Politely sleeps between *actual* downloads only; cache hits are instant.
    """
    dates = trading_date_range(start, end)
    frames: list[pd.DataFrame] = []
    holidays: list[dt.date] = []

    if progress:
        import typer

        cm = typer.progressbar(dates, label="Fetching bhavcopy")
    else:
        cm = contextlib.nullcontext(dates)

    with cm as bar:
        for date in bar:
            dest = raw_path(date, raw_dir)
            already_cached = dest.exists() and not force
            try:
                path = download_bhavcopy(date, raw_dir, force=force)
            except BhavcopyNotAvailable:
                holidays.append(date)
                continue
            if not already_cached:
                time.sleep(sleep)
            frames.append(parse_bhavcopy(path))

    if not frames:
        return pd.DataFrame(columns=HISTORY_COLUMNS), holidays

    history = pd.concat(frames, ignore_index=True)
    history = history.sort_values(["symbol", "date"]).reset_index(drop=True)
    return history, holidays


def save_history(history: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    history.to_parquet(path, index=False)


def load_history(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
