"""Download and parse NSE's corporate actions feed (dividends, bonuses,
splits, etc.) for equities.

Source: https://www.nseindia.com/api/corporates-corporateActions
Unlike the bhavcopy archive, this lives on the main nseindia.com domain,
which normally has bot-blocking in front of it -- but this endpoint answers
a plain GET with a browser User-Agent + Accept header, no cookie handshake
needed (verified live). It accepts a from_date/to_date range in DD-MM-YYYY
and comfortably returns a full 2-year window in one call.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import requests

CORPORATE_ACTIONS_URL = "https://www.nseindia.com/api/corporates-corporateActions"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "application/json",
}


def raw_path(start: dt.date, end: dt.date, raw_dir: Path) -> Path:
    return raw_dir / f"corporate_actions_{start:%Y%m%d}_{end:%Y%m%d}.json"


def download_corporate_actions(
    start: dt.date, end: dt.date, raw_dir: Path, *, force: bool = False
) -> Path:
    """Fetch the raw corporate-actions JSON for [start, end] and save it
    untouched. Returns the local path; skips the network call if already
    cached, unless force=True."""
    dest = raw_path(start, end, raw_dir)
    if dest.exists() and not force:
        return dest

    resp = requests.get(
        CORPORATE_ACTIONS_URL,
        params={
            "index": "equities",
            "from_date": start.strftime("%d-%m-%Y"),
            "to_date": end.strftime("%d-%m-%Y"),
        },
        headers=HEADERS,
        timeout=30,
    )
    resp.raise_for_status()

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)
    return dest


def parse_corporate_actions(path: Path) -> pd.DataFrame:
    """Parse the raw JSON into a clean per-event dataframe. `subject` is
    left as free text -- extracting bonus/split ratios out of it is
    src/adjust's job, not ingest's."""
    records = json.loads(path.read_text())

    def _parse_date(s: str) -> dt.date | None:
        if not s or s == "-":
            return None
        return dt.datetime.strptime(s, "%d-%b-%Y").date()

    out = pd.DataFrame(
        {
            "symbol": [r["symbol"] for r in records],
            "isin": [r["isin"] for r in records],
            "series": [r["series"] for r in records],
            "company": [r["comp"] for r in records],
            "ex_date": [_parse_date(r["exDate"]) for r in records],
            "record_date": [_parse_date(r["recDate"]) for r in records],
            "subject": [r["subject"].strip() for r in records],
        }
    )
    out = out.dropna(subset=["ex_date"])
    return out.sort_values(["symbol", "ex_date"]).reset_index(drop=True)


def fetch_corporate_actions(start: dt.date, end: dt.date, raw_dir: Path, *, force: bool = False) -> pd.DataFrame:
    path = download_corporate_actions(start, end, raw_dir, force=force)
    return parse_corporate_actions(path)
