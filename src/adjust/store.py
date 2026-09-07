"""Persist adjusted price data to DuckDB.

One table, `prices`, indexed by (symbol, date). Writes are idempotent: a
re-run over an overlapping date range replaces those rows rather than
duplicating them, so `build_and_store` can be called repeatedly (e.g. daily)
without hand-managing what's already there.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import duckdb
import pandas as pd

from adjust.split_bonus import apply_adjustment, build_adjustment_events
from ingest.corporate_actions import fetch_corporate_actions
from ingest.history import build_history

DEFAULT_DB_PATH = Path("data/processed/nse.duckdb")


def write_prices(adjusted: pd.DataFrame, db_path: Path = DEFAULT_DB_PATH) -> None:
    """Upsert `adjusted` (output of apply_adjustment) into the prices table,
    keyed on (symbol, date). Existing rows for the same key are replaced."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        has_table = con.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = 'prices'"
        ).fetchone()

        if has_table:
            existing = con.execute("SELECT * FROM prices").df()
            # DuckDB round-trips DATE columns as pandas Timestamps, not
            # python date objects -- normalize so the dedup below actually
            # matches rows against `adjusted` (which uses python date).
            existing["date"] = pd.to_datetime(existing["date"]).dt.date
            combined = pd.concat([existing, adjusted], ignore_index=True)
        else:
            combined = adjusted

        combined = combined.drop_duplicates(subset=["symbol", "date"], keep="last")
        combined = combined.sort_values(["symbol", "date"]).reset_index(drop=True)

        con.execute("CREATE OR REPLACE TABLE prices AS SELECT * FROM combined")
        con.execute("CREATE INDEX IF NOT EXISTS idx_prices_symbol_date ON prices(symbol, date)")
    finally:
        con.close()


def read_prices(
    db_path: Path = DEFAULT_DB_PATH,
    symbol: str | None = None,
    start: dt.date | None = None,
    end: dt.date | None = None,
) -> pd.DataFrame:
    """Query the prices table, optionally filtered to one symbol and/or a
    date range. Returns an empty dataframe if the table doesn't exist yet."""
    if not db_path.exists():
        return pd.DataFrame()

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        has_table = con.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = 'prices'"
        ).fetchone()
        if not has_table:
            return pd.DataFrame()

        clauses, params = [], []
        if symbol is not None:
            clauses.append("symbol = ?")
            params.append(symbol)
        if start is not None:
            clauses.append("date >= ?")
            params.append(start)
        if end is not None:
            clauses.append("date <= ?")
            params.append(end)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"SELECT * FROM prices {where} ORDER BY symbol, date"
        result = con.execute(query, params).df()
        result["date"] = pd.to_datetime(result["date"]).dt.date
        return result
    finally:
        con.close()


def build_and_store(
    start: dt.date,
    end: dt.date,
    *,
    raw_bhavcopy_dir: Path = Path("data/raw/bhavcopy"),
    raw_corporate_actions_dir: Path = Path("data/raw/corporate_actions"),
    db_path: Path = DEFAULT_DB_PATH,
    force: bool = False,
) -> tuple[pd.DataFrame, list[dt.date]]:
    """Full pipeline: raw prices -> corporate actions -> adjustment ->
    DuckDB. Returns (adjusted dataframe written, holidays skipped)."""
    raw_prices, holidays = build_history(start, end, raw_bhavcopy_dir, force=force)
    corporate_actions = fetch_corporate_actions(start, end, raw_corporate_actions_dir, force=force)
    events = build_adjustment_events(corporate_actions)
    adjusted = apply_adjustment(raw_prices, events)

    write_prices(adjusted, db_path)
    return adjusted, holidays
