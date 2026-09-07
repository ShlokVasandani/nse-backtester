"""Pull and store many years of history, chunked by year.

Why this isn't just `cli.py store --start 2015-01-01`: that holds every
row in memory at once (~4.5M rows for a decade). This walks year by year
so peak memory stays at roughly one year, and each year is persisted as it
completes, so an interruption doesn't lose everything.

The correctness trap it avoids: corporate actions must be fetched ONCE for
the whole span and applied to every chunk. Adjusting a 2015 price requires
knowing about a split that happened in 2020 -- fetching events per-chunk
would leave early years unadjusted for later splits, silently.

Usage: python scripts/pull_history.py 2015 2025-09-05
"""

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from adjust.split_bonus import apply_adjustment, build_adjustment_events, find_unparsed_suspects
from adjust.store import DEFAULT_DB_PATH, write_prices
from ingest.corporate_actions import fetch_corporate_actions
from ingest.history import build_history

RAW_BHAVCOPY = Path("data/raw/bhavcopy")
RAW_CORPORATE_ACTIONS = Path("data/raw/corporate_actions")


def main() -> None:
    start_year = int(sys.argv[1])
    end_date = dt.date.fromisoformat(sys.argv[2])
    start_date = dt.date(start_year, 1, 1)

    print(f"[pull] range {start_date} .. {end_date}", flush=True)

    # One global fetch, so every chunk is adjusted for every later event.
    print("[pull] fetching corporate actions for the full span...", flush=True)
    corporate_actions = fetch_corporate_actions(start_date, end_date, RAW_CORPORATE_ACTIONS)
    events = build_adjustment_events(corporate_actions)
    suspects = find_unparsed_suspects(corporate_actions)
    print(
        f"[pull] {len(corporate_actions)} corporate actions -> {len(events)} adjustment events, "
        f"{len(suspects)} unparsed suspects",
        flush=True,
    )
    if len(suspects):
        for subject in suspects["subject"].unique()[:20]:
            print(f"[pull]   UNPARSED: {subject!r}", flush=True)

    total_rows = 0
    for year in range(start_year, end_date.year + 1):
        chunk_start = max(dt.date(year, 1, 1), start_date)
        chunk_end = min(dt.date(year, 12, 31), end_date)
        if chunk_start > chunk_end:
            continue

        raw, holidays = build_history(chunk_start, chunk_end, RAW_BHAVCOPY, progress=False)
        if raw.empty:
            print(f"[pull] {year}: no data", flush=True)
            continue

        adjusted = apply_adjustment(raw, events)
        write_prices(adjusted, DEFAULT_DB_PATH)
        total_rows += len(adjusted)
        print(
            f"[pull] {year}: {len(adjusted)} rows, {raw['date'].nunique()} trading days, "
            f"{len(holidays)} skipped  (running total {total_rows})",
            flush=True,
        )

    print(f"[pull] DONE. {total_rows} rows into {DEFAULT_DB_PATH}", flush=True)


if __name__ == "__main__":
    main()
