"""Command-line entry point for nse-backtester."""

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import typer

from ingest.bhavcopy import BhavcopyNotAvailable, fetch_bhavcopy
from ingest.history import build_history, save_history

app = typer.Typer()

RAW_DIR = Path("data/raw/bhavcopy")
HISTORY_PATH = Path("data/raw/history.parquet")


@app.command()
def ingest_bhavcopy(
    date: dt.datetime = typer.Option(..., formats=["%Y-%m-%d"], help="Trading date, e.g. 2025-09-05"),
    force: bool = typer.Option(False, help="Re-download even if the raw file already exists"),
):
    """Download and parse one day's NSE bhavcopy."""
    try:
        df = fetch_bhavcopy(date.date(), RAW_DIR, force=force)
    except BhavcopyNotAvailable as e:
        typer.echo(str(e))
        raise typer.Exit(code=1)

    typer.echo(f"{len(df)} equity rows for {date:%Y-%m-%d}")
    typer.echo(df.head())


@app.command(name="build-history")
def build_history_cmd(
    start: dt.datetime = typer.Option(
        None, formats=["%Y-%m-%d"], help="Start date, e.g. 2023-09-07 (default: 2 years ago)"
    ),
    end: dt.datetime = typer.Option(
        None, formats=["%Y-%m-%d"], help="End date, e.g. 2025-09-07 (default: today)"
    ),
    force: bool = typer.Option(False, help="Re-download days already cached"),
):
    """Loop the bhavcopy downloader over a date range and save the combined
    raw historical store to data/raw/history.parquet."""
    end_date = end.date() if end else dt.date.today()
    start_date = start.date() if start else end_date - dt.timedelta(days=365 * 2)

    history, holidays = build_history(start_date, end_date, RAW_DIR, force=force)
    save_history(history, HISTORY_PATH)

    typer.echo(
        f"{len(history)} rows across {history['date'].nunique()} trading days "
        f"({start_date} to {end_date}), {len(holidays)} non-trading days skipped"
    )
    typer.echo(f"Saved to {HISTORY_PATH}")


if __name__ == "__main__":
    app()
