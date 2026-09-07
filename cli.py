"""Command-line entry point for nse-backtester."""

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import typer

from ingest.bhavcopy import BhavcopyNotAvailable, fetch_bhavcopy

app = typer.Typer()

RAW_DIR = Path("data/raw/bhavcopy")


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


if __name__ == "__main__":
    app()
