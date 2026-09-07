# nse-backtester

A from-scratch backtesting engine for NSE (India) equities.

## Layout

- `data/raw/` — downloaded bhavcopy archives, untouched
- `data/processed/` — adjusted, cleaned parquet/duckdb files
- `src/ingest/` — download + parse bhavcopy, corporate actions
- `src/adjust/` — split/bonus/dividend adjustment logic
- `src/strategy/` — strategy base class + implementations
- `src/engine/` — backtest loop, execution simulator
- `src/costs/` — brokerage, STT, slippage models
- `src/metrics/` — CAGR, Sharpe, drawdown, etc.
- `src/analysis/` — notebook-facing helper functions
- `tests/` — pytest suite
- `notebooks/` — exploration only, nothing production lives here
- `configs/` — strategy configs (yaml)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
python cli.py --date 2025-09-05
```

(Currently a single-command CLI, so Typer runs it without a subcommand name.
This will become `python cli.py ingest-bhavcopy --date ...` once more
subcommands — `adjust`, `backtest`, etc. — are added.)

## Status

Milestone 1 (data ingestion) in progress: bhavcopy download + parse for a single day.
