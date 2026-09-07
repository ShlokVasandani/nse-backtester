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
# Single day
python cli.py ingest-bhavcopy --date 2025-09-05

# Date range -> combined raw store at data/raw/history.parquet
python cli.py build-history --start 2023-09-07 --end 2025-09-07
```

## Status

Milestone 1 (data ingestion) in progress: bhavcopy download + parse for a single day.
