# nse-backtester

A from-scratch backtesting engine for NSE (India) equities: data ingestion,
corporate-action adjustment, strategy simulation, realistic cost modelling,
and performance metrics.

Built to answer one question honestly — **would this rule actually have made
money, after real costs, or does it just look good on a chart?**

## What it does

- **Ingests** NSE daily bhavcopy back to 2000, across two incompatible archive
  formats (the modern UDiFF feed and the retired legacy one), plus the
  corporate-actions feed.
- **Adjusts** prices for splits and bonuses, parsing them out of NSE's
  free-text filings — and refusing to guess on the ones it can't read rather
  than silently corrupting the series.
- **Stores** the adjusted series in DuckDB, keyed on (symbol, date).
- **Backtests** both single-symbol and portfolio strategies with no lookahead
  (a signal from day D's close executes at day D+1's open) and real Indian
  delivery-trading costs — STT, exchange fees, stamp duty, GST, DP charges,
  plus slippage.
- **Reports** CAGR, Sharpe, max drawdown and win rate against a matched
  benchmark.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Getting data

The DuckDB store is not committed (it's ~5M rows). Build it:

```bash
# One day, to check connectivity
python cli.py ingest-bhavcopy --date 2025-09-05

# Multiple years, chunked and resumable (this takes a while: ~2600 trading days)
python scripts/pull_history.py 2015 2025-09-05
```

## Using it

```bash
python cli.py show RELIANCE                      # stored history summary
python cli.py backtest RELIANCE --fast 20 --slow 50   # single-symbol
python cli.py portfolio --top-n 20 --cash 1000000     # portfolio momentum vs benchmark
```

## Layout

| Path | Purpose |
|---|---|
| `src/ingest/` | Bhavcopy download/parse (both archive formats), corporate actions |
| `src/adjust/` | Split/bonus adjustment, DuckDB store |
| `src/strategy/` | Single-symbol and portfolio strategy interfaces + implementations |
| `src/engine/` | Backtest loops and execution simulation |
| `src/costs/` | Brokerage, STT, slippage models |
| `src/metrics/` | CAGR, Sharpe, drawdown, win rate |
| `src/analysis/` | Parameter sweeps, train/test validation, data-quality gates |
| `scripts/` | Long-running data pulls |

## Findings so far

Measured over ~9.4 active years (2015–2025), 100-stock point-in-time universe,
monthly rebalance, real costs, against a **matched** equal-weight benchmark
(same universe, same rebalance dates, same costs — only the selection rule
differs):

| | Momentum (top 20, 12mo) | Benchmark |
|---|---|---|
| CAGR | 14.2% | 8.1% |
| Sharpe | 0.69 | 0.52 |
| Max drawdown | −49.5% | −57.8% |

Things that turned out to matter more than the headline number:

- **A 9-month backtest lied.** An earlier sweep found a moving-average rule
  beating buy-and-hold on 65/100 stocks (z≈3.0). Widening the window to span
  more than one market regime erased it completely. That rule was a regime
  artifact, not an edge.
- **Survivorship bias was worth ~10 points of CAGR.** Selecting the universe
  once from the whole window (so it can only contain companies that survived
  and had grown liquid by 2025) put the benchmark at 18.5% CAGR. Selecting it
  point-in-time instead dropped it to 8.1%.
- **Six-month momentum loses; twelve-month wins.** This matches the published
  momentum literature rather than contradicting it, which is mild evidence
  the result isn't just noise.
- **Both strategy and benchmark draw down ~50%.** Whatever the CAGR says,
  this is not a smooth ride.

## Caveats

One market, one decade, one strategy family. The universe excludes 66 symbols
whose adjusted series contain unexplained price discontinuities (see
`src/analysis/data_quality.py`) — NSE's corporate-actions feed is incomplete
for older history, and guessing correction factors would be worse than
excluding the names.

Backtest results are the output of a mechanical rule over historical data.
They are not investment advice.
