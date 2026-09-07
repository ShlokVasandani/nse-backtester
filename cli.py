"""Command-line entry point for nse-backtester."""

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import typer

from adjust.store import DEFAULT_DB_PATH, build_and_store, read_prices
from engine.backtest import run_backtest
from ingest.bhavcopy import BhavcopyNotAvailable, fetch_bhavcopy
from ingest.history import build_history, save_history
from strategy.momentum import Momentum
from strategy.moving_average import MovingAverageCrossover

app = typer.Typer()

RAW_DIR = Path("data/raw/bhavcopy")
HISTORY_PATH = Path("data/raw/history.parquet")


def _build_strategy(name: str, fast: int, slow: int, lookback: int):
    """Shared by `signal` and `backtest` so both commands pick a strategy
    the same way."""
    if name == "ma":
        return MovingAverageCrossover(fast_window=fast, slow_window=slow), f"MA crossover ({fast}/{slow})"
    if name == "momentum":
        return Momentum(lookback_days=lookback), f"Momentum ({lookback}d)"
    typer.echo(f"Unknown strategy '{name}'. Choose 'ma' or 'momentum'.")
    raise typer.Exit(code=1)


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


@app.command()
def store(
    start: dt.datetime = typer.Option(
        None, formats=["%Y-%m-%d"], help="Start date, e.g. 2023-09-07 (default: 2 years ago)"
    ),
    end: dt.datetime = typer.Option(
        None, formats=["%Y-%m-%d"], help="End date, e.g. 2025-09-07 (default: today)"
    ),
    force: bool = typer.Option(False, help="Re-download days already cached"),
):
    """Full pipeline: raw bhavcopy -> corporate actions -> split/bonus
    adjustment -> DuckDB (data/processed/nse.duckdb, table `prices`)."""
    end_date = end.date() if end else dt.date.today()
    start_date = start.date() if start else end_date - dt.timedelta(days=365 * 2)

    adjusted, holidays = build_and_store(start_date, end_date, force=force)

    typer.echo(
        f"{len(adjusted)} rows across {adjusted['date'].nunique()} trading days "
        f"({start_date} to {end_date}), {len(holidays)} non-trading days skipped"
    )
    typer.echo(f"Saved to {DEFAULT_DB_PATH}")


@app.command()
def show(symbol: str = typer.Argument(..., help="NSE symbol, e.g. RELIANCE")):
    """Print a quick summary of one symbol's stored adjusted price history."""
    symbol = symbol.upper()
    df = read_prices(symbol=symbol)
    if df.empty:
        typer.echo(f"No stored data for {symbol}. Run `store` first.")
        raise typer.Exit(code=1)

    latest = df.iloc[-1]
    n_events = (df["adj_multiplier"].diff().fillna(0) != 0).sum()

    typer.echo(f"{symbol} -- {len(df)} trading days from {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
    typer.echo(f"Latest close (adjusted): {latest['adj_close']:.2f} on {latest['date']}")
    typer.echo(f"Period high / low (adjusted close): {df['adj_close'].max():.2f} / {df['adj_close'].min():.2f}")
    period_return = (latest["adj_close"] / df["adj_close"].iloc[0] - 1) * 100
    typer.echo(f"Return over stored period: {period_return:+.1f}%")
    typer.echo(f"Split/bonus adjustments applied in this window: {n_events}")


@app.command()
def signal(
    symbol: str = typer.Argument(..., help="NSE symbol, e.g. RELIANCE"),
    strategy: str = typer.Option("ma", help="Which rule to run: 'ma' or 'momentum'"),
    fast: int = typer.Option(50, help="ma: fast SMA window (days)"),
    slow: int = typer.Option(200, help="ma: slow SMA window (days)"),
    lookback: int = typer.Option(90, help="momentum: lookback window (days)"),
):
    """Show a strategy's current signal for a symbol and its recent
    changes. This is mechanical rule output, not investment advice --
    it hasn't been backtested yet, so don't trust it with money until it
    has been (that's what src/engine + src/metrics, still to come, are for)."""
    symbol = symbol.upper()
    df = read_prices(symbol=symbol)
    if df.empty:
        typer.echo(f"No stored data for {symbol}. Run `store` first.")
        raise typer.Exit(code=1)

    strat, label = _build_strategy(strategy, fast, slow, lookback)

    changes = strat.find_signal_changes(df)
    current = changes.iloc[-1]
    state = "LONG" if current["signal"] == 1 else "FLAT"

    typer.echo(f"{symbol} -- {label} [unvalidated rule, not advice]")
    typer.echo(f"Current signal: {state} (since {current['date']})")
    typer.echo("Recent signal changes:")
    typer.echo(changes.tail(5).to_string(index=False))


@app.command()
def backtest(
    symbol: str = typer.Argument(..., help="NSE symbol, e.g. RELIANCE"),
    strategy: str = typer.Option("ma", help="Which rule to run: 'ma' or 'momentum'"),
    fast: int = typer.Option(50, help="ma: fast SMA window (days)"),
    slow: int = typer.Option(200, help="ma: slow SMA window (days)"),
    lookback: int = typer.Option(90, help="momentum: lookback window (days)"),
    cash: float = typer.Option(100_000.0, help="Starting capital"),
):
    """Backtest a strategy against a symbol's full stored history, with
    real NSE delivery trading costs and slippage -- this is what tells you
    whether a signal is worth trusting, not just what it currently says."""
    symbol = symbol.upper()
    df = read_prices(symbol=symbol)
    if df.empty:
        typer.echo(f"No stored data for {symbol}. Run `store` first.")
        raise typer.Exit(code=1)

    strat, label = _build_strategy(strategy, fast, slow, lookback)
    result = run_backtest(df, strat, initial_cash=cash)

    final_equity = result.equity_curve["equity"].iloc[-1]
    total_return = (final_equity / cash - 1) * 100
    buy_hold_equity = cash / df["adj_close"].iloc[0] * df["adj_close"].iloc[-1]
    buy_hold_return = (buy_hold_equity / cash - 1) * 100

    typer.echo(f"{symbol} -- {label} backtest, {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
    typer.echo(f"Starting capital: {cash:,.2f}")
    typer.echo(f"Strategy final equity: {final_equity:,.2f} ({total_return:+.1f}%)")
    typer.echo(f"Buy & hold would be:  {buy_hold_equity:,.2f} ({buy_hold_return:+.1f}%)")
    typer.echo(f"Trades: {len(result.trades)}")
    typer.echo("[return only -- Sharpe/drawdown/win-rate coming with src/metrics]")


if __name__ == "__main__":
    app()
