"""Multi-stock, multi-parameter backtest sweep.

Answers "does any (strategy, parameters) combination actually beat
buy-and-hold" -- not on one hand-picked stock, but broadly, since a rule
that only wins on one or two symbols out of a hundred is far more likely
to be curve-fit noise than a real edge. `aggregate_by_strategy` is the
important output here, not the single best row: an edge worth trusting
should show up as a consistent tendency across many stocks under the SAME
parameters, not as an isolated lucky hit.
"""

from __future__ import annotations

import pandas as pd

from adjust.store import read_prices
from engine.backtest import run_backtest
from metrics.performance import summarize
from strategy.base import Strategy
from strategy.momentum import Momentum
from strategy.moving_average import MovingAverageCrossover

DEFAULT_MA_GRID = [(10, 50), (10, 100), (10, 150), (20, 50), (20, 100), (20, 150), (30, 50), (30, 100), (30, 150)]
DEFAULT_MOMENTUM_GRID = [20, 30, 60, 90, 120]


def default_strategy_grid() -> list[tuple[str, Strategy]]:
    grid: list[tuple[str, Strategy]] = []
    for fast, slow in DEFAULT_MA_GRID:
        grid.append((f"ma_{fast}_{slow}", MovingAverageCrossover(fast_window=fast, slow_window=slow)))
    for lookback in DEFAULT_MOMENTUM_GRID:
        grid.append((f"momentum_{lookback}", Momentum(lookback_days=lookback)))
    return grid


def select_liquid_universe(all_prices: pd.DataFrame, top_n: int = 100) -> list[str]:
    """Symbols with a full trading-day history in `all_prices` (no listing
    gaps mid-window), ranked by average turnover and truncated to the top
    N. Restricts the sweep to stocks liquid enough to be realistic to
    trade, and keeps runtime sane."""
    full_days = all_prices["date"].nunique()
    day_counts = all_prices.groupby("symbol")["date"].nunique()
    eligible = day_counts[day_counts == full_days].index

    turnover = (
        all_prices[all_prices["symbol"].isin(eligible)]
        .groupby("symbol")["turnover"]
        .mean()
        .sort_values(ascending=False)
    )
    return turnover.head(top_n).index.tolist()


def run_sweep(
    symbols: list[str],
    all_prices: pd.DataFrame,
    strategy_grid: list[tuple[str, Strategy]] | None = None,
    *,
    initial_cash: float = 100_000.0,
) -> pd.DataFrame:
    """Backtest every (symbol, strategy) combination. `all_prices` should
    already contain every symbol in `symbols` (one shared load, not one
    DuckDB query per symbol -- this runs hundreds to thousands of
    backtests). Skips (symbol, strategy) pairs that never actually traded
    -- a strategy that never triggered isn't a data point, it's noise."""
    strategy_grid = strategy_grid or default_strategy_grid()
    grouped = {symbol: df.reset_index(drop=True) for symbol, df in all_prices.groupby("symbol")}

    rows = []
    for symbol in symbols:
        prices = grouped.get(symbol)
        if prices is None or len(prices) < 30:
            continue

        buy_hold_curve = prices[["date"]].copy()
        buy_hold_curve["equity"] = initial_cash / prices["adj_close"].iloc[0] * prices["adj_close"]
        bh = summarize(buy_hold_curve, [])

        for label, strategy in strategy_grid:
            result = run_backtest(prices, strategy, initial_cash=initial_cash)
            if not result.trades:
                continue
            s = summarize(result.equity_curve, result.trades)

            rows.append(
                {
                    "symbol": symbol,
                    "strategy": label,
                    "total_return": s.total_return,
                    "cagr": s.cagr,
                    "sharpe": s.sharpe,
                    "max_drawdown": s.max_drawdown,
                    "n_round_trips": s.n_round_trips,
                    "win_rate": s.win_rate,
                    "bh_total_return": bh.total_return,
                    "bh_sharpe": bh.sharpe,
                    "beats_bh_return": s.total_return > bh.total_return,
                    "beats_bh_sharpe": s.sharpe is not None and bh.sharpe is not None and s.sharpe > bh.sharpe,
                }
            )

    return pd.DataFrame(rows)


def aggregate_by_strategy(sweep_results: pd.DataFrame) -> pd.DataFrame:
    """Per-parameter-combination hit rate across every symbol tested. The
    honest question this answers: does this SAME rule tend to beat
    buy-and-hold broadly, or did it only win on a handful of symbols
    (consistent with luck, not edge)."""
    if sweep_results.empty:
        return pd.DataFrame()

    agg = sweep_results.groupby("strategy").agg(
        n_symbols=("symbol", "count"),
        beats_bh_return_rate=("beats_bh_return", "mean"),
        beats_bh_sharpe_rate=("beats_bh_sharpe", "mean"),
        mean_total_return=("total_return", "mean"),
        mean_bh_total_return=("bh_total_return", "mean"),
        mean_sharpe=("sharpe", "mean"),
        mean_bh_sharpe=("bh_sharpe", "mean"),
    )
    return agg.sort_values("beats_bh_sharpe_rate", ascending=False)
