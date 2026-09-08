"""Portfolio backtest engine: hold N stocks at once, rebalanced
periodically.

Same no-lookahead discipline as the single-symbol engine: targets for a
rebalance are computed from closes strictly BEFORE the execution day, and
the resulting orders fill at that day's open (with slippage). A strategy
is handed a price history slice that physically ends the day before it
trades, so it cannot peek.

What this fixes versus engine/backtest.py: that engine is all-in/all-out
on a single name, so every result it produces is a concentrated bet. This
one holds a basket, which is what actual investing looks like and what
makes a benchmark comparison meaningful.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd

from costs.fees import CostModel
from costs.slippage import SlippageModel
from engine.backtest import Trade, max_affordable_quantity
from strategy.portfolio import PortfolioStrategy


@dataclass(frozen=True)
class PortfolioBacktestResult:
    equity_curve: pd.DataFrame  # date, cash, holdings_value, equity, n_positions
    trades: list[Trade]


def to_wide(all_prices: pd.DataFrame, column: str) -> pd.DataFrame:
    """Long (symbol, date, ...) -> wide (index=date, columns=symbol)."""
    return all_prices.pivot(index="date", columns="symbol", values=column).sort_index()


def month_start_rebalance_dates(dates: list[dt.date]) -> set[dt.date]:
    """First available trading day of each calendar month."""
    seen: set[tuple[int, int]] = set()
    out: set[dt.date] = set()
    for d in dates:
        key = (d.year, d.month)
        if key not in seen:
            seen.add(key)
            out.add(d)
    return out


def _execute_rebalance(
    targets: dict[str, float],
    open_row: pd.Series,
    holdings: dict[str, int],
    cash: float,
    cost_model: CostModel,
    slippage_model: SlippageModel,
    date,
    trades: list[Trade],
) -> float:
    """Move `holdings` toward `targets` at this day's open prices.
    Returns updated cash. Sells run first so their proceeds fund the buys.
    """
    # Value the book at executable prices. A symbol with no price today
    # (didn't trade / not listed yet) is untradeable, so it's held at its
    # last known value implicitly by simply not being touched.
    portfolio_value = cash
    for symbol, shares in holdings.items():
        price = open_row.get(symbol)
        if price is not None and pd.notna(price):
            portfolio_value += shares * price

    desired: dict[str, int] = {}
    for symbol, weight in targets.items():
        price = open_row.get(symbol)
        if price is None or pd.isna(price) or price <= 0:
            continue
        exec_price = slippage_model.apply(price, "buy")
        desired[symbol] = int((portfolio_value * weight) // exec_price)

    # --- sells: full exits and trims ---
    for symbol, shares in list(holdings.items()):
        target_shares = desired.get(symbol, 0)
        if target_shares >= shares:
            continue
        price = open_row.get(symbol)
        if price is None or pd.isna(price) or price <= 0:
            continue  # can't sell what has no price today; carry the position
        quantity = shares - target_shares
        exec_price = slippage_model.apply(price, "sell")
        cost = cost_model.sell_cost(exec_price, quantity).total
        cash += exec_price * quantity - cost
        remaining = shares - quantity
        if remaining:
            holdings[symbol] = remaining
        else:
            del holdings[symbol]
        trades.append(
            Trade(
                date=date, side="sell", price=exec_price, quantity=quantity,
                cost=cost, cash_after=cash, symbol=symbol,
            )
        )

    # --- buys: new positions and top-ups, largest shortfall first ---
    shortfalls = sorted(
        ((s, q - holdings.get(s, 0)) for s, q in desired.items() if q > holdings.get(s, 0)),
        key=lambda item: item[1],
        reverse=True,
    )
    for symbol, shortfall in shortfalls:
        price = open_row.get(symbol)
        exec_price = slippage_model.apply(price, "buy")
        quantity = min(shortfall, max_affordable_quantity(cash, exec_price, cost_model))
        if quantity <= 0:
            continue
        cost = cost_model.buy_cost(exec_price, quantity).total
        cash -= exec_price * quantity + cost
        holdings[symbol] = holdings.get(symbol, 0) + quantity
        trades.append(
            Trade(
                date=date, side="buy", price=exec_price, quantity=quantity,
                cost=cost, cash_after=cash, symbol=symbol,
            )
        )

    return cash


def run_portfolio_backtest(
    all_prices: pd.DataFrame,
    strategy: PortfolioStrategy,
    *,
    cost_model: CostModel | None = None,
    slippage_model: SlippageModel | None = None,
    initial_cash: float = 100_000.0,
    rebalance_dates: set[dt.date] | None = None,
    universe_selector=None,
) -> PortfolioBacktestResult:
    """all_prices: long format with columns symbol, date, adj_open,
    adj_close (as returned by adjust.store.read_prices). Rebalances at the
    start of each month unless `rebalance_dates` is given.

    `universe_selector` (see strategy.portfolio.PointInTimeUniverse) is
    consulted at each rebalance with history up to that point, and the
    strategy only ever sees the symbols it returns. Without one, the
    strategy sees every symbol in `all_prices`, which for a fixed
    pre-filtered universe means look-ahead and survivorship bias."""
    cost_model = cost_model or CostModel()
    slippage_model = slippage_model or SlippageModel()

    closes = to_wide(all_prices, "adj_close")
    opens = to_wide(all_prices, "adj_open")
    turnovers = to_wide(all_prices, "turnover") if universe_selector is not None else None
    dates = list(closes.index)
    if rebalance_dates is None:
        rebalance_dates = month_start_rebalance_dates(dates)

    cash = initial_cash
    holdings: dict[str, int] = {}
    trades: list[Trade] = []
    equity_rows = []

    for i, date in enumerate(dates):
        # i > 0 so the strategy always sees at least one prior close, and
        # never the close of the day it trades on.
        if i > 0 and date in rebalance_dates:
            history = closes.iloc[:i]
            if universe_selector is not None:
                eligible = universe_selector.select(history, turnovers.iloc[:i])
                history = history[eligible] if eligible else history.iloc[:, :0]
            targets = strategy.select(history)
            if targets:
                cash = _execute_rebalance(
                    targets, opens.iloc[i], holdings, cash,
                    cost_model, slippage_model, date, trades,
                )

        close_row = closes.iloc[i]
        holdings_value = 0.0
        for symbol, shares in holdings.items():
            price = close_row.get(symbol)
            if price is not None and pd.notna(price):
                holdings_value += shares * price

        equity_rows.append(
            {
                "date": date,
                "cash": cash,
                "holdings_value": holdings_value,
                "equity": cash + holdings_value,
                "n_positions": len(holdings),
            }
        )

    return PortfolioBacktestResult(equity_curve=pd.DataFrame(equity_rows), trades=trades)


def equal_weight_buy_hold(all_prices: pd.DataFrame, initial_cash: float = 100_000.0) -> pd.DataFrame:
    """Benchmark: split capital equally across every symbol on day one and
    hold. Returns an equity curve comparable to run_portfolio_backtest's.

    This is the honest benchmark for a portfolio strategy -- comparing a
    20-stock basket against a single stock's buy-and-hold would flatter or
    punish it for reasons that have nothing to do with the rule.
    """
    closes = to_wide(all_prices, "adj_close")
    first_row = closes.iloc[0].dropna()
    if first_row.empty:
        return pd.DataFrame(columns=["date", "equity"])

    per_symbol_cash = initial_cash / len(first_row)
    units = per_symbol_cash / first_row  # fractional units: a pure index-like benchmark

    equity = closes[first_row.index].mul(units, axis=1).ffill().sum(axis=1)
    return pd.DataFrame({"date": closes.index, "equity": equity.to_numpy()})
