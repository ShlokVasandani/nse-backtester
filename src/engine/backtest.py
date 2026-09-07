"""Single-symbol backtest engine: walks a strategy's signals through time
and simulates real execution -- no lookahead, no free fills.

Timing: a strategy's signal for day D is computed from data available
through day D's close. It can only be acted on starting the next session,
so the position "desired as of the open of day D" is the signal from day
D-1's close. Orders execute at day D's *open* (with slippage applied), not
its close -- using the same day's close to both decide and execute would
be lookahead bias.

Position sizing: single asset, all-in/all-out. On a flip to long, buy as
many whole shares as available cash covers (price + trading costs). On a
flip to flat, sell the entire position. No fractional shares, no leverage,
no shorting.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from costs.fees import CostModel
from costs.slippage import SlippageModel
from strategy.base import Strategy


@dataclass(frozen=True)
class Trade:
    date: object
    side: str  # "buy" | "sell"
    price: float
    quantity: int
    cost: float
    cash_after: float


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: pd.DataFrame  # columns: date, cash, shares, holdings_value, equity
    trades: list[Trade]


def _max_affordable_quantity(cash: float, price: float, cost_model: CostModel) -> int:
    """Largest whole-share quantity whose price + buy-side costs fit in
    cash. Trading costs are a function of quantity (flat fees, GST), so
    this can't be solved in closed form -- decrement from the naive upper
    bound until it fits. Realistic retail order sizes make this cheap."""
    if price <= 0 or cash <= 0:
        return 0
    qty = int(cash // price)
    while qty > 0:
        if price * qty + cost_model.buy_cost(price, qty).total <= cash:
            return qty
        qty -= 1
    return 0


def run_backtest(
    prices: pd.DataFrame,
    strategy: Strategy,
    *,
    cost_model: CostModel | None = None,
    slippage_model: SlippageModel | None = None,
    initial_cash: float = 100_000.0,
) -> BacktestResult:
    """prices: one symbol's adjusted history (columns include date,
    adj_open, adj_close), sorted ascending, as returned by
    adjust.store.read_prices."""
    cost_model = cost_model or CostModel()
    slippage_model = slippage_model or SlippageModel()

    prices = prices.reset_index(drop=True)
    signals = strategy.generate_signals(prices).reset_index(drop=True)
    # Signal from day D-1's close becomes the desired position for day D;
    # day 0 has no prior signal, so it starts flat.
    target_position = signals.shift(1).fillna(0).astype(int)

    cash = initial_cash
    shares = 0
    equity_rows = []
    trades: list[Trade] = []

    for i in range(len(prices)):
        row = prices.iloc[i]
        desired = target_position.iloc[i]

        if desired == 1 and shares == 0:
            exec_price = slippage_model.apply(row["adj_open"], "buy")
            qty = _max_affordable_quantity(cash, exec_price, cost_model)
            if qty > 0:
                cost = cost_model.buy_cost(exec_price, qty)
                cash -= exec_price * qty + cost.total
                shares = qty
                trades.append(
                    Trade(date=row["date"], side="buy", price=exec_price, quantity=qty, cost=cost.total, cash_after=cash)
                )
        elif desired == 0 and shares > 0:
            exec_price = slippage_model.apply(row["adj_open"], "sell")
            cost = cost_model.sell_cost(exec_price, shares)
            cash += exec_price * shares - cost.total
            trades.append(
                Trade(date=row["date"], side="sell", price=exec_price, quantity=shares, cost=cost.total, cash_after=cash)
            )
            shares = 0

        holdings_value = shares * row["adj_close"]
        equity_rows.append(
            {
                "date": row["date"],
                "cash": cash,
                "shares": shares,
                "holdings_value": holdings_value,
                "equity": cash + holdings_value,
            }
        )

    return BacktestResult(equity_curve=pd.DataFrame(equity_rows), trades=trades)
