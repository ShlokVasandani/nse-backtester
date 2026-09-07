"""Performance metrics computed from a backtest's equity curve and trade
log. This is the piece that turns "the strategy made X%" into "is that
actually good" -- a return by itself hides whether it came from a real,
repeatable edge or one lucky month, and whether the ride there was smooth
or a string of gut-wrenching drawdowns nobody would actually sit through.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from engine.backtest import Trade

TRADING_DAYS_PER_YEAR = 252


def daily_returns(equity_curve: pd.DataFrame) -> pd.Series:
    return equity_curve["equity"].pct_change().dropna()


def cagr(equity_curve: pd.DataFrame) -> float | None:
    """Annualized compound growth rate, using actual calendar days elapsed
    (not trading-day count) between the first and last equity curve dates.
    None if there's under a day of history or a non-positive start."""
    if len(equity_curve) < 2:
        return None
    start_equity = equity_curve["equity"].iloc[0]
    end_equity = equity_curve["equity"].iloc[-1]
    days = (equity_curve["date"].iloc[-1] - equity_curve["date"].iloc[0]).days
    if days <= 0 or start_equity <= 0:
        return None
    years = days / 365.25
    return (end_equity / start_equity) ** (1 / years) - 1


def sharpe_ratio(
    equity_curve: pd.DataFrame,
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float | None:
    """Annualized Sharpe ratio of daily equity returns over a (default 0%)
    annual risk-free rate. None with fewer than 2 return observations or
    zero return variance (can't divide by zero std)."""
    returns = daily_returns(equity_curve)
    if len(returns) < 2:
        return None
    daily_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess = returns - daily_rf
    std = excess.std(ddof=1)
    if std == 0:
        return None
    return (excess.mean() / std) * (periods_per_year**0.5)


def max_drawdown(equity_curve: pd.DataFrame) -> float:
    """Worst peak-to-trough decline as a negative fraction (e.g. -0.23 for
    a 23% drawdown). 0.0 for an empty or never-declining curve."""
    if equity_curve.empty:
        return 0.0
    equity = equity_curve["equity"]
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    return float(drawdown.min())


def round_trip_pnls(trades: list[Trade]) -> list[float]:
    """Realized cash P&L for each completed buy-then-sell round trip. The
    engine trades strictly alternate buy/sell for a single asset, so
    trades are paired sequentially."""
    pnls = []
    pending_buy: Trade | None = None
    for t in trades:
        if t.side == "buy":
            pending_buy = t
        elif t.side == "sell" and pending_buy is not None:
            buy_outflow = pending_buy.price * pending_buy.quantity + pending_buy.cost
            sell_inflow = t.price * t.quantity - t.cost
            pnls.append(sell_inflow - buy_outflow)
            pending_buy = None
    return pnls


def win_rate(trades: list[Trade]) -> float | None:
    """Fraction of completed round trips with positive P&L. None if there
    are no completed round trips yet (e.g. still holding, or no trades)."""
    pnls = round_trip_pnls(trades)
    if not pnls:
        return None
    wins = sum(1 for p in pnls if p > 0)
    return wins / len(pnls)


@dataclass(frozen=True)
class PerformanceSummary:
    start_date: object
    end_date: object
    initial_equity: float
    final_equity: float
    total_return: float  # fraction, e.g. 0.05 for +5%
    cagr: float | None
    sharpe: float | None
    max_drawdown: float
    n_trades: int
    n_round_trips: int
    win_rate: float | None


def summarize(
    equity_curve: pd.DataFrame, trades: list[Trade], *, risk_free_rate: float = 0.0
) -> PerformanceSummary:
    initial_equity = equity_curve["equity"].iloc[0]
    final_equity = equity_curve["equity"].iloc[-1]
    pnls = round_trip_pnls(trades)

    return PerformanceSummary(
        start_date=equity_curve["date"].iloc[0],
        end_date=equity_curve["date"].iloc[-1],
        initial_equity=initial_equity,
        final_equity=final_equity,
        total_return=final_equity / initial_equity - 1,
        cagr=cagr(equity_curve),
        sharpe=sharpe_ratio(equity_curve, risk_free_rate=risk_free_rate),
        max_drawdown=max_drawdown(equity_curve),
        n_trades=len(trades),
        n_round_trips=len(pnls),
        win_rate=win_rate(trades),
    )
