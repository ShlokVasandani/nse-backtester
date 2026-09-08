"""Portfolio strategy interface: cross-sectional selection, not per-stock
signals.

The single-symbol Strategy in strategy/base.py answers "am I in or out of
this one stock." That shape forces 100%-in-one-name concentration, which
is what made every earlier backtest a series of concentrated bets rather
than investing. A PortfolioStrategy instead looks across the whole
universe on a rebalance date and answers "which N of these do I hold, at
what weights."
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class PointInTimeUniverse:
    """Pick the tradeable universe using ONLY data available on the
    rebalance date.

    This replaces selecting one universe up front from the whole backtest
    window, which is doubly biased: ranking by turnover over the full
    period is look-ahead (a stock that only became liquid in 2024 gets
    included in 2015), and requiring an unbroken listing history is
    survivorship (companies that were delisted or went bust are silently
    dropped, so the benchmark only ever holds winners).

    Because the bhavcopy files contain every symbol that traded on each
    day, selecting per-rebalance from what was actually trading then fixes
    both: a company that later collapses is held while it was real, and a
    company not yet liquid simply isn't eligible yet.
    """

    def __init__(self, top_n: int = 100, lookback_days: int = 252, min_history_days: int = 200):
        self.top_n = top_n
        self.lookback_days = lookback_days
        self.min_history_days = min_history_days

    def select(self, close_history: pd.DataFrame, turnover_history: pd.DataFrame) -> list[str]:
        window_close = close_history.iloc[-self.lookback_days :]
        window_turnover = turnover_history.iloc[-self.lookback_days :]
        if len(window_close) < self.min_history_days:
            return []

        # Must have actually been trading for most of the window, and be
        # trading right now (a delisted name has no current price).
        traded_enough = window_close.notna().sum() >= self.min_history_days
        trading_now = close_history.iloc[-1].notna()
        eligible = traded_enough & trading_now

        avg_turnover = window_turnover[eligible.index[eligible]].mean().dropna()
        return list(avg_turnover.nlargest(self.top_n).index)


class PortfolioStrategy(ABC):
    @abstractmethod
    def select(self, close_history: pd.DataFrame) -> dict[str, float]:
        """close_history: wide frame, index = date ascending, columns =
        symbol, values = adjusted close. It contains ONLY data up to and
        including the signal date -- the engine slices it that way, so a
        strategy physically cannot peek at future prices.

        Returns {symbol: target weight}. Weights should sum to <= 1.0;
        anything left over stays in cash. An empty dict means hold cash.
        """


class EqualWeightAll(PortfolioStrategy):
    """Hold everything in the universe, equally weighted.

    This is the benchmark that makes a fair fight: run through the SAME
    engine, the SAME point-in-time universe, the SAME monthly rebalance and
    the SAME costs as the strategy under test, so the only difference is
    the selection rule. Comparing a top-20 momentum basket against a
    buy-and-hold of every symbol in the database instead would confound
    the rule with a completely different risk profile.
    """

    def select(self, close_history: pd.DataFrame) -> dict[str, float]:
        tradeable = close_history.iloc[-1].dropna().index
        if len(tradeable) == 0:
            return {}
        weight = 1.0 / len(tradeable)
        return {symbol: weight for symbol in tradeable}


class TopNMomentum(PortfolioStrategy):
    """Hold the N stocks with the highest trailing return, equal-weighted.

    This is cross-sectional (relative) momentum: it ranks stocks against
    each other and holds the leaders, rather than asking each stock in
    isolation whether it's trending. `min_return` optionally adds an
    absolute filter on top -- set it to 0.0 to refuse to hold anything
    that fell over the lookback, which turns the strategy defensive in a
    broad selloff instead of holding "the best of a bad lot".
    """

    def __init__(self, n: int = 20, lookback_days: int = 252, min_return: float | None = None):
        if n < 1:
            raise ValueError("n must be at least 1")
        if lookback_days < 1:
            raise ValueError("lookback_days must be at least 1")
        self.n = n
        self.lookback_days = lookback_days
        self.min_return = min_return

    def select(self, close_history: pd.DataFrame) -> dict[str, float]:
        if len(close_history) <= self.lookback_days:
            return {}  # not enough history to measure the lookback yet

        past = close_history.iloc[-1 - self.lookback_days]
        now = close_history.iloc[-1]
        trailing_return = (now / past - 1).dropna()

        if self.min_return is not None:
            trailing_return = trailing_return[trailing_return > self.min_return]
        if trailing_return.empty:
            return {}

        winners = trailing_return.nlargest(self.n)
        weight = 1.0 / len(winners)
        return {symbol: weight for symbol in winners.index}
