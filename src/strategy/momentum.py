from __future__ import annotations

import pandas as pd

from strategy.base import Strategy


class Momentum(Strategy):
    """Long when the trailing N-day return is positive, flat otherwise.
    No signal until N days of history exist."""

    def __init__(self, lookback_days: int = 90):
        if lookback_days < 1:
            raise ValueError("lookback_days must be at least 1")
        self.lookback_days = lookback_days

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        close = prices["adj_close"]
        trailing_return = close.pct_change(self.lookback_days)
        return (trailing_return > 0).astype(int).rename("signal")
