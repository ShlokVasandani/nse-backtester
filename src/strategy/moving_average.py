from __future__ import annotations

import pandas as pd

from strategy.base import Strategy


class MovingAverageCrossover(Strategy):
    """Long when the fast SMA is above the slow SMA, flat otherwise.
    Classic trend-following rule; no signal until enough history exists
    for the slow window (NaN comparisons resolve to flat)."""

    def __init__(self, fast_window: int = 50, slow_window: int = 200):
        if fast_window >= slow_window:
            raise ValueError("fast_window must be shorter than slow_window")
        self.fast_window = fast_window
        self.slow_window = slow_window

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        close = prices["adj_close"]
        fast = close.rolling(self.fast_window).mean()
        slow = close.rolling(self.slow_window).mean()
        return (fast > slow).astype(int).rename("signal")
