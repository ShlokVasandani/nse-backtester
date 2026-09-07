"""Strategy interface.

A strategy takes one symbol's adjusted price history (sorted ascending by
date, as returned by adjust.store.read_prices) and emits a signal series
aligned to the same index: 1 = long, 0 = flat. No shorting -- this is for
NSE cash equities, not derivatives.

Signals are mechanical output of whatever rule the strategy encodes; they
are not investment advice. Whether a rule is worth trusting is what the
backtest engine (src/engine) and metrics (src/metrics) are for.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    @abstractmethod
    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        """prices: columns include at least 'date' and 'adj_close', one
        symbol, sorted ascending by date. Returns an int series (0/1)
        aligned to prices.index."""

    def find_signal_changes(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Dates where the signal flips, with the new signal value.
        The first row is always included (the starting state)."""
        signals = self.generate_signals(prices)
        changed = signals.ne(signals.shift(1))
        changed.iloc[0] = True

        out = pd.DataFrame(
            {
                "date": prices.loc[changed, "date"].to_numpy(),
                "signal": signals.loc[changed].to_numpy(),
            }
        )
        return out.reset_index(drop=True)
