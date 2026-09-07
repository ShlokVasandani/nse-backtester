"""Slippage: the gap between a signal's reference price and the price you'd
actually get filled at. A backtest that assumes perfect fills at the
closing price overstates returns -- this model makes that assumption
explicit and adjustable instead of silently ignoring it.

Fixed basis-points model: buys are assumed to fill worse (higher) and
sells worse (lower) than the reference price, by `bps` basis points.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Side = Literal["buy", "sell"]


@dataclass(frozen=True)
class SlippageModel:
    bps: float = 5.0  # basis points; 5 bps = 0.05%

    def apply(self, price: float, side: Side) -> float:
        factor = self.bps / 10_000
        if side == "buy":
            return price * (1 + factor)
        if side == "sell":
            return price * (1 - factor)
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
