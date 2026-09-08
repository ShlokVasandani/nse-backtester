import datetime as dt

import pandas as pd
import pytest

from strategy.momentum import Momentum
from strategy.moving_average import MovingAverageCrossover

CLOSES = [1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1]


def _prices(closes: list[float]) -> pd.DataFrame:
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(len(closes))]
    return pd.DataFrame({"symbol": ["ACME"] * len(closes), "date": dates, "adj_close": closes})


def test_moving_average_crossover_rejects_invalid_windows():
    with pytest.raises(ValueError):
        MovingAverageCrossover(fast_window=10, slow_window=10)
    with pytest.raises(ValueError):
        MovingAverageCrossover(fast_window=20, slow_window=10)


def test_moving_average_crossover_signal():
    strategy = MovingAverageCrossover(fast_window=2, slow_window=3)
    prices = _prices(CLOSES)

    signals = strategy.generate_signals(prices)

    # Uptrend crosses long at idx 2 (first date with a full slow window),
    # flat again once the downtrend catches up at idx 8.
    assert signals.tolist() == [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]


def test_momentum_signal():
    strategy = Momentum(lookback_days=3)
    prices = _prices(CLOSES)

    signals = strategy.generate_signals(prices)

    assert signals.tolist() == [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]


def test_momentum_rejects_invalid_lookback():
    with pytest.raises(ValueError):
        Momentum(lookback_days=0)


def test_find_signal_changes_reports_only_transitions():
    strategy = MovingAverageCrossover(fast_window=2, slow_window=3)
    prices = _prices(CLOSES)

    changes = strategy.find_signal_changes(prices)

    assert changes["date"].tolist() == [
        dt.date(2024, 1, 1),  # starting state, always included
        dt.date(2024, 1, 3),  # flips to long
        dt.date(2024, 1, 9),  # flips back to flat
    ]
    assert changes["signal"].tolist() == [0, 1, 0]
