import datetime as dt

import pandas as pd
import pytest

from analysis.sweep import train_test_validate
from strategy.moving_average import MovingAverageCrossover


def _trending_prices(symbol: str, start: dt.date, n_days: int, daily_return: float) -> pd.DataFrame:
    dates = [start + dt.timedelta(days=i) for i in range(n_days)]
    closes = [100.0 * (1 + daily_return) ** i for i in range(n_days)]
    return pd.DataFrame(
        {
            "symbol": [symbol] * n_days,
            "date": dates,
            "adj_open": closes,
            "adj_close": closes,
            "turnover": [1_000_000.0] * n_days,
        }
    )


@pytest.fixture
def fake_read_prices(monkeypatch):
    """Two windows: train (uptrend, long enough for a 5/10 MA to trade) and
    test (a *different* symbol set/trend), so we can check train_test_validate
    only ever touches the window it's told to."""
    train_start, train_end = dt.date(2024, 1, 1), dt.date(2024, 3, 1)
    test_start, test_end = dt.date(2024, 6, 1), dt.date(2024, 8, 1)

    train_df = pd.concat(
        [
            _trending_prices("UP", train_start, 40, 0.01),
            _trending_prices("FLAT", train_start, 40, 0.0),
        ],
        ignore_index=True,
    )
    test_df = pd.concat(
        [
            _trending_prices("UP", test_start, 40, 0.01),
            _trending_prices("FLAT", test_start, 40, 0.0),
        ],
        ignore_index=True,
    )

    def fake(symbol=None, start=None, end=None):
        if start == train_start:
            return train_df
        if start == test_start:
            return test_df
        raise AssertionError(f"unexpected read_prices call: start={start}")

    monkeypatch.setattr("analysis.sweep.read_prices", fake)
    return train_start, train_end, test_start, test_end


def test_train_test_validate_uses_train_window_to_pick_and_test_window_to_check(fake_read_prices):
    train_start, train_end, test_start, test_end = fake_read_prices
    grid = [("ma_5_10", MovingAverageCrossover(fast_window=5, slow_window=10))]

    result = train_test_validate(
        train_start, train_end, test_start, test_end, top_n=10, strategy_grid=grid
    )

    assert result.best_strategy == "ma_5_10"
    assert result.universe_size == 2
    assert not result.train_aggregate.empty
    # test_aggregate reflects the SAME strategy run on the test window only
    if not result.test_aggregate.empty:
        assert list(result.test_aggregate.index) == ["ma_5_10"]


def test_train_test_validate_raises_if_train_window_never_trades(fake_read_prices):
    train_start, train_end, test_start, test_end = fake_read_prices
    # slow_window longer than the train window's row count -> never enough
    # data for a signal -> no trades -> should fail loudly, not silently.
    grid = [("ma_5_100", MovingAverageCrossover(fast_window=5, slow_window=100))]

    with pytest.raises(ValueError):
        train_test_validate(train_start, train_end, test_start, test_end, top_n=10, strategy_grid=grid)
