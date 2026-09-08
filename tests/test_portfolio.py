import datetime as dt

import pandas as pd
import pytest

from costs.fees import CostModel
from costs.slippage import SlippageModel
from engine.portfolio_backtest import (
    equal_weight_buy_hold,
    month_start_rebalance_dates,
    run_portfolio_backtest,
)
from strategy.portfolio import PortfolioStrategy, TopNMomentum

ZERO_COST = CostModel(
    brokerage_rate=0, brokerage_flat=0, stt_rate=0, exchange_txn_rate=0,
    sebi_rate=0, stamp_duty_rate=0, gst_rate=0, dp_charge=0,
)
ZERO_SLIPPAGE = SlippageModel(bps=0)


def _long_prices(spec: dict[str, list[float]], dates: list[dt.date]) -> pd.DataFrame:
    rows = []
    for symbol, closes in spec.items():
        for d, c in zip(dates, closes):
            rows.append({"symbol": symbol, "date": d, "adj_open": c, "adj_close": c})
    return pd.DataFrame(rows)


class HoldFixed(PortfolioStrategy):
    """Test double: always targets the same fixed weights."""

    def __init__(self, targets: dict[str, float]):
        self._targets = targets
        self.seen_history_lengths: list[int] = []

    def select(self, close_history: pd.DataFrame) -> dict[str, float]:
        self.seen_history_lengths.append(len(close_history))
        return self._targets


def test_month_start_rebalance_dates_picks_first_trading_day_of_each_month():
    dates = [
        dt.date(2024, 1, 2), dt.date(2024, 1, 15), dt.date(2024, 1, 31),
        dt.date(2024, 2, 1), dt.date(2024, 2, 20),
        dt.date(2024, 3, 4),
    ]
    assert month_start_rebalance_dates(dates) == {
        dt.date(2024, 1, 2), dt.date(2024, 2, 1), dt.date(2024, 3, 4)
    }


def test_strategy_never_sees_the_close_of_the_day_it_trades_on():
    dates = [dt.date(2024, 1, 1), dt.date(2024, 2, 1), dt.date(2024, 3, 1)]
    prices = _long_prices({"A": [10.0, 20.0, 30.0]}, dates)
    strategy = HoldFixed({"A": 1.0})

    run_portfolio_backtest(
        prices, strategy, cost_model=ZERO_COST, slippage_model=ZERO_SLIPPAGE, initial_cash=1000.0
    )

    # Rebalances happen on dates[1] and dates[2] (dates[0] is skipped:
    # no prior history). History lengths must be 1 and 2 -- i.e. strictly
    # the closes BEFORE the execution day.
    assert strategy.seen_history_lengths == [1, 2]


def test_buys_at_execution_day_open_and_marks_to_close():
    dates = [dt.date(2024, 1, 1), dt.date(2024, 2, 1)]
    prices = pd.DataFrame(
        [
            {"symbol": "A", "date": dates[0], "adj_open": 10.0, "adj_close": 10.0},
            # open 20 (execution price), close 25 (mark-to-market price)
            {"symbol": "A", "date": dates[1], "adj_open": 20.0, "adj_close": 25.0},
        ]
    )
    result = run_portfolio_backtest(
        prices, HoldFixed({"A": 1.0}), cost_model=ZERO_COST,
        slippage_model=ZERO_SLIPPAGE, initial_cash=1000.0,
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.symbol == "A"
    assert trade.price == pytest.approx(20.0)  # filled at the open, not the close
    assert trade.quantity == 50  # floor(1000 / 20)

    last = result.equity_curve.iloc[-1]
    assert last["holdings_value"] == pytest.approx(50 * 25.0)  # marked at the close
    assert last["equity"] == pytest.approx(0.0 + 50 * 25.0)


def test_rebalance_sells_dropped_names_and_buys_new_ones():
    dates = [dt.date(2024, 1, 1), dt.date(2024, 2, 1), dt.date(2024, 3, 1)]
    prices = _long_prices({"A": [10.0, 10.0, 10.0], "B": [10.0, 10.0, 10.0]}, dates)

    class SwitchAtoB(PortfolioStrategy):
        def select(self, close_history):
            return {"A": 1.0} if len(close_history) == 1 else {"B": 1.0}

    result = run_portfolio_backtest(
        prices, SwitchAtoB(), cost_model=ZERO_COST, slippage_model=ZERO_SLIPPAGE, initial_cash=1000.0
    )

    sides = [(t.symbol, t.side) for t in result.trades]
    assert ("A", "buy") in sides
    assert ("A", "sell") in sides
    assert ("B", "buy") in sides
    # ends holding only B
    assert result.equity_curve.iloc[-1]["n_positions"] == 1


def test_real_costs_reduce_final_equity():
    dates = [dt.date(2024, 1, 1), dt.date(2024, 2, 1), dt.date(2024, 3, 1)]
    prices = _long_prices({"A": [100.0, 100.0, 100.0], "B": [100.0, 100.0, 100.0]}, dates)
    strategy_free = HoldFixed({"A": 0.5, "B": 0.5})
    strategy_paid = HoldFixed({"A": 0.5, "B": 0.5})

    free = run_portfolio_backtest(
        prices, strategy_free, cost_model=ZERO_COST, slippage_model=ZERO_SLIPPAGE, initial_cash=100_000.0
    )
    paid = run_portfolio_backtest(
        prices, strategy_paid, cost_model=CostModel(), slippage_model=SlippageModel(), initial_cash=100_000.0
    )

    assert paid.equity_curve["equity"].iloc[-1] < free.equity_curve["equity"].iloc[-1]


def test_top_n_momentum_picks_highest_trailing_return():
    dates = [dt.date(2024, 1, i + 1) for i in range(5)]
    closes = pd.DataFrame(
        {
            "WINNER": [100.0, 110.0, 120.0, 130.0, 200.0],
            "MIDDLE": [100.0, 100.0, 100.0, 100.0, 150.0],
            "LOSER": [100.0, 90.0, 80.0, 70.0, 50.0],
        },
        index=dates,
    )
    targets = TopNMomentum(n=2, lookback_days=4).select(closes)

    assert set(targets) == {"WINNER", "MIDDLE"}
    assert targets["WINNER"] == pytest.approx(0.5)


def test_top_n_momentum_returns_empty_without_enough_history():
    closes = pd.DataFrame({"A": [100.0, 101.0]}, index=[dt.date(2024, 1, 1), dt.date(2024, 1, 2)])
    assert TopNMomentum(n=1, lookback_days=10).select(closes) == {}


def test_top_n_momentum_absolute_filter_goes_to_cash_when_everything_fell():
    dates = [dt.date(2024, 1, i + 1) for i in range(3)]
    closes = pd.DataFrame({"A": [100.0, 90.0, 80.0], "B": [100.0, 95.0, 70.0]}, index=dates)

    assert TopNMomentum(n=2, lookback_days=2, min_return=0.0).select(closes) == {}
    # without the filter it still holds "the best of a bad lot"
    assert set(TopNMomentum(n=2, lookback_days=2).select(closes)) == {"A", "B"}


def test_equal_weight_buy_hold_benchmark():
    dates = [dt.date(2024, 1, 1), dt.date(2024, 2, 1)]
    prices = _long_prices({"A": [100.0, 200.0], "B": [100.0, 100.0]}, dates)

    curve = equal_weight_buy_hold(prices, initial_cash=1000.0)

    assert curve["equity"].iloc[0] == pytest.approx(1000.0)
    # A doubles (500 -> 1000), B flat (500) => 1500
    assert curve["equity"].iloc[-1] == pytest.approx(1500.0)


def test_point_in_time_universe_excludes_not_yet_liquid_and_delisted():
    from strategy.portfolio import PointInTimeUniverse

    dates = [dt.date(2024, 1, i + 1) for i in range(10)]
    closes = pd.DataFrame(
        {
            "LIQUID": [100.0] * 10,
            "THIN": [100.0] * 10,
            "DELISTED": [100.0] * 5 + [None] * 5,   # stopped trading
        },
        index=dates,
    )
    turnover = pd.DataFrame(
        {"LIQUID": [1e9] * 10, "THIN": [1.0] * 10, "DELISTED": [1e9] * 5 + [None] * 5},
        index=dates,
    )
    sel = PointInTimeUniverse(top_n=2, lookback_days=10, min_history_days=8)
    picked = sel.select(closes, turnover)

    # DELISTED has no current price, so it can't be bought today.
    assert "DELISTED" not in picked
    # LIQUID outranks THIN on turnover.
    assert picked[0] == "LIQUID"


def test_point_in_time_universe_returns_nothing_before_enough_history():
    from strategy.portfolio import PointInTimeUniverse

    dates = [dt.date(2024, 1, i + 1) for i in range(3)]
    closes = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=dates)
    turnover = pd.DataFrame({"A": [1e9] * 3}, index=dates)
    assert PointInTimeUniverse(top_n=5, lookback_days=252, min_history_days=200).select(
        closes, turnover
    ) == []


def test_engine_only_shows_strategy_the_point_in_time_universe():
    """The strategy must never see a symbol the universe selector excluded."""
    dates = [dt.date(2024, 1, 1), dt.date(2024, 2, 1), dt.date(2024, 3, 1)]
    prices = _long_prices({"KEEP": [10.0, 10.0, 10.0], "DROP": [10.0, 10.0, 10.0]}, dates)
    prices["turnover"] = 1e6

    class OnlyKeep:
        def select(self, close_history, turnover_history):
            return ["KEEP"]

    seen = []

    class RecordingStrategy(PortfolioStrategy):
        def select(self, close_history):
            seen.append(list(close_history.columns))
            return {}

    run_portfolio_backtest(
        prices, RecordingStrategy(), cost_model=ZERO_COST,
        slippage_model=ZERO_SLIPPAGE, universe_selector=OnlyKeep(),
    )
    assert seen and all(cols == ["KEEP"] for cols in seen)
