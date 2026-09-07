import datetime as dt

import pandas as pd
import pytest

from costs.fees import CostModel
from costs.slippage import SlippageModel
from engine.backtest import run_backtest
from strategy.base import Strategy

ZERO_COST = CostModel(
    brokerage_rate=0,
    brokerage_flat=0,
    stt_rate=0,
    exchange_txn_rate=0,
    sebi_rate=0,
    stamp_duty_rate=0,
    gst_rate=0,
    dp_charge=0,
)
ZERO_SLIPPAGE = SlippageModel(bps=0)


class FixedSignalStrategy(Strategy):
    """Test double: returns a hardcoded signal series regardless of the
    price data passed in, so engine mechanics can be tested independent of
    any real strategy's math."""

    def __init__(self, signals: list[int]):
        self._signals = signals

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        return pd.Series(self._signals, index=prices.index, name="signal")


def _prices() -> pd.DataFrame:
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(5)]
    return pd.DataFrame(
        {
            "date": dates,
            "adj_open": [100.0, 102.0, 104.0, 106.0, 108.0],
            "adj_close": [101.0, 103.0, 105.0, 107.0, 109.0],
        }
    )


def test_engine_executes_at_next_day_open_not_same_day_close():
    # signal (close-based): long,long,flat,flat,long
    # -> desired position at each day's open: flat,long,long,flat,flat
    # -> buy on day1's open (102), sell on day3's open (106)
    strategy = FixedSignalStrategy([1, 1, 0, 0, 1])
    result = run_backtest(
        _prices(), strategy, cost_model=ZERO_COST, slippage_model=ZERO_SLIPPAGE, initial_cash=10_000.0
    )

    assert len(result.trades) == 2
    buy, sell = result.trades

    assert buy.side == "buy"
    assert buy.date == dt.date(2024, 1, 2)  # day 1
    assert buy.price == pytest.approx(102.0)
    assert buy.quantity == 98  # floor(10000 / 102)
    assert buy.cash_after == pytest.approx(10_000.0 - 98 * 102.0)

    assert sell.side == "sell"
    assert sell.date == dt.date(2024, 1, 4)  # day 3
    assert sell.price == pytest.approx(106.0)
    assert sell.quantity == 98
    assert sell.cash_after == pytest.approx(buy.cash_after + 98 * 106.0)


def test_engine_equity_curve_marks_to_market_at_close():
    strategy = FixedSignalStrategy([1, 1, 0, 0, 1])
    result = run_backtest(
        _prices(), strategy, cost_model=ZERO_COST, slippage_model=ZERO_SLIPPAGE, initial_cash=10_000.0
    )
    eq = result.equity_curve

    assert eq["equity"].iloc[0] == pytest.approx(10_000.0)  # flat, no trade yet
    assert eq["shares"].iloc[1] == 98
    assert eq["equity"].iloc[1] == pytest.approx(4.0 + 98 * 103.0)  # cash + mark-to-close
    assert eq["equity"].iloc[2] == pytest.approx(4.0 + 98 * 105.0)
    assert eq["shares"].iloc[3] == 0  # sold on day 3's open
    assert eq["equity"].iloc[3] == pytest.approx(eq["cash"].iloc[3])
    assert eq["equity"].iloc[4] == pytest.approx(eq["equity"].iloc[3])  # no further trades


def test_engine_never_trades_when_always_flat():
    strategy = FixedSignalStrategy([0, 0, 0, 0, 0])
    result = run_backtest(_prices(), strategy, cost_model=ZERO_COST, slippage_model=ZERO_SLIPPAGE)

    assert result.trades == []
    assert (result.equity_curve["shares"] == 0).all()


def test_engine_deducts_real_trading_costs():
    strategy = FixedSignalStrategy([1, 1, 0, 0, 1])
    zero_cost_result = run_backtest(
        _prices(), strategy, cost_model=ZERO_COST, slippage_model=ZERO_SLIPPAGE, initial_cash=10_000.0
    )
    real_cost_result = run_backtest(
        _prices(), strategy, cost_model=CostModel(), slippage_model=SlippageModel(), initial_cash=10_000.0
    )

    # Real costs + slippage should leave strictly less final equity than
    # the zero-cost/zero-slippage baseline.
    assert real_cost_result.equity_curve["equity"].iloc[-1] < zero_cost_result.equity_curve["equity"].iloc[-1]
    assert real_cost_result.trades[0].cost > 0
    assert real_cost_result.trades[1].cost > 0


def test_engine_skips_buy_when_cash_cannot_afford_a_single_share():
    strategy = FixedSignalStrategy([1, 1])
    prices = pd.DataFrame(
        {
            "date": [dt.date(2024, 1, 1), dt.date(2024, 1, 2)],
            "adj_open": [100.0, 100.0],
            "adj_close": [100.0, 100.0],
        }
    )
    result = run_backtest(prices, strategy, cost_model=ZERO_COST, slippage_model=ZERO_SLIPPAGE, initial_cash=50.0)

    assert result.trades == []
    assert (result.equity_curve["equity"] == 50.0).all()
