import datetime as dt

import pandas as pd
import pytest

from engine.backtest import Trade
from metrics.performance import (
    cagr,
    max_drawdown,
    round_trip_pnls,
    sharpe_ratio,
    summarize,
    win_rate,
)


def _equity_curve() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [dt.date(2024, 1, 1), dt.date(2024, 3, 1), dt.date(2024, 5, 1), dt.date(2024, 7, 1)],
            "equity": [100_000.0, 110_000.0, 99_000.0, 121_000.0],
        }
    )


def test_cagr_matches_hand_computed_value():
    result = cagr(_equity_curve())
    # 182 calendar days, 100k -> 121k
    assert result == pytest.approx(0.4660180634586992)


def test_cagr_none_for_single_row():
    assert cagr(_equity_curve().iloc[:1]) is None


def test_sharpe_ratio_matches_hand_computed_value():
    result = sharpe_ratio(_equity_curve())
    assert result == pytest.approx(7.228765761341866)


def test_sharpe_ratio_none_for_zero_variance():
    flat = pd.DataFrame(
        {
            "date": [dt.date(2024, 1, 1), dt.date(2024, 1, 2), dt.date(2024, 1, 3)],
            "equity": [100_000.0, 100_000.0, 100_000.0],
        }
    )
    assert sharpe_ratio(flat) is None


def test_max_drawdown_matches_hand_computed_value():
    assert max_drawdown(_equity_curve()) == pytest.approx(-0.1)


def test_max_drawdown_zero_for_monotonic_increase():
    rising = pd.DataFrame(
        {
            "date": [dt.date(2024, 1, 1), dt.date(2024, 1, 2), dt.date(2024, 1, 3)],
            "equity": [100.0, 110.0, 120.0],
        }
    )
    assert max_drawdown(rising) == 0.0


def _trade(date, side, price, quantity, cost):
    return Trade(date=date, side=side, price=price, quantity=quantity, cost=cost, cash_after=0.0)


def test_round_trip_pnls_pairs_buys_and_sells_sequentially():
    trades = [
        _trade(dt.date(2024, 1, 1), "buy", 100.0, 10, 5.0),  # outflow 1005
        _trade(dt.date(2024, 1, 5), "sell", 110.0, 10, 5.0),  # inflow 1095 -> +90
        _trade(dt.date(2024, 2, 1), "buy", 200.0, 5, 3.0),  # outflow 1003
        _trade(dt.date(2024, 2, 10), "sell", 180.0, 5, 3.0),  # inflow 897 -> -106
    ]
    pnls = round_trip_pnls(trades)
    assert pnls == pytest.approx([90.0, -106.0])


def test_win_rate_from_round_trips():
    trades = [
        _trade(dt.date(2024, 1, 1), "buy", 100.0, 10, 0.0),
        _trade(dt.date(2024, 1, 5), "sell", 110.0, 10, 0.0),  # win
        _trade(dt.date(2024, 2, 1), "buy", 200.0, 5, 0.0),
        _trade(dt.date(2024, 2, 10), "sell", 180.0, 5, 0.0),  # loss
        _trade(dt.date(2024, 3, 1), "buy", 50.0, 20, 0.0),
        _trade(dt.date(2024, 3, 10), "sell", 60.0, 20, 0.0),  # win
    ]
    assert win_rate(trades) == pytest.approx(2 / 3)


def test_win_rate_none_when_no_round_trips():
    assert win_rate([]) is None
    assert win_rate([_trade(dt.date(2024, 1, 1), "buy", 100.0, 10, 0.0)]) is None


def test_summarize_combines_all_metrics():
    equity_curve = _equity_curve()
    trades = [
        _trade(dt.date(2024, 1, 1), "buy", 100.0, 10, 0.0),
        _trade(dt.date(2024, 1, 5), "sell", 110.0, 10, 0.0),
    ]

    summary = summarize(equity_curve, trades)

    assert summary.initial_equity == 100_000.0
    assert summary.final_equity == 121_000.0
    assert summary.total_return == pytest.approx(0.21)
    assert summary.cagr == pytest.approx(0.4660180634586992)
    assert summary.max_drawdown == pytest.approx(-0.1)
    assert summary.n_trades == 2
    assert summary.n_round_trips == 1
    assert summary.win_rate == pytest.approx(1.0)
