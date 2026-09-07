import datetime as dt

import pandas as pd
import pytest

from analysis.sweep import aggregate_by_strategy, select_liquid_universe


def test_select_liquid_universe_excludes_partial_history_and_ranks_by_turnover():
    dates = [dt.date(2024, 1, 1), dt.date(2024, 1, 2), dt.date(2024, 1, 3)]
    rows = []
    # FULL_LOW: all 3 days, low turnover
    for d in dates:
        rows.append({"symbol": "FULL_LOW", "date": d, "turnover": 1000.0})
    # FULL_HIGH: all 3 days, high turnover
    for d in dates:
        rows.append({"symbol": "FULL_HIGH", "date": d, "turnover": 9000.0})
    # PARTIAL: only 2 of 3 days -- a listing gap, should be excluded
    for d in dates[:2]:
        rows.append({"symbol": "PARTIAL", "date": d, "turnover": 50000.0})
    all_prices = pd.DataFrame(rows)

    universe = select_liquid_universe(all_prices, top_n=10)

    assert "PARTIAL" not in universe
    assert universe == ["FULL_HIGH", "FULL_LOW"]


def test_select_liquid_universe_respects_top_n():
    dates = [dt.date(2024, 1, 1), dt.date(2024, 1, 2)]
    rows = []
    for i, turnover in enumerate([100.0, 200.0, 300.0]):
        for d in dates:
            rows.append({"symbol": f"SYM{i}", "date": d, "turnover": turnover})
    all_prices = pd.DataFrame(rows)

    universe = select_liquid_universe(all_prices, top_n=2)

    assert universe == ["SYM2", "SYM1"]


def test_aggregate_by_strategy_computes_hit_rates():
    sweep_results = pd.DataFrame(
        [
            {
                "symbol": "A",
                "strategy": "ma_10_50",
                "total_return": 0.10,
                "cagr": 0.20,
                "sharpe": 1.0,
                "max_drawdown": -0.05,
                "n_round_trips": 2,
                "win_rate": 0.5,
                "bh_total_return": 0.05,
                "bh_sharpe": 0.5,
                "beats_bh_return": True,
                "beats_bh_sharpe": True,
            },
            {
                "symbol": "B",
                "strategy": "ma_10_50",
                "total_return": -0.02,
                "cagr": -0.04,
                "sharpe": -0.2,
                "max_drawdown": -0.10,
                "n_round_trips": 1,
                "win_rate": 0.0,
                "bh_total_return": 0.08,
                "bh_sharpe": 0.6,
                "beats_bh_return": False,
                "beats_bh_sharpe": False,
            },
        ]
    )

    agg = aggregate_by_strategy(sweep_results)

    assert agg.loc["ma_10_50", "n_symbols"] == 2
    assert agg.loc["ma_10_50", "beats_bh_return_rate"] == pytest.approx(0.5)
    assert agg.loc["ma_10_50", "beats_bh_sharpe_rate"] == pytest.approx(0.5)


def test_aggregate_by_strategy_empty_input():
    assert aggregate_by_strategy(pd.DataFrame()).empty
