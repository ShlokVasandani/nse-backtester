import pytest

from costs.fees import CostModel
from costs.slippage import SlippageModel


def test_buy_cost_breakdown_matches_expected_rates():
    cm = CostModel()
    b = cm.buy_cost(price=100.0, quantity=100)  # turnover = 10,000

    assert b.turnover == pytest.approx(10_000.0)
    assert b.brokerage == 0.0
    assert b.exchange_txn_charge == pytest.approx(0.307)
    assert b.sebi_charge == pytest.approx(0.01)
    assert b.stt == pytest.approx(10.0)
    assert b.stamp_duty == pytest.approx(1.5)
    assert b.dp_charge == 0.0  # no DP charge on buys
    assert b.gst == pytest.approx(0.05706)
    assert b.total == pytest.approx(11.87406)


def test_sell_cost_breakdown_has_dp_charge_and_no_stamp_duty():
    cm = CostModel()
    s = cm.sell_cost(price=100.0, quantity=100)

    assert s.stamp_duty == 0.0  # stamp duty is buy-side only
    assert s.dp_charge == pytest.approx(15.34)
    assert s.total == pytest.approx(25.71406)


def test_round_trip_cost_is_sum_of_buy_and_sell():
    cm = CostModel()
    buy = cm.buy_cost(100.0, 100)
    sell = cm.sell_cost(100.0, 100)

    assert cm.round_trip_cost(100.0, 100.0, 100) == pytest.approx(buy.total + sell.total)


def test_flat_brokerage_feeds_into_gst():
    cm = CostModel(brokerage_flat=20.0)
    b = cm.buy_cost(price=100.0, quantity=100)

    assert b.brokerage == pytest.approx(20.0)
    # GST base now includes the flat brokerage fee too.
    expected_gst_base = 20.0 + b.exchange_txn_charge + b.sebi_charge
    assert b.gst == pytest.approx(expected_gst_base * 0.18)


def test_sell_zero_quantity_has_no_dp_charge():
    cm = CostModel()
    s = cm.sell_cost(price=100.0, quantity=0)
    assert s.dp_charge == 0.0
    assert s.turnover == 0.0


def test_slippage_buy_increases_price_and_sell_decreases():
    model = SlippageModel(bps=10.0)  # 0.10%
    assert model.apply(100.0, "buy") == pytest.approx(100.10)
    assert model.apply(100.0, "sell") == pytest.approx(99.90)


def test_slippage_invalid_side_raises():
    model = SlippageModel()
    with pytest.raises(ValueError):
        model.apply(100.0, "hold")
