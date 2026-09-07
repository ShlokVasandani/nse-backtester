"""NSE equity delivery (CNC) trading cost model.

Defaults match current discount-broker (Zerodha-style) delivery charges:
zero brokerage, plus the government/exchange/regulatory charges that apply
regardless of broker:

  - STT (Securities Transaction Tax): 0.1% of turnover, BOTH buy and sell
  - Exchange transaction charge (NSE): 0.00307% of turnover, both sides
  - SEBI turnover fee: Rs 10 per crore (0.0001%), both sides
  - Stamp duty: 0.015% of turnover, BUY side only
  - GST: 18% on (brokerage + exchange transaction charge + SEBI fee) --
    not charged on STT or stamp duty, which are taxes themselves
  - DP (depository participant) charge: flat fee per scrip on SELL only,
    independent of quantity/value (Zerodha: Rs 15.34)

All rates are constructor parameters so a different broker's fee schedule
(or a future budget change to STT/stamp duty) doesn't require touching the
engine that calls this.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostBreakdown:
    turnover: float
    brokerage: float
    exchange_txn_charge: float
    sebi_charge: float
    stt: float
    stamp_duty: float
    dp_charge: float
    gst: float

    @property
    def total(self) -> float:
        return (
            self.brokerage
            + self.exchange_txn_charge
            + self.sebi_charge
            + self.stt
            + self.stamp_duty
            + self.dp_charge
            + self.gst
        )


@dataclass(frozen=True)
class CostModel:
    brokerage_rate: float = 0.0  # fraction of turnover; 0.0 = discount-broker delivery default
    brokerage_flat: float = 0.0  # flat fee per trade, added on top of brokerage_rate
    stt_rate: float = 0.001  # 0.1%, both sides
    exchange_txn_rate: float = 0.0000307  # 0.00307%, both sides
    sebi_rate: float = 0.000001  # Rs 10/crore, both sides
    stamp_duty_rate: float = 0.00015  # 0.015%, buy side only
    gst_rate: float = 0.18  # on brokerage + exchange txn + SEBI charge
    dp_charge: float = 15.34  # flat, sell side only

    def _brokerage(self, turnover: float) -> float:
        return self.brokerage_flat + turnover * self.brokerage_rate

    def buy_cost(self, price: float, quantity: int) -> CostBreakdown:
        turnover = price * quantity
        brokerage = self._brokerage(turnover)
        exchange_txn_charge = turnover * self.exchange_txn_rate
        sebi_charge = turnover * self.sebi_rate
        stt = turnover * self.stt_rate
        stamp_duty = turnover * self.stamp_duty_rate
        gst = (brokerage + exchange_txn_charge + sebi_charge) * self.gst_rate

        return CostBreakdown(
            turnover=turnover,
            brokerage=brokerage,
            exchange_txn_charge=exchange_txn_charge,
            sebi_charge=sebi_charge,
            stt=stt,
            stamp_duty=stamp_duty,
            dp_charge=0.0,
            gst=gst,
        )

    def sell_cost(self, price: float, quantity: int) -> CostBreakdown:
        turnover = price * quantity
        brokerage = self._brokerage(turnover)
        exchange_txn_charge = turnover * self.exchange_txn_rate
        sebi_charge = turnover * self.sebi_rate
        stt = turnover * self.stt_rate
        dp_charge = self.dp_charge if quantity > 0 else 0.0
        gst = (brokerage + exchange_txn_charge + sebi_charge) * self.gst_rate

        return CostBreakdown(
            turnover=turnover,
            brokerage=brokerage,
            exchange_txn_charge=exchange_txn_charge,
            sebi_charge=sebi_charge,
            stt=stt,
            stamp_duty=0.0,
            dp_charge=dp_charge,
            gst=gst,
        )

    def round_trip_cost(self, buy_price: float, sell_price: float, quantity: int) -> float:
        """Total cost of buying and later selling `quantity` shares.
        Convenience for quick "how much do costs eat into this trade"
        checks; the engine calls buy_cost/sell_cost separately per fill."""
        return (
            self.buy_cost(buy_price, quantity).total
            + self.sell_cost(sell_price, quantity).total
        )
