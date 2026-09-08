"""Capital gains tax on Indian equity.

This is not a broker fee, so it doesn't belong in CostModel -- it lands
once a year on realised gains rather than per trade. But leaving it out
flatters high-turnover strategies badly, and by exactly the amount that
matters when comparing them to buy-and-hold:

  - Short-term (held <= 12 months): 20%
  - Long-term  (held >  12 months): 12.5%, and only when you actually sell

A monthly-rebalanced portfolio realises nearly everything short-term, and
pays every year. Buy-and-hold defers the whole bill to the end, so it keeps
compounding on money the taxman hasn't taken yet. That deferral is worth
real money over a decade, independently of the rate difference.

Rates are parameters, not constants -- Indian capital gains rates were
changed in 2024 and will change again.
"""

from __future__ import annotations

from dataclasses import dataclass

SHORT_TERM_RATE = 0.20
LONG_TERM_RATE = 0.125


@dataclass(frozen=True)
class TaxModel:
    short_term_rate: float = SHORT_TERM_RATE
    long_term_rate: float = LONG_TERM_RATE

    def after_tax_multiple_realised_annually(self, annual_return: float, years: float) -> float:
        """Growth multiple for a strategy that realises its gains every
        year (i.e. turns over faster than 12 months), so each year's gain
        is taxed at the short-term rate as it happens."""
        return (1 + annual_return * (1 - self.short_term_rate)) ** years

    def after_tax_multiple_deferred(self, annual_return: float, years: float) -> float:
        """Growth multiple for buy-and-hold: compounds untaxed, then pays
        the long-term rate once on the whole gain at the end."""
        gross = (1 + annual_return) ** years
        return 1 + (gross - 1) * (1 - self.long_term_rate)

    def after_tax_cagr(self, multiple: float, years: float) -> float:
        return multiple ** (1 / years) - 1
