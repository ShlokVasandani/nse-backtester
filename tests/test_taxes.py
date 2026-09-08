import pytest

from costs.taxes import TaxModel


def test_short_term_realisation_is_taxed_every_year():
    tm = TaxModel()
    # 10% a year, realised annually at 20% -> effectively 8% compounding
    assert tm.after_tax_multiple_realised_annually(0.10, 1) == pytest.approx(1.08)
    assert tm.after_tax_multiple_realised_annually(0.10, 2) == pytest.approx(1.08**2)


def test_deferred_holding_compounds_untaxed_then_pays_once():
    tm = TaxModel()
    # 10% for 2 years = 1.21x gross; gain of 0.21 taxed once at 12.5%
    assert tm.after_tax_multiple_deferred(0.10, 2) == pytest.approx(1 + 0.21 * 0.875)


def test_deferral_beats_annual_realisation_at_the_same_gross_return():
    """The whole reason turnover matters: same pre-tax return, worse
    outcome if you keep realising it."""
    tm = TaxModel()
    realised = tm.after_tax_multiple_realised_annually(0.12, 10)
    deferred = tm.after_tax_multiple_deferred(0.12, 10)
    assert deferred > realised


def test_high_turnover_edge_shrinks_but_survives_at_measured_returns():
    """Our measured 14.2% (monthly rebalance) vs 8.1% (hold) over 9.4y."""
    tm = TaxModel()
    mom = tm.after_tax_multiple_realised_annually(0.142, 9.4)
    hold = tm.after_tax_multiple_deferred(0.081, 9.4)
    assert mom > hold
    # tax narrows the edge from ~1.68x to ~1.4x, it does not erase it
    assert 1.3 < mom / hold < 1.5


def test_rates_are_configurable():
    tm = TaxModel(short_term_rate=0.30, long_term_rate=0.20)
    assert tm.after_tax_multiple_realised_annually(0.10, 1) == pytest.approx(1.07)
