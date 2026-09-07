import datetime as dt

import pandas as pd
import pytest

from adjust.split_bonus import (
    apply_adjustment,
    build_adjustment_events,
    extract_factors,
    find_unparsed_suspects,
    parse_bonus,
    parse_split,
)

# Real subject strings pulled live from NSE's corporate-actions API
# (2018-2025 samples), used verbatim as regression fixtures.
PLAIN_BONUS = "Bonus 3:1"
BONUS_LEADING_SPACE = " Bonus  1:4"
NCRPS_BONUS = "Scheme Of Arrangement - Bonus Ncrps 4:1"
DEBENTURE_BONUS = "Scheme Of Arangement- Bonus - 1 Debenture For 1 Equity Share Held"
BONUS_WITH_DIVIDEND = "Bonus 1:1/Dividend- Rs 5 Per Share"

SPLIT_SPACED = "Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 5/- Per Share"
SPLIT_NO_SPACE = "Face Value Split (Sub-Division) - From Rs10/- Per Share To Re 1/- Per Share"
SPLIT_EXTRA_SPACE = "Face Value Split (Sub-Division) - From Rs 10 /- Per Share To Rs 2/- Per Share"
CONSOLIDATION = "Consolidation Of Equity Shares From Re 1 Per Share To Rs 10 Per Share"
BONUS_PLUS_SPLIT = "Bonus 1:2/Face Value Split (Sub-Division) From Rs 2/- Per Share To Re 1/- Per Share"

CAPITAL_REDUCTION = "Capital Reduction Rs 10 To Rs 3.30 / Consolidation Rs 3.30 To Rs.10"
PLAIN_DIVIDEND = "Dividend - Rs 10 Per Share"

# Older (2015-2020 era) wordings found in the real feed that an earlier
# version of this parser silently missed. A missed 10:1 split leaves a fake
# -90% crash in the adjusted series, so these are regression fixtures.
SPLIT_NO_PER_SHARE = "Face Value Split From Rs 10 To Rs 1"
SPLIT_NO_FROM = "Face Value Split Rs 10 To Rs 1"
SPLIT_RE_TWO = "Face Value Split From Rs 10 To Re 2"
SPLIT_TRUNCATED_PER = "Face Valus Split (Sub-Division) - From Rs 10/- Per To Rs 2/- Per Share"
SPLIT_RECORD_DATE_SUFFIX = "Face Value Split From Rs 10 To Re 1 (Record Date Revised)"
BONUS_HYPHEN_NO_SPACE = "Bonus- 1:2"
BONUS_IN_COMPOUND_SUBJECT = "Annual General Meeting / Dividend - Rs 3/- Per Share / Bonus - 1:2"

# Must stay excluded even with the looser patterns.
CAPITAL_REDUCTION_WITH_VALUES = "Capital Reduction -  From Rs 10/- To Rs 4/- Per Share"
CAPITAL_REDUCTION_BARE = "Capital Reduction"
CAPITAL_REDUCTION_NCLT = "Capital Reduction Pursuant To Nclt Order"
BONUS_DEBENTURES = "Scheme Of Arrangement - Bonus Debentures 1:1"
BONUS_NCRPS_SMALL = "Bonus Ncrps 1:116"


def test_parse_bonus_plain():
    result = parse_bonus(PLAIN_BONUS)
    assert result.kind == "bonus"
    assert result.factor == pytest.approx(1 / 4)  # 1 held / (3 new + 1 held)
    assert result.detail == "3:1"


def test_parse_bonus_handles_double_space_and_leading_space():
    result = parse_bonus(BONUS_LEADING_SPACE)
    assert result.factor == pytest.approx(4 / 5)


@pytest.mark.parametrize("subject", [NCRPS_BONUS, DEBENTURE_BONUS])
def test_parse_bonus_rejects_non_equity_bonus(subject):
    assert parse_bonus(subject) is None


def test_parse_bonus_ignores_trailing_dividend_text():
    result = parse_bonus(BONUS_WITH_DIVIDEND)
    assert result.factor == pytest.approx(0.5)


@pytest.mark.parametrize(
    "subject,expected_factor,expected_kind",
    [
        (SPLIT_SPACED, 0.5, "split"),
        (SPLIT_NO_SPACE, 0.1, "split"),
        (SPLIT_EXTRA_SPACE, 0.2, "split"),
        (CONSOLIDATION, 10.0, "consolidation"),
    ],
)
def test_parse_split_variants(subject, expected_factor, expected_kind):
    result = parse_split(subject)
    assert result.factor == pytest.approx(expected_factor)
    assert result.kind == expected_kind


def test_parse_split_returns_none_for_non_split_text():
    assert parse_split(PLAIN_DIVIDEND) is None


@pytest.mark.parametrize(
    "subject,expected_factor",
    [
        (SPLIT_NO_PER_SHARE, 0.1),
        (SPLIT_NO_FROM, 0.1),
        (SPLIT_RE_TWO, 0.2),
        (SPLIT_TRUNCATED_PER, 0.2),
        (SPLIT_RECORD_DATE_SUFFIX, 0.1),
    ],
)
def test_parse_split_handles_older_wording_variants(subject, expected_factor):
    result = parse_split(subject)
    assert result is not None, f"missed a real split: {subject!r}"
    assert result.factor == pytest.approx(expected_factor)


@pytest.mark.parametrize(
    "subject,expected_factor",
    [
        (BONUS_HYPHEN_NO_SPACE, 2 / 3),  # Bonus 1:2
        (BONUS_IN_COMPOUND_SUBJECT, 2 / 3),
    ],
)
def test_parse_bonus_handles_hyphen_separator_and_compound_subjects(subject, expected_factor):
    result = parse_bonus(subject)
    assert result is not None, f"missed a real bonus: {subject!r}"
    assert result.factor == pytest.approx(expected_factor)


@pytest.mark.parametrize(
    "subject",
    [
        CAPITAL_REDUCTION_WITH_VALUES,
        CAPITAL_REDUCTION_BARE,
        CAPITAL_REDUCTION_NCLT,
        BONUS_DEBENTURES,
        BONUS_NCRPS_SMALL,
    ],
)
def test_looser_patterns_still_refuse_non_equity_and_complex_actions(subject):
    """The loosened regexes must not start swallowing the cases we
    deliberately refuse to guess at."""
    assert extract_factors(subject) == []


def test_extract_factors_combines_bonus_and_split():
    factors = extract_factors(BONUS_PLUS_SPLIT)
    kinds = {f.kind for f in factors}
    assert kinds == {"bonus", "split"}
    combined = 1.0
    for f in factors:
        combined *= f.factor
    # Bonus 1:2 -> 2/3, Split 2->1 -> 0.5
    assert combined == pytest.approx((2 / 3) * 0.5)


def test_extract_factors_refuses_capital_reduction():
    assert extract_factors(CAPITAL_REDUCTION) == []


def test_extract_factors_empty_for_plain_dividend():
    assert extract_factors(PLAIN_DIVIDEND) == []


def test_find_unparsed_suspects_flags_capital_reduction_but_not_dividend():
    df = pd.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "subject": [CAPITAL_REDUCTION, PLAIN_DIVIDEND, PLAIN_BONUS],
        }
    )
    suspects = find_unparsed_suspects(df)
    assert list(suspects["symbol"]) == ["A"]


def test_build_adjustment_events_only_keeps_real_adjustments():
    df = pd.DataFrame(
        {
            "symbol": ["ACME", "ACME", "OTHER"],
            "ex_date": [dt.date(2024, 1, 10), dt.date(2024, 6, 1), dt.date(2024, 3, 1)],
            "subject": [PLAIN_BONUS, PLAIN_DIVIDEND, SPLIT_SPACED],
        }
    )
    events = build_adjustment_events(df)
    # dividend-only row drops out; bonus and split rows remain
    assert list(events["symbol"]) == ["ACME", "OTHER"]
    assert events.iloc[0]["factor"] == pytest.approx(0.25)
    assert events.iloc[1]["factor"] == pytest.approx(0.5)


def test_apply_adjustment_scales_history_before_ex_date_only():
    prices = pd.DataFrame(
        {
            "symbol": ["ACME"] * 4,
            "date": [dt.date(2024, 1, 1), dt.date(2024, 1, 9), dt.date(2024, 1, 10), dt.date(2024, 1, 11)],
            "open": [100.0, 100.0, 25.0, 25.0],
            "high": [100.0, 100.0, 25.0, 25.0],
            "low": [100.0, 100.0, 25.0, 25.0],
            "close": [100.0, 100.0, 25.0, 25.0],
            "prev_close": [100.0, 100.0, 100.0, 25.0],
            "volume": [1000, 1000, 4000, 4000],
        }
    )
    events = pd.DataFrame(
        {"symbol": ["ACME"], "ex_date": [dt.date(2024, 1, 10)], "factor": [0.25]}
    )

    adjusted = apply_adjustment(prices, events)

    # Bonus 3:1 on 2024-01-10: prices before that date scale by 0.25,
    # prices on/after stay as-is.
    assert adjusted.loc[adjusted["date"] < dt.date(2024, 1, 10), "adj_close"].tolist() == [
        pytest.approx(25.0),
        pytest.approx(25.0),
    ]
    assert adjusted.loc[adjusted["date"] >= dt.date(2024, 1, 10), "adj_close"].tolist() == [
        pytest.approx(25.0),
        pytest.approx(25.0),
    ]
    # Volume divided by the same factor so traded value is preserved.
    assert adjusted.loc[adjusted["date"] < dt.date(2024, 1, 10), "adj_volume"].tolist() == [
        pytest.approx(4000.0),
        pytest.approx(4000.0),
    ]
    assert adjusted.loc[adjusted["date"] >= dt.date(2024, 1, 10), "adj_volume"].tolist() == [
        pytest.approx(4000.0),
        pytest.approx(4000.0),
    ]


def test_apply_adjustment_passes_through_symbols_with_no_events():
    prices = pd.DataFrame(
        {
            "symbol": ["QUIET"],
            "date": [dt.date(2024, 1, 1)],
            "open": [50.0],
            "high": [51.0],
            "low": [49.0],
            "close": [50.5],
            "prev_close": [49.5],
            "volume": [100],
        }
    )
    events = pd.DataFrame(columns=["symbol", "ex_date", "factor"])

    adjusted = apply_adjustment(prices, events)
    assert adjusted["adj_close"].iloc[0] == pytest.approx(50.5)
    assert adjusted["adj_multiplier"].iloc[0] == 1.0


# Abbreviated NSE filings. These escaped both the parser AND
# find_unparsed_suspects (which shared its vocabulary), and were only
# caught by the independent price-jump scan -- JSWSTEEL's real 10:1 split
# sat in the store as a fake -90% crash.
SPLIT_ABBREVIATED_RE1 = "Fv Splt Frm Rs 10 To Re 1"
SPLIT_ABBREVIATED_RS2 = "Fv Splt Frm Rs 10 To Rs 2"
SPLIT_ABBREVIATED_RS5 = "Fv Splt Frm Rs 10 To Rs 5"


@pytest.mark.parametrize(
    "subject,expected_factor",
    [
        (SPLIT_ABBREVIATED_RE1, 0.1),
        (SPLIT_ABBREVIATED_RS2, 0.2),
        (SPLIT_ABBREVIATED_RS5, 0.5),
    ],
)
def test_parse_split_handles_abbreviated_filings(subject, expected_factor):
    result = parse_split(subject)
    assert result is not None, f"missed a real split: {subject!r}"
    assert result.factor == pytest.approx(expected_factor)


def test_abbreviated_splits_are_also_flagged_as_suspects_when_unparsed():
    """The suspect detector must recognise abbreviated wording as
    split-shaped too, so a future variant it can't parse still surfaces."""
    df = pd.DataFrame({"symbol": ["X"], "subject": ["Fv Splt Frm Rs 10 To Something Weird"]})
    assert len(find_unparsed_suspects(df)) == 1
