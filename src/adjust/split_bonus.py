"""Split/bonus price adjustment.

NSE's corporate-actions feed describes bonuses and splits as free text in a
`subject` field (e.g. "Bonus 3:1", "Face Value Split (Sub-Division) - From
Rs 10/- Per Share To Rs 5/- Per Share"). This module extracts a price
adjustment factor from that text and applies it to historical OHLCV data.

Convention: `factor` is the multiplier applied to prices *before* ex_date so
they're comparable to prices on/after ex_date (which are left untouched --
the most recent price is always "truth"). Volume is divided by the same
factor so traded value (price * volume) is preserved.

  - Bonus X:Y (X new shares issued for every Y held): factor = Y / (X + Y)
    e.g. Bonus 1:1 -> holder's share count doubles -> factor = 0.5
  - Face value split/consolidation, old face value A -> new face value B:
    factor = B / A (same arithmetic effect as a bonus, since share count
    scales inversely with face value)

Rows whose subject clearly *looks* like a bonus/split/consolidation but
doesn't match the expected wording are deliberately left unparsed rather
than guessed at -- see `find_unparsed_suspects`. Silently mis-adjusting a
price series is worse than leaving a gap that surfaces in review.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Subjects containing these are bonus-shaped text but aren't a plain equity
# bonus (debentures, preference shares, etc.) -- don't treat as a share
# count change.
_NON_EQUITY_BONUS = re.compile(r"debenture|ncrps|preference|nca|ncd", re.IGNORECASE)

# Compound/ambiguous actions we refuse to guess at.
_COMPLEX_ACTION = re.compile(r"capital reduction|scheme of (?:ar+angement|amalgamation)", re.IGNORECASE)

# Separator after "Bonus" varies: "Bonus 1:2", "Bonus- 1:2", "Bonus - 1:2".
_BONUS_RE = re.compile(r"\bBonus\b\s*[-:]?\s*(\d+)\s*:\s*(\d+)\b", re.IGNORECASE)

# A face-value change is only interpreted as one when the subject actually
# says so. This gate is what makes the value-pair regex below safe to keep
# loose -- and _COMPLEX_ACTION has already removed capital reductions and
# schemes of arrangement before either runs.
_FACE_VALUE_CONTEXT = re.compile(r"split|sub-?division|consolidat", re.IGNORECASE)

# The "<Rs A> ... To ... <Rs B>" pair, tolerant of the wording drift found
# across 2015-2025 filings: "Per Share" present or absent ("From Rs 10 To
# Rs 1"), truncated ("Rs 10/- Per To Rs 2/-"), "/-" present or absent, and
# "Rs"/"Re" used interchangeably. The gap between the two values is capped
# at a few non-digit characters so this can't span unrelated clauses.
_FACE_VALUE_RE = re.compile(
    r"R[se]\.?\s*(\d+(?:\.\d+)?)\s*(?:/-)?[^0-9]{0,20}?\bTo\b[^0-9]{0,12}?R[se]\.?\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedFactor:
    kind: str  # "bonus" | "split" | "consolidation"
    factor: float
    detail: str


def parse_bonus(subject: str) -> ParsedFactor | None:
    if _NON_EQUITY_BONUS.search(subject):
        return None
    m = _BONUS_RE.search(subject)
    if not m:
        return None
    new_shares, held_shares = int(m.group(1)), int(m.group(2))
    factor = held_shares / (new_shares + held_shares)
    return ParsedFactor(kind="bonus", factor=factor, detail=f"{new_shares}:{held_shares}")


def parse_split(subject: str) -> ParsedFactor | None:
    if not _FACE_VALUE_CONTEXT.search(subject):
        return None
    m = _FACE_VALUE_RE.search(subject)
    if not m:
        return None
    old_fv, new_fv = float(m.group(1)), float(m.group(2))
    if old_fv == 0:
        return None
    factor = new_fv / old_fv
    kind = "split" if new_fv < old_fv else "consolidation"
    return ParsedFactor(kind=kind, factor=factor, detail=f"{old_fv:g}->{new_fv:g}")


def extract_factors(subject: str) -> list[ParsedFactor]:
    """A single corporate action can bundle a bonus AND a split in one
    subject string (e.g. "Bonus 1:2/Face Value Split ... From Rs 2/- ...
    To Re 1/-"). Extract every component that matches, independently."""
    subject = subject.strip()
    if _COMPLEX_ACTION.search(subject):
        return []

    factors = []
    b = parse_bonus(subject)
    if b:
        factors.append(b)
    s = parse_split(subject)
    if s:
        factors.append(s)
    return factors


# Subjects that mention these words but produced zero parsed factors are
# suspicious -- likely a wording variant this parser doesn't handle yet.
_SUSPECT_KEYWORDS = re.compile(
    r"\bbonus\b|\bsplit\b|sub-division|consolidat|capital reduction", re.IGNORECASE
)


def find_unparsed_suspects(corporate_actions: pd.DataFrame) -> pd.DataFrame:
    """Rows that look bonus/split-shaped but extract_factors() found
    nothing for. Should be reviewed (and the regexes extended) before
    trusting an adjustment run over new data."""
    mask = corporate_actions["subject"].apply(
        lambda s: bool(_SUSPECT_KEYWORDS.search(s)) and not extract_factors(s)
    )
    return corporate_actions[mask]


def build_adjustment_events(corporate_actions: pd.DataFrame) -> pd.DataFrame:
    """One row per symbol/ex_date corporate action that yields a real price
    adjustment (factor != 1.0), with the combined factor if multiple
    components (e.g. bonus + split together) were found."""
    rows = []
    for row in corporate_actions.itertuples(index=False):
        parsed = extract_factors(row.subject)
        if not parsed:
            continue
        factor = 1.0
        kinds = []
        details = []
        for p in parsed:
            factor *= p.factor
            kinds.append(p.kind)
            details.append(p.detail)
        if factor == 1.0:
            continue
        rows.append(
            {
                "symbol": row.symbol,
                "ex_date": row.ex_date,
                "factor": factor,
                "kind": "+".join(kinds),
                "detail": "; ".join(details),
                "raw_subject": row.subject,
            }
        )
    events = pd.DataFrame(
        rows, columns=["symbol", "ex_date", "factor", "kind", "detail", "raw_subject"]
    )
    return events.sort_values(["symbol", "ex_date"]).reset_index(drop=True)


def _cumulative_multipliers(dates: np.ndarray, event_dates: list, event_factors: list[float]) -> np.ndarray:
    """multiplier(D) = product of factor for every event with ex_date > D.
    event_dates must be sorted ascending."""
    if not event_dates:
        return np.ones(len(dates))

    suffix = np.ones(len(event_factors) + 1)
    for i in range(len(event_factors) - 1, -1, -1):
        suffix[i] = suffix[i + 1] * event_factors[i]

    idx = np.searchsorted(event_dates, dates, side="right")
    return suffix[idx]


def apply_adjustment(prices: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `prices` (columns: symbol, date, open, high, low,
    close, prev_close, volume, ...) with adjusted OHLC + volume added.
    Symbols with no adjustment events pass through with multiplier 1.0."""
    out = prices.copy()
    out["adj_multiplier"] = 1.0

    for symbol, group in events.groupby("symbol"):
        mask = out["symbol"] == symbol
        if not mask.any():
            continue
        event_dates = group["ex_date"].tolist()
        event_factors = group["factor"].tolist()
        dates = out.loc[mask, "date"].to_numpy()
        out.loc[mask, "adj_multiplier"] = _cumulative_multipliers(dates, event_dates, event_factors)

    for col in ("open", "high", "low", "close", "prev_close"):
        out[f"adj_{col}"] = out[col] * out["adj_multiplier"]
    out["adj_volume"] = out["volume"] / out["adj_multiplier"]

    return out
