"""Independent data-quality gate for the adjusted price series.

Why this exists, concretely: adjust/split_bonus.py parses corporate actions
out of free text, and find_unparsed_suspects() flags text it can't read.
But that detector shares its vocabulary with the parser, so a filing worded
"Fv Splt Frm Rs 10 To Re 1" escaped BOTH -- neither recognised it as
split-shaped. JSWSTEEL's 2017 10:1 split sat in the store as a fake -90%
crash and nothing in the text pipeline noticed.

This module checks the prices themselves instead of the words. An
overnight move of -90% in a liquid stock is not a price move; it is an
unadjusted corporate action. That is a completely different detection
mechanism, which is the point -- it catches what the parser's blind spots
let through.

It deliberately does NOT try to infer and apply a correction factor. Some
of these gaps are splits (a clean 0.1 ratio), but others are demergers,
schemes of arrangement and rights issues whose true factor simply isn't
derivable from the price alone, and a stock really can fall 50% in a day.
Guessing would reintroduce exactly the silent corruption this is meant to
surface. Instead it produces an exclusion list, so analysis runs on data
that is trustworthy and says out loud what it dropped.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from adjust.store import DEFAULT_DB_PATH

# A liquid stock gapping below 55% or above 190% overnight is almost
# always an unadjusted corporate action rather than a real move.
DEFAULT_DROP_RATIO = 0.55
DEFAULT_JUMP_RATIO = 1.9


def find_unexplained_jumps(
    db_path: Path = DEFAULT_DB_PATH,
    *,
    min_avg_turnover: float = 20_000_000,
    min_days: int = 200,
    drop_ratio: float = DEFAULT_DROP_RATIO,
    jump_ratio: float = DEFAULT_JUMP_RATIO,
) -> pd.DataFrame:
    """Day-over-day moves in the ADJUSTED close that are too large to be
    real. Restricted to reasonably liquid, long-listed symbols, since thin
    scrips gap wildly for legitimate reasons."""
    import duckdb

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.execute(
            """
            WITH liquid AS (
                SELECT symbol FROM prices
                GROUP BY symbol
                HAVING avg(turnover) > ? AND count(*) > ?
            ),
            moves AS (
                SELECT p.symbol, p.date, p.adj_close,
                       lag(p.adj_close) OVER (PARTITION BY p.symbol ORDER BY p.date) AS prev_adj
                FROM prices p JOIN liquid USING (symbol)
            )
            SELECT symbol, date, prev_adj, adj_close,
                   adj_close / prev_adj AS ratio
            FROM moves
            WHERE prev_adj IS NOT NULL AND prev_adj > 0
              AND (adj_close / prev_adj < ? OR adj_close / prev_adj > ?)
            ORDER BY symbol, date
            """,
            [min_avg_turnover, min_days, drop_ratio, jump_ratio],
        ).df()
    finally:
        con.close()


def quarantined_symbols(db_path: Path = DEFAULT_DB_PATH, **kwargs) -> set[str]:
    """Symbols whose adjusted history contains at least one unexplained
    discontinuity. Exclude these from analysis: their returns are fiction
    wherever the gap sits."""
    jumps = find_unexplained_jumps(db_path, **kwargs)
    if jumps.empty:
        return set()
    return set(jumps["symbol"].unique())
