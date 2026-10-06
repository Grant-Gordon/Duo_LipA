"""Exact feasibility checks behind the grammar's no-dead-end property (decoder spec 4.1).

Per-chain domains are the "never broken" rules only (decoder spec 8.2): carbons, double bonds that
fit distinct positions, oxygens that fit distinct modification positions. Tendencies (odd chains,
methylene-interrupted double bonds, ...) are not encoded.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

C_MAX, DB_MAX, OX_MAX = 40, 12, 6
SUM_C_MAX, SUM_DB_MAX, SUM_OX_MAX = 120, 30, 10
POS_MAX = 39
MIN_C = {"acyl": 2, "ether": 2, "lcb": 4}
OX_RANGE = {"acyl": (0, OX_MAX), "ether": (0, OX_MAX), "lcb": (1, 4)}
MAX_MOD_COST = 2


def db_max(c: int) -> int:
    """Double bonds must take distinct positions in 1..min(c-1, 39)."""
    return max(0, min(DB_MAX, c - 1, POS_MAX))


def ox_max(kind: str, c: int) -> int:
    lo, hi = OX_RANGE[kind]
    if kind == "lcb":
        return hi
    return min(hi, MAX_MOD_COST * min(c, POS_MAX))


def in_domain(kind: str, c: int, db: int, ox: int) -> bool:
    if not (MIN_C[kind] <= c <= C_MAX):
        return False
    if not (0 <= db <= db_max(c)):
        return False
    lo = OX_RANGE[kind][0]
    return lo <= ox <= ox_max(kind, c)


@lru_cache(maxsize=None)
def chain_domain(kind: str) -> np.ndarray:
    a = np.zeros((SUM_C_MAX + 1, SUM_DB_MAX + 1, SUM_OX_MAX + 1), dtype=bool)
    for c in range(MIN_C[kind], C_MAX + 1):
        for db in range(0, db_max(c) + 1):
            lo = OX_RANGE[kind][0]
            a[c, db, lo: ox_max(kind, c) + 1] = True
    return a


def _minkowski(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros_like(a)
    C, D, O = a.shape
    for c, d, o in zip(*np.nonzero(b)):
        out[c:, d:, o:] |= a[: C - c, : D - d, : O - o]
    return out


@lru_cache(maxsize=None)
def sums_feasible(kinds: tuple[str, ...]) -> np.ndarray:
    """Boolean (C, DB, OX) array: which sum compositions some assignment of chains of these kinds
    reaches. Chain order is irrelevant to existence, so the unordered sum is exact here."""
    out = np.zeros((SUM_C_MAX + 1, SUM_DB_MAX + 1, SUM_OX_MAX + 1), dtype=bool)
    if not kinds:
        out[0, 0, 0] = True
        return out
    out = chain_domain(kinds[0])
    for k in kinds[1:]:
        out = _minkowski(out, chain_domain(k))
    return out


def _bounds_ok(m: int, lo: tuple[int, int, int], R: tuple[int, int, int]) -> bool:
    C, D, O = R
    if C < 0 or D < 0 or O < 0:
        return False
    if C < m * lo[0] or C > m * C_MAX:
        return False
    if D > m * DB_MAX or D > C - m:
        return False
    return O <= m * OX_MAX


@lru_cache(maxsize=None)
def acyl_feasible(m: int, lo: tuple[int, int, int], R: tuple[int, int, int]) -> bool:
    """Do m acyl chains, each lexicographically >= lo in (C, DB, OX), sum exactly to R?"""
    if m == 0:
        return R == (0, 0, 0)
    if not _bounds_ok(m, lo, R):
        return False
    C, D, O = R
    if m == 1:
        return R >= lo and in_domain("acyl", C, D, O)
    # the first (smallest) chain x satisfies lo <= x and x.c <= C / m
    for c in range(lo[0], C // m + 1):
        for db in range(0, min(db_max(c), D) + 1):
            for ox in range(0, min(ox_max("acyl", c), O) + 1):
                x = (c, db, ox)
                if x < lo:
                    continue
                if acyl_feasible(m - 1, x, (C - c, D - db, O - ox)):
                    return True
    return False


@lru_cache(maxsize=None)
def rest_feasible(typed: tuple[str, ...], m_acyl: int, lo: tuple[int, int, int],
                  R: tuple[int, int, int]) -> bool:
    """Remaining typed chains (any values in their domains) plus m_acyl ordered acyl chains >= lo."""
    if not typed:
        return acyl_feasible(m_acyl, lo, R)
    kind, rest = typed[0], typed[1:]
    C, D, O = R
    for c in range(MIN_C[kind], min(C_MAX, C) + 1):
        for db in range(0, min(db_max(c), D) + 1):
            olo = OX_RANGE[kind][0]
            for ox in range(olo, min(ox_max(kind, c), O) + 1):
                if rest_feasible(rest, m_acyl, lo, (C - c, D - db, O - ox)):
                    return True
    return False
