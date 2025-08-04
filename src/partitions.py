from __future__ import annotations

from itertools import combinations
from typing import List, Tuple
from .types import Alpha

__all__ = ["alpha_bs", "split_index_sets"]


def alpha_bs(n: int, D: int) -> List[Alpha]:
    """
    Generate canonical α-tuples of length n with nonnegative entries,
    nonincreasing order, and total sum ≤ D.

    Returns
    -------
    List[Alpha]
        Deterministically ordered in reverse-lex (largest entries first).
    """
    if n < 0:
        raise ValueError("n must be ≥ 0")
    if D < 0:
        return []

    cur = [0] * n
    out: List[Alpha] = []

    def rec(i: int, remaining: int, max_part: int) -> None:
        if i == n:
            out.append(tuple(cur))  # already canonical (nonincreasing)
            return
        ub = min(max_part, remaining)
        for v in range(ub, -1, -1):  # descend to keep reverse-lex order
            cur[i] = v
            rec(i + 1, remaining - v, v)

    rec(0, D, D)
    return out


def split_index_sets(n_minus_1: int):
    """
    Unordered bipartitions (I, J) of {0,..,n_minus_1-1}, unique up to swap.
    """

    inds = tuple(range(n_minus_1))
    out = []
    half = n_minus_1 // 2

    for r in range(0, half + 1):      # <-- start at 0, not 1
        for I in combinations(inds, r):
            Iset = set(I)
            J = tuple(x for x in inds if x not in Iset)
            if (len(I), I) <= (len(J), J):  # keep a canonical representative
                out.append((list(I), list(J)))
    return out
