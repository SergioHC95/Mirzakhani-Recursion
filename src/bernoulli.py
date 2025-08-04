from __future__ import annotations

from fractions import Fraction
from math import comb
from typing import List

__all__ = ["bernoulli_numbers", "bernoulli_even"]


def bernoulli_numbers(n: int) -> List[Fraction]:
    """
    Bernoulli numbers B_0..B_n as exact Fractions via Akiyama–Tanigawa.
    Convention: “second” Bernoulli numbers (B1 = +1/2).
    """
    if n < 0:
        raise ValueError("n must be >= 0")

    A = [Fraction(0)] * (n + 1)
    out: List[Fraction] = []
    Fr = Fraction  # local binding for speed
    for m in range(n + 1):
        A[m] = Fr(1, m + 1)
        # in-place triangular update
        for j in range(m, 0, -1):
            A[j - 1] = j * (A[j - 1] - A[j])
        out.append(A[0])  # A[0] == B_m
    return out


def bernoulli_even(k: int) -> Fraction:
    """
    Return B_{2k} as an exact Fraction using the binomial recurrence,
    skipping all odd Bernoulli numbers (they are 0 for n>=3).
    This is typically ~2× faster than running the full AT loop to 2k.

    Note: The binomial recurrence
        sum_{r=0}^{m} C(m+1, r) B_r = 0   (m >= 1)
    is stated for the “first” convention (B1 = -1/2). Even-index values
    are identical under both conventions, so we plug B1 = -1/2 here.
    """
    if k < 0:
        raise ValueError("k must be >= 0")
    if k == 0:
        return Fraction(1)

    B_even: List[Fraction] = [Fraction(1)]  # B_0
    Fr = Fraction
    for m in range(1, k + 1):
        n = 2 * m
        s = Fr(0)
        # sum over even indices < n
        for j in range(m):
            s += Fr(comb(n + 1, 2 * j)) * B_even[j]
        # add the B1 term from the “first” convention (B1 = -1/2)
        s += Fr(n + 1) * Fr(-1, 2)
        B_2m = -s / Fr(n + 1)
        B_even.append(B_2m)
    return B_even[k]
