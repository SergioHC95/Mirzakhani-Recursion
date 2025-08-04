from __future__ import annotations

from fractions import Fraction
from math import factorial
from typing import List
from .types import Alpha

__all__ = ["genus0_x"]


def _poly_mul_trunc(p: List[Fraction], q: List[Fraction], L: int) -> List[Fraction]:
    """Truncated product (mod u^{L+1})."""
    out = [Fraction(0)] * (L + 1)
    for i, pi in enumerate(p):
        if pi == 0:
            continue
        imax = min(L, i + len(q) - 1)
        for j in range(min(L - i, len(q) - 1) + 1):
            out[i + j] += pi * q[j]
    return out


def _series_inverse(a: List[Fraction], L: int) -> List[Fraction]:
    """
    Invert a power series A(u) with A(0)=1 up to order L:
        B(u) = 1 / A(u)  (mod u^{L+1}).
    Recurrence: b0=1; b_m = -sum_{k=1..m} a_k b_{m-k}.
    """
    if not a or a[0] != 1:
        raise ValueError("series_inverse requires a[0] == 1")
    b = [Fraction(0)] * (L + 1)
    b[0] = Fraction(1)
    for m in range(1, L + 1):
        s = Fraction(0)
        for k in range(1, min(m, len(a) - 1) + 1):
            s += a[k] * b[m - k]
        b[m] = -s
    return b


def _poly_pow_trunc(base: List[Fraction], exp: int, L: int) -> List[Fraction]:
    """Raise a series to an integer power exp (≥0) mod u^{L+1} via binary exponentiation."""
    if exp < 0:
        raise ValueError("exp must be >= 0")
    # res = 1
    res = [Fraction(0)] * (L + 1)
    res[0] = Fraction(1)
    if exp == 0:
        return res
    # b = base
    b = base[: L + 1]
    e = exp
    while e:
        if e & 1:
            res = _poly_mul_trunc(res, b, L)
        e >>= 1
        if e:
            b = _poly_mul_trunc(b, b, L)
    return res


def genus0_x(l: int, alpha: Alpha) -> Fraction:
    r"""
    Exact genus-0 closed form for
        X(0, n, α) = ⟨ τ_{α_1}\cdots τ_{α_n} ⟩_{g=0}
    at ψ-degree ℓ = l, where n = l + 3 + Σα_i.

    Using the series identity
        J₁(2x) = Σ_{k≥0} (-1)^k x^{2k+1}/(k!(k+1)!)
    one has (t x)/J₁(2x) = t / S(x²) with
        S(u) = Σ_{k≥0} (-1)^k u^k / (k!(k+1)!).
    Let H(u) = 1/S(u). Then the generating function gives
        X = (n-3)! * l! / ∏ α_i! * [u^l] H(u)^{n-2},
    so we only need formal series in u with exact rationals.

    Parameters
    ----------
    l : int
        Non-negative ψ-degree.
    alpha : Alpha
        Non-negative multi-index (length n), canonical or not.

    Returns
    -------
    Fraction
        The exact intersection number.
    """
    if l < 0 or any(a < 0 for a in alpha):
        return Fraction(0)

    A = sum(alpha)
    n = l + 3 + A
    r = n - 2  # exponent on H(u)

    # Build S(u) = sum_{k=0..l} (-1)^k / (k!(k+1)!) * u^k, truncated at u^l.
    # Note S(0) = 1, so inversion is valid.
    S: List[Fraction] = [Fraction(0)] * (l + 1)
    for k in range(l + 1):
        coeff = Fraction((-1) ** k, factorial(k) * factorial(k + 1))
        S[k] = coeff
    S[0] = Fraction(1)  # explicit, for clarity

    # H(u) = 1 / S(u) mod u^{l+1}
    H = _series_inverse(S, l)

    # Take H(u)^(n-2) mod u^{l+1} and read off the u^l coefficient
    H_pow = _poly_pow_trunc(H, r, l)
    coeff_ul = H_pow[l]  # [u^l] H(u)^{n-2}

    # Prefactor: (n-3)! * l! / ∏ α_i!
    numer = factorial(n - 3) * factorial(l)
    denom = 1
    for a in alpha:
        denom *= factorial(a)

    return Fraction(numer, denom) * coeff_ul
