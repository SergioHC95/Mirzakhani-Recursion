from __future__ import annotations

from fractions import Fraction
from math import factorial, prod
from typing import Iterable, Tuple

from .types import Alpha

__all__ = ["double_factorial_odd", "l_degree", "rx_from_x", "x_from_rx"]


def _validate_alpha(alpha: Alpha) -> None:
    """Ensure α is a finite iterable of non-negative integers."""
    for a in alpha:
        if not isinstance(a, int) or a < 0:
            raise ValueError(f"alpha must contain non-negative ints; got {alpha!r}")


def double_factorial_odd(n: int) -> int:
    r"""
    Compute the odd double factorial (2n+1)!! for n ≥ 0 as an exact integer.

    Uses the identity
        (2n+1)!! = (2n+1)! / (2^n n!)
    which is typically faster than a Python loop for large n.
    """
    if n < 0:
        raise ValueError("double_factorial_odd expects n >= 0")
    # (2n+1)! // (2^n n!) — exact division
    return factorial(2 * n + 1) // ((1 << n) * factorial(n))


def l_degree(g: int, n: int, alpha: Alpha) -> int:
    r"""
    Cohomological degree of the λ–descendant term:
        ell = 3g - 3 + n - Σ a_i
    """
    return 3 * g - 3 + n - sum(alpha)


def _odd_df_product_and_pow2(alpha: Alpha) -> tuple[int, int]:
    """
    Compute:
      P = ∏_i (2a_i + 1)!!    and
      S = Σ_i a_i
    but using the identity (2a+1)!! = (2a+1)! / (2^a a!) so that

        ∏ (2a_i + 1)!! * 4^{Σ a_i}
      = ( 2^{Σ a_i} * ∏ (2a_i + 1)! / a! )

    Returns (P_times_4S, S) with P_times_4S = ∏ (2a_i+1)!! * 4^{Σ a_i}.
    """
    _validate_alpha(alpha)
    s = sum(alpha)
    # ∏ ((2a+1)! // a!)
    fact_ratio = prod(factorial(2 * a + 1) // factorial(a) for a in alpha)
    # multiply by 2^s (since 4^s / 2^s = 2^s)
    num = (1 << s) * fact_ratio
    return num, s


def rx_from_x(g: int, n: int, alpha: Alpha, X: Fraction | int) -> Fraction:
    r"""
    Convert normalized intersection number X to R_X:

        R_X = [ (∏ (2a_i + 1)!!) * 4^{Σ a_i} / ell! ] * X,

    with ell = 3g - 3 + n - Σ a_i. Returns 0 if ell < 0.
    """
    ell = l_degree(g, n, alpha)
    if ell < 0:
        return Fraction(0)

    num, _ = _odd_df_product_and_pow2(alpha)  # ∏ (2a+1)!! * 4^{Σ a}
    den = factorial(ell)
    return Fraction(num, den) * Fraction(X)


def x_from_rx(g: int, n: int, alpha: Alpha, RX: Fraction | int) -> Fraction:
    r"""
    Inverse transform: recover X from R_X:

        X = [ ell! / ( (∏ (2a_i + 1)!!) * 4^{Σ a_i} ) ] * R_X,

    with ell = 3g - 3 + n - Σ a_i. Returns 0 if ell < 0.
    """
    ell = l_degree(g, n, alpha)
    if ell < 0:
        return Fraction(0)

    num = factorial(ell)
    den, _ = _odd_df_product_and_pow2(alpha)  # ∏ (2a+1)!! * 4^{Σ a}
    return Fraction(num, den) * Fraction(RX)
