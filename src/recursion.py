from __future__ import annotations
from fractions import Fraction
from math import factorial
from typing import Callable, Iterable, Tuple

from .types import Alpha
from .normalization import l_degree
from .partitions import split_index_sets
from .bernoulli import bernoulli_even

# ---- Bernoulli numbers B_{2k} and A(k) exact ----

def bernoulli_numbers_even(max_k: int):
    """Return dict {2m: B_{2m}} for m=0..max_k using Akiyama–Tanigawa."""
    
    B = {}
    for m in range(0, max_k+1):
        A = [Fraction(0)]*(m+1)
        for j in range(0, m+1):
            A[j] = Fraction(1, j+1)
            for k in range(j, 0, -1):
                A[k-1] = k*(A[k-1] - A[k])
        B[2*m] = A[0]
    return B

_B_CACHE = {}

def A_k(k: int) -> Fraction:
    if k == 0:
        return Fraction(4, 1)  # by definition: ζ(0) = -1/2 ⇒ A(0) = 4
    B2k = bernoulli_even(k)  # B_{2k} as a Fraction
    factor = 1 - Fraction(1, 1 << (2*k - 1))  # 1 - 2^{1 - 2k}
    num = ((-1)**(k + 1)) * (1 << (k + 2)) * B2k * factor
    den = factorial(2 * k)
    return Fraction(num, den)

# ---- RX recursion ----

def rx_rec_opt(g: int, n: int, alpha: Alpha, rx_lookup: Callable[[int,int,Alpha], Fraction]) -> Fraction:
    """Compute RX(g,n,alpha) by Mirzakhani-style recursion using a lookup for dependencies.

    Requirements:
    - Must NEVER call rx_lookup(g,n,alpha) for the same triple (no self-lookups).
    - Only depends on keys with strictly smaller D=3g-3+n (by construction of the recursion).
    - Returns Fraction exacts.
    """
    alpha = tuple(sorted(alpha, reverse=True))
    # Base seeds
    if g == 0 and n == 3 and alpha == (0,0,0):
        return Fraction(1)
    if g == 1 and n == 1 and alpha == (0,):
        return Fraction(1,24)
    if g == 1 and n == 1 and alpha == (1,):
        return Fraction(1,2)

    # Stability check
    if not (n >= 1 and (2*g - 2 + n) > 0):
        return Fraction(0)

    l = 3*g - 3 + n - sum(alpha)
    if l < 0:
        return Fraction(0)
    # If l==0, the recursion sum will have k from 0..0 and many internal l' constraints
    # reduce to base/vanishing terms. We do NOT delegate to rx_lookup(g,n,alpha).

    total = Fraction(0)

    # Term 1: sum over j (moving one marking to combine with alpha1)
    # Follow the Mathematica structure: alpha1 = min(alpha), alphas sorted desc.
    # We'll use alpha[ -1 ] as the minimum.
    alpha_sorted = list(alpha)
    alpha1 = alpha_sorted[-1]

    for k in range(0, l+1):
        coeff_k = A_k(k)
        if coeff_k == 0:
            continue

        # --- (g, n-1) contribution
        if n - 1 >= 1:
            inner_sum = Fraction(0)
            for j in range(0, n-1):
                a2 = alpha1 + alpha_sorted[j] + k - 1
                if a2 >= 0:
                    # build new alpha: drop last (min) and position j, append a2
                    new_vec = [alpha_sorted[i] for i in range(n-1) if i != j]
                    # note: alpha_sorted has length n; last index (n-1) is the min; we delete it.
                    # remove the min (last entry)
                    # we've already not included index j among first n-1 entries above
                    new_vec.append(a2)
                    new_alpha = tuple(sorted(new_vec, reverse=True))
                    inner_sum += (2*alpha_sorted[j] + 1) * rx_lookup(g, n-1, new_alpha)
            total += coeff_k * inner_sum

        # --- (g-1, n+1) contribution
        m = k + alpha1 - 2
        if g > 0 and m >= 0:
            inner = Fraction(0, 1)
            # only d1 <= d2: d1 = 0..floor(m/2)
            for d1 in range(0, m // 2 + 1):
                d2 = m - d1
                sym = Fraction(1, 2) if d1 == d2 else Fraction(1, 1)
                new_vec = alpha_sorted[:-1] + [d1, d2]
                new_alpha = tuple(sorted(new_vec, reverse=True))
                inner += sym * rx_lookup(g - 1, n + 1, new_alpha)
            total += coeff_k * 4 * inner


        # --- Splitting contribution
        fac_n1 = Fraction(1,2) if n == 1 else Fraction(1,1)
        split_sum = Fraction(0)
        # indices refer to first n-1 entries (excluding the min at the end)
        for I, J in split_index_sets(n-1):
            # sum over d from 0..k+alpha1-2
            up = k + alpha1 - 2
            if up >= 0:
                for d in range(0, up+1):
                    vecI = [alpha_sorted[i] for i in I]
                    vecJ = [alpha_sorted[j] for j in J]
                    alpha1_vec = alpha_sorted[:-1]  # first n-1 entries
                    # Append d to side 1, (k + alpha1 - 2 - d) to side 2
                    a1 = tuple(sorted(vecI + [d], reverse=True))
                    a2 = tuple(sorted(vecJ + [up - d], reverse=True))
                    g1_min, g1_max = 0, g
                    for g1 in range(g1_min, g1_max+1):
                        g2 = g - g1
                        n1, n2 = len(a1), len(a2)
                        l1 = 3*g1 - 3 + n1 - sum(a1)
                        l2 = 3*g2 - 3 + n2 - sum(a2)
                        if l1 >= 0 and l2 >= 0:
                            split_sum += rx_lookup(g1, n1, a1) * rx_lookup(g2, n2, a2)

        if split_sum:
            total += coeff_k * 4 * fac_n1 * split_sum

    return total




from fractions import Fraction
from typing import Tuple, Callable
from sympy import Symbol, Rational
from sympy.core.expr import Expr

Alpha = Tuple[int, ...]  # or import your Alpha
# You must have these available from your codebase:
#   - A_k(k: int) -> Fraction
#   - split_index_sets(n_minus_1: int) -> Iterable[Tuple[List[int], List[int]]]

def _rat(x: Fraction | int) -> Rational:
    """Convert Python int/Fraction -> SymPy Rational."""
    if isinstance(x, Fraction):
        return Rational(x.numerator, x.denominator)
    return Rational(x, 1)

# Optional: a symbol factory so every (g,n,alpha) maps to a single Symbol
_X_cache: dict[tuple[int,int,Alpha], Symbol] = {}
def X_sym(g: int, n: int, alpha: Alpha) -> Symbol:
    key = (g, n, alpha)
    s = _X_cache.get(key)
    if s is None:
        s = Symbol(f"X[{g},{n},{alpha}]")
        _X_cache[key] = s
    return s

def rx_rec_opt_symbolic(g: int, n: int, alpha: Alpha) -> Expr:
    """
    Symbolic version of the Mirzakhani-style RX recursion:
    replaces every dependency RX(g',n',alpha') with the symbol X[g',n',alpha'].
    Returns a SymPy expression with exact rationals.
    """
    alpha = tuple(sorted(alpha, reverse=True))

    # Base seeds (as exact rationals)
    if g == 0 and n == 3 and alpha == (0, 0, 0):
        return Rational(1, 1)
    if g == 1 and n == 1 and alpha == (0,):
        return Rational(1, 24)
    if g == 1 and n == 1 and alpha == (1,):
        return Rational(1, 2)

    # Stability of the target
    if not (n >= 1 and (2*g - 2 + n) > 0):
        return Rational(0, 1)

    l = 3*g - 3 + n - sum(alpha)
    if l < 0:
        return Rational(0, 1)

    total: Expr = Rational(0, 1)

    alpha_sorted = list(alpha)
    alpha1 = alpha_sorted[-1]

    for k in range(0, l + 1):
        ck = _rat(A_k(k))
        if ck == 0:
            continue

        # --- Term (g, n-1)
        if n - 1 >= 1:
            inner1: Expr = Rational(0, 1)
            for j in range(0, n - 1):
                a2 = alpha1 + alpha_sorted[j] + k - 1
                if a2 >= 0:
                    new_vec = [alpha_sorted[i] for i in range(n - 1) if i != j]
                    new_vec.append(a2)
                    new_alpha = tuple(sorted(new_vec, reverse=True))
                    coeff = Rational(2*alpha_sorted[j] + 1, 1)
                    inner1 += coeff * X_sym(g, n - 1, new_alpha)
            total += ck * inner1

        # --- Term (g-1, n+1)
        m = k + alpha1 - 2
        if g > 0 and m >= 0:
            inner2 = Rational(0, 1)
            for d1 in range(0, m // 2 + 1):     # <= half to avoid double counting
                d2 = m - d1
                sym = Rational(1, 2) if d1 == d2 else Rational(1, 1)
                new_vec = alpha_sorted[:-1] + [d1, d2]
                new_alpha = tuple(sorted(new_vec, reverse=True))
                inner2 += sym * X_sym(g - 1, n + 1, new_alpha)
            total += ck * Rational(4, 1) * inner2


        # --- Splitting term
        fac_n1 = Rational(1,2) if n == 1 else Rational(1,1)
        up = k + alpha1 - 2
        if up >= 0:
            split_sum: Expr = Rational(0, 1)
            for I, J in split_index_sets(n - 1):   # should mirror your Mathematica
                vecI = [alpha_sorted[i] for i in I]
                vecJ = [alpha_sorted[j] for j in J]
                for d in range(0, up + 1):
                    a1 = tuple(sorted(vecI + [d], reverse=True))
                    a2 = tuple(sorted(vecJ + [up - d], reverse=True))
                    for g1 in range(0, g + 1):
                        g2 = g - g1
                        n1, n2 = len(a1), len(a2)
                        l1 = 3*g1 - 3 + n1 - sum(a1)
                        l2 = 3*g2 - 3 + n2 - sum(a2)
                        if l1 >= 0 and l2 >= 0:
                            split_sum += X_sym(g1, n1, a1) * X_sym(g2, n2, a2)
            if split_sum != 0:
                total += ck * Rational(4, 1) * fac_n1 * split_sum

    return total
