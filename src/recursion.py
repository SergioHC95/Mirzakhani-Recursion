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
    B2k = bernoulli_even(k)
    exp = 2*k - 1
    # handle negative shifts by inverting the power
    if exp >= 0:
        two_pow = Fraction(1, 1 << exp)
    else:
        two_pow = Fraction(1 << (-exp), 1)
    factor = 1 - two_pow
    num = ((-1)**(k+1)) * (1 << (k+2)) * B2k * factor
    return Fraction(num, 1)

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
        if n-1 >= 1 and (2*g - 2 + (n-1)) > 0:
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
        if g > 0:
            dm = (k + alpha1 - 2)
            if dm % 2 == 0 and dm >= 0:
                dm //= 2
                inner = Fraction(0)
                for d1 in range(0, dm+1):
                    d2 = 2*dm - d1
                    sym = Fraction(1,2) if d1 == d2 else Fraction(1,1)
                    new_vec = alpha_sorted[:-1] + [d1, d2]
                    new_alpha = tuple(sorted(new_vec, reverse=True))
                    term = rx_lookup(g-1, n+1, new_alpha)
                    inner += sym * term
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
                        if l1 >= 0 and l2 >= 0 and (n1 >= 1 and (2*g1 - 2 + n1) > 0) and (n2 >= 1 and (2*g2 - 2 + n2) > 0):
                            split_sum += rx_lookup(g1, n1, a1) * rx_lookup(g2, n2, a2)
        if split_sum:
            total += coeff_k * 4 * fac_n1 * split_sum

    return total
