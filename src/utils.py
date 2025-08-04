from __future__ import annotations

from fractions import Fraction
from typing import Iterator, List, Tuple
from collections import OrderedDict

from .store import load_rx, dump_rx_atomic, canonical
from .normalization import x_from_rx
from .partitions import alpha_bs
from .compute import degree, is_stable, _compute_gn  # _compute_gn is the new driver

from pathlib import Path

ROOT = Path.cwd().parent                 # repo/
CACHE = ROOT / "data" / "wp_rx.pkl"
CACHE.parent.mkdir(parents=True, exist_ok=True)

Alpha = Tuple[int, ...]
Key = tuple[int, int, tuple[int, ...]]
__all__ = ["intersection_number"]

def intersection_number(
    g: int,
    n: int,
    alpha: Alpha,
    cache_path: str = CACHE,
    update: bool = True,
    max_workers: int | None = None,
) -> Fraction:
    """
    Compute ⟨τ_{α₁}⋯τ_{αₙ}⟩_g:
      - loads the R_X cache
      - computes missing R_X(g,n,α) (parallel if max_workers>1)
      - optionally persists the expanded cache
      - returns the normalized intersection number as Fraction
    """
    D = degree(g, n)
    if D < 0 or not is_stable(g, n):
        return Fraction(0)

    alpha_can = canonical(alpha)
    if sum(alpha_can) != D:
        return Fraction(0)

    rx = load_rx(cache_path)
    key = (g, n, alpha_can)

    if key not in rx:
        # Compute just the needed α; dependencies are handled by the worker.
        new_vals = _compute_gn(g, n, [alpha_can], rx, max_workers)
        for a, val in new_vals.items():
            rx[(g, n, a)] = val
        if update and new_vals:
            dump_rx_atomic(cache_path, rx)

    RX = rx[key]
    return x_from_rx(g, n, alpha_can, RX)

def degree_key(item: tuple[Key, Fraction]) -> tuple[int, int, int, tuple[int, ...]]:
    """
    Sorting key for (g, n, α) entries by:
    - degree(g, n) = 3g - 3 + n
    - then (g, n)
    - then α
    """
    g, n, alpha = item[0]
    return (3 * g - 3 + n, g, n, alpha)

def sorted_rx(rx: dict[Key, Fraction]) -> OrderedDict[Key, Fraction]:
    """
    Return an OrderedDict version of rx sorted by (degree, g, n, α).
    """
    return OrderedDict(sorted(rx.items(), key=degree_key))
