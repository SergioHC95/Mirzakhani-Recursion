"""
Compute driver for Weil–Petersson intersection numbers.

- Orchestrates generation of R_X(g, n; α) values by (g, n) slices.
- Supports parallel evaluation with a per-process read-only snapshot to
  avoid re-sending the cache on every task (Windows-safe).
- Exposes a simple iterator `iterate(...)` that incrementally fills and
  checkpoints a pickle cache after each (g, n).

Conventions
-----------
- α (Alpha) is always handled in canonical nonincreasing order.
- Exact arithmetic via fractions.Fraction throughout.
"""

from __future__ import annotations
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from typing import Dict, Iterable, List, Optional, Set, Tuple, Callable
import time

from .logging_config import get_logger
from .partitions import alpha_bs
from .recursion import rx_rec_opt
from .store import canonical, dump_rx_atomic, load_rx
from .types import Alpha, Job, Key

__all__ = [
    "degree",
    "is_stable",
    "ordered_pairs",
    "known_pairs",
    "missing_alphas",
    "compute_missing_gn",
    "compute_pair_gn",
    "iterate",
]

log = get_logger(__name__)

# ------------------------------------------------------------------------------
# Shared snapshot for workers (installed once per process via pool initializer)
# ------------------------------------------------------------------------------
_SNAPSHOT: Optional[Dict[Key, Fraction]] = None


def _init_snapshot(snap: Dict[Key, Fraction]) -> None:
    """Pool initializer: install the read-only snapshot once per worker."""
    global _SNAPSHOT
    _SNAPSHOT = snap


# --------------------------- basic geometry helpers ---------------------------

def degree(g: int, n: int) -> int:
    """Total degree D = 3g - 3 + n."""
    return 3 * g - 3 + n


def is_stable(g: int, n: int) -> bool:
    """Deligne–Mumford stability: 2g - 2 + n > 0 and n ≥ 1."""
    return (2 * g - 2 + n) > 0 and n >= 1


def ordered_pairs(dmax: int) -> List[Tuple[int, int]]:
    """
    All stable (g, n) such that degree(g, n) ≤ dmax,
    sorted by degree, then lexicographically by (g, n).
    """
    pairs = [(g, n)
             for g in range(dmax + 1)
             for n in range(1, 3 * dmax + 4)  # generous upper bound on n
             if is_stable(g, n) and degree(g, n) <= dmax]
    return sorted(pairs, key=lambda t: (degree(*t), *t))

def known_pairs(rx: Dict[Key, Fraction]) -> Set[Tuple[int, int]]:
    """Set of (g, n) that already have at least one α in `rx`."""
    return {(g, n) for (g, n, _) in rx.keys()}


def missing_alphas(g: int, n: int, rx: Dict[Key, Fraction]) -> List[Alpha]:
    """
    Canonical α’s required for (g, n) at total degree D = 3g - 3 + n,
    minus those already present in `rx`. Deterministic reverse-lex order.
    """
    D = degree(g, n)
    need = set(alpha_bs(n, D))           # already canonical, sum(α) ≤ D
    have = {a for (gg, nn, a) in rx if gg == g and nn == n}
    return sorted(need - have, reverse=True)


# -------------------------------- worker logic --------------------------------

def _worker(job: Job) -> Tuple[Alpha, Fraction]:
    """
    Compute a single R_X(g, n; α).

    Uses a per-process read-only snapshot (_SNAPSHOT) for dependency lookups and
    a small per-task local cache for transitive dependencies within the same job.
    """
    snapshot: Dict[Key, Fraction] = _SNAPSHOT or {}
    alpha_can = canonical(job.alpha)
    target_key: Key = (job.g, job.n, alpha_can)

    local_cache: Dict[Key, Fraction] = {}
    inflight: Set[Key] = set()

    def compute_key(key: Key) -> Fraction:
        if key in local_cache:
            return local_cache[key]
        if key in snapshot:
            return snapshot[key]
        if key in inflight:
            raise RuntimeError(f"Dependency cycle detected at {key}")
        inflight.add(key)

        gk, nk, ak = key

        def dep_lookup(g: int, n: int, alpha: Alpha) -> Fraction:
            k = (g, n, canonical(alpha))
            if k in local_cache:
                return local_cache[k]
            if k in snapshot:
                return snapshot[k]
            if k in inflight:
                raise RuntimeError(f"Dependency cycle detected at {k}")
            return compute_key(k)

        val = rx_rec_opt(gk, nk, ak, dep_lookup)
        local_cache[key] = val
        inflight.remove(key)
        return val

    def rx_lookup(g: int, n: int, alpha: Alpha) -> Fraction:
        key = (g, n, canonical(alpha))
        if key in local_cache:
            return local_cache[key]
        if key in snapshot:
            return snapshot[key]
        if key == target_key:  # defensive: recurrence must not self-query
            raise KeyError(key)
        return compute_key(key)

    # Always call the recurrence with the canonical alpha
    val = rx_rec_opt(job.g, job.n, alpha_can, rx_lookup)
    local_cache[target_key] = val
    return alpha_can, val


# ------------------------------- driver routines ------------------------------

def _compute_gn(
    g: int,
    n: int,
    alphas: Iterable[Alpha],
    rx: Dict[Key, Fraction],
    max_workers: Optional[int],
    checkpoint_fn: Optional[Callable[[int, Alpha], None]] = None,
) -> Dict[Alpha, Fraction]:
    """
    Compute R_X(g, n; α) for the provided α’s, using `rx` as read-only snapshot.

    - If `max_workers == 1`, run sequentially in-process (no spawn cost).
    - Else, start a fresh process pool and install the snapshot once per worker.
    - Optionally call `checkpoint_fn(i, α)` after computing each α.

    Returns a dict {α: R_X(g, n; α)} with canonical α.
    """
    alpha_list = [canonical(a) for a in alphas]
    if not alpha_list:
        return {}

    snapshot = dict(rx)  # Ensure isolation from mutations

    results: Dict[Alpha, Fraction] = {}

    # Sequential path
    if max_workers == 1:
        snapshot_local = snapshot  # avoid using global
        def worker_local(job: Job) -> Tuple[Alpha, Fraction]:
            alpha_can = canonical(job.alpha)
            target_key: Key = (job.g, job.n, alpha_can)

            local_cache: Dict[Key, Fraction] = {}
            inflight: Set[Key] = set()

            def compute_key(key: Key) -> Fraction:
                if key in local_cache:
                    return local_cache[key]
                if key in snapshot_local:
                    return snapshot_local[key]
                if key in inflight:
                    raise RuntimeError(f"Dependency cycle detected at {key}")
                inflight.add(key)

                gk, nk, ak = key

                def dep_lookup(g: int, n: int, alpha: Alpha) -> Fraction:
                    k = (g, n, canonical(alpha))
                    if k in local_cache:
                        return local_cache[k]
                    if k in snapshot_local:
                        return snapshot_local[k]
                    if k in inflight:
                        raise RuntimeError(f"Dependency cycle detected at {k}")
                    return compute_key(k)

                val = rx_rec_opt(gk, nk, ak, dep_lookup)
                local_cache[key] = val
                inflight.remove(key)
                return val

            def rx_lookup(g: int, n: int, alpha: Alpha) -> Fraction:
                key = (g, n, canonical(alpha))
                if key in local_cache:
                    return local_cache[key]
                if key in snapshot_local:
                    return snapshot_local[key]
                if key == target_key:  # defensive: recurrence must not self-query
                    raise KeyError(key)
                return compute_key(key)

            # Always call the recurrence with the canonical alpha
            val = rx_rec_opt(job.g, job.n, alpha_can, rx_lookup)
            local_cache[target_key] = val
            return alpha_can, val

        try:
            for i, a in enumerate(alpha_list):
                alpha, val = worker_local(Job(g, n, a))
                results[alpha] = val
                if checkpoint_fn is not None:
                    checkpoint_fn(i, alpha)
        finally:
            pass  # no global to clean up
        return results

    # Parallel path
    try:
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_snapshot,
            initargs=(snapshot,),
        ) as ex:
            futs = {ex.submit(_worker, Job(g, n, a)): (i, a) for i, a in enumerate(alpha_list)}
            for fut in tqdm(as_completed(futs), total=len(futs), desc=f"(g={g}, n={n})"):
                i, label = futs[fut]
                try:
                    alpha, val = fut.result()
                except Exception as e:
                    raise RuntimeError(f"Failed at (g={g}, n={n}, α={label}): {e}") from e
                results[alpha] = val
                if checkpoint_fn is not None:
                    checkpoint_fn(i, alpha)
    except KeyboardInterrupt:
        raise  # ensure Ctrl+C still works
    return results

def compute_pair_gn(g: int, n: int, rx: Dict[Key, Fraction], max_workers: Optional[int]) -> Dict[Alpha, Fraction]:
    """Compute all α with sum(α) ≤ D for the given (g, n)."""
    return _compute_gn(g, n, alpha_bs(n, degree(g, n)), rx, max_workers)


def compute_missing_gn(g: int, n: int, rx: Dict[Key, Fraction], max_workers: Optional[int]) -> Dict[Alpha, Fraction]:
    """Compute only the α not present in `rx` for the given (g, n)."""
    return _compute_gn(g, n, missing_alphas(g, n, rx), rx, max_workers)


import time
from typing import Optional, Dict
from fractions import Fraction

def iterate(
    dmax: int,
    cache_path: str,
    max_workers: Optional[int] = None,
    fill_missing: bool = True,
) -> Dict[Key, Fraction]:
    """
    Build / extend the on-disk table of R_X values up to `dmax`,
    checkpointing after each (g, n) and also during long runs.

    Parameters
    ----------
    dmax : int
        Max dimension of moduli space.
    cache_path : str
        Pickle path for the table; atomically updated after each (g, n).
    max_workers : int | None
        If None, let the executor choose. If 1, run sequentially (no processes).
    fill_missing : bool
        If True (default), only compute α’s not present for (g, n).
        If False, recompute the full (g, n) slice (sum(α) ≤ D).

    Returns
    -------
    Dict[Key, Fraction]
        The full table mapping (g, n, α) → Fraction (α canonical).
    """
    rx: Dict[Key, Fraction] = load_rx(cache_path)

    # Track time for periodic checkpointing
    last_save_time = time.monotonic()
    checkpoint_interval = 600 ###3600  # seconds (1 hour)

    for (g, n) in ordered_pairs(dmax):
        want = missing_alphas(g, n, rx) if fill_missing else list(alpha_bs(n, degree(g, n)))
        if not want:
            continue

        mode = "missing" if fill_missing else "full"
        workers_str = "seq" if max_workers == 1 else (str(max_workers) if max_workers else "default")
        log.info("Computing (g=%d, n=%d): %d α’s (mode=%s, workers=%s)", g, n, len(want), mode, workers_str)

        # Define checkpoint callback function to be called during _compute_gn
        def checkpoint_fn(i: int, alpha: Alpha, g=g, n=n):
            nonlocal last_save_time
            now = time.monotonic()
            if now - last_save_time > checkpoint_interval:
                dump_rx_atomic(cache_path, rx)
                last_save_time = now
                log.info("⏳ Periodic checkpoint at α #%d = %s during (g=%d, n=%d)", i, alpha, g, n)

        # Compute with periodic checkpointing
        res = _compute_gn(g, n, want, rx, max_workers, checkpoint_fn=checkpoint_fn)

        # Store final results for this (g, n)
        for a, v in res.items():
            rx[(g, n, a)] = v

        dump_rx_atomic(cache_path, rx)
        last_save_time = time.monotonic()  # reset timer after full (g, n) checkpoint
        log.info("✅ Final checkpoint saved for (g=%d, n=%d)", g, n)

    return rx
