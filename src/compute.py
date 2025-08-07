"""
Compute driver for Weil–Petersson intersection numbers.

- Orchestrates generation of R_X(g, n; α) values by (g, n) slices.
- Supports parallel evaluation with a per-process read-only snapshot to
  avoid re-sending the cache on every task (Windows-safe).
- Exposes an iterator `iterate(...)` that fills and checkpoints a pickle cache.

Conventions
-----------
- α (Alpha) is always handled in canonical nonincreasing order.
- Exact arithmetic via fractions.Fraction throughout.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED, Future
from fractions import Fraction
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TaskProgressColumn,
    TextColumn,
)
try:
    # Rich >=13 provides MofNColumn for x/y; if unavailable, we'll emulate via description text.
    from rich.progress import MofNColumn
    _HAVE_MOFN = True
except Exception:
    _HAVE_MOFN = False

from .logging_config import get_logger
from .partitions import alpha_bs
from .recursion import rx_rec_opt
from .store import canonical, dump_rx_atomic, load_rx
from .types import Alpha, Job, Key
from .genus0 import genus0_x
from .normalization import rx_from_x


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
    pairs = [
        (g, n)
        for g in range(dmax + 1)
        for n in range(1, 3 * dmax + 4)  # generous upper bound on n
        if is_stable(g, n) and degree(g, n) <= dmax
    ]
    return sorted(pairs, key=lambda t: (degree(*t), *t))


def known_pairs(rx: Dict[Key, Fraction]) -> Set[Tuple[int, int]]:
    """Set of (g, n) that already have at least one α in `rx`."""
    return {(g, n) for (g, n, _) in rx.keys()}


def missing_alphas(g: int, n: int, rx: Dict[Key, Fraction]) -> List[Alpha]:
    """
    Canonical α’s required for (g, n) at total degree D = 3g - 3 + n,
    minus those already present in `rx`. Deterministic reverse-lex order.

    NOTE: We include all α with sum(α) ≤ D for all genus, so genus-0
    is treated uniformly (your closed form handles all α).
    """
    D = degree(g, n)
    need = set(alpha_bs(n, D))  # already canonical, sum(α) ≤ D
    have = {a for (gg, nn, a) in rx if gg == g and nn == n}
    return sorted(need - have, reverse=True)


# -------------------------------- worker logic --------------------------------

def _worker(job: Job) -> Tuple[Alpha, Fraction, float, int]:
    """
    Compute a single R_X(g, n; α) value using the recursive optimizer.

    This function runs in a separate process (via ProcessPoolExecutor).
    Uses a global read-only snapshot (_SNAPSHOT) for previously computed values
    and maintains a local cache for intermediate dependencies within this task.

    Returns
    -------
    (alpha_can, result, elapsed_seconds, pid)
    """
    t0 = time.time()
    pid = os.getpid()

    alpha_can = canonical(job.alpha)
    target_key: Key = (job.g, job.n, alpha_can)

    snapshot: Dict[Key, Fraction] = _SNAPSHOT or {}  # shared read-only cache
    local_cache: Dict[Key, Fraction] = {}            # local per-job cache
    inflight: Set[Key] = set()                       # detect circular dependencies

    def compute_key(key: Key) -> Fraction:
        """Compute value for (g, n, α) recursively, with memoization."""
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
        """Lookup for rx_rec_opt: canonicalize, prevent self-query, then compute."""
        key = (g, n, canonical(alpha))
        if key in local_cache:
            return local_cache[key]
        if key in snapshot:
            return snapshot[key]
        if key == target_key:
            raise KeyError(f"Recurrence attempted to look up itself: {key}")
        return compute_key(key)

    # Final evaluation of R_X
    val = rx_rec_opt(job.g, job.n, alpha_can, rx_lookup)
    local_cache[target_key] = val
    elapsed = time.time() - t0

    return alpha_can, val, elapsed, pid


# ------------------------------- driver routines ------------------------------

def _build_worker_table(g: int, n: int, stats: Dict[int, Tuple[int, float, float]]) -> Table:
    """
    Build a Rich table summarizing per-worker activity.

    Parameters
    ----------
    stats : dict pid -> (count, total_time, last_time)

    Returns
    -------
    Table
    """
    # Build enhanced worker stats table (sorted + styled)
    table = Table(title=f"Worker Stats", box=box.MINIMAL_DOUBLE_HEAD)
    table.add_column("PID", justify="right")
    table.add_column("Tasks", justify="right")
    table.add_column("Total (s)", justify="right")
    table.add_column("Avg (s)", justify="right")
    table.add_column("Last (s)", justify="right")

    # Highlighting thresholds
    avg_thresh = 100.0
    last_thresh = 500.0 

    # Sort by task count descending
    sorted_items = sorted(stats.items(), key=lambda x: x[1][0], reverse=True)

    for pid, (count, total, last) in sorted_items:
        avg = total / count if count > 0 else 0.0

        avg_str = f"[yellow]{avg:.2f}[/yellow]" if avg > avg_thresh else f"{avg:.2f}"
        last_str = f"[red]{last:.2f}[/red]" if last > last_thresh else f"{last:.2f}"

        table.add_row(
            str(pid),
            str(count),
            f"{total:.2f}",
            avg_str,
            last_str,
        )

    return table


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

    # Genus-0: closed form valid for all α with sum(α) ≤ D
    if g == 0:
        results: Dict[Alpha, Fraction] = {}
        for i, alpha in enumerate(alpha_list):
            X = genus0_x(n, alpha)             # your closed form
            RX = rx_from_x(0, n, alpha, X)     # normalized R_X
            results[alpha] = RX
            if checkpoint_fn:
                checkpoint_fn(i, alpha)
        return results

    # Snapshot for workers (ensure isolation from mutations)
    snapshot = dict(rx)
    results: Dict[Alpha, Fraction] = {}

    # Sequential path (helpful for debugging)
    if max_workers == 1:
        for i, a in enumerate(alpha_list):
            alpha, val, _, _ = _worker(Job(g, n, a))
            results[alpha] = val
            rx[(g, n, alpha)] = val
            if checkpoint_fn: checkpoint_fn(i, alpha)
        return results

    # ---------------------------- Parallel path --------------------------------
    # Robust Rich rendering on Windows / IDE terminals: force a real terminal.
    console = Console(force_terminal=True, highlight=False, soft_wrap=False)

    # Global progress bar configuration
    progress_columns = [
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
    ]
    if _HAVE_MOFN:
        progress_columns.append(MofNColumn())  # shows x/y
    progress_columns.extend([
        TaskProgressColumn(),                  # percentage
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ])

    progress = Progress(*progress_columns, transient=False, console=console)
    task_id = progress.add_task(f"Computing {len(alpha_list)} α’s for (g={g}, n={n})", total=len(alpha_list))

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_snapshot,
        initargs=(snapshot,),
    ) as ex:

        futures: List[Future] = []
        fut_to_label: Dict[Future, Tuple[int, Alpha]] = {}
        for i, a in enumerate(alpha_list):
            fut = ex.submit(_worker, Job(g, n, a))
            futures.append(fut)
            fut_to_label[fut] = (i, a)

        worker_stats: Dict[int, Tuple[int, float, float]] = {}  # pid -> (count, total_time, last_time)

        # Live layout: progress + table; refresh even when no futures complete.
        with Live(Group(progress, _build_worker_table(g, n, worker_stats)),
                  refresh_per_second=5, console=console, screen=False) as live:

            remaining: Set[Future] = set(futures)
            t_start = time.time()

            while remaining:
                done, _ = wait(remaining, timeout=0.2, return_when=FIRST_COMPLETED)
                if not done:
                    # No completions yet; just refresh the display.
                    live.update(Group(progress, _build_worker_table(g, n, worker_stats)))
                    continue

                for fut in done:
                    remaining.remove(fut)
                    i, label = fut_to_label[fut]
                    try:
                        alpha, val, elapsed, pid = fut.result()
                    except Exception as e:
                        raise RuntimeError(f"Failed at (g={g}, n={n}, α={label}): {e}") from e

                    # Update results and checkpoint
                    results[alpha] = val
                    rx[(g, n, alpha)] = val
                    if checkpoint_fn: checkpoint_fn(i, alpha)

                    # Update worker stats
                    count, total_t, _ = worker_stats.get(pid, (0, 0.0, 0.0))
                    worker_stats[pid] = (count + 1, total_t + elapsed, elapsed)

                    # Advance progress
                    progress.update(task_id, advance=1)

                    # Rebuild table and refresh UI
                    table = _build_worker_table(g, n, worker_stats)

                    # Optional: add a tiny throughput/summary line below the bar
                    done_so_far = len(futures) - len(remaining)
                    elapsed_all = max(1e-9, time.time() - t_start)
                    throughput = done_so_far / elapsed_all
                    summary = TextColumn(f"[grey70]Throughput: {throughput:.2f} tasks/s")
                    # Render progress + summary + table
                    live.update(Group(progress, table))

    return results


def compute_pair_gn(g: int, n: int, rx: Dict[Key, Fraction], max_workers: Optional[int]) -> Dict[Alpha, Fraction]:
    """Compute all α with sum(α) ≤ D for the given (g, n)."""
    return _compute_gn(g, n, alpha_bs(n, degree(g, n)), rx, max_workers)


def compute_missing_gn(g: int, n: int, rx: Dict[Key, Fraction], max_workers: Optional[int]) -> Dict[Alpha, Fraction]:
    """Compute only the α not present in `rx` for the given (g, n)."""
    return _compute_gn(g, n, missing_alphas(g, n, rx), rx, max_workers)


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

    # Periodic checkpointing
    last_save_time = time.monotonic()
    checkpoint_interval = 3600  # seconds (1 hour)

    for (g, n) in ordered_pairs(dmax):
        # Only proceed if there’s work to do
        want = missing_alphas(g, n, rx) if fill_missing else list(alpha_bs(n, degree(g, n)))
        if not want:
            continue  # skip fully computed (g, n)

        mode = "missing" if fill_missing else "full"
        workers_str = "seq" if max_workers == 1 else (str(max_workers) if max_workers else "default")
        log.info(f"Computing (g={g}, n={n}): {len(want)} α’s (workers={workers_str})")

        # Periodic checkpoint callback, closed over `rx`
        def checkpoint_fn(i: int, alpha: Alpha, g=g, n=n) -> None:
            nonlocal last_save_time
            now = time.monotonic()
            if now - last_save_time > checkpoint_interval:
                dump_rx_atomic(cache_path, rx)
                last_save_time = now
                log.info(f"⏳ Periodic checkpoint at α #{i} = {alpha} during (g={g}, n={n})")

        # Compute with periodic checkpointing
        res = _compute_gn(g, n, want, rx, max_workers, checkpoint_fn=checkpoint_fn)

        # Store final results for this (g, n)
        for a, v in res.items():
            rx[(g, n, a)] = v

        dump_rx_atomic(cache_path, rx)
        last_save_time = time.monotonic()  # reset timer after full (g, n) checkpoint
        log.info(f"✅ Final checkpoint saved for (g={g}, n={n})")

    return rx
