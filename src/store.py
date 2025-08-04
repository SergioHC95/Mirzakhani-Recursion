from __future__ import annotations

import os
import pickle
import tempfile
import time
from fractions import Fraction
from pathlib import Path
from typing import Dict, Mapping

from .types import Alpha, Key

__all__ = ["canonical", "load_rx", "dump_rx_atomic", "seed_rx", "normalize_rx"]


# ------------------------------- basic utilities -------------------------------

def canonical(alpha: Alpha) -> Alpha:
    """Return α sorted in nonincreasing (canonical) order."""
    return tuple(sorted(alpha, reverse=True))


def seed_rx() -> Dict[Key, Fraction]:
    """Minimal seed table for R_X, using canonical α."""
    return {
        (0, 3, (0, 0, 0)): Fraction(1, 1),
        (1, 1, (0,)): Fraction(1, 24),
        (1, 1, (1,)): Fraction(1, 2),
    }


def normalize_rx(raw: Mapping[Key, Fraction | int]) -> Dict[Key, Fraction]:
    """
    Canonicalize keys and coerce values to Fraction.
    Duplicate keys that collide after canonicalization are summed.
    """
    out: Dict[Key, Fraction] = {}
    for k, v in raw.items():
        try:
            g, n, a = k  # type: ignore[misc]
        except Exception as e:
            raise ValueError(f"Invalid key shape: {k!r}") from e
        if not isinstance(g, int) or not isinstance(n, int):
            raise ValueError(f"(g,n) must be ints; got {k!r}")
        if not isinstance(a, tuple) or not all(isinstance(x, int) and x >= 0 for x in a):
            raise ValueError(f"alpha must be a tuple of nonnegative ints; got {k!r}")

        key = (g, n, canonical(a))
        val = v if isinstance(v, Fraction) else Fraction(v)
        out[key] = out.get(key, Fraction(0)) + val
    return out


# ---------------------------------- I/O layer ----------------------------------

def load_rx(path: str | os.PathLike[str]) -> Dict[Key, Fraction]:
    """Load R_X table from `path` if it exists; else return seeds. Always normalized."""
    p = Path(path)
    if not p.exists():
        return seed_rx()
    with p.open("rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        return normalize_rx(obj)
    raise ValueError(f"Unsupported cache format in {p}: expected dict, got {type(obj).__name__}")


def _fsync_dir_safe(dir_path: Path) -> None:
    """
    Best-effort directory fsync.
    - On POSIX, use os.O_DIRECTORY when available.
    - On Windows or filesystems without it, silently skip.
    """
    try:
        flag = getattr(os, "O_DIRECTORY", None)
        if flag is None:
            return  # not supported (e.g., Windows)
        fd = os.open(dir_path, flag)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except Exception:
        # Best-effort only; ignore on platforms/filesystems that don't support it.
        pass


from collections import OrderedDict
import pickle
import tempfile
import os
from pathlib import Path
from typing import Mapping
from fractions import Fraction

def dump_rx_atomic(path: str | os.PathLike[str], rx: Mapping[Key, Fraction | int]) -> None:
    """
    Atomically write `rx` (plain dict) to `path`, with retries for Windows file locks.
    Entries are sorted by (degree, g, n, α) to make output stable and readable.

    Steps:
      1) normalize and sort data
      2) write to temp file in same dir, flush + fsync
      3) atomic replace (retry a few times on PermissionError)
      4) best-effort fsync of containing directory
    """
    def degree_key(item: tuple[Key, Fraction]) -> tuple[int, int, int, tuple[int, ...]]:
        g, n, alpha = item[0]
        return (3 * g - 3 + n, g, n, alpha)

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Normalize and sort the data
    raw_data = dict(rx)
    data = normalize_rx(raw_data)
    data_sorted = OrderedDict(sorted(data.items(), key=degree_key))

    # 1) Write temp file
    with tempfile.NamedTemporaryFile(
        mode="wb", delete=False, dir=p.parent, prefix=p.name + ".", suffix=".tmp"
    ) as tf:
        tmp_name = tf.name
        try:
            pickle.dump(data_sorted, tf, protocol=pickle.HIGHEST_PROTOCOL)
            tf.flush()
            os.fsync(tf.fileno())
        except Exception:
            try:
                tf.close()
            finally:
                try:
                    os.unlink(tmp_name)
                except Exception:
                    pass
            raise

    # 2) Atomic replace with small retry loop (helps on OneDrive/AV lock)
    try:
        _atomic_replace_with_retry(tmp_name, p)
    finally:
        if Path(tmp_name).exists():
            try:
                os.unlink(tmp_name)
            except Exception:
                pass

    # 3) Best-effort dir fsync (POSIX only)
    _fsync_dir_safe(p.parent)



def _atomic_replace_with_retry(src_tmp: str | os.PathLike[str], dst: str | os.PathLike[str], retries: int = 6) -> None:
    """
    os.replace with exponential backoff on PermissionError (WinError 32 etc.).
    Delays: 5ms, 10ms, 20ms, 40ms, 80ms, 160ms.
    """
    delay = 0.005
    last_err: Exception | None = None
    for _ in range(max(1, retries)):
        try:
            os.replace(src_tmp, dst)
            return
        except PermissionError as e:
            last_err = e
            time.sleep(delay)
            delay *= 2
        except Exception as e:
            # Non-permission errors: don't spin
            last_err = e
            break
    assert last_err is not None
    raise last_err
