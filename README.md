# Weil–Petersson Volumes and $\psi$-Class Intersection Numbers via Mirzakhani Recursion


Mirzakhani-style recursion for intersection numbers with exact rational arithmetic and a simple on-disk cache.

## Install (editable)

```bash
pip install -e .
```

## Quickstart (single notebook)

Open **`notebooks/Intersection_Numbers_Tutorial.ipynb`**. It:
- Verifies the core API.
- Builds the normalized table `R_X(g,n,alpha)` via `iterate(...)`.
- Shows how to convert to intersection numbers `⟨τ_{d1}⋯τ_{dn}⟩_g` using `x_from_rx(...)`.
- Provides helper utilities to query arbitrary `(g, n, alpha)` tuples.

## Minimal API

```python
import wp

# Fill cache up to a degree bound and return a dict { (g,n,alpha): Fraction }
rx = wp.iterate(dmax=6, cache_path="data/wp_rx.pkl", max_workers=None, fill_missing=True)

# Convert normalized value to the actual intersection number
from fractions import Fraction
g, n, alpha = 0, 4, (1,0,0,0)
X = wp.x_from_rx(g, n, alpha, rx[(g,n,tuple(sorted(alpha, reverse=True)))])
assert X == Fraction(1,1)  # ⟨τ_1 τ_0^3⟩_0 = 1
```

## Tests

```bash
python -m tests.test_layout_and_compute
python -m tests.test_normalization
```
(Or use `pytest` if available.)

## Notes

- Package name: **`wp`** (importable); project name: `wp` (PyPI-style).
- Cache file defaults to `data/wp_rx.pkl` and is safe to delete or relocate.
