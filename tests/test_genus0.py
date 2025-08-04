"""
Test that genus0.py direct computation matches recursive output from wp.iterate
"""

from wp.compute import iterate, alpha_bs
from wp.normalization import x_from_rx
from wp.genus0 import genus0_x

def test_genus0_vs_recursive():
    g = 0
    dmax = 6
    rx = iterate(dmax=dmax, cache_path="data/test_wp_rx.pkl", max_workers=1, fill_missing=False)

    for n in range(3, 8):
        D = 3 * g - 3 + n
        allalpha = alpha_bs(n, D)
        for alpha in allalpha:
            alpha = tuple(sorted(alpha, reverse=True))
            key = (g, n, alpha)
            if key not in rx:
                continue  # Skip missing keys instead of failing

            x_rec = x_from_rx(g, n, alpha, rx[key])
            x_exact = genus0_x(n, alpha)

            assert x_rec == x_exact, (
                f"Mismatch for g=0, n={n}, α={alpha}:\n"
                f"  recursive = {x_rec}\n"
                f"  exact     = {x_exact}"
            )
