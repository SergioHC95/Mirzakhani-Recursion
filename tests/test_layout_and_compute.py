import os
from wp.compute import degree, is_stable, ordered_pairs

def test_layout():
    import wp
    assert hasattr(wp, "iterate")

def test_pairs():
    pairs = ordered_pairs(2)
    assert (1,1) in pairs
    assert all(is_stable(g,n) for g,n in pairs)
    assert all(degree(g,n) == 3*g-3+n for g,n in pairs)
