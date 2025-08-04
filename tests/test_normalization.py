from fractions import Fraction
from wp.normalization import rx_from_x, x_from_rx, double_factorial_odd

def test_roundtrip():
    g,n,alpha = 0,4,(1,0,0,0)
    X = Fraction(1,1)  # with l=0 in toy genus0 model
    RX = rx_from_x(g,n,alpha,X)
    X_back = x_from_rx(g,n,alpha,RX)
    assert X_back == X

def test_double_fact():
    assert double_factorial_odd(0) == 1
    assert double_factorial_odd(1) == 3
    assert double_factorial_odd(2) == 15
