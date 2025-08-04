from .compute import iterate
from .recursion import rx_rec_opt
from .normalization import x_from_rx, rx_from_x, l_degree
from .utils import intersection_number  # if present
__all__ = ["iterate", "rx_rec_opt", "x_from_rx", "rx_from_x", "l_degree",
           "intersection_number"]
