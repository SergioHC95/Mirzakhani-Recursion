from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
from fractions import Fraction

Alpha = Tuple[int, ...]
Key = Tuple[int, int, Alpha]

@dataclass(frozen=True)
class Job:
    g: int
    n: int
    alpha: Alpha
