from dataclasses import dataclass
from typing import Sequence


@dataclass
class Ordering:
    A: Sequence[Sequence[int]]
    label_permutation: Sequence[int]