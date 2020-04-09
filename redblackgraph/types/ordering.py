from dataclasses import dataclass
from typing import Dict, Sequence


@dataclass
class Ordering:
    A: Sequence[Sequence[int]]
    label_permutation: Sequence[int]
    components: Dict[int, int]
