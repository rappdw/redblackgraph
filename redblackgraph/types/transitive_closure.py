from dataclasses import dataclass
import numpy as np

@dataclass
class TransitiveClosure:
    W: np.array
    diameter: int
