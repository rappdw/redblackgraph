import builtins
import numpy as np
import redblackgraph as rb

def print(R):
    if isinstance(R, np.ndarray) or isinstance(R, np.matrix) or isinstance(R, rb.array) or isinstance(R, rb.matrix):
        builtins.print(R)
    else:
        print(np.array(R))
