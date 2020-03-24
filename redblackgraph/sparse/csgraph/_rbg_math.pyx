import warnings

import numpy as np
cimport numpy as np

include 'parameters.pxi'
include '_rbg_math.pxi'

def py_MSB(x):
    return MSB(x)

def py_avos_sum(x, y):
    return avos_sum(x, y)

def py_avos_product(x, y):
    return avos_product(x, y)
