import math
import numpy as np
from collections import defaultdict

from polypy.expression import matrix, variable, hstack, vstack, summation
from polypy.expression import Scalar

from polypy.generator import preprint
from polypy.generator_eigen import EigenGenerator

from polypy.function import function, Function

from polypy.Index import Range

from polypy.main import Compiler

# from polypy.nlp import NLP, Inequality


unique_names = defaultdict(int)
def _get_unique_name(**kwargs):
    basename = kwargs.get('basename', 'tmp')
    name = kwargs.get('name', None)
    if name:
        return name
    unique_names[basename] = unique_names[basename] + 1
    return basename + str(unique_names[basename])


# Trig functions
def sin(expr):
    try:
        return expr.__sin__()
    except AttributeError:
        return np.sin(expr)

def cos(expr):
    try:
        return expr.__cos__()
    except AttributeError:
        return np.cos(expr)
