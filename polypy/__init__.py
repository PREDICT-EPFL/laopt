# TODO: Figure out scheme to allow python-ordering of variables to create arrow-structures, etc
# TODO: More advanced variables allowing selection of part of the variable, etc
# TODO: Simple python-defined functions (integrators, affine functions, equality)
# TODO: Sparse matrices
# TODO: Pass in VariableSets
# TODO: Detect when two arguments of a op are dependent, and compute jacobian correctly!x
# TODO: Create "virtual" function in python, which can then be treated as normal functions in C++
#       i.e., e = f(x), q(e,e,e). This prevent re-computation of e.

import math
import numpy as np
from collections import defaultdict

from polypy.expression import matrix, variable, hstack, vstack
from polypy.expression import Scalar

from polypy.generator_eigen import EigenGenerator

from polypy.function import function, Function

# from polypy.poly import NLP
from polypy.Index import Range
# from polypy.expression import variable, hstack, vstack, summation
# from polypy.expression import Expression, Identity, ConstScalar, Matrix, Scalar, ConstMatrix
# from polypy.function import Function, Jacobian, Hessian
# from polypy.generator import preprint, Generator
# from polypy.nlp import NLP, Inequality
# from polypy.generator_eigen import Eigen


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
