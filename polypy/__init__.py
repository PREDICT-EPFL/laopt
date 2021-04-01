# TODO: Figure out scheme to allow python-ordering of variables to create arrow-structures, etc
# TODO: More advanced variables allowing selection of part of the variable, etc
# TODO: Simple python-defined functions (integrators, affine functions, equality)
# TODO: Sparse matrices
# TODO: Pass in VariableSets
# TODO: Detect when two arguments of a op are dependent, and compute jacobian correctly!x
# TODO: Create "virtual" function in python, which can then be treated as normal functions in C++
#       i.e., e = f(x), q(e,e,e). This prevent re-computation of e.

from polypy.poly import NLP
from polypy.poly import Index
from polypy.expression import Variable, hstack
from polypy.expression import Expression, Identity, ConstScalar, Matrix, Scalar, ConstMatrix
from polypy.function import Function
from polypy.generator import preprint, Generator
from polypy.nlp import NLP

from collections import defaultdict

unique_names = defaultdict(int)
def _get_unique_name(basename=None):
    if basename is None:
        basename = "tmp"
    unique_names[basename] = unique_names[basename] + 1
    return basename + str(unique_names[basename])
