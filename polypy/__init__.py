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
# from polypy.generator import Generator
from polypy.expression import VarType, Variable, VariableSet
from polypy.expression import Expression, Identity, ConstantScalar

from contextlib import redirect_stdout

class PrePrint:
    """Overload print by adding a prefix to every line"""
    def __init__(self, pre):
        self.pre = pre

    def __enter__(self):
        self.pre = self.pre + "\t"
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if self.pre is not None:
            self.pre = self.pre[1:]
        return True

    def __call__(self, *args, **kwargs):
        """My custom print() op."""
        args = [self.pre + a for a in args]
        return print(*args, **kwargs)


def preprint(pre=""):
    """Factory to generate a PrePrint object"""
    return PrePrint(pre)

