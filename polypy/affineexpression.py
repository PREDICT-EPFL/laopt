from polypy.Index import Index
from string import ascii_letters, digits
from polypy.expression import Expression, Identity, ConstantScalar, isScalar
from collections import defaultdict
from copy import copy


def validate_name(name):
    if set(name).difference(ascii_letters + digits + "_") \
       or name[0] not in ascii_letters + "_":
        raise ValueError(f"""Invalid name: {name}
            Only letters, numbers and underscores allowed.
            First character must be a letter or underscore.""")


def convert(op):
    """Convert a Variable into an AffineExpression"""

    def newOp(self, other):
        # if isinstance(other, Variable):
        #     other = AffineExpression({other: Identity(len(other))})
        return op(self, other)
    return newOp


class VariableSet:
    def __init__(self, name, var_type, num_vars):
        validate_name(name)
        self.name = name
        self.var_type = var_type
        self.num_vars = num_vars

    def __getitem__(self, key):
        return Variable(None, None, self, key)

    def __str__(self):
        return self.name

    @property
    def len(self):
        return self.var_type.len

    # def __repr__(self):
    #     cols = f"{self.cols}" if self.cols > 1 else ""
    #     return f"{str(self)}[{str(self.var_type)}*{cols}]"


class VarType:
    def __init__(self, name, len):
        validate_name(name)
        self.name = name
        self.len = len

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"{self.name}[{self.len}]"

    def __len__(self):
        return self.len


class AffineExpression:
    """Expression that is affine in Variables

    expr = sum_i A(i) * x(i) + offset

    where x(i) is a Variable and A(i) is an Expression
    """

    __array_priority__ = 10  # numpy + Expression => calls Expression radd

    def __init__(self, expr=None):
        self.expr = defaultdict(lambda: ConstantScalar(0), expr)

        # Test that sizes are correct
        for x, A in self.expr.items():
            assert isScalar(A) or len(x) == 1 or len(x) == A.shape[1], ValueError("Size error in Affine Expression")
        len(self)  # Test sizes

    def __len__(self):
        # Size of the output vector of the expression
        length = max([len(A) for x, A in self.expr.items()])
        assert all(len(A) == 1 or len(A) == length for x, A in self.expr.items()),\
               ValueError("Ill-formed affine expression. Summation of vectors of different sizes.")
        return length

    def __str__(self):
        rep = [f"({str(A)})" + "\u2022" + str(x) for x, A in self.expr.items()]
        return " + ".join(rep)

    @property
    def isConstant(self):
        # Returns True if this is a constant expression, and False otherwise
        for x, A in self.expr.items():
            if x != One() and not A.isZero:
                return False
        return True

    def __pos__(self):
        return self

    def __neg__(self):
        return AffineExpression({x: -A for x, A in self.expr.items()})

    def __abs__(self):
        return AffineExpression({x: abs(A) for x, A in self.expr.items()})

    def _binary_op(self, other, op):
        # Addition / subtraction
        expr = copy(self.expr)
        if hasattr(other, 'expr'):
            for x, A in other.expr.items():
                expr[x] = eval(f"expr[x] {op} A")
        else:
            expr[One()] = eval(f"expr[One()] {op} other")
        return AffineExpression(expr)

    def __add__(self, other):
        return self._binary_op(other, '+')

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._binary_op(other, '-')

    def __rsub__(self, other):
        return (-self)._binary_op(other, '+')


    def __mul__(self, other):
        # v * (sum Ai @ xi) => sum (diag(v) @ Ai) @ xi
        expr = copy(self.expr)

        if isinstance(other, Variable):
            assert other.isConstant, ValueError(f"Multiplication would result in nonlinear expression {self} * {other}")
            other = Variable.expr[One()]
        assert isScalar(other) or len(self) == 1 or len(other) == len(self),\
               ValueError(f"Incompatible sizes in elementwise multiplication {len(self)} and {len(other)}")

        for x, A in expr.items():
            if len(other) == 1:
                expr[x] = other * expr[x]
            else:
                expr[x] = other.diag() * expr[x]
        return AffineExpression(expr)

    @convert
    def __rmul__(self, other):
        return self.__mul__(other)
        # return mulExpression(other, self)

    @convert
    def __matmul__(self, other):
        return matmulExpression(self, other)

    @convert
    def __rmatmul__(self, other):
        return matmulExpression(other, self)

    def __floordiv__(self, other):
        """Implements integer division using the // operator."""
        raise NotImplementedError

    def __div__(self, other):
        """Implements division using the / operator."""
        raise NotImplementedError

    def __truediv__(self, other):
        """Implements true division. Note that this only works when from __future__ import division is in effect."""
        raise NotImplementedError

    def __rfloordiv__(self, other):
        """Implements reflected integer division using the // operator."""
        raise NotImplementedError

    def __rdiv__(self, other):
        """Implements reflected division using the / operator."""
        raise NotImplementedError

    def __rtruediv__(self, other):
        """Implements reflected true division. Note that this only works when from __future__ import division is in effect."""
        raise NotImplementedError    

    def to_eigen(self, p=None):
        # Produce eigen code to evaluate this expression
        # 
        # Returns a string that assigns the output of this expression to the variable out
        # 
        rep = [A.to_eigen(p) + " * " + x.to_eigen(p) for x, A in self.expr.items()]
        return " + ".join(rep)



class Variable(AffineExpression):
    # A vector variable, or an index into a VectorSet

    def __init__(self, name, var_type, var_set=None, ind=None):
        if name is None:
            # We're taking a column from a VarSet
            self.name = var_set.name
            self.var_type = var_set.var_type
            self.var_set = var_set
            self.ind = ind
        else:
            # We're defining a new variable
            self.name = name
            self.var_type = var_type
            self.var_set = None
            self.ind = None

        super().__init__({self: Identity(len(self))})

    def __str__(self):
        val = f"{self.name}"
        if self.ind is not None:
            return f"{val}[{str(self.ind)}]"
        return val

    def __repr__(self):
        return str(self)

    def __len__(self):
        return self.var_type.len

    def __eq__(self, other):
        if (isinstance(other, Variable)):
            return self.name == other.name and self.var_type.name == other.var_type.name\
                and self.var_set == other.var_set and self.ind == other.ind
        return False

    def __hash__(self):
        return (hash(self.name) ^
                hash(self.var_type.name) ^
                hash(self.ind))

    @property
    def shape(self):
        return (len(self), 1)

    def to_eigen(self, p=None):
        return str(self)


class One(Variable):
    # Special variable representing a constant "1"
    def __init__(self):
        super().__init__("One", VarType("One_t", 1))
