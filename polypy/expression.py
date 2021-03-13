import numbers
import numpy as np
from functools import reduce
from string import ascii_letters, digits

# TODO Error : Shape of sliced objects is wrong


def validate_name(name):
    # Validate name as valid C++ name
    if set(name).difference(ascii_letters + digits + "_") \
       or name[0] not in ascii_letters + "_":
        raise ValueError(f"""Invalid name: {name}
            Only letters, numbers and underscores allowed.
            First character must be a letter or underscore.""")



tmp_index = 1  # Used to specify uniquely named temporary variables
def _get_temp_name(name=None):
    if name:
        return name
    else:
        global tmp_index
        name = f"tmp{tmp_index}"
        tmp_index += 1
        return name


def isScalar(arg):
    """Returns True is arg is a scalar"""
    if isinstance(arg, numbers.Number):
        return True
    if isinstance(arg, Scalar):
        return True
    if isinstance(arg, Expression) and arg.shape == (1,1):
        return True
    return False


def convert(op):
    """Convert a numpy array or scalar into a ConstMatrix"""

    def newOp(self, other):
        if isinstance(other, np.ndarray):
            other = ConstMatrix(other)
        if isinstance(other, numbers.Number):
            other = ConstantScalar(other)
        return op(self, other)
    return newOp


class Expression:
    """General nonlinear matrix-valued expression

        This just records an evaluation tree, and re-produces it in Eigen.
    """

    # TODO Expressions to add : trig expressions, transpose, all Eigen expressions...

    __array_priority__ = 10  # numpy + Expression => calls Expression radd

    def __init__(self, *args):
        self.args = args

    def __str__(self):
        args = ", ".join([str(a) for a in self.args])
        return f"{self.op}({args})"

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        # Implements slicing operations
        return sliceExpression(self, key=key)

    def __pos__(self):
        return posExpression(self)

    def __neg__(self):
        return negExpression(self)

    def __abs__(self):
        return absExpression(self)

    def diag(self):
        return diagExpression(self)

    @convert
    def __add__(self, other):
        if not isinstance(other, Expression):
            return other.__radd__(self)
        return addExpression(self, other)

    @convert
    def __radd__(self, other):
        return addExpression(other, self)

    @convert
    def __sub__(self, other):
        if not isinstance(other, Expression):
            return other.__rsub__(self)
        return subExpression(self, other)

    @convert
    def __rsub__(self, other):
        return subExpression(other, self)

    @convert
    def __mul__(self, other):
        if not isinstance(other, Expression):
            return other.__rmul__(self)
        return mulExpression(self, other)

    @convert
    def __rmul__(self, other):
        return mulExpression(other, self)

    @convert
    def __matmul__(self, other):
        if not isinstance(other, Expression):
            return other.__rmatmul__(self)
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

    def _generate(self, generator, p=None):
        arg_eval = [eval("arg." + generator + "(p)") for arg in self.args]
        for i, arg in enumerate(self.args):
            if self.priority != -1 and self.args[i].priority > self.priority:
                arg_eval[i] = "(" + arg_eval[i] + ")"
        return arg_eval

    def to_python(self, p=None):
        """Return Python code to evaluate this expression"""
        return self._generate_python(*(self._generate("to_python", p)))

    def to_eigen(self, p=None):
        """Produce Eigen code to evaluate this expression"""
        # return self._generate_eigen(*(self._generate("to_eigen", p)))
        arg_eval = [arg.to_eigen(p) for arg in self.args]
        for i, arg in enumerate(self.args):
            if self.priority != -1 and self.args[i].priority > self.priority:
                arg_eval[i] = "(" + arg_eval[i] + ")"
        # return arg_eval
        return self._generate_eigen(*arg_eval)


    @property
    def isZero(self):
        # Return True is expression is zero
        return False
    


# Priorioties for order of operations. We have to do the highest priority things first.
#  -1: abs, functions (never need brackets)
#  0: id, slice
#  1: *, @
#  2: -
#  3. +
#  4: pos, neg

# Implement arithmatic operations
class UnaryExpression(Expression):
    def __init__(self, *args):
        super().__init__(*args)
        self.priority = 4
        self.shape = args[0].shape

    def _generate_python(self, *args):
        return f"{self.python_op}{args[0]}"

    def _generate_eigen(self, *args):
        return f"{self.python_op}{args[0]}"

class BinaryExpression(Expression):
    def __init__(self, *args):
        super().__init__(*args)

    def _generate_python(self, *args):
        return f"{args[0]} {self.python_op} {args[1]}"

    def _generate_eigen(self, *args):
        return f"{args[0]} {self.python_op} {args[1]}"

class sliceExpression(UnaryExpression):
    def __init__(self, *args, key):
        super().__init__(*args)
        self.op = "slice"
        self.python_op = ""
        if isinstance(key, slice):
            key = [key, ]
        self.key = key
        self.priority = 0

        self.shape = list(self.shape)
        self.shape[0] = sliceExpression._get_len(self.key[0], self.args[0].shape[0])
        if len(self.key) > 1:
            self.shape[1] = sliceExpression._get_len(self.key[1], self.args[0].shape[1])
        self.shape = tuple(self.shape)


    @staticmethod
    def _get_len(slice, length):
        # Compute the length of the slice 
        return len(range(*slice.indices(length)))


    @staticmethod
    def _slice_to_python(key):
        rep = ""
        if isinstance(key, slice):
            rep += str(key.start) if key.start else ""
            rep += ":"
            rep += str(key.stop) if key.stop else ""
            rep += ":" + str(key.step) if key.step else ""
        else:
            rep = str(key)
        return rep

    def _generate_python(self, *args):
        rep = ", ".join(sliceExpression._slice_to_python(k) for k in self.key)
        return f"{args[0]}[{rep}]"

    @staticmethod
    def _slice_to_eigen(key):
        rep = ""
        if isinstance(key, slice):
            rep += str(key.start) if key.start else ""
            rep += ":"
            rep += str(key.stop) if key.stop else ""
            rep += ":" + str(key.step) if key.step else ""
        else:
            rep = str(key)
        return rep

    @staticmethod
    def _slice_to_offset(key, length):
        # Convert a slice object key to a (len, offset)
        # length is the size of the object being sliced
        assert key.step == None, NotADirectoryError("Cannot have slices with steps until Eigen 4")

        start = key.start if key.start else 0
        stop = key.stop if key.stop else length

        if start < 0:
            start = length + start
        if stop < 0:
            stop = length + stop

        return (stop - start, start)

    def _generate_eigen(self, *args):
        if len(self.key) == 1:  # Segment of a vector
            size, offset = sliceExpression._slice_to_offset(self.key[0], self.args[0].shape[0])
            rep = f"segment<{size}>({offset})"
        else:  # Block of a matrix
            x_size, x_offset = sliceExpression._slice_to_offset(self.key[0], self.args[0].shape[0])
            y_size, y_offset = sliceExpression._slice_to_offset(self.key[1], self.args[0].shape[1])
            rep = f"block<{x_size}, {y_size}>({x_offset}, {y_offset})"

        return f"{args[0]}.template {rep}"

class posExpression(UnaryExpression):
    def __init__(self, *args):
        super().__init__(*args)
        self.op = "pos"
        self.python_op = "+"

class negExpression(UnaryExpression):
    def __init__(self, *args):
        super().__init__(*args)
        self.op = "neg"
        self.python_op = "-"

class absExpression(UnaryExpression):
    def __init__(self, *args):
        super().__init__(*args)
        self.op = "abs"
        self.priority = -1

    def _generate_python(self, *args):
        return f"abs({args[0]})"

    def _generate_eigen(self, *args):
        return f"abs({args[0]})"

class diagExpression(UnaryExpression):
    def __init__(self, *args):
        super().__init__(*args)
        self.op = "diag"
        self.priority = -1
        assert args[0].shape[1] == 1, ValueError("Attempt to convert a matrix to a diagonal matrix. Input must be a column vector.")
        self.shape = (len(args[0]), len(args[0]))

    def _generate_python(self, *args):
        return f"diag({args[0]})"

    def _generate_eigen(self, *args):
        return f"({args[0]}).asDiagonal()"

class addExpression(BinaryExpression):
    def __init__(self, *args):
        super().__init__(*args)
        self.op = "add"
        self.shape = args[0].shape if isinstance(args[1], Scalar) else args[1].shape
        assert all(isScalar(arg) or self.shape == arg.shape for arg in args), \
            TypeError(f"Adding matrices of incompatible sizes {args[0]}({args[0].shape}) vs {args[1]}({args[1].shape})")
        self.priority = 3
        self.python_op = "+"

class subExpression(BinaryExpression):
    def __init__(self, *args):
        super().__init__(*args)
        self.op = "sub"
        self.shape = args[0].shape if isinstance(args[1], Scalar) else args[1].shape        
        assert all(isScalar(arg) or self.shape == arg.shape for arg in args), \
            TypeError(f"Subtracting matrices of incompatible sizes {args[0]} vs {args[1]}")
        self.priority = 2
        self.python_op = "-"

class mulExpression(BinaryExpression):
    def __init__(self, *args):
        super().__init__(*args)
        self.op = "mul"
        self.shape = args[0].shape if isinstance(args[1], Scalar) else args[1].shape
        assert all(isScalar(arg) or self.shape == arg.shape for arg in args), \
            TypeError(f"Elementwise multiplication of matrices of incompatible sizes {args[0]} vs {args[1]}")
        self.priority = 1
        self.python_op = "*"

    def _generate_eigen(self, *args):
        return f"({args[0]}.array() * {args[1]}.array()).matrix()"

class matmulExpression(BinaryExpression):
    def __init__(self, *args):
        super().__init__(*args)
        self.op = "matmul"
        if isinstance(args[0], Scalar):
            self.shape = args[1].shape
        elif isinstance(args[1], Scalar):
            self.shape = args[0].shape
        else:
            self.shape = (args[0].shape[0], args[1].shape[1])
        assert any(isScalar(arg) for arg in args) or \
            args[0].shape[1] == args[1].shape[0],\
            TypeError(f"Multipying matrices of incompatible sizes {args[0]} vs {args[1]}")
        self.priority = 1
        self.python_op = "@"

    def _generate_eigen(self, *args):
        return f"{args[0]} * {args[1]}"


class AtomicExpression(Expression):
    def __init__(self):
        super().__init__()
        self.priority = 0

    def _generate_python(self, *args):
        return str(self)

    def _generate_eigen(self, *args):
        return str(self)


class Matrix(AtomicExpression):
    """Symbolic matrix. Elements can be changed at runtime"""

    def __init__(self, shape, name=None):
        super().__init__()
        self.name = _get_temp_name(name)
        self.shape = shape
        self.op = "id"

    # def __repr__(self):
    #     return f"{self.name}{self.shape}"

    def __str__(self):
        return self.name

    def __len__(self):
        return self.shape[0]


class ConstMatrix(Matrix):
    """Matrix whose elements are known at generation time and are fixed"""

    def __init__(self, M, name=None):
        if isinstance(M, numbers.Number):  # Convert to numpy array
            if not name:
                name = str(M)
            M = np.array([M])
        if M.ndim == 1:
            M = M.reshape(len(M), 1)  # Convert to column vector
        super().__init__(M.shape, name)
        self.M = M
        self.shape = M.shape

    @property
    def isZero(self):
        return np.all((self.M == 0))

    # @convert
    # def __matmul__(self, other):
    #     """Matrix multiplication"""
    #     assert isScalar(other) or self.shape[1] == other.shape[0], \
    #         ValueError(f"Incorrect matrix sizes for matrix multiplication: {self}({self.shape}) vs {other}({other.shape})")

    #     if isinstance(other, ConstMatrix):
    #         return ConstMatrix(self.M @ other.M)
    #     if isinstance(other, numbers.Number):
    #         return ConstMatrix(self.M * other)

    #     return super().__matmul__(other)

    # @convert
    # def __rmatmul__(self, other):
    #     print("ConstMatrix.rmatmul")
    #     """Reflected matrix multiplication"""
    #     assert isScalar(other) or self.shape[0] == other.shape[1], \
    #         ValueError(f"Incorrect matrix sizes for matrix multiplication: {self}({self.shape}) vs {other}({other.shape})")

    #     if isinstance(other, ConstMatrix):
    #         return ConstMatrix(other.M @ self.M)
    #     if isinstance(other, numbers.Number):
    #         return ConstMatrix(other * self.M)

    #     return super().__rmatmul__(other)


class Identity(ConstMatrix):
    """An identity matrix"""

    def __init__(self, n):
        super(Identity, self).__init__(np.identity(n), f"I_{n}")
        self.n = n
        self.shape = (n,n)

    # @convert
    # def __matmul__(self, other):
    #     print("Identity.matmul")
    #     """Matrix multiplication"""
    #     assert isScalar(other) or self.shape[1] == other.shape[0], \
    #         ValueError(f"Incorrect matrix sizes for matrix multiplication: {self}({self.shape}) vs {other}({other.shape})")

    #     if isScalar(other):  # scalar * Identity != scalar
    #         return super().__matmul__(other)

    #     return other

    # @convert
    # def __rmatmul__(self, other):
    #     print("Identity.rmatmul")
    #     """Reflected matrix multiplication"""
    #     assert isScalar(other) or self.shape[0] == other.shape[1], \
    #         ValueError(f"Incorrect matrix sizes for matrix multiplication: {self}({self.shape}) vs {other}({other.shape})")

    #     if isScalar(other):  # scalar * Identity != scalar
    #         return super().__rmatmul__(other)

    #     return other


class Scalar(AtomicExpression):
    """A scalar value. Treated as a variable, and changable at runtime"""

    def __init__(self, value, name):
        self.value = value
        self.op = 'id'
        self.name = name
        self.args = []
        self.shape = (1, 1)
        self.priority = 0

    def __str__(self):
        return self.name

    def to_python(self, p=None):
        return str(self)

    def to_eigen(self, p=None):
        return str(self)


class ConstantScalar(Scalar):
    """A scalar value. Treated as a constant."""

    def __init__(self, value):
        super().__init__(value, str(value))

    @property
    def isZero(self):
        return self.value == 0

    def to_python(self, p=None):
        return str(self.value)

    def to_eigen(self, p=None):
        return str(self.value)


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


class Variable(AtomicExpression):
    # A vector variable, or an index into a VectorSet

    def __init__(self, name, var_type, var_set=None, ind=None):
        super().__init__()
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
