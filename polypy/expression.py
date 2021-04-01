from enum import Enum, auto
import numbers
import numpy as np
from polypy.generator import validate_name, preprint, PrePrint
import polypy
import copy
import itertools


# TODO Errors
# - Shape of sliced objects is wrong
# - Call a function with some args being parameters and some differentiable vars
# - Deal with inf in variable bounds properly
# - Remove the _is_equal process and replace with "Freeze"

# TODO Features / improvements
# - functions with multiple outputs (?)
# - add generation code and wrappers to call from python\
# - add variable sets back in
# - drop all variables from the generation that aren't used some constraint
# - basic nonlinear functions (sin, cos, division, etc)
# - collect dependencies into different print buffers so they can be split out in order
# - add a __le__ and __ge__ to generate inequalties
# - allow setting of initial dual points

# Optimizations
# - look for repeated expressions and buffer them into temporary variables
# - reuse temporary variables and allocate at the top of the function
# - switch to a delegate formulation
# - keep a set of generated constants to re-use, rather than generating more (e.g., for lower bounds from numpy, etc)

# Checks
# - Confirm that functions only use the variables in their argument list
# - Confirm that substitutions are valid sizes (perhaps shift to a post-construction validation?)

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
    """Returns True if arg is a scalar"""
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
            other = ConstScalar(other)
        return op(self, other)
    return newOp


class Expression:
    """General nonlinear matrix-valued expression"""

    # TODO Expressions to add : trig expressions, transpose, all Eigen expressions...

    __array_priority__ = 10  # numpy + Expression => calls Expression radd

    # State of the expression. During construction, the == and <= operators produce 
    # constraints. Once frozen, equality converts to a comparison operator, and <= 
    # raises an error.
    class State(Enum):
        Construction = auto()  # Building an expression
        Frozen = auto()  # Expression finalized

    def __init__(self, *args):
        self.args = tuple(args)
        self._state = Expression.State.Construction

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

    # def _generate(self, generator, p=None):
    #     arg_eval = [eval("arg." + generator + "(p)") for arg in self.args]
    #     for i, arg in enumerate(self.args):
    #         if self.priority != -1 and self.args[i].priority > self.priority:
    #             arg_eval[i] = "(" + arg_eval[i] + ")"
    #     return arg_eval

    # def to_python(self, p=None):
    #     """Return Python code to evaluate this expression"""
    #     return self._generate_python(*(self._generate("to_python", p)))

    def generate(self, p):
        """Produce Eigen code to evaluate this expression"""
        # print(f"generate : {self}")
        arg_eval = [arg.generate(p) for arg in self.args]
        # print("arg_eval = " + f"{arg_eval}")
        for i, arg in enumerate(self.args):
            try:
                if self.priority != -1 and self.args[i].priority > self.priority:
                    arg_eval[i] = "(" + arg_eval[i] + ")"
            except AttributeError:
                print("ATTRIBUTE ERROR")
                pass
        # print("returning : " + self._generate_eigen(p, *arg_eval))
        # print(f"_generate_eigen for {self}")
        return self._generate_eigen(p, *arg_eval)

    def generate_array(self):
        # Convert this expression to an eigen array
        return ".array()"

    # @property
    # def isZero(self):
    #     # Return True is expression is zero
    #     return False

    def is_equal(self, other):
        # Test recursively if two expressions are equal
        if type(self) != type(other):
            return False
        if len(self.args) != len(other.args):
            return False
        if not self._is_equal(other):  # Test any local data
            return False
        for s, o in zip(self.args, other.args):
            if isinstance(s, Expression):
                if s.is_equal(o) == False:
                    return False
            else:
                if s != o:
                    return False
        return True

    def _is_equal(self, other):
        # Specialized in children if there are additional member elements to test for equality
        return True

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, newState):
        # Change the state recursively
        self._state = newState
        for arg in self.args:
            arg.state = newState

    def get_by_property(self, property):
        # Return a set of nodes for which the property(node) returns true
        return set(self._get_by_property(property))

    def _get_by_property(self, property):
        # Return a list of nodes for which the property(node) returns true
        rec = sum([arg._get_by_property(property) for arg in self.args], [])
        try:
            if property(self):
                rec.append(self)
        except AttributeError:
            # print("ATTRIBUTE ERROR")
            pass
        return rec

    def count_nodes(self):
        return 1 + sum([arg.count_nodes() for arg in self.args])

    # @property
    # def parameters(self):
    #     # Return a set of variable data in this expression
    #     return self.get_by_property(lambda n: n.isParameter)

    def __hash__(self):
        # return hash(self.name)
        return hash(id(self))

    @convert
    def __eq__(self, other):
        # Return a constraint that imposes equality between the two arguments
        # Note: This constraint will evaluate to True if equality can be determined
        #       at compile time
        if self.state == Expression.State.Frozen:
            return self.is_equal(other)
        elif self.state == Expression.State.Construction:
            return polypy.nlp.Equality(self, other)
        raise ValueError("Testing equality while in unknown state.")

    def substitute(self, vars, subs):
        # Return a new expression where variables in the list vars are replaced by the expressions in the list subs        

        if not isinstance(self, Variable):
            ret = copy.copy(self)
            ret.args = [arg.substitute(vars, subs) for arg in self.args]
            return ret

        # print(f"Searching for {self} in vars")
        for var, sub in zip(vars, subs):
            if var.is_equal(self):
                return sub
        return self


def hstack(*args):
    """Return an hstack expression"""
    return HStack(*args)


class HStack(Expression):
    def __init__(self, *args):
        """Stack a list of expressions horizontally"""
        super().__init__(*args)
        len(self)  # Check the sizes

    def __str__(self):
        args = [str(x) for x in self.args]
        return f"hstack({' '.join(args)})"

    def _generate_eigen(self, p, *args):
        tmp = _get_temp_name()
        shape = self.shape
        p(f"Matrix<{p.numberType}, {shape[0]}, {shape[1]}> {tmp};")
        offset = 0
        for arg, strArg in zip(self.args, args):
            p(f"{tmp}.template block<{arg.shape[0]}, {arg.shape[1]}>(0, {offset}) = {strArg};")
            offset += arg.shape[1]

        return tmp

    def __len__(self):
        l = len(self.args[0])
        for arg in self.args:
            assert l == len(arg), "All arguments in an hstack must have the same height"
        return l

    @property
    def shape(self):
        return (len(self), sum(arg.shape[1] for arg in self.args))


def vstack(*args):
    """Return an vstack expression"""
    return VStack(*args)


class VStack(Expression):
    def __init__(self, *args):
        """Stack a list of expressions vertically"""
        super().__init__(*args)

        # Check the shape
        w = self.args[0].shape[1]
        for arg in self.args:
            assert w == arg.shape[1], "All arguments in a vstack must have the same width"

    def __str__(self):
        args = [str(x) for x in self.args]
        return f"vstack({' '.join(args)})"

    def _generate_eigen(self, p, *args):
        tmp = _get_temp_name()
        shape = self.shape
        p(f"Matrix<{p.numberType}, {shape[0]}, {shape[1]}> {tmp};")
        offset = 0
        for arg, strArg in zip(self.args, args):
            p(f"{tmp}.template block<{arg.shape[0]}, {arg.shape[1]}>({offset}, 0) = {strArg};")
            offset += arg.shape[0]
        return tmp

    def __len__(self):
        return sum(arg.shape[0] for arg in self.args)

    @property
    def shape(self):
        return (len(self), self.args[0].shape[1])


# Priorioties for order of operations. We have to do the highest priority things first.
#  -1: abs, functions (never need brackets)
#  0: id, slice
#  1: *, @
#  2: -
#  3. +
#  4: pos, neg

class functionExpression(Expression):
    # Evaluation of a l'opt function
    def __init__(self, function, *args):
        super().__init__(*args)
        self.priority = 0
        self.function = function
        self.shape = (len(function), 1)

    def __repr__(self):
        return str(self)

    def __str__(self):
        args = [str(x) for x in self.args]
        return f"{self.function.name}({', '.join(args)})"

    def _generate_eigen(self, p, *args):
        p.add_dependency(self.function, 'Function')  # Register this function for generation
        out = p.get(self)  # Test if this function has been generated before

        if not out:
            out = _get_temp_name()
            p(f"Matrix<{p.number_type}, {len(self.function)}, 1> {out};")
            p(f"{self.function.wrapped_name}<{p.number_type}>({', '.join(args)}, {out});")
            p.add(self, out)

        return out

    def _is_equal(self, other):
        if id(self.function) != id(other.function):
            return False
        return True

    # def get_by_property_function(self, property):
    #     # Return a set of nodes for which the property(node) returns true in the expression of this function
    #     return self.function.expression.get_by_property(property)
    #     # return super()._get_by_property(property) + self.function.expression._get_by_property(property)

    #     # rec = super().get_by_property(property) 
    #     # rec.update(self.function.expression.get_by_property(property))
    #     # return rec

    # def __hash__(self):
    #     return hash((self.function, self.args))


class UnaryExpression(Expression):
    def __init__(self, *args):
        super().__init__(*args)
        self.priority = 4
        self.shape = args[0].shape

    def _generate_python(self, *args):
        return f"{self.python_op}({args[0]})"

    def _generate_eigen(self, p, *args):
        return f"{self.python_op}({args[0]})"

class BinaryExpression(Expression):
    def __init__(self, *args):
        super().__init__(*args)

    def _generate_python(self, *args):
        return f"{args[0]} {self.python_op} {args[1]}"

    def _generate_eigen(self, p, *args):
        return f"{args[0]} {self.python_op} {args[1]}"

class sliceExpression(UnaryExpression):
    def __init__(self, *args, key):
        super().__init__(*args)
        self.op = "slice"
        self.python_op = ""
        if isinstance(key, slice):
            key = (key, )
        self.key = key
        self.priority = 0

        # print(f"key = {key}")
        # print(f"type(key) = {type(key)}")

        self.shape = list(self.shape)
        if isinstance(key, int):
            self.shape = (1, 1)
        if isinstance(key, tuple):
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

    def _generate_python(self, p, *args):
        rep = ", ".join(sliceExpression._slice_to_python(k) for k in self.key)
        return f"{args[0]}[{rep}]"

    @staticmethod
    def _slice_generate(key):
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

    def _generate_eigen(self, p, *args):
        if isinstance(self.key, int):
            rep = f"segment<1>({self.key})"
        if isinstance(self.key, tuple):    
            if len(self.key) == 1:  # Segment of a vector
                size, offset = sliceExpression._slice_to_offset(self.key[0], self.args[0].shape[0])
                rep = f"segment<{size}>({offset})"
            else:  # Block of a matrix
                x_size, x_offset = sliceExpression._slice_to_offset(self.key[0], self.args[0].shape[0])
                y_size, y_offset = sliceExpression._slice_to_offset(self.key[1], self.args[0].shape[1])
                rep = f"block<{x_size}, {y_size}>({x_offset}, {y_offset})"

        return f"{args[0]}.template {rep}"

    def _is_equal(self, other):
        if self.key != other.key:
            return False
        return True


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

    def _generate_eigen(self, p, *args):
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

    def _generate_eigen(self, p, *args):
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

    def _generate_eigen(self, p, *args):
        if isScalar(self.args[0]) or isScalar(self.args[1]):
            return f"{args[0]} * {args[1]}"
        else:
            return f"({args[0]}{self.args[0].generate_array()} * {args[1]}{self.args[1].generate_array()}).matrix()"

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

    def _generate_eigen(self, p, *args):
        return f"{args[0]} * {args[1]}"


class AtomicExpression(Expression):
    def __init__(self):
        super().__init__()
        self.priority = 0
        self.op = "id"

    def _generate_python(self, *args):
        return str(self)

    def _generate_eigen(self, p, *args):
        return str(self)

    def generate_declaration(self, p):
        # Return string to declare this variable / constant
        raise NotImplementedError("_generate_declaration must be implemented in all AtomicExpression's")


class Matrix(AtomicExpression):
    """Symbolic matrix. Elements can be changed at runtime"""

    isParameter = True

    def __init__(self, shape, name=None, initial=None):
        super().__init__()
        self.name = _get_temp_name(name)
        self.shape = shape
        self.op = "id"
        if initial is not None:
            self.initial = initial  # Initial value of the matrix
        else:
            self.initial = np.zeros(shape)

    def __repr__(self):
        return f"{self.name}"

    def __str__(self):
        return self.name

    def __len__(self):
        return self.shape[0]

    def __hash__(self):
        return (hash((self.name, self.shape)))

    def generate_declaration(self, p):
        # Return string to declare this variable / constant
        p(f"Matrix<scalar_t, {self.shape[0]}, {self.shape[1]}> {self.name};")

    def generate_initialization(self, p):
        # Initialize the variable if initial is specified
        print(f"initializing {self}")
        M = self.initial
        if np.all(M == np.ravel(M)[0]):  # All values are the same
            p(f"{self.name}.array() = {M[0][0]};")
        else:
            ret = f"{self.name} << "
            rows = []
            for row in M:
                rows.append(", ".join(str(x) for x in row))
            ret += ", ".join(rows) + ";"
            p(ret)

    def _is_equal(self, other):
        if self.name != other.name:
            return False
        return np.array_equiv(self.initial, other.initial)

    def _generate_eigen(self, p, *args):
        p.add_dependency(self, 'VariableMatrix')  # Register this matrix for generation
        return super()._generate_eigen(p, *args)


class ConstMatrix(Matrix):
    """Matrix whose elements are known at generation time and are fixed"""

    isParameter = False
    isConstant = True

    def __init__(self, M, name=None):
        if isinstance(M, numbers.Number):  # Convert to numpy array
            if not name:
                name = str(M)
            M = np.array([M])
        if M.ndim == 1:
            M = M.reshape(len(M), 1)  # Convert to column vector
        super().__init__(M.shape, name=name, initial=M)
        self.M = M
        self.shape = M.shape

    @property
    def isZero(self):
        return np.all((self.M == 0))

    def generate_declaration(self, p):
        # Return string to declare this constant
        ret = f"using M_{self.name} = Matrix<scalar_t, {self.shape[0]}, {self.shape[1]}>;\n"
        ret += f"const M_{self.name} {self.name} = (M_{self.name}() << "
        rows = []
        for row in self.M:
            rows.append(", ".join(str(x) for x in row))
        ret += ", ".join(rows) + ").finished();"
        p(ret)

    def generate_initialization(self, p):
        pass  # No initialization in the constructor - it's in the declaration

    def _is_equal(self, other):
        return np.array_equiv(self.M, other.M)

    def _generate_eigen(self, p, *args):
        M = self.M
        if np.all(M == np.ravel(M)[0]):
            return f"Matrix<{p.number_type}, {M.shape[0]}, {M.shape[1]}>::Constant({M[0][0]})"
        # if self.shape[0] * self.shape[1] > 30:
        #     # Buffer large matrices for re-use
        # print(f"Adding dependency for {self}")
        p.add_dependency(self, 'ConstantMatrix')  # Register this matrix for generation
        return str(self)

    def generate_scalar(self, p, *args):
        """Generate a scalar assignment - x.array() = value"""
        if np.all(self.M == np.ravel(self.M)[0]):
            return str(self.M[0][0])
        raise AttributeError(f"Elements of ConstMatrix {self} are not all the same")


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

    isParameter = False

    def __init__(self, n):
        super(Identity, self).__init__(np.identity(n), f"I_{n}")
        self.n = n
        self.shape = (n,n)

    def _is_equal(self, other):
        if self.n != other.n:
            return False

        return True

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

    isParameter = True

    def __init__(self, value, name):
        self.value = value
        self.op = 'id'
        self.name = name
        self.args = ()
        self.shape = (1, 1)
        self.priority = 0

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def to_python(self, p=None):
        return str(self)

    # def generate(self, p):
    #     return str(self)

    def generate_array(self):
        # Convert this expression to an eigen array
        return ""

    def __hash__(self):
        return hash((self.name, self.value))

    def generate_declaration(self, p):
        # Return string to declare this variable / constant
        p(f"scalar_t {self.name} = {self.value};")

    def _is_equal(self, other):
        if self.value != other.value:
            return False
        if self.name != other.name:
            return False
        return True

    def _generate_eigen(self, p, *args):
        p.add_dependency(self, 'VariableScalar')  # Register this matrix for generation
        return super()._generate_eigen(p, *args)

    def generate_scalar(self, p, *args):
        """Generate a scalar assignment - x.array() = value"""
        p.add_dependency(self, 'VariableScalar')  # Register this matrix for generation
        return str(self)


class ConstScalar(Scalar):
    """A scalar value. Treated as a constant."""

    isParameter = False
    isConstant = True

    def __init__(self, value):
        super().__init__(value, str(value))

    @property
    def isZero(self):
        return self.value == 0

    # def to_python(self, p=None):
    #     return str(self.value)

    def _generate_eigen(self, p, *args):
        return str(self.value)

    def generate_scalar(self, p, *args):
        """Generate a scalar assignment - x.array() = value"""
        return str(self.value)

    # def generate_declaration(self, p):
    #     # print("==========================================")
    #     # Return string to declare this variable / constant
    #     p(f"const scalar_t {self.name} = {self.value};")

    def _is_equal(self, other):
        if self.value != other.value:
            return False
        return True

# class VarType:
#     def __init__(self, name, len):
#         validate_name(name)
#         self.name = name
#         self.len = len

#     def __str__(self):
#         return self.name

#     def __repr__(self):
#         return f"{self.name}[{self.len}]"

#     def __len__(self):
#         return self.len


# class VariableSet(list):
#     def __init__(self, name, length, num_vars):
#         validate_name(name)
#         self.name = name
#         self._len = length
#         self.num_vars = num_vars

#     def __getitem__(self, key):
#         return Variable(None, None, self, key)

#     def __repr__(self):
#         return str(self)

#     def __str__(self):
#         return self.name

#     def __len__(self):
#         return self._len

#     # def __repr__(self):
#     #     cols = f"{self.cols}" if self.cols > 1 else ""
#     #     return f"{str(self)}[{str(self.var_type)}*{cols}]"


class Variable(AtomicExpression):
    # A vector variable, or an index into a VectorSet

    # def __init__(self, name, length, var_set=None, ind=None):
    def __init__(self, name, length, **kwargs):
        super().__init__()
        # if name is None:
        #     # We're taking a column from a VarSet
        #     self.name = var_set.name
        #     self._len = len(var_set)
        #     self.var_set = var_set
        #     self.ind = ind
        # else:
        # We're defining a new variable
        self.name = name
        self._len = length
        # self.var_set = None
        # self.ind = None

        self.lb = kwargs.get('lb', -2e20)  # Deal with INF properly...
        self.ub = kwargs.get('ub', 2e20)

        if isinstance(self.lb, numbers.Number):
            self.lb = np.ones((length)) * self.lb
        if isinstance(self.ub, numbers.Number):
            self.ub = np.ones((length)) * self.ub

        assert len(self.lb) == len(self), ValueError(f"Lower bound must be a vector of length {len(self)}")
        assert len(self.ub) == len(self), ValueError(f"Upper bound must be a vector of length {len(self)}")

        if isinstance(self.lb, np.ndarray):
            self.lb = ConstMatrix(self.lb)
        if isinstance(self.ub, np.ndarray):
            self.ub = ConstMatrix(self.ub)

    def __str__(self):
        val = f"{self.name}"
        # if self.ind is not None:
        #     return f"{val}[{str(self.ind)}]"
        return val

    def __repr__(self):
        return str(self)

    def __len__(self):
        return self._len

    # def __eq__(self, other):
    #     if (isinstance(other, Variable)):
    #         return self.name == other.name and len(self) == len(other) and self.ind == other.ind
    #     return False

    # def __hash__(self):
    #     return hash((self.name, self._len, self.ind))

    @property
    def shape(self):
        return (len(self), 1)

    def _generate_eigen(self, p, *args):
        return str(self)

    def _is_equal(self, other):
        # return id(self) == id(other)
        if self.name != other.name:
            return False
        if self._len != other._len:
            return False
        return True

    @property
    def state(self):
        return super(Variable, self).state

    @state.setter
    def state(self, newState):
        # Ensure that any expressions in our bounds have their state changed too
        super(Variable, self.__class__).state.fset(self, newState)
        self.lb.state = newState
        self.ub.state = newState
