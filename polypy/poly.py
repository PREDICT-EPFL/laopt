from polypy.Index import Index
import numpy as np
from string import ascii_letters, digits
import numbers
from collections import defaultdict
from functools import reduce

def validate_name(name):
    if set(name).difference(ascii_letters + digits + "_") \
        or name[0] not in ascii_letters + "_":
        raise ValueError(f"""Invalid name: {name}
            Only letters, numbers and underscores allowed.
            First character must be a letter or underscore.""")


# A decorator that is used so that children of Expression will have priority in arithmetic operations
def testPriority(op):
    def newOp(self, other):
        if hasattr(other, 'precedence') and hasattr(self, 'precedence'):
            if other.precedence > self.precedence:
                return NotImplemented
        return op(self, other)
    return newOp



# Global variable used to track the index of temporary variables during generation 
tmp_index = 0

class Expression:
    # An expression is made up of:
    #     Operation (id,select,*,-,/,+,Function)
    #     list of args
    #
    # arg must implement
    #  len (size of output)
    #  vars (list of all base Variables)
    #
    # Operation must implement
    #  validate - return true if args are valid, false otherwise
    #  len - size of the output

    precedence = 0

    def __init__(self, op, *args):
        self.op = op
        self.args = args

    def __str__(self):
        args = ", ".join(str(a) for a in self.args)
        return f"{str(self.op)}({args})"

    @property
    def vars(self):
        # Return an ordered list of all Variables involved in this expression
        vars = set()
        for a in self.args:
            try:
                vars = vars.union(a.vars)
            except AttributeError:
                pass
        return sorted(vars, key=str)

    def __len__(self):
        # Size of the output vector
        if self.op == 'id':
            assert len(self.args) == 0, "Identity can't have any arguments"
            return len(self)

        if self.op in ('+', '-', '*'):
            l = len(self.args[0])
            return l

        if self.op in ('@', ):
            l = len(self.args[0])
            return l

        assert type(self.op) is not str, f"Unknown operation {self.op}"

        return len(self.op)

    @property
    def shape(self):
        if self.op == 'id':
            assert len(self.args) == 0, "Identity can't have any arguments"
            raise TypeError("shape should be overwritten in the children. We should never be here.")

        if self.op in ('+', '-', '*'):
            return self.args[0].shape

        if self.op in ('@', ):
            return (self.args[0].shape[0], self.args[1].shape[1])

        assert type(self.op) is not str, f"Unknown operation {self.op}"

        return self.shape

    # Implement basic operations
    @staticmethod
    def _validate_sizes(op, check, self, other):
        assert check, f"Wrong sizes for {op}: {str(self)}({self.shape}) and {str(other)}({other.shape})"

    @testPriority
    def __add__(self, other):
        if isinstance(other, numbers.Number):
            other = _WrappedNumber(other)
        Expression._validate_sizes('addition', len(self) == len(other) or len(other) == 1, self, other)
        return Expression('+', self, other)

    @testPriority
    def __radd__(self, other):
        if isinstance(other, numbers.Number):
            other = _WrappedNumber(other)
        Expression._validate_sizes('addition', len(self) == len(other) or len(other) == 1, self, other)
        return Expression('+', other, self)

    @testPriority
    def __sub__(self, other):
        if isinstance(other, numbers.Number):
            other = _WrappedNumber(other)
        Expression._validate_sizes('subtraction', len(self) == len(other) or len(other) == 1, self, other)
        return Expression('-', self, other)

    @testPriority
    def __rsub__(self, other):
        if isinstance(other, numbers.Number):
            other = _WrappedNumber(other)
        Expression._validate_sizes('subtraction', len(self) == len(other) or len(other) == 1, self, other)
        return Expression('-', other, self)

    @testPriority
    def __matmul__(self, other):
        if isinstance(other, numbers.Number):
            other = _WrappedNumber(other)
        Expression._validate_sizes('multiplcation', self.shape[1] == other.shape[0] or other.shape == (1,1), self, other)
        return Expression('@', self, other)

    @testPriority
    def __rmatmul__(self, other):
        if isinstance(other, numbers.Number):
            other = _WrappedNumber(other)
        Expression._validate_sizes('multiplcation', self.shape[0] == other.shape[1] or other.shape == (1,1), self, other)
        return Expression('@', other, self)

    @testPriority
    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            other = _WrappedNumber(other)
        Expression._validate_sizes('elementwise multiplcation', self.shape == other.shape or other.shape == (1,1), self, other)
        return Expression('*', self, other)

    @testPriority
    def __rmul__(self, other):
        if isinstance(other, numbers.Number):
            other = _WrappedNumber(other)
        Expression._validate_sizes('elementwise multiplcation', self.shape == other.shape or other.shape == (1,1), self, other)
        return Expression('*', other, self)

    def cpp_generate_call(self, p, function_name, gen_jacobian=False):
        """Print C++ function into p that will evaluate this expression"""

        vars = self.vars
        sp = " " * (len(function_name) - 2)
        p(f"EIGEN_STRONG_INLINE void {function_name}(")
        with p:
            for v in vars:
                p(f"const Eigen::Ref<const {v.var_type}<scalar_t>>& {v.cpp_name}, ")
            sep = "," if gen_jacobian else ") noexcept"
            p(f"Eigen::Ref<Eigen::Matrix<scalar_t, {len(self)}, 1>> out{sep}")
            if gen_jacobian:
                for v in vars:
                    sep = ","
                    if v == vars[-1]:
                        sep = ") noexcept"
                    p(f"Eigen::Ref<Eigen::Matrix<scalar_t, {len(self)}, {len(v.var_type)}>> J_{v.cpp_name}{sep}")
        p("{")

        # Generate expression calls recursively in a depth-first fashion
        with p:
            p(f"// {self}")
            p("")
            out_eval, out_jacobian = self._cpp_evaluate(p, gen_jacobian)
            p(f"out = {out_eval};")

            print("===========================================")
            print(out_jacobian)
            print("===========================================")
        p("}")
 
    def _cpp_evaluate(self, p, gen_jacobian=False):
        """Generate code to evaluate this expression recursively
           Evaluate the expression into output
           Returns (out, {x1: Jx1, x2:, Jx2})
           out: string to evaluate this expression
           Jxi: string to evaluate the jacobian wrt xi

           self.vars == (x1, x2, ...) Order is uncertain
        """

        pass
        # # Generate children first
        # print(self)
        # args_eval, jac_eval = zip(*[arg._cpp_evaluate(p) for arg in self.args])
        # print(jac_eval)

        # jac = defaultdict(lambda: "0") # Jacobian is zero by default
        # if self.op == '+':
        #     eval = " + ".join(args_eval)
        #     for x in self.vars:
        #         # Jacobian wrt x_jac
        #         print('*************************************')
        #         print(f"jac_eval = {jac_eval}")
        #         x_jac = [[J for v, J in arg_jac.items() if v == x] for arg_jac in jac_eval]
        #         print(x_jac)

        #         print('*************************************')

        #         x_jac = defaultdict(lambda: "0") # Jacobian is zero by default
        #         for J in jac_eval:
        #             print(f"x = {x}")
        #             print(f"J.keys() = {J.keys()}")
        #             if x in J.keys():
        #                 print("x in J.keys()")
        #                 x_jac.append(J[x])
        #                 print(f"x_jac = {x_jac}")
        #         print("HERE")
        #         print(x)
        #         print(jac_eval)

        #         print(f"x_jac = {x_jac}")
        #         print("--------------------------------")
        #         jac[x] = "+".join(x_jac)
        #         # jac[x] = "+ ".join(J[x] for J in jac_eval if x in J.keys())
        #         print("--------------------------------")


        # elif self.op == '-':
        #     eval = " - ".join(args_eval)
        # elif self.op == '@':
        #     eval = " * ".join(args_eval)
        # elif self.op == '*':
        #     args_eval = [arg if isinstance(a, _WrappedNumber) else arg + ".array()" for arg, a in zip(args_eval, self.args)]
        #     eval = " * ".join(args_eval)
        # elif isinstance(self.op, Function):
        #     args_eval = ", ".join(args_eval)
        #     global tmp_index
        #     tmp = f"tmp{tmp_index}"
        #     tmp_index += 1
        #     p(f"Eigen::Matrix<scalar_t, {len(self.op)}, 1> {tmp};")
        #     p(f"{self.op.name}({args_eval}, {tmp});")
        #     eval = tmp
        # else:
        #     raise ValueError("Should never get here...")

        # return (f"({eval})", jac)

    @property    
    def data(self):
        """Return the set with all Matrix's used in this expression"""
        dat = set()
        for arg in self.args:
            dat = dat.union(arg.data)
        return dat


class FunctionExpression(Expression):
    """Evaluation of a function"""

    precedence = 1

    @property
    def shape(self):
        return (len(self.op), 1)

class Variable(Expression):
    # A vector variable, or an index into a VectorSet

    precedence = 1

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

        # This is also an expression "identity(self)"
        super().__init__('id')
        # self.op = 'id'
        # self.args = (self,)

    def __str__(self):
        val = f"{self.name}"
        if self.ind is not None:
            return f"{val}[{str(self.ind)}]"
        return val

    def __repr__(self):
        return str(self)

    def __len__(self):
        return self.var_type.len

    @property
    def shape(self):
        return (len(self), 1)

    @property
    def num_vars(self):
        return 1

    # def generate(self, var):
    #     # Produce C++ code to evaluate this variable as an offset into var
    #     if self.ind is None:
    #         return f"{self.name}({var})"
    #     else:
    #         return f"{self.name}({var}, {self.ind})"

    @property    
    def cpp_name(self, var=None):
        """Evaluate this variable in C++"""
        if var:
            # Evaluate this variable as an offset into var
            if self.ind is None:
                return f"{self.name}({var})"
            else:
                return f"{self.name}({var}, {self.ind})"
        else:
            # A unique name for this variable
            ind = ""
            if isinstance(self.ind, Index):
                ind = self.ind.cpp_name
            elif self.ind is not None:
                ind = str(self.ind)

            return f"{self.name}{ind}"

    def _cpp_evaluate(self, p, gen_jacobian=False):
        """Return a string evaluating this variable"""
        return (self.cpp_name, {self: ConstMatrix(np.identity(len(self)), f"I_{len(self)}")})

    @property
    def indices(self):
        # Return set of indices for this variable
        if isinstance(self.ind, Index):
            return self.ind.indices
        else:
            return set()

    @property
    def vars(self):
        return {self}

    def __eq__(self, other):
        if (isinstance(other, Variable)):
            return self.name == other.name and self.var_type == other.var_type and \
                   self.var_set == other.var_set and self.ind == other.ind
        return False

    def __hash__(self):
        return (hash(self.name) ^
                hash(self.var_type) ^
                hash(self.ind))

    @property
    def data(self):
        return set()

    

class FConstant(float):
    def __new__(cls, name, num):
        return super(FConstant, cls).__new__(cls, num)

    def __init__(self, name, num):
        validate_name(name)
        self.name = name


class IConstant(int):
    def __new__(cls, name, num):
        return super(IConstant, cls).__new__(cls, num)

    def __init__(self, name, num):
        validate_name(name)
        self.name = name


class Matrix(Expression):
    """Wrapper for a Eigen Matrix that will be generated in the C++ code.
    This matrix is not assumed to be constant, and so is changable at run-time.
    """

    precedence = 2

    def __init__(self, mat, name=None):
        if name:
            validate_name(name)
        else:
            global tmp_index
            name = f"tmp{tmp_index}"
            tmp_index += 1

        self.mat = mat
        self.name = name
        super(Matrix, self).__init__('id')

    def __str__(self):
        return self.name

    def __len__(self):
        return len(self.mat)

    @property
    def shape(self):
        return self.mat.shape

    def _cpp_evaluate(self, p, gen_jacobian=False):
        """Return an expression evaluating this matrix"""
        return (self.name, {})

    @property
    def data(self):
        return {self}

    def __eq__(self, other):
        if (isinstance(other, Matrix)):
            return self.name == other.name and self.mat == other.mat
        return False

    def __hash__(self):
        return hash(self.name)

    @property
    def cpp_name(self):
        return str(self)
    
    # @property
    # def jacobian(self):
    #     if self.shape[1] > 1:
    #         raise RuntimeError("Trying to take the jacobian of a constant matrix... we should not be able to get here")
    #     return Jacobian(self)  # Create a "zero" jacobian


class ConstMatrix(Matrix):

    precedence = 10

    """A matrix that is guaranteed to be constant.
    As a result, we can do calculations involving this matrix at generation time.
    """
    def __add__(self, other):
        if not self.mat.any(): # Matrix is zero
            return other
        if isinstance(other, numbers.Number):
            return ConstMatrix(self.mat + other)
        if isinstance(other, ConstMatrix):
            return ConstMatrix(self.mat + other.mat)
        return Expression('+', self, other)

    def __radd__(self, other):
        if not self.mat.any(): # Matrix is zero
            return other
        if isinstance(other, numbers.Number):
            return ConstMatrix(self.mat + other)
        if isinstance(other, ConstMatrix):
            return ConstMatrix(self.mat + other.mat)
        return Expression('+', self, other)

    def __sub__(self, other):
        if not self.mat.any(): # Matrix is zero
            return -other  # TODO: Implement negation operator
        if isinstance(other, numbers.Number):
            return ConstMatrix(self.mat - other)
        if isinstance(other, ConstMatrix):
            return ConstMatrix(self.mat - other.mat)
        return Expression('-', self, other)

    def __rsub__(self, other):
        if not self.mat.any(): # Matrix is zero
            return other
        if isinstance(other, numbers.Number):
            return ConstMatrix(other - self.mat)
        if isinstance(other, ConstMatrix):
            return ConstMatrix(other.mat - self.mat)
        return Expression('-', other, self)

    def __matmul__(self, other):        
        M = self.mat 
        if not M.any():  # Matrix is zero
            return ConstMatrix(np.zeros((M.shape[0], other.shape[1])))
        if (M.shape[0] == M.shape[1]) and (M == np.eye(M.shape[0])).all(): 
            return other  # self is the identity

        if isinstance(other, numbers.Number):
            return other * self
        if isinstance(other, ConstMatrix):
            O = other.mat
            if (O.shape[0] == O.shape[1]) and (O == np.eye(O.shape[0])).all():  # Other is identity
                return self
            return ConstMatrix(self.mat @ other.mat)

        Expression._validate_sizes('multiplcation', self.shape[1] == other.shape[0] or other.shape == (1,1), self, other)
        return Expression('@', self, other)

    def __rmatmul__(self, other):
        M = self.mat
        if not M.any():  # Matrix is zero
            return ConstMatrix(np.zeros((other.shape[0], M.shape[1])))
        M = self.mat 
        if (M.shape[0] == M.shape[1]) and (M == np.eye(M.shape[0])).all():
            return other  # self is the identity

        if isinstance(other, numbers.Number):
            return other * self
        if isinstance(other, ConstMatrix):
            O = other.mat
            if (O.shape[0] == O.shape[1]) and (O == np.eye(O.shape[0])).all():  # Other is identity
                return self
            return ConstMatrix(other.mat @ self.mat)

        Expression._validate_sizes('multiplcation', self.shape[0] == other.shape[1] or other.shape == (1,1), self, other)
        return Expression('@', other, self)

    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            return ConstMatrix(self.mat * other)
        if isinstance(other, ConstMatrix):
            return ConstMatrix(self.mat * other)
        return Expression('*', self, other)

    def __rmul__(self, other):
        if isinstance(other, numbers.Number):
            return ConstMatrix(self.mat * other)
        if isinstance(other, ConstMatrix):
            return ConstMatrix(other * self.mat)
        return Expression('*', other, self)


class _WrappedNumber(Expression):
    """Wraps a scalar in an Expression, so we can handle it normally"""

    def __init__(self, val):
        assert isinstance(val, numbers.Number), "Val must be a scalar number"
        self.val = val
        super(_WrappedNumber, self).__init__('id')

    def __str__(self):
        return str(self.val)

    def __len__(self):
        return 1

    @property
    def shape(self):
        return (1,1)

    def _cpp_evaluate(self, p, gen_jacobian=False):
        return (str(self), {})

    @property
    def data(self):
        return set()

    @property
    def jacobian(self):
        return Jacobian(self)  # Create a "zero" jacobian

    @property
    def cpp_name(self):
        return str(self)
    


class Jacobian:
    """Represents a Jacobian object"""

    def __init__(self, expr):
        """
        expr : expression whose jacobian we want to compute
        """
        self.expr = expr
        self.tmpVars = [] # List of temporary variables required to evaluate this expression
        self.tmpJacobians = [] # List of temporary matrices required to evaluate this jacobian

        # Build Jacobians recursively
        if expr.op == 'id':
            self.Jargs = []
        else:
            self.Jargs = [Jacobian(arg) for arg in expr.args]

        # Generate any temporary vectors and matrices required
        if isinstance(expr.op, Function):
            global tmp_index

            # Temporary variable for the output
            self.outputName = f"tmp{tmp_index}"
            tmp_index += 1
            self.tmpVars.append(f"Eigen::Matrix<scalar_t, {len(self.expr.op)}, 1> {self.outputName};")

            # Temporary matrices for the jacobians
            self.argJacobians = []
            for var in self.expr.op.input_types:
                Jtmp = f"Jtmp{tmp_index}"
                shape = (len(self.expr.op), len(var))
                self.tmpJacobians.append(f"Eigen::Matrix<scalar_t, {shape[0]}, {shape[1]}> {Jtmp};")
                self.argJacobians.append(Matrix(np.ones(shape), Jtmp))  # Create a Matrix placeholder 
                tmp_index += 1

    def cpp_generate_expression(self, p, gen_jacobian=False):
        """Generate C++ statement to compute the expression into the variable 'out'"""
        self._cpp_generate_temporary_vars(p, gen_jacobian)
        eval = self._cpp_generate_expression(p, gen_jacobian)
        p(f"out = {eval};")

    def _cpp_generate_expression(self, p, gen_jacobian=False):
        args_eval = [arg._cpp_generate_expression(p, gen_jacobian) for arg in self.Jargs]
        op = self.expr.op

        if op == "+":
            eval = " + ".join(args_eval)
        elif op == '-':
            eval = " - ".join(args_eval)
        elif op == '@':
            args_eval = ["(" + arg + ")" for arg in args_eval]
            eval = " * ".join(args_eval)
        elif op == '*':
            args_eval = ["(" + arg + ")" for arg in args_eval]
            args_eval = [arg if isinstance(a, _WrappedNumber) else arg + ".array()" for arg, a in zip(args_eval, self.expr.args)]
            eval = " * ".join(args_eval)
        elif isinstance(op, Function):
            args_eval = ["(" + arg + ")" for arg in args_eval]
            args_eval = ", ".join(args_eval)
            J_args = ""
            if gen_jacobian:
                J_args = ", " + ", ".join([J.name for J in self.argJacobians])
            p(f"{op.name}({args_eval}, {self.outputName}{J_args});")
            eval = self.outputName
        elif op == "id":
            eval = self.expr.cpp_name
        else:
            raise TypeError(f"Unknown operation {op}")

        return eval

    def compute_jacobian(self, x):
        """Compute the jacobian of the expression wrt Variable x"""

        # Compute the jacobian wrt x of each of the arguments
        args_eval = [arg.compute_jacobian(x) for arg in self.Jargs]
        op = self.expr.op
        # print("============= HERE ===============")
        # print(str(self.expr))
        # print("args_eval = [" + ", ".join([str(arg) for arg in args_eval]) + "]")

        # eval = ""
        if op == "+":
            eval = reduce(lambda a, b: a + b, args_eval)

        elif op == '-':
            eval = reduce(lambda a, b: a - b, args_eval)

        elif op == '@':
            # We must have two arguments - one a constant expression, and one an expression
            args = self.expr.args
            assert len(args) == 2, TypeError("More than two arguments in a matrix multiplication")
            
            if args[0].vars:
                eval = args_eval[0] @ args[1]
            elif args[1].vars:
                eval = args[0] @ args_eval[1]
            else:
                eval = ConstMatrix(np.zeros((args[0].shape[0], args[1].shape[1])), f"Z_{args[0].shape[0]}_{args[1].shape[1]}")

        elif op == '*':
            # print("HEREHEREHERE")
            print("* SYMBOL")
            # args_eval = [arg if isinstance(a, _WrappedNumber) else arg + ".array()" for arg, a in zip(args_eval, self.expr.args)]
            # eval = " * ".join(args_eval)

        elif isinstance(op, Function):
            eval = reduce(lambda a, b: a + b, [func_jac @ arg for func_jac, arg in zip(self.argJacobians, args_eval)])

        elif op == "id":
            if x == self.expr:
                eval = ConstMatrix(np.identity(len(x)), f"I_{len(x)}")
            else:
                eval = ConstMatrix(np.zeros((len(self.expr), len(x))), f"Z_{len(self.expr)}_{len(x)}")

        else:
            raise TypeError(f"Unknown operation {op}")

        # print(f"eval = {str(eval)}")
        # print("==================================")

        return eval

    def _cpp_generate_temporary_vars(self, p, gen_jacobian=False):
        [p(var) for var in self.tmpVars]
        if gen_jacobian:
            [p(var) for var in self.tmpJacobians]

        for arg in self.Jargs:
            arg._cpp_generate_temporary_vars(p, gen_jacobian)

    def __call__(self, x):
        """
        Compute Jacobian wrt x
        """
        print("Building jacobian of")
        print(expr)

        op = self.expr.op
        if op == "+":
            pass
            # for arg in self.expr.args:
            #     if arg.
        elif op == "-":
            pass
        elif op == "@":
            pass
        elif op == "*":
            pass
        elif isinstance(op, Function):
            pass
        else:
            raise TypeError(f"Unknown operation {op}")

        # if Jargs:
        #     self.Jargs = Jargs
        # else:
        #     self.Jargs = {arg: arg.jacobian for arg in self.expr.args}















class Function:
    def __init__(self, name, size_output, *input_types):
        validate_name(name)
        self.name = name
        self.size_output = size_output  # If None, then this is a scalar-output op
        self.input_names, self.input_types = zip(*input_types)  # List of VarTypes

    def __call__(self, *args):
        return FunctionExpression(self, *args)

    def __str__(self):
        return self.name
        # args = [f"{i}" for i in self.input_types]
        # return f"{self.name}(out, {', '.join(args)})"

    def __repr__(self):
        args = [f"{repr(i)}" for i in self.input_types]
        return f"{self.name}(out[{self.size_output}], {', '.join(args)})"

    def __len__(self):
        return self.size_output


class Constraint:
    """Describe a constraint: f(x) == 0, or lb <= f(x) <= ub

    Contains an evaluation f(x), a name and upper/lower bounds (equal to None if eq)
    """

    def __init__(self, name, evaluation, lb=None, ub=None):
        validate_name(name)
        self.evaluation = evaluation
        self.name = name
        self.lb = lb
        self.ub = ub

    def __str__(self):
        return 'blah'
        # if self.lb is None:
        #     return str(self.evaluation)
        # return f"{str(self.lb)} <= {repr(self.evaluation)} <= {str(self.ub)}"

    def generate(self, var, output="output"):
        # Produce C++ code
        return self.evaluation.generate(var, output)

    def generate_args(self, var):
        # Produce C++ code
        return self.evaluation.generate_args(var)

    @property
    def size_output(self):
        return self.evaluation.size_output

    @property
    def indices(self):
        # Returns the index for this constraint, or None
        return self.evaluation.indices

    @property
    def num_iterations(self):
        # Return the number of iterations done by any index in this constraint
        num_iterations = 1
        if list(self.indices):
            i = list(self.indices)[0]
            num_iterations = len(list(i.rng))
        return num_iterations


class VariableSet:
    def __init__(self, name, var_type, num_vars):
        assert isinstance(num_vars, int), "num_vars must be a strictly positive integer"
        assert num_vars > 1, "num_vars must be a strictly positive integer"
        validate_name(name)
        self.name = name
        self.var_type = var_type
        self.num_vars = num_vars

    def __getitem__(self, key):
        assert type(key) in (int, Index), "Index must be an integer or an Index object"
        if type(key) == int:
            assert (key >= 0 and key < self.num_vars), "Index is not a valid column of the variable"
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



    

class NLP:
    def __init__(self, name="MyNLP"):
        validate_name(name)
        self.name = name
        self.ni = 0
        self.scalar = "double"

        self._data = []  # Constant matrices that are also expressions
        self._constants = []  # Scalar _constants that will be #define'd in c++
        self.vars = []
        self.var_types = []  # List of vector types
        self.functions = []

        self.equalities = []  # Constraints
        self.inequalities = []  # Constraints
        # self.var_bounds = []  # (Variable, lb, ub) or (Variable, lb, ub)

        self.objective = None

    @property
    def nx(self):
        # Compute the number of variables
        return sum(v.var_type.len * v.num_vars for v in self.vars)

    @property
    def ne(self):
        # Compute the number of equality constraints
        return sum(e.size_output * e.num_iterations for e in self.equalities)

    def define_constant(self, name, value, number_type=None):
        """Define a scalar that can be used in python and c++"""
        if number_type is None:
            number_type = type(value)
        if number_type == float:
            c = FConstant(name, value)
        elif number_type == int:
            c = IConstant(name, value)
        else:
            raise ValueError("Unknown number type")

        self._constants.append(c)
        return c

    def data(self, name, mat):
        """Define a constant matrix expression. mat = numpy array"""
        assert type(mat) == np.ndarray, "mat must be a numpy array"
        c = Matrix(mat, name)
        self._data.append(c)
        return c

    def var(self, name, vartype, cols=None):
        if cols == None or cols == 1:
            v = Variable(name, vartype)
        else:
            v = VariableSet(name, vartype, cols)
        self.vars.append(v)
        return v

    def var_type(self, name, rows):
        v = VarType(name, rows)
        self.var_types.append(v)
        return v

    def function(self, name, size_output, *input_types):
        f = Function(name, size_output, *input_types)
        self.functions.append(f)
        return f

    def equality(self, name, expr):
        # Add an equality constraint to the NLP
        assert type(expr) == Expression, "Can only constrain evaluation"
        assert type(name) == str, "Name must be a string"
        assert name not in [e.name for e in self.equalities], f"Duplicate name for constraint ({name})"
        c = Constraint(name, expr)
        self.equalities.append(c)
        return c

    def inequality(self, name, expr, lb, ub):
        # Add an inequality constraint to the NLP lb <= expr <= ub
        assert type(expr) == Expression, "Can only constrain evaluation"
        assert type(name) == str, "Name must be a string"
        c = Constraint(name, expr, lb, ub)
        self.inequalities.append(c)
        return c

    def objective(self, expr):
        assert type(expr) == Expression, "Objective must be an evaluation"
        self.objective = expr

    @property
    def __str__(self):
        str = "NLP"
        str += f"\n\tEqualities:"
        for e in self.equalities:
            str += f"\t\t{str(e)}\n"
        # str += f"\n\tInequalities = {self.inequalities}"
        # str += f"\n\tVariable bounds = {self.var_bounds}"
        return str
