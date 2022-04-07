from dataclasses import dataclass
import ast
import inspect
import functools
import textwrap as tw
from enum import Enum
import copy
import numpy as np
from typing import Callable, Any
import os
import sys, subprocess
from collections import OrderedDict

# TODOs
# - generate lambas for internal functions
# - deal with constness of args, etc
# - improve bracket handling
# - add annotation for assign so that the user can specify desired type and/or size when it's not obvious
# - if/for
# - list -> array (?)


#
# generate_ functions return strings
# make_ functions return Expression-like objects
#

class NUMTYPE(Enum):

    # Value determines the promotion order. 
    # Real can become a DIFF, but not the other way around
    DIFF = 1  
    REAL = 2
    INT = 3

    def __str__(self):
        cpp_types = {
            NUMTYPE.DIFF: 'diff_t',
            NUMTYPE.REAL: 'scalar_t',
            NUMTYPE.INT: 'int'
        }
        return cpp_types[self]

    @staticmethod
    def get_type(object):
        """Return the NUMTYPE of the given object"""
        type_map = {
            type(1.1): NUMTYPE.REAL,
            type(1): NUMTYPE.INT
        }
        return type_map[type(object)]

    @staticmethod
    def lowest_common_type(*args):
        """Finds the lowest-common denomenator of 
           types of args [Expr]

        Returns dtype : Lowest type that all args can be promoted to"""
        return min(arg.dtype for arg in args)

    def __lt__(self, other):
        return self.value < other.value




class TranspliationError(Exception):
    """Something that's fine in python, but that we can't transpile to C++"""
    pass

class Indenter:
    """Class that can indent / outdent as you write"""
    def __init__(self):
        self._stack = [""]

    def write(self, s):
        """Append the line s to the output, indenting every line"""
        self._stack[-1] += s

    def writeln(self, s):
        """Append the line s to the output, indenting every line"""
        self._stack[-1] += s + "\n"

    def __iadd__(self, s):
        self.writeln(s)
        return self

    def indent(self):
        self._stack.append("")

    def dedent(self):
        if len(self._stack) == 1:
            return
        self.write(tw.indent(self._stack.pop(), '  '))

    def _flatten(self):
        """Return a flattened copy"""
        s = copy.deepcopy(self._stack)
        while len(s) > 1:
            p = s.pop()
            s[-1] += tw.indent(p, '  ')
        return s[0]

    def __str__(self):
        return self._flatten().rstrip()

    def __repr__(self):
        return str(self)


class Context:
    """Variables that are known in the current scope"""
    def __init__(self, context = None):
        self.variables = {}
        self.functions = {}

        if context is not None:
            # Append the current context to this one
            for key, item in context.variables.items():
                self.variables[key] = item
            for key, item in context.functions.items():
                self.functions[key] = item

    def register_variable(self, name, var):
        self.variables[name] = var

    def register_function(self, name, var):
        self.functions[name] = var

    def is_known(self, val):
        """Returns true if the name val is in this context"""
        return val in self.variables.keys()

    def __str__(self):
        s  = "Variables in context: " + ",".join(self.variables.keys()) + "\n"
        s += "Functions in context: " + ",".join(self.functions.keys())
        return s


class RecursiveVisitor(ast.NodeVisitor):

    def __init__(self):
        self._context = [Context()]  # Stack to store the current context

    @property
    def context(self):
        """Return current context"""
        return self._context[-1]

    def new_context(self):
        """Add a new context and return it"""
        self._context.append(Context(self.context))
        return self.context

    def pop_context(self):
        """Remove current context and return previous one"""
        return self._context.pop()

    def recursive(func):
        """ decorator to make visitor work recursive """
        def wrapper(self,node):
            func(self,node)
            for child in ast.iter_child_nodes(node):
                self.visit(child)
        return wrapper

    def visit_Assign(self, node):
        assert len(node.targets) == 1, "Cannot handle multiple assignment"

        # Assign in python is responsible both for declaring new variables
        # as well as copying them. Here we check if a variable exists, and
        # if it doesn't, then we declare a variable of the appropriate type.
        # If it does, then we confirm that it's the right size to copy
        # the given expression, and otherwise raise a transiplation error
        # because even though this is valid python code, it can't be done
        # in C++

        value = self.visit(node.value)
        target = node.targets[0]

        # If the variable isn't known, then we can create it
        if isinstance(target, ast.Name) and not self.context.is_known(target.id):
            self.context.register_variable(target.id, Matrix(value.shape, target.id, dtype=value.dtype))
            target = self.visit(target)  # The visitor will work, since it's now been registered
            return Expr(value.generate_declaration(target, initialize=True), dtype=value.dtype)

        # Check that the type of the target and value are the same
        target = self.visit(target)
        if target.dtype != value.dtype:
            raise TranspliationError(f"Reuse of variable names in C++ is not allowed if the type changes ({target.value})")
        if list(target.shape) != list(value.shape):
            op = ast.unparse(node)
            raise TranspliationError(f"Could not broadcast input array from shape {value.shape} into shape {target.shape} in operation {op}")
            # if not(target.ndim == 2 and value.ndim == 1 and target.shape[0] == value.shape[0]):
        return Expr(value.generate_assignment(target), dtype=target.dtype)

    def visit_BinOp(self, node):
        """ visit a BinOp node and visits it recursively"""
        left = self.visit(node.left)
        right = self.visit(node.right)
        dtype = NUMTYPE.lowest_common_type(left, right)
        left.cast(dtype)
        right.cast(dtype)
        return self.binop(node.op, left, right)

    @functools.singledispatchmethod
    def binop(self, op, left, right):
        raise NotImplementedError(f"Unknown binary operation {type(op)}")

    def broadcast(self, op, left, right):
        left = left.squeeze()
        right = right.squeeze()

        if left.ndim > 0 and right.ndim == 0:
            return Expr(f"(({left}).array() {op} ({right})).matrix()", left.shape, dtype=left.dtype)
        if right.ndim > 0 and left.ndim == 0:
            return Expr(f"(({left}) {op} ({right}).array()).matrix()", right.shape, dtype=left.dtype)
        if right.ndim == right.ndim and right.shape == left.shape:
            return Expr(f"({left}) {op} ({right})", left.shape, dtype=left.dtype)
        raise TranspliationError(f"Wrong sizes for the operation {left}{op}{right}")
    
    @binop.register(ast.Add)
    def _(self, op, left, right):
        return self.broadcast("+", left, right)

    @binop.register(ast.Sub)
    def _(self, op, left, right):
        return self.broadcast("-", left, right)

    @binop.register(ast.Mult)
    def _(self, op, left, right):
        return self.broadcast("*", left, right)

    @binop.register(ast.MatMult)
    def _(self, op, left, right):
        # Check that sizes are compatible
        if right.ndim == 1:
            shape = (left.shape[0], )
        if right.ndim == 2:
            shape = (left.shape[0], right.shape[1])
        if right.ndim == 0:
            raise TranspliationError(f"matmul: Input {right} does not have enough dimensions")
        return Expr(f"({left}) * ({right})", shape)

    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        return self.unaryop(node.op, self.visit(node.operand))

    @functools.singledispatchmethod
    def unaryop(self, op, operand):
        raise NotImplementedError(f"Unknown unary operation {type(op)}")

    @unaryop.register(ast.UAdd)
    def _(self, op, operand):
        return operand

    @unaryop.register(ast.USub)
    def _(self, op, operand):
        e = copy.copy(operand)
        try:
            e.value = -e.value
        except TypeError:
            e.value = "-" + str(e.value)
        return e

    # @unaryop.register(ast.Not)
    # def _(self, op, operand):

    # @unaryop.register(ast.Invert) # Logical inversion
    # def _(self, op, operand):


    def visit_ClassDef(self, node):
        print("ClassDef")
        print(ast.dump(node, indent=2))

    def visit_Attribute(self, node):
        print("Attribute")
        print(ast.dump(node, indent=2))

    def visit_Name(self, node):
        if self.context.is_known(node.id):
            return self.context.variables[node.id]

        raise(NameError(f"Unknown variable {node.id}"))

    def visit_Return(self, node):
        return Expr(f"return {self.visit(node.value)};")

    def visit_FunctionDef(self,node):
        # Create a new context for this function
        self.new_context()

        returns = eval(ast.unparse(node.returns))
        args = self.visit(node.args)

        function = Indenter()
        args_str = [f"{arg.generate_type(reference=True, const=True)} {str(arg)}" for arg in args]
        function += f"template<typename {NUMTYPE.REAL}=double, typename {NUMTYPE.DIFF}={NUMTYPE.REAL}>"
        function += f"{returns.generate_type(reference=False, const=False)} {node.name}({', '.join(args_str)})"
        function += "{"
        function.indent()
        for b in node.body:
            function += str(self.visit(b))
        function.dedent()
        function += "}"

        self.pop_context()
        # print("TODO: Add the declared function output to the parent context")
        return str(function)

    def visit_arguments(self, node):
        """Arguments of a function defintiion"""
        assert len(node.posonlyargs) == 0, "Cannot handle position-only arguments"
        assert len(node.kwonlyargs) == 0, "Cannot handle keyword-only arguments"
        assert node.vararg == None, "Cannot handle variable-length arguments"
        assert node.kwarg == None, "Cannot handle keyword arguments"
        return [self.visit(arg) for arg in node.args]

    def visit_arg(self, node):
        arg = eval(ast.unparse(node.annotation)) 
        arg.value = node.arg
        self.context.register_variable(node.arg, arg)
        return arg

    def visit_Module(self,node):
        """ visit a Module node and the visits recursively"""
        return [self.visit(child) for child in node.body]

    def visit_Call(self, node):
        # Pull the function name including its object if required
        funcname = ast.unparse(node.func)
        args = [self.visit(arg) for arg in node.args]
        return self.context.functions[funcname](*args)

    def visit_List(self, node):
        return List((self.visit(e) for e in node.elts))

    def visit_Tuple(self, node):
        return List((self.visit(e) for e in node.elts))

    def visit_Constant(self, node):
        return Expr(value=node.value, dtype=NUMTYPE.get_type(node.value))

    def visit_Subscript(self, node):
        target = self.visit(node.value)
        slice = self.visit(node.slice)

        # slice is an Expr or list of Expr's per dimension
        # Convert to a list of length ndim
        if not isinstance(slice, List): 
            slice = List((slice, ))

        slice_str = "(" + ", ".join(map(str, slice)) + ")"

        def slice_to_size(target_size, slice):
            """Use the slice_info to compute the shape of the resulting matrix"""
            if target_size == Expr.DYNAMIC or str(slice) == 'all':
                return target_size

            try:
                info = slice.slice_info
                if not all(isinstance(s, int) for s in info):
                    return Expr.DYNAMIC

                return len(range(target_size)[info[0]:info[1]:info[2]])

            except AttributeError:
                # This is a constant - just a single value
                return 1

        shape = [slice_to_size(t, s) for t, s in zip(target.shape, slice)]

        # Follow the numpy convention of loosing a dimension if we slice to a column or row
        return Expr(str(target.value) + slice_str, shape, dtype=target.dtype).squeeze()

    def visit_Slice(self, node):
        """Convert the slice field to an Eigen seq statement
        
        Returns scalar expression whose value is the Eigen seq statement
        for this slice.

        An auxillary attribute slice_info is set containing (lower, upper, shape)
        information numerically
        """

        if node.lower is None and node.upper is None:
            e = Expr(value="all")
            e.slice_info = (0, 1000000, 1)
            return e

        def index_to_eigen(node, default, default_val):
            """Convert a single index to eigen"""
            if node is None:
                return default, default_val
            val = self.visit(node).value
            if isinstance(val, int):
                if val < 0:
                    return f"last-fix<{-val}>", val
                return f"fix<{val}>", val
            return val, Expr.DYNAMIC

        lower, lower_shape = index_to_eigen(node.lower, 'fix<0>', 0)
        upper, upper_shape = index_to_eigen(node.upper, 'last', 1000000)  # Value larger than any matrix
        incr, incr_shape = index_to_eigen(node.step, 'fix<1>', 1)

        e = Expr(value=f"seq({lower},{upper},{incr})") 
        e.slice_info = (lower_shape, upper_shape, incr_shape)
        return e

    def generic_visit(self,node):
        print(f"type(node) = {type(node)}")
        raise NotImplementedError(f"Unknown node type {type(node)}")


class staticproperty(staticmethod):
    """Decorator to make a static method a property"""
    def __get__(self, *_):         
        return self.__func__()

@dataclass
class Expr:
    """An expression of the form op(*args)

    We copy the numpy array interface

    ndim : number of dimensions (0 for a scalar)
    dtype : data type
    """

    value: str = ""  # The evaluation of this expression
    shape: tuple = ()  # tuple of dimension sizes
                       # () == scalar value
                       # (n, ) == 1d vector or list
                       # (n, m) == 2d matrix
                       #
                       # If size is DYNAMIC, then it's a dynamic-sized matrix
    dtype: NUMTYPE = NUMTYPE.REAL

    def __len__(self):
        return self.shape[0]

    @property
    def ndim(self):
        return len(self.shape)

    @staticproperty
    def DYNAMIC():
        """Dyanmic-sized matrix"""
        return "Dynamic"

    def __str__(self):
        return str(self.value)

    def squeeze(self):
        """Return a copy of the expression with singleton dimensions removed"""
        e = copy.deepcopy(self)
        e.shape = list(filter(lambda a: a != 1, self.shape))
        return e

    def cast(self, dtype):
        """Cast this expression to the given type"""
        if self.dtype == dtype:
            return self
        if self.ndim == 0:
            self.value = f"static_cast<{str(dtype)}>({str(self.value)})"
        if self.ndim > 0:
            self.value = f"{str(self.value)}.template cast<{str(dtype)}>()"
        self.dtype = dtype
        return self

    # TODO: change this to a @singledispatchmethod so that we can handle list initialization
    def generate_declaration(self, target, initialize=False) -> str:
        """Returns a string declaring the target as the type of this expression

        If initialize == True, then we copy the value of this expression to target.
        """

        s = f"{self.generate_type()} {target.value}"

        if initialize:
            try:
                return s + str(self.initialize_data) + ";"
            except AttributeError:
                pass

            if isinstance(self.value, List):  # Do initializer_list initialization
                return s + str(self.value) + ";"

            if self.ndim == target.ndim:
                return s + f" = {str(self.value)};"

            if self.ndim == 0 and target.ndim > 0:  # Broadcast
                return s + f" = {self.generate_type()}::Constant({str(self.value)});"

            assert TranspliationError("Invalid dimensions in assignment")
        return s + ";"


    def generate_assignment(self, target) -> str:
        """Returns a string assigning the value of this expression to the target

        Assumption: Declaration has already been done.
        """

        # print(f"self.shape = {self.shape}, target.shape = {target.shape}, target.value = {target.value}, self.value = {self.value}")

        if self.ndim == target.ndim:
            return f"{target.value} = {str(copy.deepcopy(self).cast(target.dtype))};"

        if self.ndim == 0 and target.ndim > 0:  # Broadcast
            e = copy.deepcopy(self)
            e.value = f"{self.generate_type()}::Constant({self.value})"
            return f"{target.value} = {str(e.cast(target.dtype))};"

        assert Exception("Invalid dimensions in assignment")


    def generate_type(self, reference=False, const=False) -> str:
        """Return a reference type to this argument"""
        const_str = "const " if const else ""
        ref_str = "&" if reference else ""

        if self.ndim == 0:  # Scalar
            return f"{const_str}{str(self.dtype)}{ref_str}"

        if self.ndim == 1:
            if reference:
                return f"{const_str}Ref<{const_str}Vector<{str(self.dtype)}, {len(self)}>>&"
            else:
                return f"{const_str}Vector<{self.dtype}, {len(self)}>"

        if self.ndim == 2:
            if reference:
                return f"{const_str}Ref<{const_str}Matrix<{str(self.dtype)}, {self.shape[0]}, {self.shape[1]}>>&"
            else:
                return f"{const_str}Matrix<{str(self.dtype)}, {self.shape[0]}, {self.shape[1]}>"

    def generate_as_array(self) -> str:
        """Convert the matrix expression to an array"""
        return f"({str(self)}).array()"



class Matrix(Expr):
    def __init__(self, shape: tuple, name: str="", dtype=NUMTYPE.REAL):
        self.shape = shape
        self.value = name
        self.dtype = dtype

    # def generate_assign(self, value):
    #     """Return code to assign value to self"""
    #     return f"{self} << {value};"

    # def generate_declaration(self):
    #     """Return a declaration to store this vector"""
    #     return self.generate_type() + f" {str(self)};"


class Vector(Matrix):
    def __init__(self, length: int, name: str="", dtype=NUMTYPE.REAL):
        self.shape = (length, )
        self.value = name
        self.dtype = dtype


class List:
    """List-like object"""

    def __init__(self, data, dtype=NUMTYPE.REAL):
        self.data = list(data)
        self.dtype = dtype

    def __str__(self):
        """Return a CSV initializer for this list"""
        return "{" + ",".join(map(str, self.data)) + "}"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def to_list(self):
        """Convert this List of Expr's to a python list

        This is not recursive.
        """
        return [d.value for d in self.data]
        

    # def generate_type(self) -> str:
    #     """Return a declaration to store this list as an array"""
    #     return f"std::array<{self.dtype}, {len(self.data)}>"

    # def generate_assign(self, target) -> str:
    #     """Return code to assign this list to an object named target"""
    #     return f"{self.dtype} {target.name}[{len(self.data)}] = {{{','.join(map(str, self.data))}}};"


# @dataclass
# class FunctionInfo:
#     name: str
#     make_call: Callable[[Expr], str]


def unary_array_factory(unary_func: str) -> tuple[str, Callable[[...], Expr]]:
    """Function taking one argument that operates elementwise on a matrix"""
    def unary_array_call_generator(*args):
        assert len(args) == 1, f"Unary array function {unary_func} only take one argument"
        arg = args[0]
        if arg.ndim == 0:
            return Expr(f"{unary_func}({str(arg)})")
        return Expr(f"{unary_func}({arg.generate_as_array()}).matrix()", arg.shape)

    return unary_func, unary_array_call_generator

def flatten(t):
    """Flatten a list of lists"""
    flat = []
    for e in t:
        if isinstance(e, list):
            flat.extend(flatten(e))
        else:
            flat.append(e)
    return flat


def numpy_expr(*args) -> Expr:
    """Return a numpy expression from the given arguments

    Argument can either be array-like, or a scalar
    """
    data = args[0]  # List of elements
    try:
        shape = (len(data), len(data[0]))
    except TypeError:
        try:
            shape = (len(data), )
        except TypeError:
            shape = ()
    except IndexError:
        try:
            shape = (len(data), )
        except TypeError:
            shape = ()

    e = Expr(value=data, shape=shape)
    e.value = e.generate_type() + "(" + str(data) + ")"
    e.initialize_data = data
    return e

def numpy_full(shape, value) -> Expr:
    """Produce a matrix of shape shape filled with value value"""
    e = Expr(shape=shape.to_list())
    return Expr(e.generate_type() + f"::Constant({value})", shape.to_list())


class Transpiler:
    def __init__(self):
        self.visitor = RecursiveVisitor()

        # Add standard functions
        self.visitor.context.register_function(*unary_array_factory('sin'))
        self.visitor.context.register_function(*unary_array_factory('cos'))
        self.visitor.context.register_function(*unary_array_factory('tan'))

        self.visitor.context.register_function('np.array', numpy_expr)
        self.visitor.context.register_function('np.full', numpy_full)

    def generate_function(self, func) -> str:
        tree = ast.parse(tw.dedent(inspect.getsource(func)))
        return transpiler.visitor.visit(tree)[0]

    def generate_pybind(self, func, filename):
        """Produce C++ code with pybind linkage"""
        funcname = func.__name__
        os.system(f'rm -f example.cpp')
        with open(filename, 'w') as f:
            f.write(tw.dedent(f"""\
            #include <pybind11/embed.h>
            #include <pybind11/eigen.h>

            #include <Eigen/Dense>
            using namespace Eigen;

            template<typename {NUMTYPE.DIFF}, typename {NUMTYPE.REAL}={NUMTYPE.DIFF}>\n"""))
            f.write(self.generate_function(func))
            f.write(tw.dedent(f"""\n
            PYBIND11_MODULE(example, m) {{
                m.doc() = "Blah blah blah";
                m.def("{funcname}", &{funcname}<double>, "{func.__doc__}");
            }}"""))


if __name__ == "__main__":

    @dataclass
    class MyClass:
        x: Matrix((3,4), dtype=NUMTYPE.REAL)

        def __init__(self, test):
            pass


    # @template("scalar_t")
    def sys(x: Matrix((2,1), dtype=NUMTYPE.DIFF), u: Vector(1, dtype=NUMTYPE.DIFF)) -> Vector(2, dtype=NUMTYPE.DIFF):
        p = pi*3
        A = np.array([[2,2],[3,4]]) * p
        B = A[0,:]
        f = A[:,1] + x
        q = np.array([5,6])
        q = B @ u
        t = u[0] * (sin(A) @ x + B @ u + 3)
        return t

    def sys(x: Matrix((2,1), dtype=NUMTYPE.DIFF), u: Vector(1, dtype=NUMTYPE.DIFF)) -> Vector(2, dtype=NUMTYPE.DIFF):
        A = np.array([[2,2],[3,4]])
        B = A[:,0]
        return x[0] * A @ x + B @ u

    @sparse_jacobian
    def constraints(x: Matrix((2,N), dtype=NUMTYPE.REAL), u: Matrix((1,N-1), dtype=NUMTYPE.REAL)):
        out = []
        for i in range(N-1):
            out.append(A@x[i,:] + B@u[i,:] - 3)
        return out.jacobian

    transpiler = Transpiler()
    transpiler.visitor.context.register_variable("pi", Expr(value=3.14159265359))
    print(transpiler.generate_function(sys))
    # print(transpiler.generate_function(MyClass))

    # transpiler.generate_pybind(sys, 'example.cpp')    
    # out = subprocess.run('make example', shell=True)
    # print(f"RESULT OF COMPILE : {out}")
