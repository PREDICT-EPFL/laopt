from polypy.expression import Variable, Expression, functionExpression
from polypy.generator import preprint, validate_name
import polypy as pp
import types
import numpy as np

#########################################
# Public functions to create function 
#########################################

def function(f):
    """Decorator to convert python function to polypy function

    Used as:
    @pp.function
    def func(x: n, u: m):
        expr = A @ x + B @ u
        return expr

    The sizes of the input variables must be specified in the function annotations.
    """

    # if not f.__annotations__:
        

    # Create polypy input variables of the right size
    args = (pp.variable(key, val) for key, val in f.__annotations__.items())

    # Create a polypy Function object
    try:
        return Function(f.__name__, args, f(*args))
    except TypeError:
        raise TypeError("Annotations must be added to all arguments to specify variable sizes") from None


class Function:

    def __init__(self, name, inputs, expression):
        validate_name(name)
        self.wrapped_name = "_" + name  # Private name used for all internal purposes
        self.name = name  # Public name
        self.inputs = inputs  # List of Variables
        self.output = pp.variable("out", len(expression))  # Variable
        self.expression = expression

        for x in inputs:
            assert isinstance(x, Variable), TypeError("Inputs must be a list of variables")
        # assert isinstance(output, Variable), TypeError("Output must be a variable")
        assert isinstance(expression, Expression), TypeError("Expression must be an expression")

    def __repr__(self):
        return str(self)

    def __str__(self):
        args = [str(x) for x in self.inputs]
        return f"{self.name}({', '.join(args)})"

    def __len__(self):
        return len(self.output)

    def num_inputs(self):
        return sum(len(x) for x in self.inputs)

    def generate_declaration(self, p):
        # Generate an eigen function to evaluate this Function
        # Assumptions: 
        #  - All parameters and constants have already been defined in the containing class
        #  - All functions that this function calls exist

        print(f"GENERATING {self}")

        # Produce function signature
        p(f"template<typename T>")
        p(f"EIGEN_STRONG_INLINE void {self.wrapped_name}("
            + ", ".join(f"cVec<T, {x.shape[0]}>& {x}" for x in self.inputs)
            + f", Vec<T, {self.output.shape[0]}> {self.output}) const noexcept")

        for x in self.inputs:
            assert x.shape[1] == 1, "Can only handle vector inputs"
        assert self.output.shape[1] == 1, "Output must be a vector"
        # with p:
        #     for x in self.inputs:
        #         p(f"const Ref<const Matrix<T, {x.shape[0]}, {x.shape[1]}>>& {x}, ")
        #     p(f"Ref<Matrix<T, {self.output.shape[0]}, {self.output.shape[1]}>> {self.output}) const noexcept")
        # Evaluate expression
        self.expression.state = Expression.State.Frozen
        with p.function(number_type="T"):
            p(f"{self.output} = {self.expression.generate(p)};")

    @property
    def parameters(self):
        # Return a set of variable data that needs to be written out for this function
        return self.expression.get_by_property(lambda n: n.isParameter)

    @property
    def constants(self):
        # Return a set of constant data that needs to be written out for this function
        return self.expression.get_by_property(lambda n: n.isConstant)

    @property
    def constantMatrices(self):
        # Return a set of constant data that needs to be written out for this function
        return self.expression.get_by_property(lambda n: n.isConstant and isinstance(n.M, np.ndarray))

    def _functions(self):
        # Return a set of functions that this function calls (recursively)
        funcs = list(set([f.function for f in self.expression.get_by_property(lambda n: n.function)]))
        return list(set(sum([f._functions() for f in funcs], funcs) + [self, ]))

    @property
    def functions(self):
        # Return a set of functions that this function calls (recursively)
        return set(self._functions())

    def __call__(self, *args):
        # Evaluate this function and return the expression
        return functionExpression(self, *args)

    def __eq__(self, other):
        # Returns true if two functions would evaluate to the same value for all inputs
        if not isinstance(other, Function):
            return False
        if len(self.inputs) != len(other.inputs):
            return False
        for arg1, arg2 in zip(self.inputs, other.inputs):
            if len(arg1) != len(arg2):
                return False

        # Create matching input variables for both functions
        args = [pp.variable("var" + str(i), len(arg)) for i, arg in enumerate(self.inputs)]
        eval_self = self(*args)
        eval_other = other(*args)
        print(f"eval_self = {eval_self}")
        print(f"eval_other = {eval_other}")
        return eval_self.is_equal(eval_other)

    def __hash__(self):
        return hash(id(self))


class Jacobian(Function):
    """Generates jacobian for the function during generation"""

    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)

    def generate_jacobian(self, classname, p=preprint()):
        # Wrapper for Jacobian call
        sizes = [str(len(x)) for x in self.inputs]
        p(f"MAKE_JACOBIAN({self.name}, {classname}<scalar_t>, {len(self.output)}, {', '.join(sizes)});")


class Hessian(Function):
    """Generates hessian for the function during generation"""

    def __init__(self, parent):
        self.__dict__.update(parent.__dict__)

    def generate_hessian(self, classname, p=preprint()):
        # Wrapper for Jacobian call
        sizes = [str(len(x)) for x in self.inputs]
        p(f"MAKE_HESSIAN({classname}, {self.name});")
        p(f"{self.name}Hessian<{', '.join(sizes)}> {self.name};")

    def generate_declaration(self, p):
        # Generate an eigen function to evaluate this Function
        # Assumptions: 
        #  - All parameters and constants have already been defined in the containing class
        #  - All functions that this function calls exist

        # Produce function signature
        p(f"template<typename T>")
        p(f"EIGEN_STRONG_INLINE void {self.wrapped_name}("
            + ", ".join(f"cVec<T, {x.shape[0]}>& {x}" for x in self.inputs)
            + f", T& {self.output}) const noexcept")
        with p:
            for x in self.inputs:
                assert x.shape[1] == 1, "Can only handle vector variables"

            
            # p(f"const Ref<const Matrix<T, {x.shape[0]}, {x.shape[1]}>>& {x}, ")
            # p(f"T& {self.output}) const noexcept")

        # Evaluate expression
        self.expression.state = Expression.State.Frozen
        with p.function(number_type="T", cast_constants=True):
            p(f"{self.output} = ({self.expression.generate(p)})[0];")
