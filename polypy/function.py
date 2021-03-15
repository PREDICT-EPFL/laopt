from polypy.expression import Variable, Expression, functionExpression
from polypy.generator import preprint, validate_name
import types

class Function:

    def __init__(self, name, inputs, output, expression):
        validate_name(name)
        self.name = "_" + name  # Private name used for all internal purposes
        self.wrapped_name = name  # Public name
        self.inputs = inputs  # List of Variables
        self.output = output  # Variable
        self.expression = expression

        for x in inputs:
            assert isinstance(x, Variable), TypeError("Inputs must be a list of variables")
        assert isinstance(output, Variable), TypeError("Output must be a variable")
        assert isinstance(expression, Expression), TypeError("Expression must be an expression")

    def __repr__(self):
        return str(self)

    def __str__(self):
        args = [str(x) for x in self.inputs]
        return f"{self.output} = {self.name}({', '.join(args)})"

    def __len__(self):
        return len(self.output)

    def generate(self, p=preprint()):
        # Generate an eigen function to evaluate this Function
        # Assumptions: 
        #  - All parameters and constants have already been defined in the containing class
        #  - All functions that this function calls exist

        p = FunctionGenerator(p)

        # Produce function signature
        p(f"template<typename T>")
        p(f"EIGEN_STRONG_INLINE void {self.name}(")
        with p:
            for x in self.inputs:
                p(f"const Eigen::Ref<const Eigen::Matrix<T, {x.shape[0]}, {x.shape[1]}>>& {x}, ")
            p(f"Eigen::Ref<Eigen::Matrix<T, {self.output.shape[0]}, {self.output.shape[1]}>> {self.output}) const noexcept")
        p("{")

        # Evaluate expression
        # TODO: Change this function to accept a normal preprint, and then wrap it with a FunctionGenerator here
        with p:
            p(f"{self.output} = {self.expression.generate(p)};")

        p("}")

        return p

    def generate_jacobian(self, classname, p=preprint()):
        # Wrapper for Jacobian call
        sizes = [str(len(x)) for x in self.inputs]
        p(f"MAKE_JACOBIAN({self.wrapped_name}, {classname}<scalar_t>, {len(self.output)}, {', '.join(sizes)});")


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
        return self.expression.get_by_property(lambda n: n.isConstant and n.shape != (1, 1))

    @property
    def functions(self):
        # Return a set of functions that this function calls (recursively)
        functions = self.expression.get_by_property(lambda n: n.function)  # True if has a function attribute
        return set(f.function for f in functions)

    def __call__(self, *args):
        # Evaluate this function and return the expression
        return functionExpression(self, *args)


class FunctionGenerator:

    def __init__(self, p=preprint()):
        self.evaluations = []  # List of sub-expressions (functions) that have already been evaluated
        self.constants = []
        self.p = p  # PrePrint object

    def __enter__(self):
        return self.p.__enter__()

    def __exit__(self, exc_type, exc_value, tb):
        self.p.__exit__(exc_type, exc_value, tb)
        return True

    def __call__(self, *args, **kwargs):
        return self.p(*args, **kwargs)

    def add(self, expr, name):
        # Add an expression 
        if self.evaluations:
            evaled, _ = zip(*(self.evaluations))
            assert expr not in evaled, ValueError("Adding an evaluation that has already been evaluated... shouldn't happen")
        self.evaluations.append((expr, name))

    def get(self, expr):
        # Return name of expression if it's been previously eavluated, else None
        if self.evaluations:
            name = next((name for e, name in self.evaluations if expr == e), None)
            return name
        return None

    # def __call__(self, *args):
    #     return self.p(*args)