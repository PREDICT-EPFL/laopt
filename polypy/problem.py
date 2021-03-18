# from expression import Variable
import polypy
from polypy import Variable, Function
from polypy.generator import Generator


class Constraint:
    # Describes a constraint

    def __init__(self, expr):
        # self.expr = expr

        # Convert the constraint to generic function
        # arguments to call this function with
        #
        # We've got an expression expr
        # 1. Convert this to a function f(x0,x1,...) where vars = {x0,x1,...} 
        #    are optimization variables
        # 2. Create a function g(var0, var1,...) where vari are generic variables
        # 3. The constraint is now g(x0, x1, ...)
        # 
        # We do this so that we can recognize is different constraints are calling 
        # the same function g

        # Argument list for our new function
        args = expr.get_by_property(lambda n: isinstance(n, Variable))
        self.args = sorted(args, key=lambda v: v.name)

        # Generic named variables
        vars = [Variable(f"var{i}", len(arg)) for i, arg in enumerate(self.args)]

        # Create function
        funcname = polypy._get_unique_name("func")
        out = Variable("out", len(expr))
        expr = expr.substitute(self.args, vars)        
        self.function = Function(funcname, vars, out, expr)

        self.name = None

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self.function)

class Equality(Constraint):
    # Describes an equality constraint lhs == rhs

    def __init__(self, lhs, rhs=0):
        super().__init__(lhs-rhs)

        # try:
        self._is_equal = lhs.is_equal(rhs)
        # except:
        #     self._is_equal = False

    def __bool__(self):
        # Returns True is the expression evaluates to zero
        return self._is_equal

    def __str__(self):
        return str(self.function) + " = 0"


class Inequality(Constraint):
    # Describes an inequality constraint lb <= expr <= ub
    # lb anb ub must be either constants or expressions involving only parameters

    def __init__(self, expr, lb=float('-inf'), ub=float('inf')):
        super().__init__(expr)
        self.lb = lb
        self.ub = ub

    def __str__(self):
        return str(self.lb) + " <= " + str(self.expr) + " <= " + str(self.ub)


class Problem:
    # Describes an optimization problem

    def __init__(self, name):
        self.name = name
        self.equalities = []  # List of expressions

        # List of tuples (lb, expression, ub)
        #  lb, ub are list-like objects
        self.inequalities = []

        self.variableBnds = []  # (lb, ub)

        self.variables = []  # List of variables

        self.parameters = []  # List of parameters

        self.aux_functions = []  # Additional functions that the user wants

    def variable(self, name, length, number=None):
        if number:
            x = [Variable(name + str(i), length) for i in range(number)]
            self.variables += x
        else:
            x = Variable(name, length)
            self.variables.append(x)
        return x

    def add_function(self, function):
        # Add an auxillary function to the generation list
        self.aux_functions.append(function)

    def add(self, constraint, name=None):
        # Add an inequality or equality constraint to the problem

        if name is None:
            if isinstance(constraint, Equality):
                name = polypy._get_unique_name("eq")
            elif isinstance(constraint, Inequality):
                name = polypy._get_unique_name("ineq")
            else:
                name = polypy._get_unique_name("con")
        constraint.name = name

        for eq in self.equalities + self.inequalities:
            if constraint.function.expression.is_equal(eq.function.expression):
                constraint.function = eq.function

        if isinstance(constraint, Equality):
            # Test if the function that we're calling here already exists
            self.equalities.append(constraint)

        if isinstance(constraint, Inequality):
            self.inequalities.append(constraint)

    def functions(self):
        # Return all functions that need to be generated for this problem
        funcs = set(con.function for con in self.equalities + self.inequalities)
        dependents = []
        for func in funcs:
            dependents += func.functions
        return funcs.union(dependents).union(self.aux_functions)

    def generate(self, filename="gen.hpp", classname="LOpt"):
        # Write out a class to evaluate this problem

        g = Generator()
        p = g.p
        for f in self.functions():
            g.add_function(f)
        g.generate_preamble()

        with g.generate_class(classname):
            g.generate_functions()

            p("// Define problem sizes")            
            p("enum")
            with g.function_body(post_string=";"):
                p(f"VAR_SIZE  = {sum(len(var) for var in self.variables)},")
                p(f"NUM_EQ    = {sum(len(eq) for eq in self.equalities)},")
                p(f"NUM_INEQ  = {sum(len(eq) for eq in self.inequalities)},")
                p(f"NUM_BOX   = 0,")
                p(f"DUAL_SIZE = NUM_EQ + NUM_INEQ + NUM_BOX,")
            p("")

            p("// NLP variable types")
            p(f"using nlp_variable_t    = Matrix<scalar_t, VAR_SIZE, 1>;")
            p(f"using nlp_constraints_t = Matrix<scalar_t, NUM_EQ + NUM_INEQ, 1>;")

            # For now - we're assuming dense jacobians and hessian
            p(f"using nlp_eq_jacobian_t = Matrix<scalar_t, NUM_EQ + NUM_INEQ, VAR_SIZE>;")
            p(f"using nlp_hessian_t     = Matrix<scalar_t, VAR_SIZE, VAR_SIZE>;")
            p(f"using nlp_cost_t        = scalar_t;")
            p(f"using nlp_dual_t        = Matrix<scalar_t, DUAL_SIZE, 1>;")
            p("")

            p("// Define optimization variables (offsets in var)")
            offset = 0
            for var in self.variables:
                p(f"DECLARE_VAR({var}, {offset}, {len(var)})")
                offset += len(var)
            p("")

            p("// Define constraints (offset functions)")
            offset = 0
            for con in self.equalities:
                p(f"DECLARE_CONSTRAINT({con.name}, {offset}, {len(con)})")
                offset += len(con)
            p("")

            self._generate_equalities(p)
            p("")
            self._generate_equalities_linearized(p)

        g.write_file(filename)
        return p

    def _generate_equalities(self, p):
        p("EIGEN_STRONG_INLINE void equalities(const Eigen::Ref<const nlp_variable_t>& var, ")
        p("                                    Eigen::Ref<nlp_constraints_t> _equalities) noexcept")
        p("{")
        with p:
            for con in self.equalities:
                p(f"{con.function.name}({', '.join([f'{arg.name}(var)' for arg in con.args])}, {con.name}(_equalities));")
        p("}")

    def _generate_equalities_linearized(self, p):
        p("EIGEN_STRONG_INLINE void equalities_linearised(const Eigen::Ref<const nlp_variable_t>& var,")
        p("                                               Eigen::Ref<nlp_constraints_t> equalities,")
        p("                                               Eigen::Ref<nlp_eq_jacobian_t> jacobian) noexcept")
        p("{")
        with p:
            for con in self.equalities:
                p(f"{con.function.name}({', '.join([f'{arg.name}(var)' for arg in con.args])},")
                with p:
                    p(f"{con.name}(equalities),")
                    jac = [f"jacobian.template block<{con.name}_len, {var.name}_len>({con.name}_offset(), {var.name}_offset())" for var in con.args]
                    p(",\n".join(jac) + ");")
        p("}")
