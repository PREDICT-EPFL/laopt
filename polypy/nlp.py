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
        self.expr = expr

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self.function)

    @property    
    def shape(self):
        return (len(self.function), self.function.num_inputs())

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


class NLP:
    # Describes an optimization nlp
    #
    #  min f(x)
    #  s.t. g_l <= g(x) <= g_u
    #       x_l <= x <= x_u
    #

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

    @property
    def constraints(self):
        return self.equalities + self.inequalities

    def add_function(self, function):
        # Add an auxillary function to the generation list
        self.aux_functions.append(function)

    @property
    def nnz_constraints_jacobian(self):
        # Return the number of nonzeros in the constraints jacobian
        nnz = 0
        for con in self.constraints:
            nnz += con.shape[0] * con.shape[1]
        return nnz

    def add(self, constraint, name=None):
        # Add an inequality or equality constraint to the nlp

        if name is None:
            if isinstance(constraint, Equality):
                name = polypy._get_unique_name("eq")
            elif isinstance(constraint, Inequality):
                name = polypy._get_unique_name("ineq")
            else:
                name = polypy._get_unique_name("con")
        constraint.name = name

        for eq in self.constraints:
            if constraint.function.expression.is_equal(eq.function.expression):
                constraint.function = eq.function

        if isinstance(constraint, Equality):
            # Test if the function that we're calling here already exists
            self.equalities.append(constraint)

        if isinstance(constraint, Inequality):
            self.inequalities.append(constraint)

    def functions(self):
        # Return all functions that need to be generated for this nlp
        funcs = set(con.function for con in self.constraints)
        dependents = []
        for func in funcs:
            dependents += func.functions
        return funcs.union(dependents).union(self.aux_functions)

    def generate(self, filename="gen.hpp", classname="LOpt"):
        # Write out a class to evaluate this nlp

        g = Generator()
        p = g.p
        for f in self.functions():
            g.add_function(f)
        g.generate_preamble()

        with g.generate_class(classname):
            p("// Define NLP sizes")
            p("enum")
            with g.function_body(post_string=";"):
                p(f"NUM_VARS  = {sum(len(var) for var in self.variables)},")
                p(f"NUM_CON   = {sum(len(eq) for eq in self.constraints)},")
                # p(f"NUM_INEQ  = {sum(len(eq) for eq in self.inequalities)},")
                # p(f"NUM_CON   = NUM_EQ + NUM_INEQ,")
                # p(f"NUM_BOX   = 0,")
                # p(f"DUAL_SIZE = NUM_EQ + NUM_INEQ + NUM_BOX,")
                p(f"nnz_constraints_jacobian = {self.nnz_constraints_jacobian},")
            p("")

            p("// NLP variable types")
            p(f"using variable_t   = Eigen::Matrix<scalar_t, NUM_VARS, 1>;")
            p(f"using constraint_t = Eigen::Matrix<scalar_t, NUM_CON, 1>;")

            # For now - we're assuming dense jacobians and hessian
            p(f"using constraint_jacobian_t = Eigen::Matrix<scalar_t, NUM_CON,  NUM_VARS>;")
            p(f"using cost_hessian_t        = Eigen::Matrix<scalar_t, NUM_VARS, NUM_VARS>;")
            p(f"using cost_t                = scalar_t;")
            # p(f"using dual_t       = Eigen::Matrix<scalar_t, DUAL_SIZE, 1>;")
            p("")

            p("// Define optimization variables (offsets in var)")
            offset = 0
            for var in self.variables:
                p(f"DECLARE_VAR({var}, {offset}, {len(var)})")
                offset += len(var)
            p("")

            p("// Define constraints (offset functions)")
            offset = 0
            for con in self.constraints:
                p(f"DECLARE_CONSTRAINT({con.name}, {offset}, {len(con)})")
                offset += len(con)
            p("")

            self._generate_equalities(p)
            p("")
            self._generate_equalities_linearized(p)
            p("")
            self._get_constraints_sparsity(p)
            p("")

            g.generate_functions()


        g.write_file(filename)
        return p

    def _generate_equalities(self, p):
        p("EIGEN_STRONG_INLINE void constraints(const Eigen::Ref<const variable_t>& var, ")
        p("                                     Eigen::Ref<constraint_t> constraints) noexcept")
        p("{")
        with p:
            for con in self.constraints:
                p(f"{con.function.name}({', '.join([f'{arg.name}(var)' for arg in con.args])}, {con.name}(constraints));")
        p("}")

    def _generate_equalities_linearized(self, p):
        p("EIGEN_STRONG_INLINE void constraints_jacobian(const Eigen::Ref<const variable_t>& var,")
        p("                                              Eigen::Ref<constraint_t> constraints,")
        p("                                              Eigen::Ref<constraint_jacobian_t> jacobian) noexcept")
        p("{")
        with p:

            for con in self.constraints:
                offsets = ", ".join([f"{var.name}_o()" for var in con.args])
                offsets = f"Fill_Dense<decltype(jacobian), {con.name}_o(), {offsets}>(jacobian));"
                p(f"{con.function.name}({', '.join([f'{arg.name}(var)' for arg in con.args])}, {con.name}(constraints), " + offsets)
        p("}")

    def _get_constraints_sparsity(self, p):
        # Build a dictionary mapping variables to their place in the global variable
        var_pos = dict()
        pos = 0
        for var in self.variables:
            var_pos[var] = pos
            pos += len(var)

        # Build a dictionary mapping constraints to their place in the global variable
        con_pos = dict()
        pos = 0
        for con in self.constraints:
            con_pos[con] = pos
            pos += len(con)

        # Generate function to return the sparsity structure
        #
        p("EIGEN_STRONG_INLINE void constraints_sparse_initialize(Eigen::SparseMatrix<scalar_t>& J)")
        p("{")
        with p:
            p("std::vector<Eigen::Triplet<scalar_t>> trip;")
            for con in self.constraints:
                p(f"for(int row={con_pos[con]}; row<{con_pos[con] + len(con)}; row++)  // {con.expr}")
                p("{")
                with p:
                    for var in con.args:
                        p(f"for(int col={var_pos[var]}; col<{var_pos[var]+len(var)}; col++) trip.emplace_back(row, col, 0.0); // {var}")
                p("}")
            p("J.setFromTriplets(trip.begin(), trip.end());")
            p("J.makeCompressed();")
        p("}")
        p("")

        # Generate function to fill in the jacobian
        #
        p("// J must have been initialized with initialize_constraints")
        p("EIGEN_STRONG_INLINE void constraints_sparse_jacobian(const Eigen::Ref<const variable_t>& var,")
        p("                                                     Eigen::Ref<constraint_t> constraints,")
        p("                                                     Eigen::SparseMatrix<scalar_t>& J) noexcept")
        p("{")
        with p:
            for con in self.constraints:
                p("{")
                with p:
                    # Declare dense temporaries for the jacobians
                    for var in con.args:
                        p(f"Eigen::Matrix<scalar_t, {con.name}_len, {var.name}_len> J_{var.name};")

                    # Call the function
                    call = f"{con.function.name}({', '.join([f'{arg.name}(var)' for arg in con.args])}, "
                    call += f"{con.name}(constraints), " + ", ".join([f"J_{var.name}" for var in con.args]) + ");"
                    p(call)

                    # Copy the jacobians into the sparse J
                    p(f"for(int row={con_pos[con]}, i=0; row<{con_pos[con] + len(con)}; row++, i++)")
                    p("{")
                    with p:
                        for var in con.args:
                            p(f"for(int col={var_pos[var]}, j=0; col<{var_pos[var]+len(var)}; col++, j++) J.coeffRef(row, col) = J_{var.name}(i, j);")
                    p("}")
                p("}")
        p("}")

        p("// Fills in the values of the sparse matrix in the order defined in constraints_sparse_initialize")
        p("EIGEN_STRONG_INLINE void constraints_sparse_jacobian_values(const Eigen::Ref<const variable_t>& var,")
        p("                                                            Eigen::Ref<constraint_t> constraints,")
        p("                                                            Eigen::Ref<Eigen::Matrix<scalar_t, nnz_constraints_jacobian, 1>> J) noexcept")
        p("{")
        with p:
            p("int ind = 0;")
            for con in self.constraints:
                p("{")
                with p:
                    # Declare dense temporaries for the jacobians
                    for var in con.args:
                        p(f"Eigen::Matrix<scalar_t, {con.name}_len, {var.name}_len> J_{var.name};")

                    # Call the function
                    call = f"{con.function.name}({', '.join([f'{arg.name}(var)' for arg in con.args])}, "
                    call += f"{con.name}(constraints), " + ", ".join([f"J_{var.name}" for var in con.args]) + ");"
                    p(call)

                    # Copy the jacobians into the sparse J
                    p(f"for(int row=0; row<{len(con)}; row++)")
                    p("{")
                    with p:
                        for var in con.args:
                            p(f"for(int col=0; col<{len(var)}; col++) J(ind++) = J_{var.name}(row, col);")
                    p("}")
                p("}")
        p("}")


# class Ipopt:
#     # Interface to Ipopt
    
#     def __init__(self, nlp):
#         # nlp = instance of NLP
#         self.nlp = nlp

#     def generate(self, p):
#         nlp = self.nlp

#         p("\n\n")
#         p('#include "IpIpoptApplication.hpp"')
#         p('#include "IpTNLP.hpp"')
#         p("")
#         p("template<typename scalar_t>")
#         p(f"{nlp.classname}_Ipopt : public TNLP, public {nlp.classname}")
#         
