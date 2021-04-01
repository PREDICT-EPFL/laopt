# from expression import Variable
import polypy
from polypy import Variable, Function
from polypy.generator import Generator

from collections import namedtuple
import numpy as np
import copy


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

        self.original_expr = copy.copy(expr)

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

    def __str__(self):
        if self.name:
            return self.name
        return str(self.original_expr)

    def __len__(self):
        return len(self.function)

    @property    
    def shape(self):
        return (len(self.function), self.function.num_inputs())

    @property
    def state(self):
        return self.expr.state

    @state.setter
    def state(self, newState):
        self.expr.state = newState


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

    # def __str__(self):
    #     return str(self.function) + " = 0"


class Inequality(Constraint):
    # Describes an inequality constraint lb <= expr <= ub
    # lb anb ub must be either constants or expressions involving only parameters

    def __init__(self, expr, lb=float('-inf'), ub=float('inf')):
        super().__init__(expr)
        self.lb = lb
        self.ub = ub

    # def __str__(self):
    #     return str(self.lb) + " <= " + str(self.expr) + " <= " + str(self.ub)


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

        self.variables = []  # List of variables

        self.parameters = []  # List of parameters

        self.aux_functions = []  # Additional functions that the user wants

    def freeze(self):
        # Freeze all expressions
        for x in self.equalities + self.inequalities \
               + self.variables + self.parameters + self.aux_functions:
            x.state = polypy.Expression.State.Frozen

    def unfreeze(self):
        # Freeze all expressions
        for x in self.equalities + self.inequalities \
               + self.variables + self.parameters + self.aux_functions:
            x.state = polypy.Expression.State.Frozen

    def variable(self, name, length, number=None, **kwargs):
        if number:
            x = [Variable(name + str(i), length, **kwargs) for i in range(number)]
            self.variables += x
        else:
            x = Variable(name, length, **kwargs)
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
        return funcs.union(self.aux_functions)  # Dependencies will be discovered during generation
        # dependents = []
        # for func in funcs:
        #     dependents += func.functions
        # return funcs.union(dependents).union(self.aux_functions)

    def generate(self, p): #filename="gen.hpp", classname="LOpt"):
        # Write out a class to evaluate this nlp

        # Freeze everything to make it easier to do set operations
        self.freeze()

        for f in self.functions():
            p.add_dependency(f, "Function")

        # with g.generate_class(classname):
        p("// Define NLP sizes")
        p("enum")
        with p.function(post_string=";"):
            p(f"NUM_VARS  = {sum(len(var) for var in self.variables)},")
            p(f"NUM_CON   = {sum(len(eq) for eq in self.constraints)},")
            # p(f"NUM_INEQ  = {sum(len(eq) for eq in self.inequalities)},")
            # p(f"NUM_CON   = NUM_EQ + NUM_INEQ,")
            # p(f"NUM_BOX   = 0,")
            # p(f"DUAL_SIZE = NUM_EQ + NUM_INEQ + NUM_BOX,")
            p(f"nnz_constraints_jacobian = {self.nnz_constraints_jacobian},")
        p("")

        p("// NLP variable types")
        p(f"using variable_t   = Matrix<scalar_t, NUM_VARS, 1>;")
        p(f"using constraint_t = Matrix<scalar_t, NUM_CON, 1>;")

        # For now - we're assuming dense jacobians and hessian
        p(f"using constraint_jacobian_t = Matrix<scalar_t, NUM_CON,  NUM_VARS>;")
        p(f"using cost_hessian_t        = Matrix<scalar_t, NUM_VARS, NUM_VARS>;")
        p(f"using cost_t                = scalar_t;")
        # p(f"using dual_t       = Matrix<scalar_t, DUAL_SIZE, 1>;")
        p("")

        p("// Define optimization variables (offsets in var)")
        offset = 0
        for var in self.variables:
            setattr(var, 'offset', offset)
            p(f"DECLARE_VAR({var}, {var.offset}, {len(var)})")
            offset += len(var)
        p("")

        p("// Define constraints (offset functions)")
        offset = 0
        for con in self.constraints:
            setattr(con, 'offset', offset)
            p(f"DECLARE_CONSTRAINT({con.name}, {con.offset}, {len(con)})")
            offset += len(con)
        p("")

        self._generate_constraints(p)
        p("")
        self._generate_bounds(p)

        # Freeze everything to make it easier to do set operations
        self.unfreeze()

        return p

    def _generate_bounds(self, p):
        # Generate variable bounds

        p("// Evaluates the upper and lower bounds for the optimization variable into x_l and x_u")
        p("// Can access the resulting bounds with the macros var_get(x_l), where var is the variable")
        p("EIGEN_STRONG_INLINE void variable_bounds(Ref<variable_t> x_l, ")
        p("                                         Ref<variable_t> x_u) noexcept")
        with p.function():
            p("using T = scalar_t;")  # All function calls will not use derivatives
            for var in self.variables:
                # print(f"Generating {var}")
                # print(f"\tlb = {var.lb}")
                p(f"{str(var)}_get(x_l) = {var.lb.generate(p)};")
                # print(f"\tub = {var.ub}")
                p(f"{str(var)}_get(x_u) = {var.ub.generate(p)};")


    def _generate_constraints(self, p):       
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

        # We assume that our jacobian is stored in compressed column format
        # The structure is aligned blockwise
        #
        #         var1 | var2 | var3 | ... | varN
        #  con1 |      |      |      |     |
        #  con2 |      |      |      |     |
        #  con3 |      |      |      |     |
        #  con4 |      |      |      |     |
        #  con5 |      |      |      |     |
        #
        # Where each of the con/var blocks is either dense, or zero.
        #
        # In compressed column format, we have a Value vector of length NNZ (number of non-zeros).
        # For each non-zero con/var pair, we store three things:
        # - 
        Block = namedtuple("Block", [
            "index",  # index into Value of the top-left element
            "nnz_col",  # Number of non-zeros in the var column
            "shape",  # shape of the block (row, col, len(con), len(var))
            "var", "con"  # The variable and constraint involved
            ])

        index = 0  # Index into the Value vector
        blocks = np.zeros(shape=(len(self.constraints), len(self.variables)), dtype=Block)
        for var_index, var in enumerate(self.variables):
            cons = [con for con in self.constraints if var in con.args]
            nnz_col = sum(len(con) for con in cons)  # Number of non-zeros per column

            block_start_index = index

            for con_index, con in enumerate(self.constraints):
                if var in con.args:

                    # Non-zero blocks for this variable
                    row = sum(len(con) for con in self.constraints[:con_index])
                    col = sum(len(var) for var in self.variables[:var_index])
                    blocks[con_index, var_index] = \
                        Block(block_start_index, nnz_col, \
                            (row, col, len(con), len(var)), \
                            var, con)

                    block_start_index += len(con)

            # Increment Value index
            index += nnz_col * len(var)

        # Generate function to return the sparsity structure
        #
        p("EIGEN_STRONG_INLINE void constraints_sparse_initialize(SparseMatrix<scalar_t>& J)")
        p("{")
        with p:
            p("set_nonzero_blocks<scalar_t>(J, {BlockInfo")
            blk_info = [f"{{{', '.join(str(i) for i in blk.shape)}}}" for blk in blocks.flatten() if blk]
            blk_info = [", ".join(blk_info[i:i+6]) for i in range(0, len(blk_info), 6)]
            p(",\n".join(blk_info))
            p("});")
        p("}\n")

        p("// Forwarding calls for dense and sparse jacobians. Will be moved to parent class later.")
        p("EIGEN_STRONG_INLINE void constraints(const Ref<const variable_t>& var, ")
        p("                                     Ref<constraint_t> constraints) noexcept")
        p("{this->template constraints_impl<int>(var, constraints, 0);}")
        p("EIGEN_STRONG_INLINE void constraints(const Ref<const variable_t>& var,")
        p("                                     Ref<constraint_t> constraints,")
        p("                                     Ref<constraint_jacobian_t> jacobian) noexcept")
        p("{this->template constraints_impl<Ref<constraint_jacobian_t>>(var, constraints, jacobian);}")
        p("EIGEN_STRONG_INLINE void constraints(const Ref<const variable_t>& var, ")
        p("                                     Ref<constraint_t> constraints, ")
        p("                                     Ref<SparseMatrix<scalar_t>> jacobian) noexcept")
        p("{this->template constraints_impl<Ref<SparseMatrix<scalar_t>>>(var, constraints, jacobian);}")
        p("")

        p("template<typename jacobian_t>")
        p("EIGEN_STRONG_INLINE void constraints_impl(const Ref<const variable_t>& var,")
        p("                                          Ref<constraint_t> constraints,")
        p("                                          jacobian_t jacobian) noexcept")
        p("{")
        with p:
            for con in self.constraints:
                col_offset = []
                for var in con.args:
                    iVar = self.variables.index(var)
                    iCon = self.constraints.index(con)
                    col_offset.append(sum([len(blk.con) for blk in blocks[:iCon, iVar] if blk]))

                p(f"{con.function.name}({{{', '.join([f'{arg.name}' for arg in con.args])}}}, {{{', '.join(str(i) for i in col_offset)}}}, {con.name}, var, constraints, jacobian);")
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
