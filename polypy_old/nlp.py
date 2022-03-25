# from expression import Variable
import polypy
import polypy as pp
from polypy.expression import Variable
from polypy import variable
from polypy import Function
from polypy.function import Jacobian, Hessian
from polypy.generator import Generator

from collections import namedtuple
from collections.abc import Iterable
import numpy as np
import copy
import numbers


class Constraint:
    # Describes a constraint

    def __init__(self, expr, lb=None, ub=None):
        # self.expr = expr

        # Convert the constraint to generic function
        # and arguments to call this function with
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
        vars = [pp.variable(f"var{i}_", len(arg)) for i, arg in enumerate(self.args)]

        # Create function
        funcname = polypy._get_unique_name(basename="func")
        expr = expr.substitute(self.args, vars)        
        self.function = Function(funcname, vars, expr)

        self.name = None
        self.expr = expr

        self.lb = lb
        self.ub = ub

        if isinstance(self.lb, numbers.Number):
            self.lb = np.ones((len(expr))) * self.lb
        if isinstance(self.ub, numbers.Number):
            self.ub = np.ones((len(expr))) * self.ub

        if isinstance(self.lb, np.ndarray):
            self.lb = polypy.expression.ConstMatrix(self.lb)
        if isinstance(self.ub, np.ndarray):
            self.ub = polypy.expression.ConstMatrix(self.ub)

        # Test if this constraint is a set of constraints indexed by an Index
        try:
            self.index = self.original_expr.get_index()
        except IndexError:
            self.index = None

    def __repr__(self):
        return str(self)

    def __str__(self):
        if self.name:
            return self.name
        return str(self.original_expr)

    def __len__(self):
        # try:
        #     return len(self.function) * self.index.num_iterations
        # except AttributeError:
        return len(self.function)

    @property    
    def shape(self):
        try:
            return (len(self.function), self.index.num_iterations)
        except AttributeError:
            return (len(self.function), 1)
        # return (len(self.function), self.function.num_inputs())

    def freeze(self):
        return self.expr.freeze()

    def unfreeze(self):
        return self.expr.unfreeze()

    def generate_sparsity(self, p):
        """Print expression to write sparsity structure for this constraint into p"""
        vars = ", ".join(p.get_var_info(var) for var in self.args)
        if self.index:
            p(f"for(int i={self.index.start}, eq_ind=0; i<{self.index.stop}; i+={self.index.step}, eq_ind++)")
            p(f"\tset_equation_sparsity(trip, {self.name}.info(eq_ind), {{{vars}}});")
        else:
            p(f"set_equation_sparsity(trip, {self.name}.info(), {{{vars}}});")

    @property
    def nnz(self):
        """Return the number of non-zeros in the constraint

        Note: This is one constraint, not the series of constraints if index is not None
        """
        return sum(len(var) for var in self.args) * len(self)

class Equality(Constraint):
    # Describes an equality constraint lhs == rhs

    def __init__(self, lhs, rhs):

        # Check if either the lhs or the rhs are constants
        if not lhs.get_by_property(lambda x: isinstance(x, Variable)):
            super().__init__(rhs, lb=lhs, ub=lhs)
        elif not rhs.get_by_property(lambda x: isinstance(x, Variable)):
            super().__init__(lhs, lb=rhs, ub=rhs)
        else:
            super().__init__(lhs - rhs, lb=0, ub=0)

    def __bool__(self):
        # Returns True is the expression evaluates to zero
        return self._is_equal


class Inequality(Constraint):
    # Describes an inequality constraint lb <= expr <= ub
    # lb anb ub must be either constants or expressions involving only parameters

    def __init__(self, expr, **kwargs):
        super().__init__(expr, lb=kwargs.get('lb', -2e20), ub=kwargs.get('ub', 2e20))

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
        self.constraints = []  # List of constraints
        self.aux_functions = []  # Additional functions that the user wants
        
        self.variables = None  # List of variables with VariableSets expanded
        self.compressed_vars = None  # List of variables with VariableSets not expanded

    def freeze(self):
        # Freeze all expressions
        for x in self.constraints + self.variables + self.aux_functions:
            x.freeze()
            # x.state = polypy.Expression.State.Frozen

    def unfreeze(self):
        # Freeze all expressions
        for x in self.constraints + self.variables + self.aux_functions:
            x.unfreeze()
            # x.state = polypy.Expression.State.Frozen

    # def variable(self, name, shape, **kwargs):
    #     x = pp.variable(name, shape, **kwargs)
    #     self.variables += x
    #     return x
    #     # if number:
    #     #     x = [variable(name + str(i), length, **kwargs) for i in range(number)]
    #     #     self.variables += x
    #     # else:
    #     #     x = variable(name, length, **kwargs)
    #     #     self.variables.append(x)
    #     # return x

    def add_function(self, function):
        # Add an auxillary function to the generation list
        self.aux_functions.append(function)

    @property
    def nnz_constraints_jacobian(self):
        # Return the number of nonzeros in the constraints jacobian
        nnz = 0
        for con in self.constraints:
            nnz += con.nnz * con.shape[1]
            # nnz += con.shape[0] * con.shape[1]xxxxx
        return nnz

    def add(self, constraints, name=None):
        """Add an inequality or equality constraint to the NLP.

        constraint can be an element, or an iterable
        """

        if not isinstance(constraints, Iterable):
            constraints = (constraints, )

        for constraint in constraints:
            if name is None:
                if isinstance(constraint, Equality):
                    name = polypy._get_unique_name(basename="eq")
                elif isinstance(constraint, Inequality):
                    name = polypy._get_unique_name(basename="ineq")
                else:
                    name = polypy._get_unique_name(basename="con")
            constraint.name = name

            for eq in self.constraints:
                if constraint.function.expression.is_equal(eq.function.expression):
                    constraint.function = eq.function

            self.constraints.append(constraint)

    def minimize(self, expr):
        """Set objective function to minimize"""
        self.obj = Constraint(expr)
        self.obj.name = "obj"

    def functions(self):
        # Return all functions that need to be generated for this nlp
        funcs = set(con.function for con in self.constraints)
        return funcs.union(self.aux_functions)  # Dependencies will be discovered during generation
        # dependents = []
        # for func in funcs:
        #     dependents += func.functions
        # return funcs.union(dependents).union(self.aux_functions)

    def get_variables(self, expand=True):
        """Search through every equation in the NLP and extract all used Variables
        """

        # Collect all Variables and compress into VariableSets (so we can get the contiguous ordering correct)
        compressed_vars = set()
        for con in [*self.constraints, self.obj]:
            for var in con.original_expr.get_by_property(lambda n: isinstance(n, Variable)):
                if var.var_set:
                    compressed_vars.add(var.var_set)
                else:
                    compressed_vars.add(var)
        return list(compressed_vars)

    def _expand_variables(self, compressed_vars):
        """Take a list of Variables and VariableSets, and expands the VariableSets in place"""
        vars = []
        for v in compressed_vars:
            try:
                vars.extend(v.expand())
            except AttributeError:
                vars.append(v)
        return vars

    def set_variables(self, variables=None):
        """Set the variable for this NLP, and the desired order.

        If variables is not specfied, then we'll search through the NLP and extract all variables.

        Note: VariableSets must appear in contiguous order.
        """
        if not variables:
            variables = self.get_variables()
        else:
            # Convert lists to VariableSets
            for i in range(len(variables)):
                if isinstance(variables[i], list):
                    variables[i] = variables[i][0].var_set

        self.compressed_vars = variables
        self.variables = self._expand_variables(variables)

    def generate(self, p):
        """Write out a class to evaluate this nlp"""

        if not self.variables:  # If the variable order hasn't been set, detect and set it
            self.set_variables()

        # Freeze everything to make it easier to do set operations
        self.freeze()

        for f in self.functions():
            p.add_dependency(f, "Function")

        # Create a language-specific generator
        lang = p.get_nlp_generator(self)

        p.comment("Define variable accessors and ordering")
        lang.declare_variables(p, self.compressed_vars)
        p("")

        p.comment("Define constraint accessors and ordering")
        lang.declare_constraints(p, self.constraints)
        p("")

        p.comment("Define NLP sizes")
        lang.declare_nlp_sizes(p)

        p.comment("Short names to pass constant vectors and writable vectors")
        p("template<typename T, std::size_t n>")
        p("using cVec = const Eigen::Ref<const Eigen::Matrix<T, n, 1>>;")
        p("template<typename T, std::size_t n>")
        p("using Vec = Eigen::Ref<Eigen::Matrix<T, n, 1>>;")
        p("")


        p.comment("NLP variable types")
        p(f"using variable_t            = Matrix<scalar_t, NUM_VARS, 1>;")
        p(f"using constraint_t          = Matrix<scalar_t, NUM_CON, 1>;")

        # For now - we're assuming dense jacobians and hessian
        p(f"using constraint_jacobian_t = Matrix<scalar_t, NUM_CON,  NUM_VARS>;")
        p(f"using obj_gradient_t        = Matrix<scalar_t, 1, NUM_VARS>;")
        p(f"using obj_hessian_t         = Matrix<scalar_t, NUM_VARS, NUM_VARS>;")
        p(f"using obj_t                 = scalar_t;")
        # p(f"using dual_t       = Matrix<scalar_t, DUAL_SIZE, 1>;")
        p("")


        # p.comment("================= USER ACCESS FUNCTIONS =================")
        # p("using variable_map_t = std::map<std::string, var_slow_t>;")
        # p("variable_map_t variable_map = ")
        # p("{")
        # p(f", \n".join(f'\t{{"{var}", {var}}}' for var in self.compressed_vars))
        # p("};")
        # p("")
        # p("using MatrixX = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;")
        # ("p")
        # p("template<typename T>")
        # p("auto get(T v, const Eigen::Ref<const variable_t>& var)")
        # p("{")
        # p("    return Eigen::Map<const MatrixX>(var.data() + v.offset, v.len, v.num_vars);")
        # p("}")
        # ("p")
        # p("template<>")
        # p("auto get<std::string>(std::string name, const Eigen::Ref<const variable_t>& var)")
        # p("{")
        # p("    variable_map_t::iterator it = variable_map.find(name);")
        # p("    assert(it != variable_map.end());")
        # p("    return get(it->second, var);")
        # p("}   ")
        # ("p")
        # p("template<>")
        # p("auto get<const char*>(const char* name, const Eigen::Ref<const variable_t>& var)")
        # p("{")
        # p("    return get(std::string(name), var);")
        # p("}")
        # p("")

        self._generate_constraints(p)
        p("")
        self._generate_bounds(p)
        p("")
        self._generate_objective(p)
        p("")

        # Freeze everything to make it easier to do set operations
        self.unfreeze()

        return p

    def _generate_bounds(self, p):
        # Generate variable bounds

        p.comment("Evaluates the upper and lower bounds for the optimization variable into x_l and x_u")
        p.comment("Can access the resulting bounds with the macros var_get(x_l), where var is the variable")
        p("EIGEN_STRONG_INLINE void variable_bounds(Ref<variable_t> x_l, ")
        p("                                         Ref<variable_t> x_u) noexcept")
        with p.function():
            for var in self.compressed_vars:
                try:
                    scalar = var.lb.generate_scalar(p)
                    p(f"{var.eigen_get('x_l', columnwise=False)}.array() = {scalar};")
                except AttributeError:
                    p(f"{var.eigen_get('x_l', columnwise=True)} = {var.lb.generate(p)};")

                try:
                    scalar = var.ub.generate_scalar(p)
                    p(f"{var.eigen_get('x_u', columnwise=False)}.array() = {scalar};")
                except AttributeError:
                    p(f"{var.eigen_get('x_u', columnwise=True)} = {var.ub.generate(p)};")

            for var in self.variables:
                try:
                    if not var.var_set.lb._is_equal(var.lb):
                        try:
                            scalar = var.lb.generate_scalar(p)
                            p(f"{var.eigen_get('x_l', columnwise=False)}.array() = {scalar};")
                        except AttributeError:
                            p(f"{var.eigen_get('x_l', columnwise=True)} = {var.lb.generate(p)};")

                    if not var.var_set.ub._is_equal(var.ub):
                        try:
                            scalar = var.ub.generate_scalar(p)
                            p(f"{var.eigen_get('x_u', columnwise=False)}.array() = {scalar};")
                        except AttributeError:
                            p(f"{var.eigen_get('x_u', columnwise=True)} = {var.ub.generate(p)};")
                except AttributeError:
                    pass

        p("")
        p.comment("Evaluates the upper and lower bounds for the constraints variable into g_l and g_u")
        p.comment("Can access the resulting bounds with the macros con_get(g_l), where con is the variable")
        p("EIGEN_STRONG_INLINE void constraint_bounds(Ref<constraint_t> g_l, ")
        p("                                           Ref<constraint_t> g_u) noexcept")
        with p.function():
            for con in self.constraints:
                try:
                    p(f"{str(con)}.get(g_l).array() = {con.lb.generate_scalar(p)};")
                except AttributeError:
                    p(f"{str(con)}.get(g_l) = {con.lb.generate(p)};")

                try:
                    p(f"{str(con)}.get(g_u).array() = {con.ub.generate_scalar(p)};")
                except AttributeError:
                    p(f"{str(con)}.get(g_u) = {con.ub.generate(p)};")



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

        blocks = self._sparsity_structure(self.constraints)

        # Generate function to return the sparsity structure
        #
        p.comment("Set non-zeros of J to match the sparsity structure of the constraint Jacobian")
        p("EIGEN_STRONG_INLINE void constraints_sparse_initialize(SparseMatrix<scalar_t>& J)")
        with p.function():
            p("std::vector<Eigen::Triplet<scalar_t>> trip;")
            p("")
            for con in self.constraints:
                con.generate_sparsity(p)
            p("")
            p("J.setFromTriplets(trip.begin(), trip.end());")
            p("J.makeCompressed();")

        p.comment("Forwarding calls for dense and sparse jacobians. Will be moved to parent class later.")
        p("EIGEN_STRONG_INLINE void constraints(const Ref<const variable_t>& var, Ref<constraint_t> constraints) noexcept")
        p("\t{this->template constraints_impl<int>(var, constraints, 0);}")
        p("EIGEN_STRONG_INLINE void constraints(const Ref<const variable_t>& var, Ref<constraint_t> constraints, Ref<constraint_jacobian_t> jacobian) noexcept")
        p("\t{this->template constraints_impl<Ref<constraint_jacobian_t>>(var, constraints, jacobian);}")
        p("EIGEN_STRONG_INLINE void constraints(const Ref<const variable_t>& var, Ref<constraint_t> constraints, Ref<SparseMatrix<scalar_t>> jacobian) noexcept")
        p("\t{this->template constraints_impl<Ref<SparseMatrix<scalar_t>>>(var, constraints, jacobian);}")
        p("")

        p("template<typename jacobian_t>")
        p("EIGEN_STRONG_INLINE void constraints_impl(const Ref<const variable_t>& var,")
        p("                                          Ref<constraint_t> constraints,")
        p("                                          jacobian_t jacobian) noexcept")
        p("{")
        with p:
            for con in self.constraints:
                print(f"Processing constraint {con}")
                p.add_dependency(Jacobian(con.function), "Function")
                # p.eval_constraint(con)

                col_offset = []
                for var in con.args:
                    print(con)
                    print(var)
                    print("HERE")

                    iVar = self.variables.index(var)
                    iCon = self.constraints.index(con)
                    col_offset.append(sum([len(blk.con) for blk in blocks[:iCon, iVar] if blk]))

                p(f"{con.function.name}({{{', '.join([p.get_var_offset(arg) for arg in con.args])}}}, {{{', '.join(str(i) for i in col_offset)}}}, {con.name}(), var, constraints, jacobian);")
        p("}")

    def _generate_objective(self, p):
        """Generate objective function"""
        # p.add_dependency(self.obj.function, "Function")

        # Decompose our objective into a summation of objectives
        try:
            obj = self.obj.original_expr._decompose()
        except AttributeError:
            obj = [self.obj.original_expr, ]

        # Create objective function
        obj_func = [Constraint(expr) for expr in obj]

        # Identify unique function calls and add then to the generation list
        unique_funcs = []
        for i, o in enumerate(obj_func):
            found = False
            for u in unique_funcs:
                if o.function.expression.is_equal(u.function.expression):
                    o.function = u.function
                    found = True
                    break
            if not found:
                unique_funcs.append(o)
                p.add_dependency(Hessian(o.function), "Hessian")

        # Compute interactions between variables
        vars = self.variables
        H = np.zeros((len(vars), len(vars)))  # One if this element of the hessian is filled in
        for o in obj_func:
            for v_col in o.args:
                iv_col = self.variables.index(v_col)
                for v_row in o.args:
                    iv_row = self.variables.index(v_row)
                    H[iv_row, iv_col] = 1

        # Compute offsets into the Values vector for each block
        H_offset = np.zeros((len(vars), len(vars)), dtype=int)  # Offset into the Values vector where this variable block starts
        offset = 0  # Offset into the Values vector
        for col in range(len(vars)):
            col_offset = offset
            for row in range(len(vars)):
                if H[row, col]:
                    H_offset[row, col] = col_offset
                    col_offset += len(self.variables[row])
                    offset += len(self.variables[row]) * len(self.variables[col])

        # Build a dictionary mapping variables to their place in the global variable
        var_pos = dict()
        pos = 0
        for var in self.variables:
            var_pos[var] = pos
            pos += len(var)


        # Generate function to return the sparsity structure
        #
        p("EIGEN_STRONG_INLINE void objective_sparse_initialize(SparseMatrix<scalar_t>& H)")
        p("{")
        with p:
            # BlockInfo = row, col, num_rows, num_cols
            p("set_nonzero_blocks<scalar_t>(H, {BlockInfo")
            blk_info = []
            for iRow, row_var in enumerate(self.variables):
                for iCol, col_var in enumerate(self.variables):
                    if H[iRow, iCol]:
                        row = var_pos[row_var]
                        num_rows = len(row_var)
                        col = var_pos[col_var]
                        num_cols = len(col_var)
                        blk_info.append(f"{{{row}, {col}, {num_rows}, {num_cols}}}")

            # blk_info = [f"{{{', '.join(str(i) for i in blk.shape)}}}" for blk in blocks.flatten() if blk]
            blk_info = [", ".join(blk_info[i:i+6]) for i in range(0, len(blk_info), 6)]
            p(",\n".join(blk_info))
            p("});")
        p("}\n")

        # p("EIGEN_STRONG_INLINE scalar_t objective(const Ref<const variable_t>& var, Ref<obj_gradient_t> gradient, Ref<obj_hessian_t> hessian) noexcept")
        # p("\t{return this->template objective_impl<Ref<obj_gradient_t>, Ref<obj_hessian_t>>(var, gradient, hessian);}")
        # p("EIGEN_STRONG_INLINE scalar_t objective(const Ref<const variable_t>& var, Ref<obj_gradient_t> gradient) noexcept")
        # p("\t{return this->template objective_impl<Ref<obj_gradient_t>, int>(var, gradient, 0);}")
        # p("EIGEN_STRONG_INLINE scalar_t objective(const Ref<const variable_t>& var) noexcept")
        # p("\t{return this->template objective_impl<int, int>(var, 0, 0);}")
        # p("")

        p.comment("Locations to store the computed hessians")
        for i, o in enumerate(obj_func):
            offset = np.zeros((len(o.args), len(o.args)), dtype=int)
            for iCol, col in enumerate(o.args):
                for iRow, row in enumerate(o.args):
                    ind_row = self.variables.index(row)
                    ind_col = self.variables.index(col)
                    offset[iRow, iCol] = H_offset[ind_row, ind_col]
            str_offset = ", ".join("{" + ", ".join(str(offset[iRow, iCol]) for iCol in range(len(o.args))) + "}" for iRow in range(len(o.args)))
            if len(o.args) == 1:
                p(f"const Matrix<Eigen::Index, {len(o.args)}, {len(o.args)}> H_offset_{i} = {str_offset};")
            else:
                p(f"const Matrix<Eigen::Index, {len(o.args)}, {len(o.args)}> H_offset_{i} = {{{str_offset}}};")
        p("")

        p("EIGEN_STRONG_INLINE scalar_t objective(const Ref<const variable_t>& var) noexcept")
        with p.function():
            p("scalar_t val = 0.0;")
            p("")

            for i, o in enumerate(obj_func):
                # var_offsets = "{" + ', '.join(str(var_pos[var]) for var in o.args) + "}"
                var_offsets = "{" + ', '.join(p.get_var_offset(var) for var in o.args) + "}"
                p(f"val += {o.function.name}({var_offsets}, var);")
            p("return val;")
        p("")

        p("EIGEN_STRONG_INLINE scalar_t objective(const Ref<const variable_t>& var,")
        p("                                       Ref<obj_gradient_t> gradient) noexcept")
        with p.function():
            p("scalar_t val = 0.0;")
            p("gradient.setZero();")
            p("")

            for i, o in enumerate(obj_func):
                # var_offsets = "{" + ', '.join(str(var_pos[var]) for var in o.args) + "}"
                var_offsets = "{" + ', '.join(p.get_var_offset(var) for var in o.args) + "}"
                p(f"val += {o.function.name}({var_offsets}, var, gradient, true);")
            p("return val;")
        p("")

        p("EIGEN_STRONG_INLINE scalar_t objective(const Ref<const variable_t>& var,")
        p("                                       Ref<obj_gradient_t> gradient,")
        p("                                       SparseMatrix<scalar_t>& hessian) noexcept")
        with p.function():
            p("scalar_t val = 0.0;")
            p("gradient.setZero();")
            p.comment("Set non-zero elements of hessian to zero, but don't change nnz")
            p("Eigen::Map<Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>> (hessian.valuePtr(), hessian.nonZeros(), 1).array() = 0.0;")
            p("")

            for i, o in enumerate(obj_func):
                var_offsets = "{" + ', '.join(p.get_var_offset(var) for var in o.args) + "}"
                p(f"val += {o.function.name}({var_offsets}, "
                    + f"var, gradient, H_offset_{i}, hessian, true);")
            p("return val;")


    def _sparsity_structure(self, constraints):
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
        blocks = np.zeros(shape=(len(constraints), len(self.variables)), dtype=Block)
        for var_index, var in enumerate(self.variables):
            cons = [con for con in constraints if var in con.args]
            nnz_col = sum(len(con) for con in cons)  # Number of non-zeros per column

            block_start_index = index

            for con_index, con in enumerate(constraints):
                if var in con.args:

                    # Non-zero blocks for this variable
                    row = sum(len(con) for con in constraints[:con_index])
                    col = sum(len(var) for var in self.variables[:var_index])
                    blocks[con_index, var_index] = \
                        Block(block_start_index, nnz_col, \
                            (row, col, len(con), len(var)), \
                            var, con)

                    block_start_index += len(con)

            # Increment Value index
            index += nnz_col * len(var)

        return blocks



