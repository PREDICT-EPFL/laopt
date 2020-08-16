import cog
import copy
from collections import namedtuple

# TODOS

# Refactoring...
# seperate "functions" from the NLP - just make things that have eval, jacobian and hessian operators
# from this build dense QP, dense NLP, blockwise QP, blockwise NLP, etc
# layer integrators and collocation on top

# Critical path
# update RHS of equalities and inequalities
# create a QP version of the NLP and attach to QP solver
# hook up the NLP to IPOPT 

# Nice to have
# assert no spaces in any names
# check that all the "functions" classes have the right member functions
# allow specification of variable bounds?
# add a bunch of auto-generated constraints via python? (A*x <= b) type stuff?
# allow for some of the jacobians to be specified manually and only generate the missing bits
# implement BFGS

class Variable:
    def __init__(self, name, offset, rows, cols=1, col=None):
        self.name = name
        self.rows = rows
        self.cols = cols
        self.offset = offset

        self.col = col

    def __getitem__(self, key):
        assert type(key) in (int, Index), "Index must be an integer or an Index object"

        # Gets a column
        assert(self.cols > 1) # Can't get a column of a single vector

        newvar = copy.copy(self)
        newvar.col = key
        return newvar

    def __str__(self):
        if self.cols > 1:
            if self.col is None: # Return the whole matrix as a vector
                return f"{self.name}Mat()"
            else:
                return f"{self.name}({self.col})" # Return the requested column
        else:
            return f"{self.name}()"

    def gen_define(self, size = False, offset = False, func = False):
        # Set to true the element to generate
        # Generate var(i) const function
        if size:
            cog.outl(f"static constexpr auto s{self.name} = {self.rows};")

            if self.cols > 1:
                cog.outl(f"static constexpr auto s{self.name}Mat = {self.rows * self.cols};")


        if self.cols == 1:
            if offset:
                cog.outl(f"constexpr auto o{self.name}() {{return {self.offset};}};")
            if func:
                cog.outl(f"constexpr auto  {self.name}() {{return x.template segment<s{self.name}>(o{self.name}());}};")
        else:
            # Accessors for a column of the variable
            if offset:
                cog.outl(f"constexpr auto o{self.name}(int col) {{return {self.offset}+{self.rows}*col;}};")
            if func:
                cog.outl(f"constexpr auto  {self.name}(int col) {{return x.template segment<s{self.name}>(o{self.name}(col));}};")

            # Accessors for the whole matrix
            if offset:
                cog.outl(f"constexpr auto o{self.name}Mat() {{return {self.offset};}};")
            if func:
                cog.outl(f"constexpr auto  {self.name}Mat() {{return x.template segment<s{self.name}Mat>(o{self.name}Mat());}};")

    @property
    def offset_func(self):
        """Return the function to compute the offset"""
        if self.cols == 1:
            return f"o{self.name}()"
        if self.col is None:
            return f"o{self.name}Mat()"
        return f"o{self.name}({self.col})"

    @property
    def size_func(self):
        """Return the constant computing the size of the variable"""
        if self.cols == 1:
            return f"s{self.name}"    # There's only one column
        if self.col is None:
            return f"s{self.name}Mat" # Return the whole matrix
        return f"s{self.name}"        # Size of one column

    @property
    def offset(self):
        if self.col is None:
            return self.__offset
        else:
            return self.__offset + self.rows * self.col

    @offset.setter
    def offset(self, offset):
        self.__offset = offset

    # @property
    # def seg(self):
    #     # return a x.SEG(size, offset) form
    #     return f"x.SEG({self.rows}, {self.offset})"


class Constraint:
    # Evaluation of a Function
    # e.g. sys(X, U)
    # and associated constraint
    # Has a short-name that we can refer to with an index
    def __init__(self, function, *args):
        # name - [string] Short name used to refer to the constraint
        # function - [Function] to be evaluated
        # args - [*Variable] to be passed to the function
        #        If args contain an index, then this is a looped constraint

        self.name = None # Set later
        self.offset = None # Set later
        self.function = function
        
        # Extract index (assuming that there's only one)
        self.index = None
        for arg in args:
            assert(type(arg) == Variable)
            if type(arg.col) == Index:
                self.index = arg.col
        self.args = args

    def __eq__(self, other):
        # Add an equality constraint
        assert(other == 0)
        self.function.nlp.constraints.append(self)

    @property
    def total_size(self):
        if self.index is None:
            return self.function.size_output
        else:
            return self.function.size_output * self.index.num_iterations

    def gen_define(self, size = False, offset = False, func = False):
        # Produce short name for this constraint
        # Set to true the element to generate
        if size:
            cog.outl(f"static constexpr auto s{self.name} = {self.function.size_output};")

        if self.index is None:
            if offset:
                cog.outl(f"constexpr auto o{self.name}() {{return {self.offset};}};")
            if func:
                cog.outl(f"constexpr auto  {self.name}() {{return g.template segment<s{self.name}>(o{self.name}());}};")
        else:
            if offset:
                cog.outl(f"constexpr auto o{self.name}(int ind) {{return {self.offset}+{self.function.size_output}*ind;}};")
            if func:
                cog.outl(f"constexpr auto  {self.name}(int ind) {{return g.template segment<s{self.name}>(o{self.name}(ind));}};")

    def __str__(self):
        # Return the short name for the constraint
        if self.index is None:
            return f"{self.name}()"
        else:
            return f"{self.name}(i)"

    def gen_eval(self):
        f = self.function
        if self.index is not None:
            idx = self.index
            cog.outl(f"\tfor(int i={idx.rng.start}; i<{idx.rng.stop}; i++)")
            cog.out("\t")
        cog.outl(f"\t{f.name}(param, {str(self)}, " + ", ".join(map(str, self.args)) + f");")
        return f.size_output

    def gen_eval_jacobian(self, offset):
        """Generate Jacobian evaluation"""

        pre = ""
        f = self.function
        num_iterations = 1
        offsetFunc = f"o{self.name}()"
        if self.index is not None:
            rng = self.index.rng
            cog.outl(f"\tfor(int i={rng.start}; i<{rng.stop}; i++)")
            pre = "\t"
            num_iterations = self.index.num_iterations
            offsetFunc = f"o{self.name}(i)"
        cog.out(f"{pre}\t")
        func_name = f"{f.name}("
        pre = pre + "\t" + " " * len(func_name)
        cog.outl(func_name + f"param, {str(self)}, " + ", ".join(str(v) for v in self.args) + ", ")
        Jargs = ", ".join(f"J.BLK(s{self.name},{v.size_func},{offsetFunc},{v.offset_func})" for v in self.args)
        cog.outl(pre + Jargs + ");")
        return f.size_output * num_iterations

    def gen_eval_hessian(self):
        """Generate exact Hessian evaluation"""

        def H_name(rVar,cVar):
            return f"H_{rVar.name}{cVar.name}"

        def H_blk(rowVar, colVar): 
            return f"Ref<Matrix<Scalar,{rowVar.size_func},{colVar.size_func}>> {H_name(rowVar,colVar)} = " + \
                   f"hessian_f.BLK({rowVar.size_func},{colVar.size_func},{rowVar.offset_func},{colVar.offset_func})"

        for rVar in self.args:
            for cVar in self.args:
                cog.outl("\t" + H_blk(rVar,cVar) + ";")
            args = ",".join(H_name(rVar,cVar) for cVar in self.args)
            cog.outl("\t" + f"auto H_{rVar.name} = forward_as_tuple(" + args + f"); // Row {rVar.name} of the hessian")
            cog.outl()

        for v in self.args:
            cog.out("\t" + f"Ref<Matrix<Scalar,{v.size_func},1>> grad_{v.name} = ")
            cog.outl(f"gradient_f.SEG({v.size_func},{v.offset_func});")
        cog.outl()


        name = f"\t{self.function.name}"
        pre = "\t" + " " * len(name)
        cog.outl(f"{name}(param, f,")

        # Variables
        cog.outl(pre + ", ".join(str(v) for v in self.args) + ", ")  

        # Gradients
        cog.outl(pre + ", ".join(f"grad_{v.name}" for v in self.args) + ",")

        # Hessian
        cog.outl(pre + ", ".join(f"H_{v.name}" for v in self.args) + ");")


class NLP:
    def __init__(self):
        # List of vars and functions
        self.vars = []
        self.num_vars = 0
        self.constraints = [] # Evaluations of functions
        self.functions = [] # List of defined functions
        self.objective = None

    def var(self, name, n, m = 1, lb=None, ub=None):
        var = Variable(name, self.num_vars, n, m)
        self.vars.append(var)
        self.num_vars = self.num_vars + n * m
        return var

    def finalize_constraints(self):
        # Set constraint names and offsets into the global variable
        offset = 0
        for (num, con) in enumerate(self.constraints):
            con.name = f"c{num}"
            con.offset = offset
            offset = offset + con.total_size

    def generate(self):
        self.finalize_constraints()

        cog.outl("// Bring NLP names into this namespace")
        cog.outl("using Base = NLP< MyNLP<Scalar, Traits> >;")
        cog.outl("using Base::x;")
        cog.outl("using Base::J;")
        cog.outl("using Base::g;")
        cog.outl("using Base::f;")
        cog.outl("using Base::gradient_f;")
        cog.outl("using Base::hessian_f;")
        cog.outl()

        cog.outl("// Define variables data and accessors")
        cog.outl("// Sizes")
        [var.gen_define(size=True) for var in self.vars]
        cog.outl("// Offsets")
        [var.gen_define(offset=True) for var in self.vars]
        cog.outl("// Accessor")
        [var.gen_define(func=True) for var in self.vars]
        cog.outl()

        cog.outl("// Define short names for constraints")
        cog.outl("// Sizes")
        [con.gen_define(size=True) for con in self.constraints]
        cog.outl("// Offsets")
        [con.gen_define(offset=True) for con in self.constraints]
        cog.outl("// Accessor")
        [con.gen_define(func=True) for con in self.constraints]
        cog.outl()

        cog.outl("// Instantiate function, jacobians and hessian")
        for f in self.functions:
            if f == self.objective.function:
                continue
            f.instantiate()
        self.objective.function.instantiate(hessian = True)
        cog.outl()

        cog.outl("// Evaluate constraints")
        cog.outl("inline void eval()\n{")
        for con in self.constraints:
            con.gen_eval()
        cog.outl("}\n")

        cog.outl("// Evaluate jacobians")
        cog.outl("inline void eval_jacobian()\n{")
        offset = 0
        for con in self.constraints:
            offset = offset + con.gen_eval_jacobian(offset)
        cog.outl("}\n")

        cog.outl("// Evaluate hessian of objective")
        cog.outl("inline void eval_objective()\n{")
        self.objective.gen_eval_hessian();
        cog.outl("}\n")


    def generate_traits(self, class_name = "MyTraits"):
        # Produce a traits class with the required sizes
        cog.outl(f"struct {class_name}")
        cog.outl("{")
        cog.outl("    enum {")
        cog.outl(f"        num_vars = {self.num_vars},")
        cog.outl(f"        num_eq = {sum(con.total_size for con in self.constraints)}")
        cog.outl("    };")
        cog.outl("};")

    def scalar_function(self, function_name, *input_types):
        f = Function(self, function_name, None, *input_types)
        self.functions.append(f)
        return f

    def function(self, function_name, size_output, *input_types):
        """Define a vector valued function"""
        f = Function(self, function_name, size_output, *input_types)
        self.functions.append(f)
        return f

    def const(self, name, val):
        if type(val) == int:
            t = "int"
        elif type(val) == float:
            t = "Scalar"
        cog.outl(f"constexpr {t} {name} = {val};")
        return val

    def minimize(self, objective):
        self.objective = objective


class Index:
    def __init__(self, rng, op = 'i'):
        self.op = op
        self.rng = rng

    @property
    def num_iterations(self):
        # Compute the number of iterations that this index represents 
        # i.e., max(i) - min(i)
        return self.rng.stop - self.rng.start

    def __str__(self):
        return self.op

    def makeop(self, other, op):
        if op in ('+', '-'):
            return Index(self.rng, f"({str(self)}{op}{str(other)})")
        else:
            return Index(self.rng, f"{str(self)}{op}{str(other)}")

    def __add__(self, other):
        return self.makeop(other, '+')

    def __sub__(self, other):
        return self.makeop(other, '-')

    def __mul__(self, other):
        return self.makeop(other, '*')

    def __div__(self, other):
        return self.makeop(other, '/')

    def __radd__(self, other):
        return self.makeop(other, '+')

    def __rsub__(self, other):
        return self.makeop(other, '-')

    def __rmul__(self, other):
        return self.makeop(other, '*')

    def __rdiv__(self, other):
        return self.makeop(other, '/')


class Function:
    def __init__(self, nlp, function_name, size_output, *input_types):
        self.nlp = nlp
        self.name = function_name
        self.size_output = size_output # If None, then this is a scalar-output function
        self.input_types = input_types

        self.generate_signature()

    @staticmethod
    def var_sig(var):
        # Return Vec<T, n> varname or Mat<T, n, m> varname
        # var = (varname, n) or (varname, n, m)
        if len(var) == 2:
            return f"Vec<T,{var[1]}> {var[0]}" 
        elif len(var) == 3:
            return f"Mat<T,{var[1]},{var[2]}> {var[0]}" 
        else:
            raise TypeError("Input variable description wrong")

    def generate_signature(self):
        # Generate function signature
        inputs = ", ".join(f"RC{self.var_sig(var)}" for var in self.input_types)
        if self.size_output is None:
            outputs = "T& out"
        else:
            outputs = f"R{self.var_sig(('out', self.size_output))}"
        params = "param_t& param"

        cog.outl(f"template <typename T, typename param_t>")
        cog.outl(f"inline void _{self.name}(" + ", ".join([params, outputs, inputs]) + ")")

    def __call__(self, *args):
        return Constraint(self, *args)

    def instantiate(self, hessian=False):
        arg_sizes = ", ".join(f"{var[1]}" for var in self.input_types)
        if hessian == False:
            assert self.size_output is not None, "Cannot generate jacobian for scalar function (use a vector valued function of length 1)"
            cog.outl(f"make_jacobian({self.name}, {self.size_output}, {arg_sizes});")
        else:
            assert self.size_output is None, "Can only generate hessian for scalar functions"
            cog.outl(f"make_hessian({self.name}, {arg_sizes});")
