import cog
import copy

# TODOS

# Critical path
# pass in matrix variables (for the cost function)
# add hessian for cost
# update RHS of equalities and inequalities
# create a QP version of the NLP and attach to QP solver
# hook up the NLP to IPOPT 
# get and set the variables by name. Generate code for this where?
# change all defines to constexpr

# Nice to have
# assert no spaces in any names
# check that all the "functions" classes have the right member functions
# think on how to generalize to eigen AD
# check that if a function is called with different variables, that the variable sizes match
# allow specification of variable bounds?
# add a bunch of auto-generated constraints via python? (A*x <= b) type stuff?
# change back to the CRTP format
# allow for some of the jacobians to be specified manually and only generate the missing bits

class Variable:
    def __init__(self, name, offset, rows, cols=1, col=0):
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
            return f"{self.name}({self.col})"
        else:
            return f"{self.name}()"
        # assert self.cols == 1, "Can only generate with vectors"
        # return self.name

    def gen_define(self, var_name = "x"):
        # Generate var(i) const function
        if self.cols == 1:
            return f"constexpr auto {self.name}() {{return {var_name}.SEG({self.rows}, {self.offset});}};"
        else:
            return f"constexpr auto {self.name}(int col) {{return {var_name}.SEG({self.rows}, {self.offset} + {self.rows} * col);}};"

    @property
    def offset(self):
        return self.__offset + self.rows * self.col

    @offset.setter
    def offset(self, offset):
        self.__offset = offset

    @property
    def seg(self):
        # return a x.SEG(size, offset) form
        return f"x.SEG({self.rows}, {self.offset})"

class NLP:
    def __init__(self):
        # List of vars and constraints
        self.vars = []
        self.num_vars = 0
        self.constraints = Functions(self)

    def var(self, name, n, m = 1, lb=None, ub=None):
        var = Variable(name, self.num_vars, n, m)
        self.vars.append(var)
        self.num_vars = self.num_vars + n * m
        return var

    def generate(self):
        # Produce short names for everything we're going to use
        cog.outl("using Base = NLP< MyNLP<Scalar, Traits> >;")
        cog.outl("using Base::x;")
        cog.outl("using Base::J;")
        cog.outl("using Base::g;")
        for var in self.vars:
            cog.outl(var.gen_define())
        # self.generate_eval() 

    def generate_eval(self):
        # Generate the "evaluation" functions

        # Evaluation of constraints
        cog.outl("inline void eval()")
        cog.outl("{")
        cog.outl("\t")
        cog.outl(")")


    def generate_traits(self, class_name = "MyTraits"):
        # Produce a traits class with the required sizes
        cog.outl(f"struct {class_name}")
        cog.outl("{")
        cog.outl("    enum {")
        cog.outl(f"        num_vars = {self.num_vars},")
        cog.outl(f"        num_eq = {self.constraints.get_num_functions()}")
        cog.outl("    };")
        cog.outl("};")

    def begin_func(self, name, varnames):
        self.constraints.begin_func(name, varnames)
    
    def end_func(self, name):
        self.constraints.end_func(name)

    def equality(self, function_name, size_output, vars, index = None):
        self.constraints.append("equality", function_name, size_output, vars, index)

    def inequality(self, function_name, size_output, vars, index=None, lb=None, ub=None):
        self.constraints.append("inequality", function_name, size_output, vars, index)

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
    def __init__(self, function_type, function_name, size_output, vars, index = None):
        self.function_type = function_type
        self.name = function_name
        self.size_output = size_output
        self.vars = vars
        self.index = index

    @property
    def total_size(self):
        if self.index is None:
            return self.size_output
        else:
            rng = self.index.rng
            return self.size_output * (rng.stop - rng.start)

    def gen_sig(self, varnames):
        # Generate function signature
        cog.outl(f"template <typename T> struct {self.name}_t {{")
        func_name = f"inline void operator()"
        cog.out(func_name)
        inputs = ", ".join(f"RCVec<T,{v.rows}> {vname}" for (vname, v) in zip(varnames, self.vars))
        outputs = f"RVec<T,{self.size_output}> out"
        params = "param_t& param"
        cog.outl("(\n\t" + ",\n\t".join([inputs, outputs, params]) + ")")

    def instantiate(self):
        arg_sizes = ", ".join(str(v.rows) for v in self.vars)
        cog.outl(f"Jacobian<{self.name}_t, Scalar, param_t, {self.size_output}, {arg_sizes}> J_{self.name};")
        cog.outl(f"{self.name}_t<Scalar> {self.name};")

    def gen_eval(self, offset):
        offset_str = str(offset)
        if self.index is not None:
            idx = self.index
            cog.outl(f"\tfor(int i={idx.rng.start}; i<{idx.rng.stop}; i++)")
            cog.out("\t")
            offset_str = f"{offset}+i*{self.size_output}"
        output = f"g.SEG({self.size_output},{offset_str})"
        cog.outl(f"\t{self.name}(" + ", ".join(map(str, self.vars)) + f", {output}, param);")
        return self.size_output

        # J_dynamics(x.SEG(2,0), x.SEG(2,2), x.SEG(1,4),
        #     g.SEG(2,0), 
        #     J.BLK(2,2,0,0), J.BLK(2,2,0,2), J.BLK(2,1,0,4));


    def gen_eval_jacobian(self, offset):
        pre = ""
        offset_str = str(offset)
        num_iterations = 1
        if self.index is not None:
            idx = self.index
            cog.outl(f"\tfor(int i={idx.rng.start}; i<{idx.rng.stop}; i++)")
            pre = "\t"
            offset_str = f"{offset}+i*{self.size_output}"
            num_iterations = self.index.num_iterations
        cog.out(f"{pre}\t")
        func_name = f"J_{self.name}("
        pre = pre + "\t" + " " * len(func_name)
        cog.out(func_name)
        cog.outl(f"g.SEG({self.size_output}, {offset_str}),")
        args = ", ".join(str(v) for v in self.vars)
        cog.outl(pre + args + ",")
        Jargs = ", ".join(f"J.BLK({self.size_output},{v.rows},{offset_str},{v.offset})" for v in self.vars)
        cog.outl(pre + Jargs + f",\n{pre}param"");")
        return self.size_output * num_iterations

    def gen_jacobian(self, varnames, nvars):
        # Generate a set of functions returning the Jacobian wrt each vector var

        # Function sig
        func_name = f"inline void J_{self.name}("
        pre = ",\n" + (" " * len(func_name))
        J_args = pre.join(f"Ref<Matrix<Scalar, {self.size_output}, {v.rows}>> J_{vname}" for (vname, v) in zip(varnames, self.vars))
        args = pre.join(f"const Ref<const Matrix<Scalar, {v.rows}, 1>> {vname}" for (vname, v) in zip(varnames, self.vars))
        cog.outl(f"{func_name}{args}{pre}Ref<Matrix<Scalar, {self.size_output}, 1>> val{pre}{J_args})")
        cog.outl("{")

        # Declare AD variables
        num_inputs = sum(v.rows for v in self.vars)
        cog.outl(f"\tusing input_t = Matrix<Scalar, {num_inputs}, 1>;")
        cog.outl(f"\tusing ADScalar = AutoDiffScalar<input_t>;")
        cog.outl(f"\tMatrix<ADScalar, {num_inputs}, 1> _x;")
        cog.outl(f"\tMatrix<ADScalar, {self.size_output}, 1> _out;")

        cog.outl("\t// Copy current value into dual variables")
        seg_names = []
        row = 0
        for (vname, v) in zip(varnames, self.vars):
            seg_names.append(f"_x.SEG({v.rows},{row})")
            row = row + v.rows
            cog.outl(f"\t{seg_names[-1]} = {vname};")

        cog.outl("\t// Compute the Jacobian")
        cog.outl("\tAD_seed(_x);")
        cog.outl(f"\tthis->{self.name}<ADScalar>({', '.join(seg_names)}, _out);")

        cog.outl(f"\tfor(int i=0; i<{self.size_output}; i++) // Copy Jacobian into output variables")
        cog.outl("\t{")
        cog.outl("\t\tval(i) = _out[i].value();")
        cog.outl("\t\tRef<input_t> deriv = _out[i].derivatives();")
        row = 0
        for (vname, v) in zip(varnames, self.vars):
            cog.outl(f"\t\tJ_{vname}.row(i) = deriv.SEG({v.rows},{row});")
            row = row + v.rows
        cog.outl("\t}")

        cog.outl("};")
        cog.outl()


class Functions:
    def __init__(self, nlp):
        self.functions = []
        self.nlp = nlp

    def append(self, function_type, function_name, size_output, vars, index = None):
        # function_type - "equality" or "inequality"
        # function_name - C++ function call
        # size_output - size of output
        # vars - list of vars to call with
        self.functions.append(Function(function_type, function_name, size_output, vars, index))

    def __iter__(self):
        return iter(self.functions)

    def get_num_functions(self):
        if not self.functions:
            return 0
        return sum([func.total_size for func in self.functions])

    def gen_sizes(self):
        # Produce the nvars and nfuncs lines
        cog.outl("enum {")
        nfunc = sum([func.total_size for func in self.functions])
        cog.outl(f'\tnfuncs = {nfunc},')
        cog.outl(f'\tnvars = {self.nlp.num_vars}')
        cog.outl("};")
        cog.outl()
        cog.outl("Eigen::Matrix<Scalar, nfuncs, nvars> J; // Jacobian of function")
        cog.outl("Eigen::Matrix<Scalar, nfuncs, 1>     f; // Value of function")
        cog.outl()

    def gen_eval(self):
        # Evaluate the functions given the variable var_name into the vector func_name
        cog.outl(f"inline void eval()")
        cog.outl("{")
        offset = 0
        for func in self.functions:
            offset = offset + func.gen_eval(offset)
        cog.outl("}")
        cog.outl()

    def gen_jacobian(self):
        cog.outl("inline void eval_jacobian()")
        cog.outl("{")

        offset = 0
        for func in self.functions:
            offset = offset + func.gen_eval_jacobian(offset)
        cog.outl("}")
        cog.outl()

    def begin_func(self, name, varnames):
        # Generates the functor signature for the named function
        f = next(f for f in self.functions if f.name == name)
        f.gen_sig(varnames)

    def end_func(self, name):
        cog.outl("};")

        # Instantiates the jacobian and functor
        f = next(f for f in self.functions if f.name == name)
        f.instantiate()
