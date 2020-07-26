import cog
import copy

# TODOS
# assert no spaces in any names
# check that all the "functions" classes have the right member functions
# pass in matrix variables (for the cost function)
# think on how to generalize to eigen AD
# add hessian for cost
# hook up the NLP to IPOPT 
# create a QP version of the NLP and attach to QP solver
# update RHS of equalities and inequalities
# check that if a function is called with different variables, that the variable sizes match

class Variable:
    def __init__(self, name, offset, rows, cols, col=0):
        self.name = '_' + name
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
            return f"{self.name}"
        # assert self.cols == 1, "Can only generate with vectors"
        # return self.name

    def gen_define(self, var_name = "x"):
        # Generate #define var(i) index_offset
        if self.cols == 1:
            return f'#define {self.name} {var_name}.SEG({self.rows}, {self.offset})'
        else:
            return f'#define {self.name}(col) {var_name}.SEG({self.rows}, {self.offset} + {self.rows} * col)'

    @property
    def offset(self):
        return self.__offset + self.rows * self.col

    @offset.setter
    def offset(self, offset):
        self.__offset = offset


class NLP:
    def __init__(self):
        # List of vars and constraints
        self.vars = []
        self.num_vars = 0

    def var(self, name, n, m = 1):
        var = Variable(name, self.num_vars, n, m)
        self.vars.append(var)
        self.num_vars = self.num_vars + n * m
        return var

    def gen_variables(self):
        # Produce short names macros for everything we're going to use
        cog.outl('#define SEG(size, offset) template segment<size>(offset) // Segment of an Eigen vector')
        cog.outl('#define BLK(x_size, y_size, x_offset, y_offset) template block<x_size, y_size>(x_offset, y_offset) // Block of an eigen matrix')
        cog.outl("#define Vec(Scalar, size) Eigen::Matrix<Scalar, size, 1> // Eigen vector")
        cog.outl("#define RVec(Scalar, size) Eigen::Ref<Eigen::Matrix<Scalar, size, 1>> // Reference to eigen vector")
        for var in self.vars:
            cog.outl(var.gen_define())


class Index:
    def __init__(self, rng, op = 'i'):
        self.op = op
        self.rng = rng

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
    def __init__(self, function_name, size_output, vars, index = None):
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


class Functions:
    def __init__(self, nlp):
        self.functions = []
        self.nlp = nlp

    def append(self, function_name, size_output, vars, index = None):
        # function_name - C++ function call
        # size_output - size of output
        # vars - list of vars to call with
        self.functions.append(Function(function_name, size_output, vars, index))

    def gen(self):
        self.gen_sizes()

        cog.outl("void initialize() {")
        cog.outl("\tJ.setZero();")
        cog.outl("\tf.setZero();")
        cog.outl("}")
        cog.outl()

        self.gen_eval()
        self.gen_jacobian()

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

    def gen_eval(self, func_name = "f", var_name = "x"):
        # Evaluate the functions given the variable var_name into the vector func_name
        cog.outl(f"void eval(RVec(Scalar, nvars) {var_name})")
        cog.outl("{")
        offset = 0
        for func in self.functions:
            offset_str = str(offset)
            if func.index is not None:
                idx = func.index
                cog.outl(f"\tfor(int i={idx.rng.start}; i<{idx.rng.stop}; i++)")
                cog.out("\t")
                offset_str = f"{offset}+i*{func.size_output}"
            cog.out(f"\t{func_name}.SEG({func.size_output},{offset_str}) = ")
            cog.out(f"{func.name}<Scalar>")
            cog.outl('(' + ", ".join(map(str, func.vars)) + ");")
            offset = offset + func.total_size
        cog.outl("}")
        cog.outl()

    def gen_jacobian(self, jacobian_name = 'J', var_name = 'x'):
        # Evaluate the jacobian of the functions at the variable var_name 
        # into the vector jacobian_name
        cog.outl(f"void eval_jacobian(RVec(dual, nvars) {var_name})")
        cog.outl("{")

        # Produce lambda functions for all the member functions that we want to pass
        fnames = [f.name for f in self.functions]
        indices = [fnames.index(x) for x in set(fnames)]
        unique_funcs = [self.functions[i] for i in indices]

        cog.outl("\t// Lambda functions to capture object, so we can get pointers to member functions")
        for f in unique_funcs:
            args = ", ".join([f"RVec(dual, {v.rows}) x{i}" for (i,v) in enumerate(f.vars)])
            cog.outl(f"\tstatic const auto _{f.name} = [this]({args})")
            args = ", ".join([f"x{i}" for (i,v) in enumerate(f.vars)])
            cog.outl(f"\t\t{{return this->{f.name}<dual>({args});}};")
        cog.outl()

        cog.outl("\t// Compute Jacobians block-wise")
        offset = 0
        for func in self.functions:
            pre = ""
            offset_str = str(offset)
            if func.index is not None:
                idx = func.index
                cog.outl(f"\tfor(int i={idx.rng.start}; i<{idx.rng.stop}; i++)")
                cog.outl("\t{")
                pre = "\t"
                offset_str = f"{offset}+i*{func.size_output}"
            for var in func.vars:
                cog.out(f"{pre}\t{jacobian_name}.")
                cog.out(f"BLK({func.size_output},{var.rows},{offset_str},{var.offset}) = ")
                cog.out(f"jacobian(")
                cog.out(f"_{func.name}, ")
                cog.out("wrt(" + str(var) + "), ")
                cog.outl('at(' + ",".join(map(str, func.vars)) + "));")
            if func.index is not None:
                cog.outl("\t}")
            offset = offset + func.total_size
        cog.outl("}")
        cog.outl()

    def gen_sig(self, name, varnames):
        # Generates the function signature for the named function
        f = next(f for f in self.functions if f.name == name)

        cog.outl("template <typename T>")
        cog.out(f"inline Vec(T, {f.size_output}) {name}")
        args = (f"RVec(T, {v.rows}) {vname}" for (vname, v) in zip(varnames, f.vars))
        cog.out(f"({', '.join(args)})")

    # template <typename T>
    # inline StateType<T> dynamics(Ref<StateType<T>> xp,
    #                              Ref<StateType<T>> x,
    #                              Ref<InputType<T>> u)
