import cog
import copy

# TODOS
# - assert no spaces in any names

# dualvar = lambda var: 'D' + str(var)

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
        cog.outl('#define SEG(size, offset) template segment<size>(offset)')
        cog.outl('#define BLK(x_size, y_size, x_offset, y_offset) template block<x_size, y_size>(x_offset, y_offset)')
        for var in self.vars:
            cog.outl(var.gen_define())

    # def add_equality(self, name, func_name, size_f, vars):
    #     self.eq.append(Constraint(name, func_name, self.num_eq, size_f, vars))
    #     self.num_eq = self.num_eq + size_f

    # def add_inequality(self, name, func_name, size_f, vars, lb, ub):
    #     if size_f > 1:
    #         assert len(lb) == len(ub), "len(lb) must equal len(ub)"
    #         assert len(lb) == size_f, "len(lb) must equal size_f"
    #     self.ineq.append(Constraint(name, func_name, self.num_ineq, size_f, vars, lb, ub))
    #     self.num_ineq = self.num_ineq + size_f

    # def gen_eval_eq(self):
    #     cog.outl('// Equality constraints')
    #     for con in self.eq:
    #         cog.outl(f'#define {con.name} {con.to_offset("g_eq")}')

    #     cog.outl()
    #     for con in self.eq:
    #         con.gen_eval("g_eq")

    # def gen_eval_jacobian(self, eq_name, ineq_name):
    #     cog.outl('// Derivative variables')
    #     for var in self.vars:
    #         for col in range(var.cols):
    #             v = var[col]
    #             cog.outl(f'#define {dualvar(v)} {v.to_offset("primal_d")}')
    #     cog.outl()

    #     cog.outl('// Equalities')
    #     for con in self.eq:
    #         con.gen_jacobian_defines(eq_name)
    #     for con in self.eq:
    #         con.gen_jacobian()

    #     cog.out('\n')
    #     cog.outl('// Inequalities')
    #     for con in self.ineq:
    #         con.gen_jacobian_defines(ineq_name)
    #     for con in self.ineq:
    #         con.gen_jacobian()

    # def gen_traits(self): # Produce traits structure
    #     cog.outl(f'num_vars = {self.num_vars},') 
    #     cog.outl(f'num_eq   = {self.num_eq},')
    #     cog.outl(f'num_ineq = {self.num_ineq}')


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
    def __init__(self, nlp, class_name):
        self.functions = []
        self.nlp = nlp
        self.class_name = class_name

    def append(self, function_name, size_output, vars, index = None):
        # function_name - C++ function call
        # size_output - size of output
        # vars - list of vars to call with
        self.functions.append(Function(function_name, size_output, vars, index))

    def gen(self):
        self.gen_sizes()

        cog.outl(f"{self.class_name}" + "() {")
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
        cog.outl(f"void eval(Ref<Matrix<Scalar, nvars, 1>> {var_name})")
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
        cog.outl(f"void eval_jacobian(Ref<Matrix<dual, nvars, 1>> {var_name})")
        cog.outl("{")
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
                cog.out(f"{func.name}<dual>, ")
                cog.out("wrt(" + str(var) + "), ")
                cog.outl('at(' + ",".join(map(str, func.vars)) + "));")
            if func.index is not None:
                cog.outl("\t}")
            offset = offset + func.total_size
        cog.outl("}")
        cog.outl()


# class Constraint:
#     def __init__(self, 
#                 name,      # Descriptor for short-name
#                 func_name, # C++ function name
#                 offset,    # Offset into g(x)
#                 size_f,    # Number of outputs
#                 vars,      # List of variables to be passed to the function
#                 lb = None, # Bounds - used for inequalities only
#                 ub = None):
#         self.name = name
#         self.func_name = func_name
#         self.offset = offset
#         self.size_f = size_f
#         self.lb = lb
#         self.ub = ub
#         for var in vars:
#             if var.cols > 1:
#                 print("Error: Can only pass a vector to functions - not a matrix")
#                 print(var)
#             assert var.cols == 1, "Can only pass a vector to functions - not a matrix"
#         self.vars = vars

    # def to_offset(self, name_assign):
    #     # Generate the descriptor for this 
    #     # Base::{name_assign}.template segment<{self.size_f}>({self.offset})
    #     return f'Base::{name_assign}.template segment<{self.size_f}>({self.offset})'

    # def gen_eval(self, name_assign): # Variable name to assign to
    #     cog.out(f'{self.name} = {self.func_name}<double>')
    #     cog.outl('(' + ",".join(map(str, self.vars)) + ");")

    # def gen_jacobian_defines(self, name_assign):
    #     for var in self.vars:
    #         cog.out(f'#define J_{self.name}_{var.name} ')
    #         cog.out(f'Base::{name_assign}.template block<{self.size_f}, {var.rows}>')
    #         cog.outl(f'({self.offset}, {var.offset})')

    # def gen_jacobian(self):
    #     for var in self.vars:
    #         cog.out(f'J_{self.name}_{var.name} = ')
    #         cog.out(f'jacobian(')
    #         cog.out(f'{self.func_name}<dual>, ')
    #         cog.out("wrt(" + dualvar(var) + "), ")
    #         cog.outl('at(' + ",".join(map(dualvar, self.vars)) + "));")
