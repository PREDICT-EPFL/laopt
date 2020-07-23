import cog

# TODOS
# - assert no spaces in any names

dualvar = lambda var: 'D' + str(var)

class Variable:
    def __init__(self, name, offset, rows, cols):
        self.name = name
        self.rows = rows
        self.cols = cols
        self.offset = offset

    def __getitem__(self, key):
        assert(type(key) == int)

        if self.cols == 1:
            return self
        else:
            return Variable(self.name + str(key), 
                self.offset + key * self.rows,
                self.rows, 1)

    def to_offset(self, name = ""):
        # return f'VAR){self.offset},{self.rows})'
        return f'Base::{name}.template segment<{self.rows}>({self.offset})'

    def __str__(self):
        assert self.cols == 1, "Can only generate with vectors"
        return self.name

    def __repr__(self):
        return f"{self.name}({self.rows}, {self.cols}, +{self.offset})"


class Constraint:
    def __init__(self, 
                name,      # Descriptor for short-name
                func_name, # C++ function name
                offset,    # Offset into g(x)
                size_f,    # Number of outputs
                vars,      # List of variables to be passed to the function
                lb = None, # Bounds - used for inequalities only
                ub = None):
        self.name = name
        self.func_name = func_name
        self.offset = offset
        self.size_f = size_f
        self.lb = lb
        self.ub = ub
        for var in vars:
            if var.cols > 1:
                print("Error: Can only pass a vector to functions - not a matrix")
                print(var)
            assert var.cols == 1, "Can only pass a vector to functions - not a matrix"
        self.vars = vars

    def to_offset(self, name_assign):
        # Generate the descriptor for this 
        # Base::{name_assign}.template segment<{self.size_f}>({self.offset})
        return f'Base::{name_assign}.template segment<{self.size_f}>({self.offset})'

    def gen_eval(self, name_assign): # Variable name to assign to
        cog.out(f'{self.name} = {self.func_name}<double>')
        cog.outl('(' + ",".join(map(str, self.vars)) + ");")

    def gen_jacobian_defines(self, name_assign):
        for var in self.vars:
            cog.out(f'#define J_{self.name}_{var.name} ')
            cog.out(f'Base::{name_assign}.template block<{self.size_f}, {var.rows}>')
            cog.outl(f'({self.offset}, {var.offset})')

    def gen_jacobian(self):
        # Output format:

        # J_eq.block<2,3>(3,4) = jacobian(DoubleIntegrator<dual>::dynamics, 
        #                                 wrt(primal.segment<2>(0)),
        #                                 at(primal.segment<2>(0), 
        #                                    primal.segment<2>(3), 
        #                                    primal.segment<1>(4)))        

        for var in self.vars:
            cog.out(f'J_{self.name}_{var.name} = ')
            cog.out(f'jacobian(')
            cog.out(f'{self.func_name}<dual>, ')
            cog.out("wrt(" + dualvar(var) + "), ")
            cog.outl('at(' + ",".join(map(dualvar, self.vars)) + "));")


class NLP:
    def __init__(self):
        # List of vars and constraints
        self.vars = []
        self.ineq = []
        self.eq = []

        self.num_vars = 0
        self.num_eq = 0
        self.num_ineq = 0

    def var(self, name, n, m = 1):
        var = Variable(name, self.num_vars, n, m)
        self.vars.append(var)
        self.num_vars = self.num_vars + n * m
        return var

    def add_equality(self, name, func_name, size_f, vars):
        self.eq.append(Constraint(name, func_name, self.num_eq, size_f, vars))
        self.num_eq = self.num_eq + size_f

    def add_inequality(self, name, func_name, size_f, vars, lb, ub):
        if size_f > 1:
            assert len(lb) == len(ub), "len(lb) must equal len(ub)"
            assert len(lb) == size_f, "len(lb) must equal size_f"
        self.ineq.append(Constraint(name, func_name, self.num_ineq, size_f, vars, lb, ub))
        self.num_ineq = self.num_ineq + size_f

    def gen_eval_eq(self):
        cog.outl('// Equality constraints')
        for con in self.eq:
            cog.outl(f'#define {con.name} {con.to_offset("g_eq")}')

        cog.outl()
        for con in self.eq:
            con.gen_eval("g_eq")

    def gen_eval_jacobian(self, eq_name, ineq_name):
        cog.outl('// Derivative variables')
        for var in self.vars:
            for col in range(var.cols):
                v = var[col]
                cog.outl(f'#define {dualvar(v)} {v.to_offset("primal_d")}')
        cog.outl()

        cog.outl('// Equalities')
        for con in self.eq:
            con.gen_jacobian_defines(eq_name)
        for con in self.eq:
            con.gen_jacobian()

        cog.out('\n')
        cog.outl('// Inequalities')
        for con in self.ineq:
            con.gen_jacobian_defines(ineq_name)
        for con in self.ineq:
            con.gen_jacobian()

    def gen_traits(self): # Produce traits structure
        cog.outl(f'num_vars = {self.num_vars},') 
        cog.outl(f'num_eq   = {self.num_eq},')
        cog.outl(f'num_ineq = {self.num_ineq}')

    def gen_variables(self):
        # Produce short names macros for everything we're going to use
        cog.outl('// Variables')
        for var in self.vars:
            for col in range(var.cols):
                v = var[col]
                cog.outl(f'#define {v.name} {v.to_offset("primal")}')
        cog.outl()
