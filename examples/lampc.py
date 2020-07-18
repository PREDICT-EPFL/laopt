import cog

class Variable:
    def __init__(self, name, offset, rows, cols):
        self.name = name
        self.rows = rows
        self.cols = cols
        self.offset = offset

    def __getitem__(self, key):
        assert(type(key) == int)

        return Variable(self.name + str(key), 
            self.offset + key * self.rows,
            self.rows, 1)

    def to_offset(self, name = "primal"):
        return f'Base::{name}.template segment<{self.rows}>({self.offset})'

    def __repr__(self):
        return f"{self.name}({self.rows}, {self.cols}, +{self.offset})"


class Constraint:
    def __init__(self, 
                func_name, # C++ function name
                offset,    # Offset into g(x)
                size_f,    # Number of outputs
                vars):     # List of variables to be passed to the function
        self.func_name = func_name
        self.offset = offset
        self.size_f = size_f
        for var in vars:
            if var.cols > 1:
                print("Error: Can only pass a vector to functions - not a matrix")
                print(var)
            assert var.cols == 1, "Can only pass a vector to functions - not a matrix"
        self.vars = vars

    def gen_eval(self, 
             name_assign): # Variable name to assign to
        cog.out(f'Base::{name_assign}.template segment<{self.size_f}>({self.offset}) = ')
        cog.outl(f'{self.func_name}<double>(')
        for (i, var) in enumerate(self.vars):
            cog.out(" " * 4 + var.to_offset("primal"))
            if i < len(self.vars) - 1:
                cog.out(",")
            else:
                cog.out(");")
            cog.outl(" " * 2 + f'// {var.name}')

    def __wrt(self, vars, var_name = "primal_d", pre_line = "", term_string = ","):
        # Create the structure
        #
        # Base::primal_d.template segment<2>(6),   // x1
        # pre_line + Base::primal_d.template segment<2>(4),   // x0
        # pre_line + Base::primal_d.template segment<1>(0) + term_string  // u0
        #
        # pre_line = text added to the start of every line
        # func_name = name - at - in the example above

        at = ""
        sep = ""
        for (i, var) in enumerate(vars):
            at = at + sep + var.to_offset(var_name)
            sep = pre_line
            if i < len(vars) - 1:
                at = at + ", "
            else:
                at = at + term_string
            at = at + "  // " + var.name
            if i < len(vars) - 1:
                at = at + "\n" + pre_line

        return at



    def gen_jacobian(self, name_assign):
        # Output format:

        # J_eq.block<2,3>(3,4) = jacobian(DoubleIntegrator<dual>::dynamics, 
        #                                 wrt(primal.segment<2>(0)),
        #                                 at(primal.segment<2>(0), 
        #                                    primal.segment<2>(3), 
        #                                    primal.segment<1>(4)))        


        indent = " " * 4
        for var in self.vars:
           cog.out(f'Base::{name_assign}.template block<{self.size_f}, {var.rows}>')
           cog.out(f'({self.offset}, {var.offset}) = ')
           cog.outl(f'jacobian(')
           cog.outl(indent + f'{self.func_name}<dual>,')
           cog.outl(indent + "wrt(" + self.__wrt([var], "primal_d", indent, '),'))
           cog.outl(indent + "at(" + self.__wrt(self.vars, "primal_d", indent, ');'))


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

    def add_equality(self, func_name, size_f, vars):
        self.eq.append(Constraint(func_name, self.num_eq, size_f, vars))
        self.num_eq = self.num_eq + size_f

    def gen_eval_eq(self, name_assign):
        for con in self.eq:
            con.gen_eval(name_assign)

    def gen_eval_jacobian_eq(self, name_assign):
        for con in self.eq:
            con.gen_jacobian(name_assign)

    def traits(self): # Produce traits structure
        cog.outl(f'num_vars = {self.num_vars},') 
        cog.outl(f'num_eq   = {self.num_eq},')
        cog.outl(f'num_ineq = {self.num_ineq}')
