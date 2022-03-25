import numpy as np
import polypy as pp

N, n, m = 5, 2, 1

A = pp.matrix([[0, 0, 1, 2], [1, 0, 3, 4]], name='A')
B = pp.matrix([1, 0], name='B', constant=False)
c = pp.matrix([[1], [2]], name='c')

@pp.function
def dynamics(x: n, u: m):
    vel = (u * pp.matrix([1, 1])) @ (np.array([[1,1]]) @ x) # (x[0] + x[1])
    xdot = A @ pp.vstack(vel, (u), x[1]) + B @ u
    return xdot

def rk4(f, x,u):
    h = pp.Scalar(0.1, 'h')
    k1 = f(x, u)
    k2 = f(x + (h * 0.5) * k1, u)
    k3 = f(x + (h * 0.5) * k2, u)
    k4 = f(x + h * k3, u)
    return x + (h * 0.1667) * (k1 + 2 * k2 + 2 * k3 + k4)

@pp.function
def assign(x: n):
    return x

@pp.function
def sys_d(x: n, u: m):    
    t = assign(pp.vstack(u, x[1]*(x[1]+3*x[0])))
    return rk4(dynamics, t, u)

def test(a:3, b:2):
    t = 4*5
    return vstack(2*a,5*t[:,-1]*5)

# @pp.function
# def sys_0(u: m):
#     return sys_d(x0, u)

# opt = pp.NLP("MyProblem")

x_lb = np.array([-1, -2e20]).T
x = pp.variable("x", n, num_vars=N, lb=x_lb)
u = pp.variable("u", m, num_vars=N - 1)
xss = pp.variable('xss', n)
uss = pp.variable('uss', m)

x0 = pp.matrix(np.zeros(n), name="x0", constant=False)

# opt.add(x[1] == dynamics(x0, u[0]))
# opt.add(xss == dynamics(xss, uss))

# opt.add(x[i + 1] == dynamics(x[i], u[i]) for i in pp.Range(1, N - 1))
# for i in range(1, N-1):
#     opt.add(x[i + 1] == dynamics(x[i], u[i]))

# C = pp.matrix([1, 1]).T
# for i in range(1, N - 1):
#     opt.add(pp.Inequality(C @ x[i], lb=-1, ub=1))

def stage_cost(x: n, u: m):
    # Define stage cost
    q = pp.matrix([1, 1], name="q", constant=False)
    # return sum((q @ x) * x)
    return q[0] * x[0] * x[0] + q[1] * x[1] * x[1] + u[0] * u[0]

# print(sum(x[1]))

# print(sum((x[1]) * x[1]))

# opt.minimize(pp.summation(*map(lambda y: stage_cost(y[0] - xss, y[1] - uss), zip(x, u))))

# print(xss == dynamics(xss, uss))

# p = pp.EigenGenerator()
# # print((x[2]*x[3]*(3+np.array([[1],[1.3]])@u[0])).generate(p))

# prob = pp.Compiler("test")

# print(sys_d.expression)

import functools
from polypy.expression import *

class EigenGenerator:
    def __init__(self):
        self._options = dict()  # Options that can be queried during generation
        self.set_option('number_type', 'scalar_t')
        self.set_option('var_name', 'var')
        self.set_option('cast_constants', True)  # 
        self.set_option('diff_number_type', 'T')

        # Functions that need to be generated
        self._dependencies = set()

        # Atoms that need to be declared
        self._declarations = set()

        # Atoms that need to be initialized in the constructor
        self._initializations = set()

        # Different areas of the class being generated
        self._constructor = pp.preprint()  # in the constructor
        self._header = pp.preprint()  # top of the class
        self._body = pp.preprint()  # main body of the class
        self._footer = pp.preprint()  # after all standard defintions, but within the class
        self._postclass = pp.preprint()  # after the class

    def option(self, option_name):
        """Return the value of the option, or None"""
        return self._options.get(option_name, None)

    @property
    def var(self):
        return self.option('var_name')

    def set_option(self, name, value):
        """Set the given option

        Note: In the current implementation, this option will be unset 
        when leaving the current "with" block
        """
        self._options[name] = value

    def generate(self, expr, p=None):
        """Generate Eigen code to evalute the expression

        Auxillary code is written to p, and then a string is returned
        that evaluates the expression
        """
        if p == None:
            self.p = pp.preprint()
        else:
            self.p = p
        return self._generate(expr), str(self.p)

    def print_generate(self, expr):
        """For debugging - just print out the generated expression"""
        val, code = self.generate(expr)
        print(code)
        print(f"Result = {val}")

    @functools.singledispatchmethod
    def _generate(self, expr):
        raise NotImplementedError(f"Do not know how to generate Eigen code for a {type(expr)}")

    # Decorator to generate the arguments before calling the function
    def _generate_args(func):
        def inner(self, expr):
            return func(self, expr, [self._generate(arg) for arg in expr.args])
        return inner

    #### Generate a callable ####

    @_generate.register(pp.Function)
    def _(self, func):
        """Write the function definition"""
        
        args = ", ".join(f"({var.name},{len(var)})" for var in func.inputs)
        self._body(f"FUNCTION({func.name}, scalar_t, param_t, ({func.output.name}, {len(func.output)}), {args})")
        with self._body.function() as p:
            s, i = self.generate(func.expression, p)
            p(f"{func.output.name} = {s};")


    #### Basic Arithmetic ####

    @_generate.register(addExpression)
    @_generate_args
    def _(self, expr, args):
        return " + ".join(f"({arg})" for arg in args)

    @_generate.register(mulExpression)
    @_generate_args
    def _(self, expr, args):
        return " * ".join(f"({arg})" for arg in args)

    @_generate.register(matmulExpression)
    @_generate_args
    def _(self, expr, args):
        return f"({args[0]}) * ({args[1]})"

    #### Atoms ####

    @_generate.register(Variable)
    def _(self, expr):
        return str(expr)
        """Return an eigen statement to access this variable as an offset into var"""
        # if expr.var_set:
        #     return f"{str(expr.var_set.name)}.SEG({self.var}, {expr.ind})"
        # else:
        #     return f"{str(expr)}.SEG({self.var})"

    @_generate.register(Scalar)
    def _(self, expr):
        self._declarations.add(expr)
        return str(expr)

    @_generate.register(ConstScalar)
    def _(self, expr):
        if self.option('cast_constants'):
            scalar_t = self.option('diff_number_type')
            return f"static_cast<{scalar_t}>({expr.value})"
        return str(expr.value)

    @_generate.register(ConstMatrix)
    def _(self, expr):
        self._declarations.add(expr)
        M = expr.M
        if np.all(M == np.ravel(M)[0]):
            return f"Matrix<{self.option('number_type')}, {M.shape[0]}, {M.shape[1]}>::Constant({M[0][0]})"

        # p.add_dependency(expr, 'ConstantMatrix')  # Register this matrix for generation
        if self.option('cast_constants'):
            return f"{expr}.template cast<{self.option('number_type')}>()"
        return str(expr)

    @_generate.register(Matrix)
    def _(self, expr):
        self._declarations.add(expr)
        self._initializations.add(expr)

        if self.option('cast_constants'):
            return f"{str(expr)}.template cast<{self.option('number_type')}>()"
        return str(expr)


    #### Slicing ####

    @_generate.register(VStack)
    @_generate.register(HStack)
    @_generate_args
    def _(self, expr, args):
        # Stack the arguments vertically
        tmp = pp._get_unique_name()
        shape = expr.shape
        self.p(f"Matrix<{self.option('number_type')}, {shape[0]}, {shape[1]}> {tmp} << {', '.join(args)};")

        # self.p(f"Matrix<{self.option('number_type')}, {shape[0]}, {shape[1]}> {tmp};")
        # self.p(f"{tmp} << {', '.join(args)};")
        # offset = 0
        # arg_shapes = (a.shape for a in expr.args)
        # for shape, strArg in zip(arg_shapes, args):
        #     if type(expr) == pp.expression.VStack:
        #         self.p(f"{tmp}.template block<{shape[0]}, {shape[1]}>({offset}, 0) = {strArg};")
        #         offset += shape[0]
        #     elif type(expr) == pp.expression.HStack:
        #         self.p(f"{tmp}.template block<{shape[0]}, {shape[1]}>(0, {offset}) = {strArg};")
        #         offset += shape[1]
        #     else:
        #         print("error")
        return tmp

    @staticmethod
    def _slice_to_offset(key, length):
        # Convert a slice object key to a (len, offset)
        # length is the size of the object being sliced
        assert key.step == None, NotADirectoryError("Cannot have slices with steps until Eigen 4")

        start = key.start if key.start else 0
        stop = key.stop if key.stop else length

        if start < 0:
            start = length + start
        if stop < 0:
            stop = length + stop

        return (stop - start, start)

    @_generate.register(sliceExpression)
    @_generate_args
    def _(self, expr, args):
        if isinstance(expr.key, int):
            rep = f"segment<1>({expr.key})"
        if isinstance(expr.key, tuple):    
            if len(expr.key) == 1:  # Segment of a vector
                size, offset = sliceExpression._slice_to_offset(expr.key[0], expr.args[0].shape[0])
                rep = f"segment<{size}>({offset})"
            else:  # Block of a matrix
                x_size, x_offset = sliceExpression._slice_to_offset(expr.key[0], expr.args[0].shape[0])
                y_size, y_offset = sliceExpression._slice_to_offset(expr.key[1], expr.args[0].shape[1])
                rep = f"block<{x_size}, {y_size}>({x_offset}, {y_offset})"

        return f"{args[0]}.template {rep}"

    #### Functions ####

    @_generate.register(functionExpression)
    @_generate_args
    def _(self, expr, args):
        self._dependencies.add(expr.function)

        # Evaluate the function into a temporary matrix, and return the temp
        out = pp._get_unique_name()
        self.p(f"Matrix<{self.option('diff_number_type')}, {len(expr.function)}, 1> {out};")
        self.p(f"{expr.function.name}::template impl<{self.option('diff_number_type')}>(p, {out}, {', '.join(args)});")
        return out


gen = EigenGenerator()

# print("===============")
# gen.print_generate(x[1] + x[4])

# print("===============")
# gen.print_generate(3*u[1])

# print("===============")
# gen.print_generate(vstack(x[1],x[3]))

# print("===============")
# gen.print_generate(sys_d.expression)

# gen.generate(dynamics)
gen.generate(sys_d)

print(gen._body)

print(sys_d.expression)



# print(dynamics.generate_declaration(p))

# print(str(p))

# with pp.EigenGenerator(filename="examples/myproblem.hpp") as generator:
#     # generator.generate_dependencies(generator)
#     with generator.generate_class('LOpt') as gen:  # <= generates class declaration at open
#         opt.generate(gen)
#         # sys.generate_declaration(gen)
#         # print(sys_d)
#         # sys_d.generate_declaration(gen)
#         # dynamics.generate_declaration(gen)
#         # sys_0.generate_declaration(gen)

#         # sys_d(x0, u[1]).generate(gen)

#         # gen(f, jacobian=True)  # <= shorthand for generate declaration
#         # gen(A)
#     # <= generates constructor at close

#     # with generator.generate_class('CBob') as gen:
#     #     gen(A)
#     #     gen(u)


# print(pp.sin(x))
# print(len(pp.sin(x)))

# exit()


# f = Function("sys", (x, u), A @ x + B @ u)

# h = Scalar(0.1, 'h')
# xp = x
# k1 = f(xp, u)
# k2 = f(xp + (h * 0.5) * k1, u)
# k3 = f(xp + (h * 0.5) * k2, u)
# k4 = f(xp + h * k3, u)
# expr = xp + (h * 0.1667) * (k1 + 2 * k2 + 2 * k3 + k4)

# rk4 = Function("rk4", (x, u), expr)

# ################ Generate optimization problem ##################

# # i = Variable("i", 2)
# # testfunc = Function("testfunc", (i, ), Variable("out", 1), i[0] + 4 * i[1])
# # q = Matrix((2, 1), 'q')

# opt = NLP("MyProblem")
# x = []
# u = []
# for i in range(N):
#     x.append(opt.variable("x" + str(i), n)) #, lb=-4 * ConstMatrix(np.ones((2, 1)), 't') * Scalar(1.2, 'd')))
#     u.append(opt.variable("u" + str(i), m)) #, ub=2))

# # x = opt.variable("x", n, N, lb=4 * ConstMatrix(np.ones((2, 1)), 't') * Scalar(1.2, 'd'))
# # u = opt.variable("u", m, N - 1, ub=2 * testfunc(q * 5 + 3.2))
# xss = opt.variable("xss", n)
# uss = opt.variable("uss", m)

# x_initial = Matrix((n, 1), 'x_initial')

# _x = Variable("x", n)
# _u = Variable("u", m)
# l = Function("stage_cost", (_x, _u), _x[0] * _x[0] + _x[1] * _x[1] + 2 * _u[0] * _u[0])

# for i in range(N - 2):
#     opt.add(rk4(x[i], u[i]) == x[i + 1])
#     # opt.add(Inequality(C @ x[i], lb=-c, ub=5))
# opt.add(x_initial == x[0])
# opt.add(rk4(xss, uss) == xss)
# # opt.add(np.zeros((2, 1)) == sum([rk4(x[i], u[i]) for i in range(N - 1)], np.array([0, 0]).T))

# # for i in range(N):
# #     val = 0
# #     for j in range(n):
# #         val += (x[i][j]) * (x[i][j])
# #     print(val)
# # opt.minimize(sum(sum(y) for y in x))

# opt.minimize(summation(*map(lambda y: l(y[0] - xss, y[1] - uss), zip(x, u)), xss[0]*xss[0] + xss[1]*xss[1] + uss[0]*uss[0]))

# # t = pp.expression.addExpression(x[0], x[1], x[2], x[3])
# # opt.add_function(Function('test', x, Variable('out', n), sum(x)))

# with Generator(filename="examples/myproblem.hpp") as gen:
#     with gen.generate_class('LOpt') as p:
#         opt.generate(p)

# #     with gen.generate_class('Test') as p:
# #         testfunc.generate(p)


# # opt.generate(filename="examples/myproblem.hpp")

# print(opt.variables)

# ################ Test ##################
