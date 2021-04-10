import numpy as np
from polypy import Variable, Matrix, Function, ConstScalar, Scalar, Generator, ConstMatrix
from polypy import Index
from polypy import preprint
from polypy.expression import AtomicExpression, functionExpression, hstack, summation
from polypy import NLP
from polypy.nlp import Inequality

import polypy as pp

N, n, m = 5, 2, 1

x = Variable("x", n)
u = Variable("u", m)

A = ConstMatrix(np.array([[0, 0], [1, 0]]), 'A')
B = Matrix((n, m), "B", initial=np.array([[1], [0]]))
c = ConstMatrix(np.array([1, 2]).T, 'c')

dx = Variable("dx", n)
f = Function("sys", (x, u), dx, A @ x + B @ u)

h = Scalar(0.1, 'h')
xp = x
k1 = f(xp, u)
k2 = f(xp + (h * 0.5) * k1, u)
k3 = f(xp + (h * 0.5) * k2, u)
k4 = f(xp + h * k3, u)
expr = xp + (h * 0.1667) * (k1 + 2 * k2 + 2 * k3 + k4)

out = Variable("out", n)
rk4 = Function("rk4", (x, u), out, expr)

################ Generate optimization problem ##################

# i = Variable("i", 2)
# testfunc = Function("testfunc", (i, ), Variable("out", 1), i[0] + 4 * i[1])
# q = Matrix((2, 1), 'q')

opt = NLP("MyProblem")
x = []
u = []
for i in range(N):
    x.append(opt.variable("x" + str(i), n)) #, lb=-4 * ConstMatrix(np.ones((2, 1)), 't') * Scalar(1.2, 'd')))
    u.append(opt.variable("u" + str(i), m)) #, ub=2))

# x = opt.variable("x", n, N, lb=4 * ConstMatrix(np.ones((2, 1)), 't') * Scalar(1.2, 'd'))
# u = opt.variable("u", m, N - 1, ub=2 * testfunc(q * 5 + 3.2))
xss = opt.variable("xss", n)
uss = opt.variable("uss", m)

x_initial = Matrix((n, 1), 'x_initial')

_x = Variable("x", n)
_u = Variable("u", m)
l = Function("stage_cost", (_x, _u), Variable("out", 1), _x[0] * _x[0] + _x[1] * _x[1] + 2 * _u[0] * _u[0])

for i in range(N - 2):
    opt.add(rk4(x[i], u[i]) == x[i + 1])
    # opt.add(Inequality(C @ x[i], lb=-c, ub=5))
opt.add(x_initial == x[0])
opt.add(rk4(xss, uss) == xss)
# opt.add(np.zeros((2, 1)) == sum([rk4(x[i], u[i]) for i in range(N - 1)], np.array([0, 0]).T))

# for i in range(N):
#     val = 0
#     for j in range(n):
#         val += (x[i][j]) * (x[i][j])
#     print(val)
# opt.minimize(sum(sum(y) for y in x))

opt.minimize(summation(*map(lambda y: l(y[0] - xss, y[1] - uss), zip(x, u)), xss[0]*xss[0] + xss[1]*xss[1] + uss[0]*uss[0]))

# t = pp.expression.addExpression(x[0], x[1], x[2], x[3])
# opt.add_function(Function('test', x, Variable('out', n), sum(x)))

with Generator(filename="examples/myproblem.hpp") as gen:
    with gen.generate_class('LOpt') as p:
        opt.generate(p)

#     with gen.generate_class('Test') as p:
#         testfunc.generate(p)


# opt.generate(filename="examples/myproblem.hpp")

print(opt.variables)

################ Test ##################
