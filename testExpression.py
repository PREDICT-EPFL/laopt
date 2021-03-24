import numpy as np
from polypy import Variable, Matrix, Function, ConstScalar, Scalar, Generator, ConstMatrix
from polypy import Index
from polypy import preprint
from polypy.expression import AtomicExpression, functionExpression
from polypy import NLP
from polypy.nlp import Inequality
import polypy
import os
import time
import copy

N, n, m = 5, 2, 1

x = Variable("x", n)
u = Variable("u", m)

A = ConstMatrix(np.array([[0, 0], [1, 0]]), 'A')
B = Matrix((n, m), "B", initial=np.array([[1], [0]]))
c = ConstMatrix(np.array([1, 2]).T, 'c')

dx = Variable("dx", n)
f = Function("sys", (x, u), dx, A @ x + B @ u)

h = 0.1
xp = x
k1 = f(xp, u)
k2 = f(xp + (h * 0.5) * k1, u)
k3 = f(xp + (h * 0.5) * k2, u)
k4 = f(xp + h * k3, u)
expr = xp + (h * 0.1667) * (k1 + 2 * k2 + 2 * k3 + k4)

out = Variable("out", n)
rk4 = Function("rk4", (x, u), out, expr)

################ Generate optimization problem ##################

opt = NLP("MyProblem")
x = opt.variable("x", n, N)
u = opt.variable("u", m, N - 1)
xss = opt.variable("xss", n)
uss = opt.variable("uss", m)

xx = Matrix((n, 1), 'xx')

C = np.array([[1, 2], [3, 4]])
c = np.array([[1], [2]])

for i in range(N - 1):
    opt.add(rk4(x[i], u[i]) == x[i+1])
    opt.add(Inequality(C @ x[i], lb=-c, ub=c))
opt.add(xx == x[0])
opt.add(rk4(xss, uss) == xss)
opt.add(np.zeros((2,1)) == sum([rk4(x[i], u[i]) for i in range(N - 1)], np.array([0, 0]).T))

opt.generate(filename="examples/myproblem.hpp")

print(opt.variables)



################ Test ##################
