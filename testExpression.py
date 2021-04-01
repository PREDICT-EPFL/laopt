import numpy as np
from polypy import Variable, Matrix, Function, ConstScalar, Scalar, Generator, ConstMatrix
from polypy import Index
from polypy import preprint
from polypy.expression import AtomicExpression, functionExpression, hstack
from polypy import NLP
from polypy.nlp import Inequality


# class A(object):
#     def __init__(self):
#         self._prop = None

#     @property
#     def prop(self):
#         return self._prop

#     @prop.setter
#     def prop(self, value):
#         print("In A")
#         self._prop = value


# class C(A):
#     pass


# class B(C):
#     @property
#     def prop(self):
#         value = super(B, self).prop
#         # do something with / modify value here
#         return value

#     @prop.setter
#     def prop(self, value):
#         print("In B")
#         super(B, self.__class__).prop.fset(self, value)

# bob = B()
# bob.prop = 4
# print(bob.prop)

# exit()

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

i = Variable("i", 2)
testfunc = Function("testfunc", (i, ), Variable("out", 1), i[0] + 4 * i[1])
q = Matrix((2, 1), 'q')

opt = NLP("MyProblem")
x = opt.variable("x", n, N, lb=4 * ConstMatrix(np.ones((2, 1)), 't') * Scalar(1.2, 'd'))
u = opt.variable("u", m, N - 1, ub=2 * testfunc(q * 5 + 3.2))
xss = opt.variable("xss", n)
uss = opt.variable("uss", m)

xx = Matrix((n, 1), 'xx')

C = np.array([[1, 2], [3, 4]])
c = np.array([[1], [2]])

for i in range(N - 2):
    opt.add(rk4(x[i], u[i]) == x[i+1])
    opt.add(Inequality(C @ x[i], lb=-c, ub=c))
    opt.add(Inequality(C @ x[i], lb=-c, ub=5))
opt.add(xx == x[0])
opt.add(rk4(xss, uss) == xss)
opt.add(np.zeros((2, 1)) == sum([rk4(x[i], u[i]) for i in range(N - 1)], np.array([0, 0]).T))

print(opt.equalities[0].original_expr)


with Generator(filename="examples/myproblem.hpp") as gen:
    with gen.generate_class('LOpt') as p:
        opt.generate(p)

#     with gen.generate_class('Test') as p:
#         testfunc.generate(p)


# opt.generate(filename="examples/myproblem.hpp")

print(opt.variables)

################ Test ##################
