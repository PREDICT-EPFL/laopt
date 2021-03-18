import numpy as np
from polypy import Variable, Matrix, Function, ConstScalar, Scalar, Generator, ConstMatrix
from polypy import Index
from polypy import preprint
from polypy.expression import AtomicExpression, functionExpression
from polypy import Problem
import polypy
import os
import time
import copy

N, n, m = 5, 2, 1

x = Variable("x", n)
u = Variable("u", m)

# A = ConstMatrix(np.array([[0, 0], [1, 0]]), "A")
A = ConstMatrix(np.array([[0, 0], [1, 0]]), 'A')
B = Matrix((n, m), "B", initial=np.array([[1], [0]]))
c = ConstMatrix(np.array([1, 2]).T, 'c')

dx = Variable("dx", n)
f = Function("sys", (x, u), dx, A @ x + B @ u)

h = 0.1

out = Variable("out", n)

xp = x
k1 = f(xp, u)
k2 = f(xp + (h * 0.5) * k1, u)
k3 = f(xp + (h * 0.5) * k2, u)
k4 = f(xp + h * k3, u)
expr = xp + (h * 0.1667) * (k1 + 2 * k2 + 2 * k3 + k4)

rk4 = Function("rk4", (x, u), out, expr)

# xp = x
# for i in range(4):
#     xp = rk4(xp, u)

# single = Function("single", (x, u), out, xp)

# g = Generator()
# g.add_function(single)

# g.generate("examples/gen.hpp")



################ Generate optimization problem ##################

opt = Problem("MyProblem")
x = opt.variable("x", n, N)
u = opt.variable("u", m, N - 1)
xss = opt.variable("xss", n)
uss = opt.variable("uss", m)

xx = Matrix((2,1), "xx")

# i = 1
# expr = A @ x[i] + B @ u[i] == x[i+1]
# print(expr)

for i in range(N - 1):
    opt.add(rk4(x[i], u[i]) == x[i+1])
    # opt.add(A @ x[i] + B @ u[i] == x[i+1])
opt.add(xx == x[0])
opt.add(rk4(xss, uss) == xss)

print(opt.functions())
opt.generate(filename="examples/myproblem.hpp")

print(opt.variables)

exit()

# # for i in range(N - 1):
# #     print(opt.equalities[i])
# #     print(opt.equalities[i].function)

# out = Variable("out", 4)
# x = Variable("x", 4)
# f = Function("bob", (x, ), out, 2*x)

# y = Variable("y", 4)
# g = Function("bill", (y, ), out, 2*y)

# y = Variable("y", 4)
# h = Function("jill", (y, ), out, 3*y)

# # print(f == g)
# # print(f == h)

# print('=====================')

# z = Variable('z', 4)
# print(x * y)
# q= f(x, u[0])
# print(f"before = {q}")
# print(f"after = {q.substitute((x, ), (5 * z + x, ))}")

# # print(opt.equalities[0])
# # print(opt.equalities[0].function)

# print(opt.variables)

# opt.equalities[0].to_function()
# hash(x[0])
# opt.add()

# # os.system('g++ -I /usr/local/include/eigen3/ -I /Users/cnjones/git/lampc/src 
# -std=c++14 -O3 -fPIC -shared examples/model.cpp -o model.so')
# # from ctypes import *
# # import ctypes
# # from numpy.ctypeslib import ndpointer


# # test = cdll.LoadLibrary("model.so")
# # # test.test()

# # c_float_p = ctypes.POINTER(ctypes.c_double)
# # # data = np.array([[1.129], [2], [3]])
# # # data = data.astype(np.double)
# # # data_p = data.ctypes.data_as(c_float_p)

# # # test.test2(data_p)

# # fun = test.callme
# # fun.restype = None
# # # fun.argtypes = [ndpointer(ctypes.c_double),
# # #                 ndpointer(ctypes.c_double),
# # #                 ndpointer(ctypes.c_double),
# # #                 ndpointer(ctypes.c_double),
# # #                 ndpointer(ctypes.c_double)]
# # fun.argtypes = [ndpointer(ctypes.c_double),
# #                 ndpointer(ctypes.c_double),
# #                 ndpointer(ctypes.c_double)]


# # x = np.array([1, 2, 3], dtype=np.float64).T
# # u = np.array([1, 2], dtype=np.float64).T

# # # xp = np.ones((3,1), order="F").astype(np.double)
# # # Jx = np.ones((3,3), order="F").astype(np.double)
# # # Ju = np.ones((3,2), order="F").astype(np.double)
# # xp = np.ones((3, 1), dtype=np.float64)
# # Jx = np.ones((3, 3), dtype=np.float64)
# # Ju = np.ones((3, 2), dtype=np.float64)

# # NUM = 10000
# # start = time.process_time()
# # for i in range(NUM):
# #     # fun(x, u, xp, Jx, Ju)
# #     fun(x, u, xp)
# # print(f"Time per integration: {(time.process_time() - start) / NUM}")

# # print(xp)
# # print(Jx)
# # print(Ju)

# # exit()

# # i = Index(name = 'i', rng = range(0, N))
# # X = VariableSet('x', x_t, N)
# # # print(expr.to_eigen())


# # # e = xp - f(A*x - c, u)

# # # Eigen:::... tmp;
# # # f(A*x - c, u, tmp)
# # # return xp - tmp

# import casadi as cas
# from casadi import *

# x = MX.sym('x', n)
# u = MX.sym('u', m)
# h = MX.sym('h', 1)

# A = np.array([[1,2,3],[4,5,6],[7,8,9]])
# B = np.array([[1,2,3],[4,5,6]]).T

# f = cas.Function("sys", (x, u), [A @ x + B @ u])

# k1 = f(x, u)
# k2 = f(x + (h * 0.5) * k1, u)
# k3 = f(x + (h * 0.5) * k2, u)
# k4 = f(x + h * k3, u)
# expr = x + (h * 0.1667) * (k1 + 2 * k2 + 2 * k3 + k4)

# g = cas.Function('g',[x, u, h],[expr])
# C = CodeGenerator('gen.c', {'cpp': True, 'with_header': True, 'with_mem': True})
# C.add(f)
# C.add(g)
# C.generate()

# os.system('g++ -O3 -fPIC -shared gen.c -o gen.so')

# cas_g = external('g', './gen.so')

# x = np.array([1,2,3]).T
# u = np.array([1,2]).T
# h = 1e-2

# NUM = 1
# start = time.process_time()
# for i in range(NUM):
#     cas_g(x, u, h)
# print(f"Time per integration: {(time.process_time() - start) / NUM}")

# # print(f(np.array([[1.2],[3.4]])))
# # print(h(np.array([[1.2],[3.4]])))
