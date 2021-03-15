import numpy as np
from polypy import Variable, VarType, Matrix, Function, ConstScalar, VariableSet, Scalar, Generator, ConstMatrix
from polypy import Index
from polypy import preprint
from polypy.expression import AtomicExpression, functionExpression
import os
import time

N, n, m = 10, 2, 1

x_t = VarType("x_t", n)
u_t = VarType("u_t", m)

x = Variable("x", x_t)
u = Variable("u", u_t)

A = ConstMatrix(np.array([[0, 0], [1, 0]]), "A")
B = Matrix((n, m), "B", initial=np.array([[1], [0]]))
c = ConstMatrix(np.array([1, 2]).T, 'c')

dx = Variable("dx", x_t)
f = Function("sys", (x, u), dx, A @ x + B @ u)

h = 0.1
k1 = f(x, u)
k2 = f(x + (h * 0.5) * k1, u)
k3 = f(x + (h * 0.5) * k2, u)
k4 = f(x + h * k3, u)
expr = x + (h * 0.1667) * (k1 + 2 * k2 + 2 * k3 + k4)

out = Variable("out", x_t)
rk4 = Function("rk4", (x, u), out, expr)

g = Generator()
g.add_function(rk4)
g.add_function(f)

g.generate("examples/gen.hpp")

exit()

os.system('g++ -I /usr/local/include/eigen3/ -I /Users/cnjones/git/lampc/src -std=c++14 -O3 -fPIC -shared examples/model.cpp -o model.so')
from ctypes import *
import ctypes
from numpy.ctypeslib import ndpointer


test = cdll.LoadLibrary("model.so")
# test.test()

c_float_p = ctypes.POINTER(ctypes.c_double)
# data = np.array([[1.129], [2], [3]])
# data = data.astype(np.double)
# data_p = data.ctypes.data_as(c_float_p)

# test.test2(data_p)

fun = test.callme
fun.restype = None
# fun.argtypes = [ndpointer(ctypes.c_double),
#                 ndpointer(ctypes.c_double),
#                 ndpointer(ctypes.c_double),
#                 ndpointer(ctypes.c_double),
#                 ndpointer(ctypes.c_double)]
fun.argtypes = [ndpointer(ctypes.c_double),
                ndpointer(ctypes.c_double),
                ndpointer(ctypes.c_double)]


x = np.array([1, 2, 3], dtype=np.float64).T
u = np.array([1, 2], dtype=np.float64).T

# xp = np.ones((3,1), order="F").astype(np.double)
# Jx = np.ones((3,3), order="F").astype(np.double)
# Ju = np.ones((3,2), order="F").astype(np.double)
xp = np.ones((3, 1), dtype=np.float64)
Jx = np.ones((3, 3), dtype=np.float64)
Ju = np.ones((3, 2), dtype=np.float64)

NUM = 10000
start = time.process_time()
for i in range(NUM):
    # fun(x, u, xp, Jx, Ju)
    fun(x, u, xp)
print(f"Time per integration: {(time.process_time() - start) / NUM}")

print(xp)
# print(Jx)
# print(Ju)

# exit()

# i = Index(name = 'i', rng = range(0, N))
# X = VariableSet('x', x_t, N)
# # print(expr.to_eigen())


# # e = xp - f(A*x - c, u)

# # Eigen:::... tmp;
# # f(A*x - c, u, tmp)
# # return xp - tmp

from casadi import *

x = MX.sym('x', n)
u = MX.sym('u', m)
h = MX.sym('h', 1)

A = np.array([[1,2,3],[4,5,6],[7,8,9]])
B = np.array([[1,2,3],[4,5,6]]).T

f = Function("sys", (x, u), [A @ x + B @ u])

k1 = f(x, u)
k2 = f(x + (h * 0.5) * k1, u)
k3 = f(x + (h * 0.5) * k2, u)
k4 = f(x + h * k3, u)
expr = x + (h * 0.1667) * (k1 + 2 * k2 + 2 * k3 + k4)

g = Function('g',[x, u, h],[expr])
C = CodeGenerator('gen.c', {'cpp': True, 'with_header': True, 'with_mem': True})
C.add(f)
C.add(g)
C.generate()

os.system('g++ -O3 -fPIC -shared gen.c -o gen.so')

cas_g = external('g', './gen.so')

x = np.array([1,2,3]).T
u = np.array([1,2]).T
h = 1e-2

NUM = 10000
start = time.process_time()
for i in range(NUM):
    cas_g(x, u, h)
print(f"Time per integration: {(time.process_time() - start) / NUM}")

# print(f(np.array([[1.2],[3.4]])))
# print(h(np.array([[1.2],[3.4]])))
