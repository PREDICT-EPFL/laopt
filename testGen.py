from polypy import NLP
from polypy import Index
from polypy import Generator
from polypy import preprint
import numpy as np
import functools

from polypy.poly import ConstMatrix
from polypy.poly import Matrix
from polypy.poly import Jacobian

from polypy.poly import Expression
from contextlib import redirect_stdout

nlp = NLP("ThisIsMyNLP")

N = nlp.define_constant("N", 5)  # Prediction horizon
n = nlp.define_constant("n", 3)  # State dimension
m = nlp.define_constant("m", 2)  # Input dimension

x_t = nlp.var_type('x_t', n)
u_t = nlp.var_type('u_t', m)
y_t = nlp.var_type('y_t', 1)

x0 = nlp.var("x0", x_t)
x = nlp.var("x", x_t, N)
u = nlp.var("u", u_t, N - 1)
xss = nlp.var("xss", x_t)
uss = nlp.var("uss", u_t)
r = nlp.var("ref", y_t)

sys = nlp.function('sys', n, ("xp", x_t), ("x", x_t), ("u", u_t))
out_bnd = nlp.function('out_bnd', 10, ("x", x_t))
stage_cost = nlp.function('stage_cost', 1, ("x", x_t), ("u", u_t), ("ref", y_t))

i = Index(rng=range(0, N - 1), name='i')

A = nlp.data('A', np.array([[1,2,3],[4,5,6]]))
B = nlp.data('B', np.array([[1,2],[3,4],[5,6]]))
c = nlp.data('c', np.array([[1,2]]).transpose())

# print(u[i+1])

# c = ConstMatrix(np.array([[1,0]]).transpose(), 'c')
# z = ConstMatrix(np.array([[0],[0]]))
# d = nlp.data('c', np.array([[1,2]]).transpose())
# A = ConstMatrix(np.identity(3))
# # q = c + 2*c - c + c
# # q = 2*c + z*8 + z*u[0] + d - z - c
# q = A @ x[0] + 10*z
# print(q)
# print(q.mat)
# print(c + c)


# e = A @ sys(x[0], B @ (u[i] - 2*u[i+1] + 3*u[i-1]) + x[0], A @ sys(xss, xss, 7*uss)+c)
# e = A @ sys(x[0], B @ (u[i] - u[i+1] + u[i-1]) + x[0], A @ sys(xss, xss, uss)+c)
# # e = A @ x[0] + c + A @ B @ u[0]
# # e = A @ B @ u[0]
# # e = A @ sys(x[0], x[1], u[0])

I = ConstMatrix(2*np.identity(3), 'II')
e = I @ x[0]
e = A@sys(x[0], I @ x[0], A@x[0] + u[0])
e = A @ sys(xss, sys(xss,xss,uss), uss) + 4@uss
print(e)

p = preprint("")
jac = Jacobian(e)

p("==========================")
jac.cpp_generate_expression(p)

p("\n\n==========================")
jac.cpp_generate_expression(p, True)
eval = jac.compute_jacobian(uss)
# p(f"J = {eval}")

# TODO: Deal with all the MatrixConst expressions that evaluate to zero and end up with a tmp name
Jacobian(eval).cpp_generate_expression(p)


# # f =  x[4] * (B@c)
# # p = preprint("")
# # with p:
# #     f.cpp_generate_call(p, "expression_1")

# # e = u[1] + u[0]
# # print(e)

# # with redirect_stdout(open('examples/testXX.hpp', 'w+')):
# #     # Write out all data
# #     # print(e.data)

# #     p = preprint("")
# #     with p:
# #         e.cpp_generate_call(p, "expression_1")

# #     with p:
# #         e.cpp_generate_call(p, "expression_1", True)


# # A = nlp.data('A', np.array([[1,2,3],[4,5,6],[7,8,9]]))
# # B = nlp.data('A', np.array([[1,2],[4,5],[7,8]]))

# # e = x[i+1] - A @ x[i] - B @ u[i]
# # with redirect_stdout(open('examples/testXX.hpp', 'w+')):
# #     p = preprint("")
# #     with p:
# #         e.cpp_generate_call(p, "expression_1")
