import numpy as np
from polypy.expression import *
from polypy import preprint
from polypy import VarType, VariableSet, Variable
# from polypy.affineexpression import AffineExpression, One
from polypy import Index

from collections import defaultdict

A = Matrix((3, 3), 'A')
C = ConstMatrix(np.ones((3,3)), 'C')
I = Identity(3)
c = Scalar(3.4, 'c')

# e = A * -(A + A)[:-1,1:-1] - A[:-1] + c

# print(e)
# print(e.generate_python())
# print(e.generate_eigen())

print("\n\n\n")

N, n, m = 10, 3, 2

x = Variable("x", n, N)
u = Variable("u", m, N - 1)

A = Matrix((2*n, 2*n), "A")
B = Matrix((n, m), "B")

i = Index(rng=range(0, N - 1), name='i')


_x = Variable('x', n)
_u = Variable('u', m)
_xp = Variable('xp', n)
f = Function(
(_x, _u),  # Inputs
_xp,  # Outputs
_xp - ((A[2:6,2:5]@_x)[:-1] + B@_u)  # Expression to evaluate
)

# print((x[i+1] - ((A[2:6,2:5]@x[i])[:-1] + B@u[i])).to_eigen())

nlp.add_constraint(x[i + 1] == f(x[i], u[i]))

# Process:
# 1. Define fake variables x, u, which gives variable types and sizes
# 2. Define functions f = function({x, u} => {xp}, expression)  # Decide multiple outputs?
# --- Generate functions ---

# 1. Define list of expressions
# 2. 
