import numpy as np
import polypy as pp

# Very simple linear MPC regulation example
N, n, m = 5, 2, 1

x = pp.Variable("x", n)
u = pp.Variable("u", m)

A = pp.ConstMatrix(np.array([[1, 0], [0.5, 1]]), 'A')
B = pp.Matrix((n, m), "B", initial=np.array([[0.5], [0.125]]))

################ Generate optimization problem ##################

opt = pp.NLP("MyProblem")
x = opt.Variable("x", n, N)
u = opt.Variable("u", m, N-1)

x_initial = pp.Matrix((n, 1), 'x_initial')

_x = pp.Variable("x", n)
_u = pp.Variable("u", m)
l = pp.Function("stage_cost", (_x, _u), pp.Variable("out", 1), _x[0] * _x[0] + _x[1] * _x[1] + 2 * _u[0] * _u[0])

for i in range(N - 2):
    opt.add(A @ x[i] + B @ u[i] == x[i + 1])
opt.add(x_initial == x[0])

opt.minimize(pp.summation(*map(lambda y: l(y[0], y[1]), zip(x, u))))

with pp.Generator(filename="examples/myproblem.hpp") as gen:
    with gen.generate_class('LOpt') as p:
        opt.generate(p)
