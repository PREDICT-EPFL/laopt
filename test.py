from polypy import NLP
from polypy import Index
from polypy import Generator
import numpy as np

nlp = NLP("ThisIsMyNLP")

N = nlp.const("N", 5)  # Prediction horizon
n = nlp.const("n", 3)  # State dimension
m = nlp.const("m", 2)  # Input dimension

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
sys = nlp.function('test', n, ("xp", x_t), ("x", x_t), ("u", u_t))
out_bnd = nlp.function('out_bnd', 10, ("x", x_t))

stage_cost = nlp.function('stage_cost', 1, ("x", x_t), ("u", u_t), ("ref", y_t))

# TODO: Confirm that the sizes of the call are valid
# TODO: Confirm that all names are valid C++ names
i = Index(rng=range(0, N - 1), name='myIndex')
nlp.equality('steady_state', sys(xss, xss, uss))
nlp.equality('dynamics', sys(x[i + 1], x[i], u[i]))
nlp.equality('testx', sys(x[0], xss, u[2]))

A = np.array([(1,2,3), (4,5,6), (7,8,9)])

nlp.objective = stage_cost

# Generator(nlp, 'test.hpp', number_type="double")
# Generator(nlp)



# Let
# A*sys(xss,xss,uss) + 3*sys(x[2],xss,u[1]) + x[0]
#
# A*sys(xss,xss,uss) + 3*sys(xss,xss,uss) ==> (A + 3I) sys(...) in python

