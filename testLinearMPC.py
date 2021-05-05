import numpy as np
import polypy as pp


class Discrete_Linear_System:
    """Very simple discrete-time linear MPC regulation example"""

    def dynamics(x, u):
        return [0.1 * sin(x[0]), 0.5 * cos(x[1])] + [0, u]

    def __init__(self, N=5):
        self.N, self.n, self.m = N, 2, 1
        N, n, m = self.N, self.n, self.m

        self.x = pp.variable("x", n)
        self.u = pp.variable("u", m)
        x, u = self.x, self.u

        # Define dynamics
        # A = np.array([[1, 0], [0.5, 1]])
        # A = pp.Matrix(A, name='A')  # If we want to be able to modify it at runtime
        # A = pp.Matrix(A, name='A', contant=True)  # If we want to be able to read it at runtime

        A = pp.ConstMatrix(np.array([[1, 0], [0.5, 1]]), 'A')  # Constant
        B = pp.Matrix((n, m), "B", initial=np.array([[0.5], [0.125]]))  # Editable at run-time

        self.dynamics = pp.Function("dynamics", (x, u), A @ x + B @ u)

        # Define stage cost
        q = pp.Matrix((n, 1), "q", initial=np.array([[1], [1]]))
        self.stage_cost = pp.Function("stage_cost", (x, u), q[0] * x[0] * x[0] + q[1] * x[1] * x[1] + u[0] * u[0])


def rk4_integration(shape, f, h_default=0.1):
    n, m = shape
    x, u = pp.variable("x", n), pp.variable("u", m)

    h = pp.Scalar(h_default, 'h')

    xp = x
    k1 = f(xp, u)
    k2 = f(xp + (h * 0.5) * k1, u)
    k3 = f(xp + (h * 0.5) * k2, u)
    k4 = f(xp + h * k3, u)
    expr = xp + (h * 0.1667) * (k1 + 2 * k2 + 2 * k3 + k4)

    return pp.Function("rk4", (x, u), expr)


class Continuous_Linear_System:
    """Very simple continuous-time linear MPC regulation example"""

    def __init__(self, N=5, h=0.1):
        self.N, self.n, self.m = N, 2, 1
        N, n, m = self.N, self.n, self.m

        self.x = pp.variable("x", n)
        self.u = pp.variable("u", m)
        x, u = self.x, self.u

        # Define dynamics
        A = np.array([[0, 0], [1, 0]])
        B = np.array([[1], [0]])
        A = pp.ConstMatrix(A, 'A')  # If you want to name the matrix in C++
        B = pp.ConstMatrix(B, "B")
        self.f = pp.Function("dynamics", (x, u), A @ x + B @ u)
        self.dynamics = rk4_integration((n, m), self.f, h)

        # Define stage-cost
        q = pp.Matrix((n, 1), "q", initial=np.array([[1], [1]]))
        self.stage_cost = pp.Function("stage_cost", (x, u), q[0] * x[0] * x[0] + q[1] * x[1] * x[1] + u[0] * u[0])


if 0:
    sys = Discrete_Linear_System(20)
else:
    sys = Continuous_Linear_System(10)

n, m, N = sys.n, sys.m, sys.N


################ Generate optimization problem ##################

opt = pp.NLP("MyProblem")

x_lb = np.array([-1, -2e20]).T
x = pp.variable("x", n, num_vars=N, lb=x_lb)
u = pp.variable("u", m, num_vars=N)
xss = pp.variable('xss', n)
uss = pp.variable('uss', m)

# i = pp.Index(name='i', rng=range(N - 1))
# j = pp.Index(name='j', rng=range(N - 1))
# print((x[2*i+3]).get_index())

# # Relax constraint at time 2
# x[2].lb = pp.ConstMatrix(np.array([-5, -2e20]).T)
# x[3].ub = pp.ConstMatrix(np.array([4, 5]).T)

x_initial = pp.Matrix((n, 1), 'x_initial')

# for i in range(N - 1):
# i = pp.Index(rng=range(N - 1), name='i')
# opt.add(sys.dynamics(x[i], u[i]) == x[i + 1])

# opt.add(sys.dynamics(x[i], u[i]) == x[i + 1] for i in pp.Index(rng=range(N - 1), name='i'))

with opt.add():
    for i in pp.Range(0, N - 1, 3):
        sys.dynamics(x[i], u[i]) == x[i + 1]
        -1 <= C @ x[i] <= 1


C = pp.ConstMatrix(np.array([[1, 1]]), 'C')
for i in range(N - 1):
    opt.add(sys.dynamics(x[i], u[i]) == x[i + 1])
    opt.add(pp.Inequality(C @ x[i], lb=-1, ub=1))

opt.add(x_initial == x[0])
opt.add(sys.dynamics(xss, uss) == xss)
opt.add(xss[1] == 0)

opt.minimize(pp.summation(*map(lambda y: sys.stage_cost(y[0] - xss, y[1] - uss), zip(x, u))))

# opt.set_variables([xss, x, uss, u])  # If you want to set the order of the variables
# print(opt.compressed_vars)

with pp.Generator(filename="examples/myproblem.hpp") as gen:
    with gen.generate_class('LOpt', pp.Eigen("")) as p:
        opt.generate(p)