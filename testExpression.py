import numpy as np
import polypy as pp

N, n, m = 5, 2, 1

A = pp.matrix([[0, 0, 1, 2], [1, 0, 3, 4]], name='A')
B = pp.matrix([1, 0], name='B', constant=False)
c = pp.matrix([[1], [2]], name='c')


@pp.function
def dynamics(x: n, u: m):
    vel = pp.matrix([1, 1]) @ pp.sin(x[0] + x[1])
    xdot = A @ pp.vstack(vel, pp.cos(u), x[1]) + B @ u
    return xdot


def rk4(f, x, u):
    h = pp.Scalar(0.1, 'h')
    xp = x
    k1 = f(xp, u)
    k2 = f(xp + (h * 0.5) * k1, u)
    k3 = f(xp + (h * 0.5) * k2, u)
    k4 = f(xp + h * k3, u)
    return xp + (h * 0.1667) * (k1 + 2 * k2 + 2 * k3 + k4)


@pp.function
def sys_d(x: n, u: m):
    return rk4(dynamics, x, u)

# @pp.function
# def sys_0(u: m):
#     return sys_d(x0, u)

opt = pp.NLP("MyProblem")

x_lb = np.array([-1, -2e20]).T
x = pp.variable("x", n, num_vars=N, lb=x_lb)
u = pp.variable("u", m, num_vars=N - 1)
xss = pp.variable('xss', n)
uss = pp.variable('uss', m)

x0 = pp.matrix(np.zeros(n), name="x0", constant=False)

opt.add(x[1] == dynamics(x0, u[0]))
opt.add(xss == dynamics(xss, uss))

# opt.add(x[i + 1] == dynamics(x[i], u[i]) for i in pp.Range(1, N - 1))
for i in range(1, N-1):
    opt.add(x[i + 1] == dynamics(x[i], u[i]))

# C = pp.matrix([1, 1]).T
# for i in range(1, N - 1):
#     opt.add(pp.Inequality(C @ x[i], lb=-1, ub=1))

def stage_cost(x: n, u: m):
    # Define stage cost
    q = pp.matrix([1, 1], name="q", constant=False)
    # return sum((q @ x) * x)
    return q[0] * x[0] * x[0] + q[1] * x[1] * x[1] + u[0] * u[0]

# print(sum(x[1]))

# print(sum((x[1]) * x[1]))

opt.minimize(pp.summation(*map(lambda y: stage_cost(y[0] - xss, y[1] - uss), zip(x, u))))

with pp.EigenGenerator(filename="examples/myproblem.hpp") as generator:
    # generator.generate_dependencies(generator)
    with generator.generate_class('LOpt') as gen:  # <= generates class declaration at open
        opt.generate(gen)
        # sys.generate_declaration(gen)
        # print(sys_d)
        # sys_d.generate_declaration(gen)
        # dynamics.generate_declaration(gen)
        # sys_0.generate_declaration(gen)

        # sys_d(x0, u[1]).generate(gen)

        # gen(f, jacobian=True)  # <= shorthand for generate declaration
        # gen(A)
    # <= generates constructor at close

    # with generator.generate_class('CBob') as gen:
    #     gen(A)
    #     gen(u)


# print(pp.sin(x))
# print(len(pp.sin(x)))

exit()


f = Function("sys", (x, u), A @ x + B @ u)

h = Scalar(0.1, 'h')
xp = x
k1 = f(xp, u)
k2 = f(xp + (h * 0.5) * k1, u)
k3 = f(xp + (h * 0.5) * k2, u)
k4 = f(xp + h * k3, u)
expr = xp + (h * 0.1667) * (k1 + 2 * k2 + 2 * k3 + k4)

rk4 = Function("rk4", (x, u), expr)

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
l = Function("stage_cost", (_x, _u), _x[0] * _x[0] + _x[1] * _x[1] + 2 * _u[0] * _u[0])

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
