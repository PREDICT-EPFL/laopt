#include "lampc.hpp"

using namespace std::placeholders;

// #include "DoubleIntegrator.hpp"
// using namespace DoubleIntegrator;

/*[[[cog
from lampc import *
import cog
nlp = NLP()

# Simple linear MPC example
N = 5
X = nlp.var("X", 2, N)
U = nlp.var("U", 1, N-1)
xss = nlp.var("xss", 2, 1)
uss = nlp.var("uss", 1, 1)

nlp.gen_variables();
]]]*/
//[[[end]]]


template<typename Scalar>
struct Equalities
{
    /*[[[cog
    funcs = Functions(nlp)

    i = Index(range(N-1))
    funcs.append("dynamics", 2, (X[i+1], X[i], U[i+1]), i)

    funcs.append("initial_state", 2, (X[0], ))

    # Force xss,uss to be a steady-state
    funcs.append("dynamics", 2, (xss, xss, uss))

    # Set X[N-1] = xss
    funcs.append("equal", 2, (X[N-1], xss))
    ]]]*/
    //[[[end]]]

    enum {
        nx = 2,
        nu = 1
    };

    template<typename T> using StateType = Matrix<T, nx, 1>;
    template<typename T> using InputType = Matrix<T, nu, 1>;

    StateType<Scalar> x0; // Initial state

    template <typename T>
    inline StateType<T> dynamics(Ref<StateType<T>> xp,
                                 Ref<StateType<T>> x,
                                 Ref<InputType<T>> u)
    {
        return StateType<T> {xp[0] - (-sin(x[1]) + x[1]*x[0]), xp[1] - cos(x[0])*u[0]};
    };
 
    template <typename T>
    inline StateType<T> initial_state(Ref<StateType<T>> x)
    {
        return x - x0;
    };

    // x == xss
    template <typename T>
    inline StateType<T> equal(Ref<StateType<T>> x,
                              Ref<StateType<T>> xss)
    {
        return x - xss;
    };

    Equalities()
    {
        x0.setZero();
        initialize(); // Must call in the constructor
    }

    /*[[[cog
    funcs.gen()
    ]]]*/
    //[[[end]]]
};

template<typename Scalar>
struct Cost
{
    /*[[[cog
    funcs = Functions(nlp)

    funcs.append("cost", 1, (X[1], U[0]))
    ]]]*/
    //[[[end]]]

    template<typename S>
    inline Matrix<S, 1, 1> cost(Ref<Matrix<S, 2, 1>> x, Ref<Matrix<S, 1, 1>> u)
    {
        return u;
    }

    /*[[[cog
    funcs.gen()
    ]]]*/
    //[[[end]]]
};

using Scalar = double;
Equalities<Scalar> eq;
Equalities<Scalar> ineq;
Cost<Scalar> cost;
NLP nlp(eq, ineq, cost);

int main()
{
    nlp.primal.setConstant(1.2);
    nlp.eval();
    nlp.eval_jacobians();

    cout << "nlp.primal = " << nlp.primal.transpose() << endl;
    cout << "nlp.eq.f = " << nlp.eq.f.transpose() << endl;
    cout << "nlp.eq.J = \n" << nlp.eq.J << endl;
    cout << "nlp.ineq.J = \n" << nlp.ineq.J << endl;

    const auto samples = 10000;
    const auto begin = std::chrono::high_resolution_clock::now();
    for (auto i = 0; i < samples; ++i)
        nlp.eval_jacobians();        
    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> duration = end - begin;
    cout << "Time per jacobian evaluation = " << duration.count() / samples << endl;

    // Change initial state and re-compute
    for(int i=0; i<10; i++)
    {
        nlp.eq.x0[0] = i;
        nlp.eq.x0[1] = i*2;
        cout << "x0 = " << nlp.eq.x0.transpose() << endl;
        nlp.eval();
        cout << "nlp.eq.f = " << nlp.eq.f.transpose() << endl;
    }

    return 0;
}
