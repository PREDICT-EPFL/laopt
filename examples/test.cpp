#define EIGEN_NO_MALLOC

#include "lampc.hpp"


/*[[[cog 
from lampc import *
nlp = NLP()

N = nlp.const("N", 5) # Prediction horizon
n = nlp.const("n", 2) # State dimension
m = nlp.const("m", 1) # Input dimension
]]]*/
//[[[end]]]


template<typename _Scalar>
struct param_t
{
    using Scalar = _Scalar;

    Vec<Scalar, n + m> p2; // Some other fake parameter
    Vec<Scalar, n> x0;
    Vec<Scalar, 15> p; // Some other fake parameter

    Vec<Scalar, n> q;
    Vec<Scalar, m> r;

    param_t() 
    {
        x0.setConstant(0);
        p2.setConstant(1.3);
        p.setConstant(0);

        for(int i=0; i<n; i++) q(i) = i;
        for(int i=0; i<m; i++) r(i) = 2*i;
    }
};

// System dynamics (need a different type for x)
template<typename T, typename param_t, typename Tx>
inline void sys_general(param_t& param, RVec<T,n> out, RCVec<T,n> xp, RCVec<Tx,n> x, RCVec<T,m> u)
{
    out(0) = param.p(1) * xp(0) - (-sin(x(1)) + x(1)*x(0)) + param.p(0);
    out(1) = param.p(2) * xp(1) - cos(x(0))*u(0) + param.p(1);
}


// System dynamics
//[[[cog sys = nlp.function("sys", n, ("xp", n), ("x", n), ("u", m)) ]]]
//[[[end]]]
{
    sys_general<T, param_t, T>(param, out, xp, x, u);
}

// System dynamics for the first state
//[[[cog sys0 = nlp.function("sys0", n, ("xp", n), ("u", m)) ]]]
//[[[end]]]
{
    // x0 is a constant, the rest we want to take the derivative 
    using Scalar = typename decltype(param.p)::Scalar;
    sys_general<T, param_t, Scalar>(param, out, xp, param.x0, u);
}

// Enforce equality between two states
//[[[cog equal = nlp.function("equal", n, ("a", n), ("b", n)) ]]]
//[[[end]]]
{
    out = a - b;
}

// Objective function
//[[[cog objective = nlp.scalar_function("objective", ("_X", n*N), ("_U", m*(N-1))) ]]]
//[[[end]]]
{
    const Map<const Matrix<T,n,N> > X(_X.data(),n,N);
    const Map<const Matrix<T,m,N-1> > U(_U.data(),m,N-1);

    static const auto Q = param.q.template cast<T>().asDiagonal();
    static const auto R = param.r.template cast<T>().asDiagonal();

    out = (X.array() * (Q * X).array()).sum() +
          (U.array() * (R * U).array()).sum() + 
          (_X.template segment<m*(N-1)>(4).array() * _U.array()).sum() - 
          (_U.sum()) * (_X.sum());
          ;
}


template <typename Scalar, typename Traits>
struct MyNLP : public NLP< MyNLP<Scalar, Traits> >
{
    using param_t = param_t<Scalar>;
    param_t param;

    /*[[[cog
    X = nlp.var("X", n, N)
    U = nlp.var("U", m, N-1)

    i = Index(range(1, N-1))
    sys0(X[0], U[0]) == 0
    sys(X[i+1], X[i], U[i+1]) == 0

    xss = nlp.var("xss", n)
    uss = nlp.var("uss", m)

    sys(xss, xss, uss) == 0
    equal(X[N-1], xss) == 0

    nlp.minimize(objective(X, U))

    # i = Index(range(1, N))
    # lb <= out_bnd(X[i], U[i]) <= "ub_func(i)"

    nlp.generate()
    ]]]*/
    //[[[end]]]
};


//[[[cog nlp.generate_traits() ]]]
//[[[end]]]


static MyNLP<float, MyTraits> nlp;


int main()
{
    cout << "Hello world\n";

    nlp.x.setRandom();
    nlp.param.x0.setRandom();
    nlp.param.p.setRandom();

{
    const auto samples = 10000;
    const auto begin = std::chrono::high_resolution_clock::now();
    for (auto i = 0; i < samples; ++i)
    {
        nlp.param.p(0) += 0.1;
        nlp.param.x0(0) += 0.1;
        nlp.eval();
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> duration = end - begin;
    cout << "Time per evaluation = " << duration.count() / samples << endl;
}
    // cout << "g = " << nlp.g.transpose() << endl;

{
    const auto samples = 10000;
    const auto begin = std::chrono::high_resolution_clock::now();
    for (auto i = 0; i < samples; ++i)
    {
        nlp.param.p(0) += 0.1;
        nlp.param.x0(0) += 0.1;
        nlp.eval_jacobian();
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> duration = end - begin;
    cout << "Time per jacobian = " << duration.count() / samples << endl;
}
    // cout << "J = " << endl << nlp.J << endl;

{
    const auto samples = 10000;
    const auto begin = std::chrono::high_resolution_clock::now();
    for (auto i = 0; i < samples; ++i)
    {
        // nlp.param.q(0) += 0.1;
        nlp.eval_objective();
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> duration = end - begin;
    cout << "Time per hessian = " << duration.count() / samples << endl;

}
    // cout << "H = " << endl << nlp.hessian_f << endl;

    return 0;
}
