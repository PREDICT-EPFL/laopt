#define EIGEN_NO_MALLOC

#include "lampc.hpp"


/*[[[cog 
from lampc import *
nlp = NLP()

N = nlp.const("N", 5) # Prediction horizon
n = nlp.const("n", 2) # State dimension
m = nlp.const("m", 1) # Input dimension
]]]*/
constexpr int N = 5;
constexpr int n = 2;
constexpr int m = 1;
//[[[end]]]


template<typename _Scalar>
struct param_t
{
    using Scalar = _Scalar;

    Vec<Scalar, n + m> p2; // Some other fake parameter
    Vec<Scalar, n> x0;
    Vec<Scalar, 15> p; // Some other fake parameter

    param_t() 
    {
        x0.setConstant(0);
        p2.setConstant(1.3);
        p.setConstant(0);
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
template <typename T, typename param_t>
inline void _sys(param_t& param, RVec<T,2> out, RCVec<T,2> xp, RCVec<T,2> x, RCVec<T,1> u)
//[[[end]]]
{
    sys_general<T, param_t, T>(param, out, xp, x, u);
}

// System dynamics for the first state
//[[[cog sys0 = nlp.function("sys0", n, ("xp", n), ("u", m)) ]]]
template <typename T, typename param_t>
inline void _sys0(param_t& param, RVec<T,2> out, RCVec<T,2> xp, RCVec<T,1> u)
//[[[end]]]
{
    // x0 is a constant, the rest we want to take the derivative 
    using Scalar = typename decltype(param.p)::Scalar;
    sys_general<T, param_t, Scalar>(param, out, xp, param.x0, u);
}

// Enforce equality between two states
//[[[cog equal = nlp.function("equal", n, ("a", n), ("b", n)) ]]]
template <typename T, typename param_t>
inline void _equal(param_t& param, RVec<T,2> out, RCVec<T,2> a, RCVec<T,2> b)
//[[[end]]]
{
    out = a - b;
}

// Objective function
//[[[cog objective = nlp.function("objective", 1, ("_X", n*N), ("_U", m*N)) ]]]
template <typename T, typename param_t>
inline void _objective(param_t& param, RVec<T,1> out, RCVec<T,10> _X, RCVec<T,5> _U)
//[[[end]]]
{
    auto X = _X.reshaped(n,N);
    auto U = _U.reshaped(m,N);
    using Scalar = typename param_t::Scalar;
    static const DiagonalMatrix<Scalar, 2> Q(1,1);
    static const DiagonalMatrix<Scalar, 1> R(10);

    out = (X.array() * (Q * X).array()).sum() +
          (U.array() * (R * U).array()).sum();
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
    sys0(X[1], U[0]) == 0
    sys(X[i+1], X[i], U[i+1]) == 0

    xss = nlp.var("xss", n)
    uss = nlp.var("uss", m)

    sys(xss, xss, uss) == 0
    equal(X[N-1], xss) == 0

    #nlp.minimize(objective(X, U))

    # i = Index(range(1, N))
    # lb <= out_bnd(X[i], U[i]) <= "ub_func(i)"

    nlp.generate()
    ]]]*/
    // Bring NLP names into this namespace
    using Base = NLP< MyNLP<Scalar, Traits> >;
    using Base::x;
    using Base::J;
    using Base::g;

    // Define variables data and accessors
    // Sizes
    static constexpr auto sX = 2;
    static constexpr auto sU = 1;
    static constexpr auto sxss = 2;
    static constexpr auto suss = 1;
    // Offsets
    constexpr auto oX(int col) {return 0+2*col;};
    constexpr auto oU(int col) {return 10+1*col;};
    constexpr auto oxss() {return 14;};
    constexpr auto ouss() {return 16;};
    // Accessor
    constexpr auto  X(int col) {return x.template segment<sX>(oX(col));};
    constexpr auto  U(int col) {return x.template segment<sU>(oU(col));};
    constexpr auto  xss() {return x.template segment<sxss>(oxss());};
    constexpr auto  uss() {return x.template segment<suss>(ouss());};

    // Define short names for constraints
    // Sizes
    static constexpr auto sc0 = 2;
    static constexpr auto sc1 = 2;
    static constexpr auto sc2 = 2;
    static constexpr auto sc3 = 2;
    // Offsets
    constexpr auto oc0() {return 0;};
    constexpr auto oc1(int ind) {return 2+2*ind;};
    constexpr auto oc2() {return 8;};
    constexpr auto oc3() {return 10;};
    // Accessor
    constexpr auto  c0() {return g.template segment<sc0>(oc0());};
    constexpr auto  c1(int ind) {return g.template segment<sc1>(oc1(ind));};
    constexpr auto  c2() {return g.template segment<sc2>(oc2());};
    constexpr auto  c3() {return g.template segment<sc3>(oc3());};

    // Instantiate functions and jacobians
    make_differentiable(sys, 2, 2, 2, 1);
    make_differentiable(sys0, 2, 2, 1);
    make_differentiable(equal, 2, 2, 2);
    make_differentiable(objective, 1, 10, 5);

    // Evaluate constraints
    inline void eval()
    {
    	sys0(param, c0(), X(1), U(0));
    	for(int i=1; i<4; i++)
    		sys(param, c1(i), X((i+1)), X(i), U((i+1)));
    	sys(param, c2(), xss(), xss(), uss());
    	equal(param, c3(), X(4), xss());
    }

    // Evaluate jacobians
    inline void eval_jacobian()
    {
    	sys0(param, c0(), X(1), U(0), 
    	     J.BLK(sc0,sX,oc0(),oX(1)), J.BLK(sc0,sU,oc0(),oU(0)));
    	for(int i=1; i<4; i++)
    		sys(param, c1(i), X((i+1)), X(i), U((i+1)), 
    		    J.BLK(sc1,sX,oc1(i),oX((i+1))), J.BLK(sc1,sX,oc1(i),oX(i)), J.BLK(sc1,sU,oc1(i),oU((i+1))));
    	sys(param, c2(), xss(), xss(), uss(), 
    	    J.BLK(sc2,sxss,oc2(),oxss()), J.BLK(sc2,sxss,oc2(),oxss()), J.BLK(sc2,suss,oc2(),ouss()));
    	equal(param, c3(), X(4), xss(), 
    	      J.BLK(sc3,sX,oc3(),oX(4)), J.BLK(sc3,sxss,oc3(),oxss()));
    }

    //[[[end]]]
};

//[[[cog nlp.generate_traits() ]]]
struct MyTraits
{
    enum {
        num_vars = 17,
        num_eq = 12
    };
};
//[[[end]]]


using Scalar = double;
static MyNLP<Scalar, MyTraits> nlp;


int main()
{
    cout << "Hello world\n";

    nlp.x.setRandom();
    nlp.param.x0.setRandom();
    nlp.param.p.setRandom();

    cout << "====================================\n";
    nlp.eval();
    cout << "g = " << nlp.g.transpose() << endl;

    nlp.eval_jacobian();
    // cout << "J = " << endl << nlp.J << endl;

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

    return 0;
}
