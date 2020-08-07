#define EIGEN_NO_MALLOC

#include "lampc.hpp"


template <typename Scalar, typename Traits>
struct MyNLP : public NLP< MyNLP<Scalar, Traits> >
{
    /*[[[cog
    from lampc import *
    nlp = NLP()

    # Simple linear MPC example
    N = 5
    X = nlp.var("X", 2, N, lb=(-5, -3), ub=10)
    U = nlp.var("U", 1, N-1, lb="lb_func")

    i = Index(range(N-1))
    nlp.equality("dynamics", 2, (X[i+1], X[i], U[i+1]), i)

    nlp.equality("initial_state", 2, (X[0], ))

    xss = nlp.var("xss", 2)
    uss = nlp.var("uss", 1)
    nlp.equality("dynamics", 2, (xss, xss, uss))
    nlp.equality("equal", 2, (X[N-1], xss))

    # i = Index(range(1, N))
    # nlp.inequality("out_bnd", 3, (X[i], U[i]), index=i, lb=0, ub="ub_func")
    ]]]*/
    //[[[end]]]

    struct param_t
    {
        VEC(Scalar, 2) x0; // Initial state

        param_t() 
        {
            x0.setConstant(0);
        }
    };
    param_t param;

    /*[[[cog nlp.begin_func("dynamics", ("xp", "x", "u")) ]]]*/
    template <typename T> struct dynamics_t {
    inline void operator()(
    	RCVec<T,2> xp, RCVec<T,2> x, RCVec<T,1> u,
    	RVec<T,2> out,
    	param_t& param)
    //[[[end]]]
    {
        out(0) = param.x0(1) * xp(0) - (-sin(x(1)) + x(1)*x(0)) + param.x0(0);
        out(1) = param.x0(0) * xp(1) - cos(x(0))*u(0) + param.x0(1);
    }
    /*[[[cog nlp.end_func("dynamics") ]]]*/
    };
    Jacobian<dynamics_t, Scalar, param_t, 2, 2, 2, 1> J_dynamics;
    dynamics_t<Scalar> dynamics;
    //[[[end]]]

    /*[[[cog nlp.begin_func("equal", ("a", "b")) ]]]*/
    template <typename T> struct equal_t {
    inline void operator()(
    	RCVec<T,2> a, RCVec<T,2> b,
    	RVec<T,2> out,
    	param_t& param)
    //[[[end]]]
    {
        out = a - b;
    };
    /*[[[cog nlp.end_func("equal") ]]]*/
    };
    Jacobian<equal_t, Scalar, param_t, 2, 2, 2> J_equal;
    equal_t<Scalar> equal;
    //[[[end]]]

    // /*[[[cog nlp.begin_func("initial_state", ("x", )) ]]]*/
    template <typename T> struct initial_state_t {
    inline void operator()(
    	RCVec<T,2> x,
    	RVec<T,2> out,
    	param_t& param)
    // //[[[end]]]
    {
        out = x - param.x0;
    };
    // /*[[[cog nlp.end_func("initial_state") ]]]*/
    };
    Jacobian<initial_state_t, Scalar, param_t, 2, 2> J_initial_state;
    initial_state_t<Scalar> initial_state;
    // //[[[end]]]

    /*[[[cog 
    nlp.generate()
    ]]]*/
    using Base = NLP< MyNLP<Scalar, Traits> >;
    using Base::x;
    using Base::J;
    using Base::g;
    constexpr auto X(int col) {return x.SEG(2, 0 + 2 * col);};
    constexpr auto U(int col) {return x.SEG(1, 100 + 1 * col);};
    constexpr auto xss() {return x.SEG(2, 149);};
    constexpr auto uss() {return x.SEG(1, 151);};
    //[[[end]]]

    /*[[[cog 
    nlp.constraints.gen_eval() 
    nlp.constraints.gen_jacobian() 
    ]]]*/
    inline void eval()
    {
    	for(int i=0; i<49; i++)
    		dynamics(X((i+1)), X(i), U((i+1)), g.SEG(2,0+i*2), param);
    	initial_state(X(0), g.SEG(2,2), param);
    	dynamics(xss(), xss(), uss(), g.SEG(2,4), param);
    	equal(X(49), xss(), g.SEG(2,6), param);
    }

    inline void eval_jacobian()
    {
    	for(int i=0; i<49; i++)
    		J_dynamics(g.SEG(2, 0+i*2),
    		           X((i+1)), X(i), U((i+1)),
    		           J.BLK(2,2,0+i*2,((i+1)*2+0)), J.BLK(2,2,0+i*2,(i*2+0)), J.BLK(2,1,0+i*2,((i+1)*1+100)),
    		           param);
    	J_initial_state(g.SEG(2, 98),
    	                X(0),
    	                J.BLK(2,2,98,0),
    	                param);
    	J_dynamics(g.SEG(2, 100),
    	           xss(), xss(), uss(),
    	           J.BLK(2,2,100,149), J.BLK(2,2,100,149), J.BLK(2,1,100,151),
    	           param);
    	J_equal(g.SEG(2, 102),
    	        X(49), xss(),
    	        J.BLK(2,2,102,98), J.BLK(2,2,102,149),
    	        param);
    }

    //[[[end]]]

};

/*[[[cog
nlp.generate_traits()
]]]*/
struct MyTraits
{
    enum {
        num_vars = 152,
        num_eq = 104
    };
};
//[[[end]]]


using Scalar = double;
MyNLP<Scalar, MyTraits> nlp;


// struct param_t
// {
//     VEC(Scalar, 2) x0; // Initial state

//     param_t() 
//     {
//         x0.setConstant(0);
//     }
// };
// param_t param;

// template <typename T> struct dynamics_t {
// inline void operator()(
//     RCVec<T,2> xp, RCVec<T,2> x, RCVec<T,1> u,
//     RVec<T,2> out,
//     param_t& param)
// {
//     out(0) = param.x0(1) * xp(0) - (-sin(x(1)) + x(1)*x(0)) + param.x0(0);
//     out(1) = param.x0(0) * xp(1) - cos(x(0))*u(0) + param.x0(1);
// }
// };
// Jacobian<dynamics_t, Scalar, param_t, 2, 2, 2, 1> J_dynamics;
// dynamics_t<Scalar> dynamics;


int main()
{
    cout << "Hello world\n";

    nlp.x.setRandom();

    cout << "====================================\n";
    nlp.eval();
    cout << "g = " << nlp.g.transpose() << endl;

    nlp.eval_jacobian();
    cout << "J = " << endl << nlp.J << endl;

    const auto samples = 1000000;
    const auto begin = std::chrono::high_resolution_clock::now();
    for (auto i = 0; i < samples; ++i)
        nlp.eval_jacobian();
    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> duration = end - begin;
    cout << "Time per jacobian = " << duration.count() / samples << endl;

    return 0;
}