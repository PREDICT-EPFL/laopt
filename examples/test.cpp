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
    //[[[end]]]
    {
        out(0) = param.x0(1) * xp(0) - (-sin(x(1)) + x(1)*x(0)) + param.x0(0);
        out(1) = param.x0(0) * xp(1) - cos(x(0))*u(0) + param.x0(1);
    }
    /*[[[cog nlp.end_func("dynamics") ]]]*/
    //[[[end]]]

    /*[[[cog nlp.begin_func("equal", ("a", "b")) ]]]*/
    //[[[end]]]
    {
        out = a - b;
    };
    /*[[[cog nlp.end_func("equal") ]]]*/
    //[[[end]]]

    // /*[[[cog nlp.begin_func("initial_state", ("x", )) ]]]*/
    // //[[[end]]]
    {
        out = x - param.x0;
    };
    // /*[[[cog nlp.end_func("initial_state") ]]]*/
    // //[[[end]]]

    /*[[[cog 
    nlp.generate()
    ]]]*/
    //[[[end]]]

    /*[[[cog 
    nlp.constraints.gen_eval() 
    nlp.constraints.gen_jacobian() 
    ]]]*/
    //[[[end]]]

};

/*[[[cog
nlp.generate_traits()
]]]*/
//[[[end]]]


using Scalar = double;
MyNLP<Scalar, MyTraits> nlp;


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