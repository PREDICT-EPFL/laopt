#include "lampc.hpp"

#include "DoubleIntegrator.hpp"
using namespace DoubleIntegrator;

/*[[[cog
from lampc import *
import cog
nlp = NLP()

# Simple linear MPC example

N = 5
X = nlp.var("x", 2, N)
U = nlp.var("u", 1, N-1)

for i in range(N-1):
    nlp.add_equality("dynamics" + str(i), "dynamics", 2, (X[i+1], X[i], U[i]))

xss = nlp.var("xss", 2, 1)
uss = nlp.var("uss", 1, 1)

# Force xss,uss to be a steady-state
nlp.add_equality("ss", "dynamics", 2, (xss, xss, uss))

# Set X[N-1] = xss
nlp.add_equality("ss_2", "steadystate", 2, (X[N-1], xss))

# Unrolls the for-loop
for i in range(N-1):
    nlp.add_inequality("input_bnds", "input_constraints", 1, (U[i], ), (-1), (1))

# Encodes the for-loop


nlp.gen_variables();
]]]*/
//[[[end]]]


template<typename Scalar>
struct Equation
{

}

template<typename Scalar, typename Traits>
struct MyNLP : public NLP< MyNLP<Scalar, Traits> >
{
    using Base = NLP< MyNLP<Scalar, Traits> >;

    public:
    enum {
        num_vars =  Traits::num_vars,
        num_eq   =  Traits::num_eq,
        num_ineq =  Traits::num_ineq
    };

    // Evaluate functions and jacobians
    void eval_impl()
    {
        /*[[[cog
        nlp.gen_eval_eq()
        ]]]*/
        //[[[end]]]
    }

    void eval_jacobians_impl()
    {
        /*[[[cog
        nlp.gen_eval_jacobian("J_eq", "J_ineq")
        ]]]*/
        //[[[end]]]
    }
};

struct MyTraits
{
    enum {
        /*[[[cog
        nlp.gen_traits()
        ]]]*/
        //[[[end]]]
    };
};

MyNLP<double, MyTraits> myNLP;

int main()
{
    myNLP.primal.setConstant(1.2);
    myNLP.eval();
    myNLP.eval_jacobians();

    cout << "myNLP.primal = " << myNLP.primal.transpose() << endl;
    cout << "myNLP.g_eq = " << myNLP.g_eq.transpose() << endl;
    cout << "myNLP.J_eq = \n" << myNLP.J_eq << endl;
    cout << "myNLP.J_ineq = \n" << myNLP.J_ineq << endl;

    const auto samples = 100000;
    const auto begin = std::chrono::high_resolution_clock::now();
    for (auto i = 0; i < samples; ++i)
        myNLP.eval_jacobians();        
    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> duration = end - begin;
    cout << "Time per jacobian evaluation = " << duration.count() / samples << endl;

    return 0;
}
