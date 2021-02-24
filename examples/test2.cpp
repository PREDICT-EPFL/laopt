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
// Variables
#define x(i) Base::primal.template segment<2>(2*i)
#define u(i) Base::primal.template segment<1>(10 + 1*i)
#define xss Base::primal.template segment<2>(14)
#define uss Base::primal.template segment<1>(16)

//[[[end]]]


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
        // Equality constraints
        #define dynamics(i) Base::g_eq.template segment<2>(2*i)
        for (int i=0; i<N; i++)
            dynamics(i) = dynamics<double>(x(i+1),x(i),u(i));


        #define ss Base::g_eq.template segment<2>(8)
        #define ss_2 Base::g_eq.template segment<2>(10)

        ss = dynamics<double>(xss,xss,uss);
        ss_2 = steadystate<double>(x(4),xss);
        //[[[end]]]
    }

    void eval_jacobians_impl()
    {
        /*[[[cog
        nlp.gen_eval_jacobian("J_eq", "J_ineq")
        ]]]*/
        // Derivative variables
        #define Dx(i) Base::primal_d.template segment<2>(i*xxx)
        #define Du(i) Base::primal_d.template segment<1>(i*xxx)
        #define Dxss Base::primal_d.template segment<2>(14)
        #define Duss Base::primal_d.template segment<1>(16)

        // Equalities
        #define J_dynamics_x(i,j) Base::J_eq.template block<2, 2>(ind_dynamics(i), ind_x(j))
        #define J_dynamics_u(i,j) Base::J_eq.template block<2, 1>(ind_dynamics(i), ind_u(j))

        #define J_ss_xss Base::J_eq.template block<2, 2>(8, 14)
        #define J_ss_uss Base::J_eq.template block<2, 1>(8, 16)
        #define J_ss_2_x(i) Base::J_eq.template block<2, 2>(10, ind_x(i))
        #define J_ss_2_xss Base::J_eq.template block<2, 2>(10, 14)

        for (int r=0; r< num_dynamis; r++)
        {
            for (int c=0; c<size_x; c++)
                J_dynamics_x(r,c) = jacobian(dynamics<dual>, wrt(Dx(c)), at(Dx(i+1),Dx(i),Du(i)));
            for (int j=0; j<size_x; j++)
                J_dynamics_u(i,j) = jacobian(dynamics<dual>, wrt(Du(j)), at(Dx1,Dx0,Du0));
        }
        J_ss_xss = jacobian(dynamics<dual>, wrt(Dxss), at(Dxss,Dxss,Duss));
        J_ss_xss = jacobian(dynamics<dual>, wrt(Dxss), at(Dxss,Dxss,Duss));
        J_ss_uss = jacobian(dynamics<dual>, wrt(Duss), at(Dxss,Dxss,Duss));
        J_ss_2_x4 = jacobian(steadystate<dual>, wrt(Dx4), at(Dx4,Dxss));
        J_ss_2_xss = jacobian(steadystate<dual>, wrt(Dxss), at(Dx4,Dxss));

        // Inequalities
        #define J_input_bnds_u(c) Base::J_ineq.template block<1, 1>(0, ind_u(c))
        for (int r=0; r<num_input_bnds; r++)
            for (int c=0; c<num_u; c++)
                J_input_bnds_u(r) = jacobian(input_constraints<dual>, wrt(Du(c)), at(Du(c)));
        J_input_bnds_u1 = jacobian(input_constraints<dual>, wrt(Du1), at(Du1));
        J_input_bnds_u2 = jacobian(input_constraints<dual>, wrt(Du2), at(Du2));
        J_input_bnds_u3 = jacobian(input_constraints<dual>, wrt(Du3), at(Du3));
        //[[[end]]]
    }
};

struct MyTraits
{
    enum {
        /*[[[cog
        nlp.gen_traits()
        ]]]*/
        num_vars = 17,
        num_eq   = 12,
        num_ineq = 4
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
