#include "lampc.hpp"

// #include "DoubleIntegrator.hpp"
// using namespace DoubleIntegrator;

/*[[[cog
from lampc import *
import cog
nlp = NLP()

# Simple linear MPC example
N = 5
x0 = nlp.var("x0", 2, 1)
U = nlp.var("U", 1, N-1)
xss = nlp.var("xss", 2, 1)
uss = nlp.var("uss", 1, 1)

nlp.gen_variables();
]]]*/
#define SEG(size, offset) template segment<size>(offset)
#define BLK(x_size, y_size, x_offset, y_offset) template block<x_size, y_size>(x_offset, y_offset)
#define _X(col) x.SEG(2, 0 + 2 * col)
#define _U(col) x.SEG(1, 10 + 1 * col)
#define _xss x.SEG(2, 14)
#define _uss x.SEG(1, 16)
//[[[end]]]


template<typename Scalar>
struct Equalities
{
    /*[[[cog
    funcs = Functions(nlp, "Equalities")

    i = Index(range(N-1))
    funcs.append("dynamics", 2, (X[i+1], X[i], U[i]), i)

    # Force xss,uss to be a steady-state
    funcs.append("dynamics", 2, (xss, xss, uss))

    # Set X[N-1] = xss
    funcs.append("steadystate", 2, (X[N-1], xss))
    ]]]*/
    //[[[end]]]

    enum {
        nx = 2,
        nu = 1
    };

    template<typename T> using StateType = Matrix<T, nx, 1>;
    template<typename T> using InputType = Matrix<T, nu, 1>;

    template <typename T>
    static StateType<T> dynamics(Ref<StateType<T>> xp,
                                 Ref<StateType<T>> x,
                                 Ref<InputType<T>> u)
    {
        StateType<T> eq;
        eq << xp[0] - (-sin(x[1]) + x[1]*x[0]), xp[1] - cos(x[0])*u[0];
        return eq;
    };
 
    // x == xss
    template <typename T>
    static StateType<T> steadystate(Ref<StateType<T>> x,
                                    Ref<StateType<T>> xss)
    {
        return x - xss;
    };

    /*[[[cog
    funcs.gen()
    ]]]*/
    enum {
    	nfuncs = 12,
    	nvars = 17
    };

    Eigen::Matrix<Scalar, nfuncs, nvars> J; // Jacobian of function
    Eigen::Matrix<Scalar, nfuncs, 1>     f; // Value of function

    Equalities() {
    	J.setZero();
    	f.setZero();
    }

    void eval(Ref<Matrix<Scalar, nvars, 1>> x)
    {
    	for(int i=0; i<4; i++)
    		f.SEG(2,0+i*2) = dynamics<Scalar>(_X((i+1)), _X(i), _U(i));
    	f.SEG(2,8) = dynamics<Scalar>(_xss, _xss, _uss);
    	f.SEG(2,10) = steadystate<Scalar>(_X(4), _xss);
    }

    void eval_jacobian(Ref<Matrix<dual, nvars, 1>> x)
    {
    	for(int i=0; i<4; i++)
    	{
    		J.BLK(2,2,0+i*2,((i+1)*2+0)) = jacobian(dynamics<dual>, wrt(_X((i+1))), at(_X((i+1)),_X(i),_U(i)));
    		J.BLK(2,2,0+i*2,(i*2+0)) = jacobian(dynamics<dual>, wrt(_X(i)), at(_X((i+1)),_X(i),_U(i)));
    		J.BLK(2,1,0+i*2,(i*1+10)) = jacobian(dynamics<dual>, wrt(_U(i)), at(_X((i+1)),_X(i),_U(i)));
    	}
    	J.BLK(2,2,8,14) = jacobian(dynamics<dual>, wrt(_xss), at(_xss,_xss,_uss));
    	J.BLK(2,2,8,14) = jacobian(dynamics<dual>, wrt(_xss), at(_xss,_xss,_uss));
    	J.BLK(2,1,8,16) = jacobian(dynamics<dual>, wrt(_uss), at(_xss,_xss,_uss));
    	J.BLK(2,2,10,8) = jacobian(steadystate<dual>, wrt(_X(4)), at(_X(4),_xss));
    	J.BLK(2,2,10,14) = jacobian(steadystate<dual>, wrt(_xss), at(_X(4),_xss));
    }

    //[[[end]]]
};

template<typename Scalar>
struct Cost
{
    /*[[[cog
    funcs = Functions(nlp, "Cost")

    funcs.append("cost", 1, (X[1], U[0]))
    ]]]*/
    //[[[end]]]

    template<typename S>
    static Matrix<S, 1, 1> cost(Ref<Matrix<S, 2, 1>> x, Ref<Matrix<S, 1, 1>> u)
    {
        return u;
    }

    /*[[[cog
    funcs.gen()
    ]]]*/
    enum {
    	nfuncs = 1,
    	nvars = 17
    };

    Eigen::Matrix<Scalar, nfuncs, nvars> J; // Jacobian of function
    Eigen::Matrix<Scalar, nfuncs, 1>     f; // Value of function

    Cost() {
    	J.setZero();
    	f.setZero();
    }

    void eval(Ref<Matrix<Scalar, nvars, 1>> x)
    {
    	f.SEG(1,0) = cost<Scalar>(_X(1), _U(0));
    }

    void eval_jacobian(Ref<Matrix<dual, nvars, 1>> x)
    {
    	J.BLK(1,2,0,2) = jacobian(cost<dual>, wrt(_X(1)), at(_X(1),_U(0)));
    	J.BLK(1,1,0,10) = jacobian(cost<dual>, wrt(_U(0)), at(_X(1),_U(0)));
    }

    //[[[end]]]
};

using Scalar = double;
NLP<Scalar, Equalities, Equalities, Cost> nlp;

int main()
{    
    nlp.primal.setConstant(1.2);
    nlp.eval();
    nlp.eval_jacobians();

    cout << "nlp.primal = " << nlp.primal.transpose() << endl;
    cout << "nlp.eq.f = " << nlp.eq.f.transpose() << endl;
    cout << "nlp.eq.J = \n" << nlp.eq.J << endl;
    cout << "nlp.ineq.J = \n" << nlp.ineq.J << endl;

    const auto samples = 1000;
    const auto begin = std::chrono::high_resolution_clock::now();
    for (auto i = 0; i < samples; ++i)
        nlp.eval_jacobians();        
    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> duration = end - begin;
    cout << "Time per jacobian evaluation = " << duration.count() / samples << endl;

    return 0;
}
