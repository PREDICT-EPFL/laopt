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
    enum {
    	nfuncs = 14,
    	nvars = 17
    };

    Eigen::Matrix<Scalar, nfuncs, nvars> J; // Jacobian of function
    Eigen::Matrix<Scalar, nfuncs, 1>     f; // Value of function

    void initialize() {
    	J.setZero();
    	f.setZero();
    }

    void eval(Ref<Matrix<Scalar, nvars, 1>> x)
    {
    	for(int i=0; i<4; i++)
    		f.SEG(2,0+i*2) = dynamics<Scalar>(_X((i+1)), _X(i), _U((i+1)));
    	f.SEG(2,8) = initial_state<Scalar>(_X(0));
    	f.SEG(2,10) = dynamics<Scalar>(_xss, _xss, _uss);
    	f.SEG(2,12) = equal<Scalar>(_X(4), _xss);
    }

    void eval_jacobian(Ref<Matrix<dual, nvars, 1>> x)
    {
    	// Lambda functions to capture object, so we can get pointers to member functions
    	static const auto _dynamics = [this](
    		Ref<Eigen::Matrix<dual,2,1>> x0,
    		Ref<Eigen::Matrix<dual,2,1>> x1,
    		Ref<Eigen::Matrix<dual,1,1>> x2)
    		{return this->dynamics<dual>(x0,x1,x2);};
    	static const auto _initial_state = [this](
    		Ref<Eigen::Matrix<dual,2,1>> x0)
    		{return this->initial_state<dual>(x0);};
    	static const auto _equal = [this](
    		Ref<Eigen::Matrix<dual,2,1>> x0,
    		Ref<Eigen::Matrix<dual,2,1>> x1)
    		{return this->equal<dual>(x0,x1);};

    	// Compute Jacobians block-wise
    	for(int i=0; i<4; i++)
    	{
    		J.BLK(2,2,0+i*2,((i+1)*2+0)) = jacobian(_dynamics, wrt(_X((i+1))), at(_X((i+1)),_X(i),_U((i+1))));
    		J.BLK(2,2,0+i*2,(i*2+0)) = jacobian(_dynamics, wrt(_X(i)), at(_X((i+1)),_X(i),_U((i+1))));
    		J.BLK(2,1,0+i*2,((i+1)*1+10)) = jacobian(_dynamics, wrt(_U((i+1))), at(_X((i+1)),_X(i),_U((i+1))));
    	}
    	J.BLK(2,2,8,0) = jacobian(_initial_state, wrt(_X(0)), at(_X(0)));
    	J.BLK(2,2,10,14) = jacobian(_dynamics, wrt(_xss), at(_xss,_xss,_uss));
    	J.BLK(2,2,10,14) = jacobian(_dynamics, wrt(_xss), at(_xss,_xss,_uss));
    	J.BLK(2,1,10,16) = jacobian(_dynamics, wrt(_uss), at(_xss,_xss,_uss));
    	J.BLK(2,2,12,8) = jacobian(_equal, wrt(_X(4)), at(_X(4),_xss));
    	J.BLK(2,2,12,14) = jacobian(_equal, wrt(_xss), at(_X(4),_xss));
    }

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
    enum {
    	nfuncs = 1,
    	nvars = 17
    };

    Eigen::Matrix<Scalar, nfuncs, nvars> J; // Jacobian of function
    Eigen::Matrix<Scalar, nfuncs, 1>     f; // Value of function

    void initialize() {
    	J.setZero();
    	f.setZero();
    }

    void eval(Ref<Matrix<Scalar, nvars, 1>> x)
    {
    	f.SEG(1,0) = cost<Scalar>(_X(1), _U(0));
    }

    void eval_jacobian(Ref<Matrix<dual, nvars, 1>> x)
    {
    	// Lambda functions to capture object, so we can get pointers to member functions
    	static const auto _cost = [this](
    		Ref<Eigen::Matrix<dual,2,1>> x0,
    		Ref<Eigen::Matrix<dual,1,1>> x1)
    		{return this->cost<dual>(x0,x1);};

    	// Compute Jacobians block-wise
    	J.BLK(1,2,0,2) = jacobian(_cost, wrt(_X(1)), at(_X(1),_U(0)));
    	J.BLK(1,1,0,10) = jacobian(_cost, wrt(_U(0)), at(_X(1),_U(0)));
    }

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
