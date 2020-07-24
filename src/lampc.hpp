#include <iostream>
#include <typeinfo>
#include <vector>
#include <tuple>
#include <functional>
#include <string>
#include <sstream>
#include <array>
#include <chrono>
#include <utility>
using namespace std;

// Eigen includes
#include <Eigen/Core>
#include <Eigen/StdVector>
using namespace Eigen;

// autodiff include
#include <autodiff/forward.hpp>
#include <autodiff/forward/eigen.hpp>
using namespace autodiff;

template<typename Scalar, 
         template<typename> typename _Eq_t, 
         template<typename> typename _Ineq_t, 
         template<typename> typename _Cost_t>
struct NLP
{
    using Eq_t = _Eq_t<Scalar>;
    using Ineq_t = _Ineq_t<Scalar>;
    using Cost_t = _Cost_t<Scalar>;

    enum {
        nvars =  Eq_t::nvars,
        num_eq   =  Eq_t::nfuncs,
        num_ineq =  Ineq_t::nfuncs
    };

    static_assert(Eq_t::nvars == Ineq_t::nvars, 
            "Number of variables in equalities, inequalities and cost must match");
    static_assert(Eq_t::nvars == Cost_t::nvars,
            "Number of variables in equalities, inequalities and cost must match");
    static_assert(Cost_t::nfuncs == 1,
            "Cost function must be scalar");

    Eigen::Matrix<Scalar, nvars, 1> primal;
    Eigen::Matrix<dual, nvars, 1> primal_d; // Dual variables for gradients

    Eq_t eq;      // Vector function equalities
    Ineq_t ineq;  // Vector function inequalities
    Cost_t cost;  // Scalar function cost

    void eval()
    {
        eq.eval(primal);
        ineq.eval(primal);
        cost.eval(primal);
    }

    void eval_jacobians()
    {
        // Copy primal to dual variables
        primal_d = primal;

        eq.eval_jacobian(primal_d);
        ineq.eval_jacobian(primal_d);
        cost.eval_jacobian(primal_d);
    }
};

// Template for a function type
// We can add all sorts of static_asserts in NLP to 
// confirm that the function types are the right structure
// template<typename Scalar>
// struct Function
// {
//     enum {
//         nvars =  ***,
//         nfuncs = ***
//     };

//     Eigen::Matrix<Scalar, nfuncs, nvars> J; // Jacobian of function
//     Eigen::Matrix<Scalar, nfuncs, 1>     f; // Value of function

//     void Function() {
//         J.setZero();
//         f.setZero();
//     }

//     void eval(Ref<Matrix<Scalar, nvars, 1>> x)
//     {
//         f(x) = ...
//     }

//     void eval_jacobian(Ref<Matrix<dual, nvars, 1>> x)
//     {
//         J(x) = ...
//     }
// }