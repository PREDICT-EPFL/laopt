#include <iostream>
#include <typeinfo>
#include <vector>
#include <tuple>
#include <functional>
#include <string>
#include <sstream>
#include <array>
#include <chrono>
// #include <array_traits>
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

#include "DoubleIntegrator.hpp"


template <typename Derived,
          typename Scalar,
          typename NLP_traits,
          typename eq_offsets // tuple(pair(offset, size), pair(offset, size), ...)
          >
struct DenseNLP : public myNLP<DenseNLP>
{
    using num_var = NLP_traits::num_var;
    using num_eq = NLP_traits::num_eq;
    using num_ineq = NLP_traits::num_ineq;

    Eigen::Matrix<Scalar, num_var, 1> primal;

    Eigen::Matrix<Scalar, num_eq, num_var> J_eq;
    Eigen::Matrix<Scalar, num_eq, 1> g_eq;

    Eigen::Matrix<Scalar, num_ineq, num_var> J_ineq;
    Eigen::Matrix<Scalar, num_ineq, 1> g_ineq;

    Eigen::Matrix<dual, num_var, 1> primal_d; // Dual variables

    void eval()
    {
        static_cast<Derived *>(this)->eval_impl();
    }
}


template <NLP>
struct myNLP
{
    // Evaluate functions and jacobians
    void eval_impl()
    {
        // DoubleIntegrator<dual>::dynamics(col(X, i + 1), col(X, i), col(U, i));

        // for (int colIndex = 0; colIndex <= length(ceq.varIndices); colIndex++)
        // {
        //     for (int row = offset<i>; row <= offset<i> + size<i>; row++)
        //     {
        //     }
        // }
    }
}

const int N = 5;
auto X = nlp.Variable<2, N>; // tuple(pair(offset, size), pair(offset, size), ...)
auto U = nlp.Variable<1, N>;

#define col(Var, i) primal.segment(get<i>(Var)::First, get<i>(Var)::Second)

template <typename... Args>
auto WRT(Args... &&args)
{
    return std::forward_as_tuple<Args...>(args...);
}

J.block template<2,3>(3,4) = jacobian(DoubleIntegrator<dual>::dynamics, WRT(col(X, i + 1), col(X, i), col(U, i)));
J.block template<2,3>(3,4) = jacobian(DoubleIntegrator<dual>::dynamics, WRT(col(X, i + 1), col(X, i), col(U, i)));
J.block template<2,3>(3,4) = jacobian(DoubleIntegrator<dual>::dynamics, WRT(col(X, i + 1), col(X, i), col(U, i)));
J.block template<2,3>(3,4) = jacobian(DoubleIntegrator<dual>::dynamics, WRT(col(X, i + 1), col(X, i), col(U, i)));
J.block template<2,3>(3,4) = jacobian(DoubleIntegrator<dual>::dynamics, WRT(col(X, i + 1), col(X, i), col(U, i)));
J.block template<2,3>(3,4) = jacobian(DoubleIntegrator<dual>::dynamics, WRT(col(X, i + 1), col(X, i), col(U, i)));

/*[[[cog
import cog

class NLP:


N = 10
X = Variable("x", 2, N)
U = Variable("u", 1, N-1)

for i = 1:N


cog.outl(f'X = "{X}"')
]]]*/
X = "testing"
//[[[end]]]


/*[[[cog


cog.outl(f'X = "{X}"')
]]]*/
X = "testing"
//[[[end]]]
