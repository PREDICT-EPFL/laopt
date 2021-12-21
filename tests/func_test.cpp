/**
 * Unit test for the construction and computation of LAMPC functions.
 */

#include "lampc.hpp"

#include <iomanip>
#include <Eigen/Eigenvalues> 
#include <type_traits>

using namespace lampc;

#include "gtest/gtest.h"

namespace {

// Build a couple functions to test
template<typename scalar_t>
struct func_t
{
    struct param_t
    {
        scalar_t q = 3;
    };

    FUNCTION(quadratic, scalar_t, param_t, (out, 2), (x, 1), (y, 2))
    {
        out << x(0)*(x(0) + y(0)), 
               p.q * x(0) + y(0) + 2 * y(1) + y(0)*y(1)*p.q;
    };

    FUNCTION(linear, scalar_t, param_t, (out, 1), (x, 1), (y, 2))
    {
        out << x(0) + y(0) + y(1);
    };


};

TEST(FunctionTest, Evaluation) {
    using scalar_t = double;
    using func = func_t<scalar_t>;

    Eigen::Matrix<scalar_t, 1, 1> x {2};
    Eigen::Matrix<scalar_t, 2, 1> y {3, 4};

    func::param_t p;
    p.q = 5;

    Eigen::Matrix<scalar_t, 2, 1> sol_eval {10, 81};
    EXPECT_EQ(func::quadratic::eval(p, x, y), sol_eval);

    func::quadratic::jacobian_return_t sol_jac;
    sol_jac.val << 10, 81;
    sol_jac.jacobian << 7, 2, 0, 
                        5, 21, 17;
    auto ret_jac = func::quadratic::jac(p, x, y);
    EXPECT_EQ(ret_jac.val, sol_jac.val);
    EXPECT_EQ(ret_jac.jacobian, sol_jac.jacobian);

    func::quadratic::hessian_return_t sol_hess;
    sol_hess.val << 10, 81;
    sol_hess.jacobian << 7, 2, 0, 
                         5, 21, 17;

    sol_hess.hessian[0] << 2, 1, 0,
                           1, 0, 0,
                           0, 0, 0;
    sol_hess.hessian[1] << 0, 0, 0,
                           0, 0, 5,
                           0, 5, 0;

    auto ret_hess = func::quadratic::hessian(p, x, y);
    EXPECT_EQ(ret_hess.val, sol_hess.val);
    EXPECT_EQ(ret_hess.jacobian, sol_hess.jacobian);
    EXPECT_EQ(ret_hess.hessian[0], sol_hess.hessian[0]);
    EXPECT_EQ(ret_hess.hessian[1], sol_hess.hessian[1]);
}

}