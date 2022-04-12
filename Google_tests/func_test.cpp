/**
 * Unit test for the construction and computation of LAMPC functions.
 */

#include <iostream>

#include "lampc.hpp"

#include "gtest/gtest.h"

namespace {

template<typename scalar_t=double, typename diff_t=scalar_t>
EIGEN_STRONG_INLINE Eigen::Vector<diff_t, 2> test_function(
        const Eigen::Ref<const Eigen::Vector<diff_t, 2>>& x, 
        const Eigen::Ref<const Eigen::Vector<diff_t, 1>>& u) noexcept
{
    Eigen::Matrix<scalar_t, 2, 2> A{{1,2},{3,4}};
    Eigen::Matrix<scalar_t, 2, 1> B{{5},{6}};
    return x(0) * (A.template cast<diff_t>() * x + B.template cast<diff_t>() * u);
}

/**
 * Compute the jacobian and hessian of a function
 */
TEST(FunctionTest, Construction) {
    using scalar_t = double;

    using Function = lampc::Function<scalar_t, 2, 2,1>;
    auto function = make_function(Function, test_function);

    Eigen::VectorX<scalar_t> x(2); x << 1,2;
    Eigen::VectorX<scalar_t> u(1); u << 3;

    // Evaluation
    EXPECT_EQ(function(x,u), (Eigen::VectorX<scalar_t>(2) << 20,29).finished());

    // Jacobian
    EXPECT_EQ(function(x,u,lampc::Jacobian()), 
        std::make_pair(
            (Eigen::Vector<scalar_t,2>() << 20,29).finished(),
            (Eigen::Matrix<scalar_t,2,3>() << 21,2,5,32,4,6).finished()
        ));

    // Hessian
    EXPECT_EQ(function(x,u,lampc::Hessian()), 
        std::make_tuple(
            (Eigen::Vector<scalar_t,2>() << 20,29).finished(),
            (Eigen::Matrix<scalar_t,2,3>() << 21,2,5,32,4,6).finished(),
            Function::hessian_t{
                (Eigen::Matrix<scalar_t,3,3>() << 2,2,5,2,0,0,5,0,0).finished(),
                (Eigen::Matrix<scalar_t,3,3>() << 6,4,6,4,0,0,6,0,0).finished()
            }
        ));
}

}