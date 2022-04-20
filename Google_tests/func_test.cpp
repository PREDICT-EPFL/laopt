/**
 * Unit test for the construction and computation of LAMPC functions.
 */

#include <iostream>

#include "lampc.hpp"

#include "gtest/gtest.h"

namespace {

template<typename scalar_t, template<typename,typename> class Type>
struct TestFunction : public Type<TestFunction<scalar_t,Type>, lampc::FuncInfo<scalar_t, 2, 2,1>>
{
    Eigen::Matrix<scalar_t, 2, 2> A{{0,1},{0,0}};
    Eigen::Matrix<scalar_t, 2, 1> B{{0},{1}};

    template<typename diff_t>
    EIGEN_STRONG_INLINE Eigen::Vector<diff_t, 2> 
    impl( const Eigen::Ref< const Eigen::Vector<diff_t, 2> >& x, 
          const Eigen::Ref< const Eigen::Vector<diff_t, 1> >& u) noexcept        
    {
        return x(0) * (A.template cast<diff_t>() * x + B.template cast<diff_t>() * u);
    }
};

/**
 * Compute the jacobian and hessian of a function
 */
TEST(FunctionTest, Eval) {
    using scalar_t = double;
    TestFunction<scalar_t, lampc::Eval> test;

    Eigen::Vector<scalar_t, 2> x;
    Eigen::Vector<scalar_t, 1> u;
    x << 1,2;
    u << 3;
    test.eval(x, u);

    Eigen::Vector<scalar_t,2> ground;
    ground << 2,3;
    EXPECT_EQ(test.value, ground);
}

TEST(FunctionTest, Jacobian) {
    using scalar_t = double;
    TestFunction<scalar_t, lampc::Jacobian> test;

    Eigen::Vector<scalar_t, 2> x;
    Eigen::Vector<scalar_t, 1> u;
    x << 1,2;
    u << 3;

    test.eval_jacobian(x, u);   

    Eigen::Vector<scalar_t,2> val;
    val << 2,3;
    EXPECT_EQ(test.value, val);

    Eigen::Matrix<scalar_t,2,3> jac;
    jac << 2,1,0,3,0,1;
    EXPECT_EQ(test.jacobian, jac);
}

TEST(FunctionTest, Hessian) {
    using scalar_t = double;
    TestFunction<scalar_t, lampc::Hessian> test;

    Eigen::Vector<scalar_t, 2> x;
    Eigen::Vector<scalar_t, 1> u;
    x << 1,2;
    u << 3;

    test.eval_hessian(x, u);   

    Eigen::Vector<scalar_t,2> val;
    val << 2,3;
    EXPECT_EQ(test.value, val);

    Eigen::Matrix<scalar_t,2,3> jac;
    jac << 2,1,0,3,0,1;
    EXPECT_EQ(test.jacobian, jac);

    Eigen::Matrix<scalar_t,3,3> H;
    H << 0,1,0,1,0,0,0,0,0;
    EXPECT_EQ(test.hessian[0], H);
    H << 0,0,1,0,0,0,1,0,0;
    EXPECT_EQ(test.hessian[1], H);
}


}