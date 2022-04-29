/**
 * Unit test for the construction and computation of LAMPC functions.
 */

#include <iostream>

#include "lampc_utility.hpp"
#include "lampc_function.hpp"
#include "bsmatrix.hpp"
#include "functions.hpp"

#include "test_utils.hpp"

#include "gtest/gtest.h"

namespace {

template<typename scalar_t>
struct TestFunction : public lampc::MakeDifferentiable<TestFunction<scalar_t>, scalar_t, 2, 2,1>
{
    // using lampc::MakeDifferentiable<TestFunction<scalar_t>, scalar_t, 2, 2,1>::operator();

    Eigen::Matrix<scalar_t, 2, 2> A{{1,2},{3,4}};
    Eigen::Matrix<scalar_t, 2, 1> B{{5},{6}};

    template<typename diff_t>
    EIGEN_STRONG_INLINE Eigen::Vector<diff_t, 2> 
    impl( const Eigen::Ref< const Eigen::Vector<diff_t, 2> >& x, 
          const Eigen::Ref< const Eigen::Vector<diff_t, 1> >& u) noexcept        
    {
        // return Eigen::Vector<diff_t,2>::Constant(static_cast<diff_t>(3));
        return (A.template cast<diff_t>() * x + B.template cast<diff_t>() * u);
    }
};

/**
 * Compute the jacobian and hessian of a function
 */

TEST(FunctionTest, Hessian) {
    using scalar_t = double;
    using test_t = TestFunction<scalar_t>;
    test_t test;

    Eigen::Vector<scalar_t, 2> x;
    Eigen::Vector<scalar_t, 1> u;
    x << 1,2;
    u << 3;

    // We compute the evaluation sequence for the hessian,
    // but can compute just the eval or jacobian too.
    test(lampc::Eval(), x, u);   
    test(lampc::Jacobian(), x, u);   
    test(lampc::Hessian(), x, u);   

    test_t::value_t value;
    test_t::jacobian_t jacobian;
    test_t::hessian_array_t hessian;

    std::tie(value, jacobian, hessian) = test(lampc::Hessian(), x, u);   

    Eigen::Vector<scalar_t,2> val;
    val << 20,29;
    EXPECT_EQ(value, val);

    Eigen::Matrix<scalar_t,2,3> jac;
    jac << 1,2,5,3,4,6;
    EXPECT_EQ(jacobian, jac);

    Eigen::Matrix<scalar_t,3,3> H;
    H << 0,0,0,0,0,0,0,0,0;
    EXPECT_EQ(hessian[0], H);
    H << 0,0,0,0,0,0,0,0,0;
    EXPECT_EQ(hessian[1], H);
}


/**************************
 * Test the RK4 integrator
 **************************/

/**
 * Continuous-time dynamics - double integrator
 */
template<typename scalar_t>
struct sys_t
{
    Eigen::Matrix<scalar_t, 2, 2> A{{0,1},{0,0}};
    Eigen::Matrix<scalar_t, 2, 1> B{{0},{1}};

    template<typename diff_t>
    EIGEN_STRONG_INLINE Eigen::Vector<diff_t, 2> 
    impl( const Eigen::Ref< const Eigen::Vector<diff_t, 2> >& x, 
          const Eigen::Ref< const Eigen::Vector<diff_t, 1> >& u) noexcept        
    {
        return A.template cast<diff_t>() * x + B.template cast<diff_t>() * u;
    }
};

TEST(RK4Test, SimpleTest) {
    using scalar_t = double;
    sys_t<scalar_t> sys;
    using dsys_t = lampc::functions::RK4<sys_t<scalar_t>, scalar_t, 2, 1>;
    dsys_t dsys(sys, 0.1);

    // Compute the discrete-time system and its jacobian
    Eigen::Vector<scalar_t, 2> x;
    Eigen::Vector<scalar_t, 1> u;
    x << 1,2;
    u << 3;

    dsys_t::value_t value;
    dsys_t::jacobian_t jacobian;

    // Eval only - doesn't compute jacobian
    value = dsys(lampc::Eval(), x, u); 

    Eigen::Vector<scalar_t,2> val;
    val << 1.215, 2.3;

    EXPECT_TRUE(val.isApprox(value, 1e-4));

    // Compute jacobian and evaluation
    std::tie(value, jacobian) = dsys(lampc::Jacobian(), x, u);   

    Eigen::Matrix<scalar_t,2,3> jac;
    jac << 1,0.1,0.005,0,1,0.1;
    EXPECT_TRUE(val.isApprox(value, 1e-4));
    EXPECT_TRUE(jac.isApprox(jacobian, 1e-4));
}


/***********************************************
 * Test the BSMatrix interface to the functions
 ***********************************************/

TEST(BSMatrixTest, Jacobian) {
    using scalar_t = double;
    TestFunction<scalar_t> test;

    Eigen::Vector<scalar_t, 2> x;
    Eigen::Vector<scalar_t, 1> u;
    x << 1,2;
    u << 3;

    // Function to run
    auto f = [&](auto& value, auto& jacobian)
    {
        for(int i=0; i<5; i++)
        {
            x(0) = i;
            test(x, u, value(seqN(i*2,2),0), jacobian(seqN(i*2,2),multiSeq(seqN(i*2,2),seqN(10+i,1))));
        }
    };

    lampc::BSMatrixSparsity sparsity_value(10, 1);
    lampc::BSMatrixSparsity sparsity_jacobian(10, 15);
    f(sparsity_value, sparsity_jacobian); // Extract sparsity pattern

    auto value_tape = sparsity_value.makeBSTape(10, 1);
    auto jacobian_tape = sparsity_jacobian.makeBSTape(10, 15);
    f(value_tape, jacobian_tape); // Extract operation sequence

    auto BS_value = value_tape.template makeBSMatrix<scalar_t>();
    auto BS_jacobian = jacobian_tape.template makeBSMatrix<scalar_t>();

    Eigen::Vector<scalar_t, 10> S_value;
    Eigen::SparseMatrix<scalar_t> S_jacobian;

    BS_value.set_target(S_value);
    BS_jacobian.initialize_matrix(S_jacobian);

    f(BS_value, BS_jacobian);

    Eigen::MatrixX<scalar_t> value_g(10,1);
    value_g << 19,26,20,29,21,32,22,35,23,38;
    EXPECT_TRUE(value_g.isApprox(S_value, 1e-4));

    auto jacobian_g = triplet_to_sparse<double>(10,15,{{0,0,1},{1,0,3},{0,1,2},{1,1,4},{2,2,1},{3,2,3},{2,3,2},{3,3,4},{4,4,1},{5,4,3},{4,5,2},{5,5,4},{6,6,1},{7,6,3},{6,7,2},{7,7,4},{8,8,1},{9,8,3},{8,9,2},{9,9,4},{0,10,5},{1,10,6},{2,11,5},{3,11,6},{4,12,5},{5,12,6},{6,13,5},{7,13,6},{8,14,5},{9,14,6}});
    EXPECT_TRUE(jacobian_g.isApprox(S_jacobian, 1e-4));
}

/**************************
 * Test custom functions
 **************************/

TEST(FunctionsTest, eq) {

    // Discrete dynamics
    using scalar_t = double;
    sys_t<scalar_t> sys;
    using dsys_t = lampc::functions::RK4<sys_t<scalar_t>, scalar_t, 2, 1>;
    dsys_t dsys(sys, 0.1);
  
    // Equality constraint dsys(x,u) - xp
    using dsys_eq_t = lampc::functions::eq<dsys_t, 2, 2, 1>;
    dsys_eq_t dsys_eq(dsys);

    dsys_eq_t::value_t value;
    dsys_eq_t::jacobian_t jacobian;

    Eigen::Vector<scalar_t, 2> xp;
    Eigen::Vector<scalar_t, 2> x;
    Eigen::Vector<scalar_t, 1> u;
    x << 1,2;
    u << 3;
    xp << 4,5;

    std::tie(value, jacobian) = dsys_eq(lampc::Jacobian(), xp,x,u);    

    std::cout << "value = " << value.transpose() << std::endl;
    std::cout << "jac \n" << jacobian << std::endl;
}

TEST(FunctionsTest, id) {

    // Discrete dynamics
    using scalar_t = double;
    lampc::functions::id<scalar_t,2> id2;
  
    Eigen::Vector<scalar_t, 2> value;
    Eigen::Matrix<scalar_t, 2, 2> jacobian;

    Eigen::Vector<scalar_t, 2> x;
    x << 1,2;

    jacobian.array() = 20;
    std::tie(value, jacobian) = id2(lampc::Jacobian(), x);

    std::cout << "value = " << value.transpose() << std::endl;
    std::cout << "jac \n" << jacobian << std::endl;
}

}