/**
 * Unit test for the construction and computation of LAMPC functions.
 */

#include <iostream>

#include "laopt/laopt.hpp"

#include "test_utils.hpp"
#include "gtest/gtest.h"

namespace {

template<typename scalar_t>
struct VectorFunction : public laopt::Differentiable<VectorFunction<scalar_t>, true>
{
    const Eigen::Matrix<scalar_t, 2, 2> A{{1,2},{3,4}};
    const Eigen::Matrix<scalar_t, 2, 1> B{{5},{6}};

    template<typename X, typename U, typename Scalar = typename Eigen::MatrixBase<X>::Scalar>
    EIGEN_STRONG_INLINE Eigen::Vector<Scalar, 2>
    function_impl(const Eigen::MatrixBase<X>& x, const Eigen::MatrixBase<U>& u) noexcept
    {
        return x(0) * (A * x + B * u);
    }
};

template<typename scalar_t>
struct VectorFunctionParameter : public laopt::Differentiable<VectorFunctionParameter<scalar_t>, true>
{
    const Eigen::Matrix<scalar_t, 2, 2> A{{1,2},{3,4}};
    const Eigen::Matrix<scalar_t, 2, 1> B{{5},{6}};

    template<typename X, typename U, typename Scalar = typename Eigen::MatrixBase<X>::Scalar>
    EIGEN_STRONG_INLINE Eigen::Vector<Scalar, 2>
    function_impl(const double p, const Eigen::MatrixBase<X>& x, const Eigen::MatrixBase<U>& u) noexcept
    {
        return p * (A * x + B * u);
    }
};

template<typename scalar_t>
struct ScalarFunction : public laopt::Differentiable<ScalarFunction<scalar_t>, true>
{
    const Eigen::Matrix<scalar_t, 2, 2> Q{{1,0},{0,1}};
    const Eigen::Matrix<scalar_t, 1, 1> R{2};

    template<typename X, typename U, typename Scalar = typename Eigen::MatrixBase<X>::Scalar>
    EIGEN_STRONG_INLINE Scalar
    function_impl(const Eigen::MatrixBase<X>& x, const Eigen::MatrixBase<U>& u) noexcept
    {
        return x.dot(Q * x) + u.dot(R * u);
    }
};


TEST(FunctionTest, FuncInfo)
{
    using scalar_t = double;

    using Arg1 = laopt::Variable<scalar_t, 2>;
    using Arg2 = laopt::Variable<scalar_t, 1>;

    {
        using Info = VectorFunction<scalar_t>::FuncInfo<laopt::DefaultTag, Arg1, Arg2>;
        int n_inputs = Info::n_inputs;
        int n_outputs = Info::n_outputs;
        EXPECT_EQ(n_inputs, 3);
        EXPECT_EQ(n_outputs, 2);
    }

    {
        using Info = VectorFunctionParameter<scalar_t>::FuncInfo<laopt::DefaultTag, scalar_t, Arg1, Arg2>;
        int n_inputs = Info::n_inputs;
        int n_outputs = Info::n_outputs;
        EXPECT_EQ(n_inputs, 3);
        EXPECT_EQ(n_outputs, 2);
    }

    {
        using Info = ScalarFunction<scalar_t>::FuncInfo<laopt::DefaultTag, Arg1, Arg2>;
        int n_inputs = Info::n_inputs;
        int n_outputs = Info::n_outputs;
        EXPECT_EQ(n_inputs, 3);
        EXPECT_EQ(n_outputs, 1);
    }
}

TEST(FunctionTest, VectorFunction)
{
    using scalar_t = double;
    VectorFunction<scalar_t> test;

    laopt::IndexedVector<Eigen::Vector<scalar_t, 2>> x;
    laopt::IndexedVector<Eigen::Vector<scalar_t, 1>> u;
    x << 2, 1;
    u << 3;

    Eigen::Vector<scalar_t, 2> value;
    Eigen::Matrix<scalar_t, 2, 3> jacobian;
    std::array<Eigen::Matrix<scalar_t, 3, 3>, 2> hessian;

    value = test(x.cast_base(), u.cast_base());

    Eigen::Vector<scalar_t, 2> val;
    val << 38, 56;
    EXPECT_EQ(value, val);

    value.setZero();
    jacobian.setZero();
    test.jacobian(value, jacobian, x, u);

    Eigen::Matrix<scalar_t, 2, 3> jac;
    jac << 21, 4, 10, 34, 8, 12;
    EXPECT_EQ(jacobian, jac);
}

TEST(FunctionTest, VectorFunctionParameter)
{
    using scalar_t = double;
    VectorFunctionParameter<scalar_t> test;

    scalar_t p = 5;
    laopt::IndexedVector<Eigen::Vector<scalar_t, 2>> x;
    laopt::IndexedVector<Eigen::Vector<scalar_t, 1>> u;
    x << 2, 1;
    u << 3;

    Eigen::Vector<scalar_t, 2> value;
    Eigen::Matrix<scalar_t, 2, 3> jacobian;
    std::array<Eigen::Matrix<scalar_t, 3, 3>, 2> hessian;

    value = test(p, x.cast_base(), u.cast_base());

    Eigen::Vector<scalar_t, 2> val;
    val << 95, 140;
    EXPECT_EQ(value, val);

    value.setZero();
    jacobian.setZero();
    test.jacobian(value, jacobian, p, x, u);

    Eigen::Matrix<scalar_t, 2, 3> jac;
    jac << 5, 10, 25, 15, 20, 30;
    EXPECT_EQ(jacobian, jac);
}

TEST(FunctionTest, ScalarFunction)
{
    using scalar_t = double;

    using Arg1 = laopt::IndexedVector<Eigen::Vector<scalar_t, 2>>;
    using Arg2 = laopt::IndexedVector<Eigen::Vector<scalar_t, 1>>;

    ScalarFunction<scalar_t> f;
    Arg1 x;
    Arg2 u;
    x << 1, 2;
    u << 3;
    Eigen::Vector<scalar_t, 1> weight;
    weight << 1;

    using info = ScalarFunction<scalar_t>::FuncInfo<laopt::DefaultTag, Arg1, Arg2>;
    info::scalar_t value;
    info::gradient_t gradient;
    info::hessian_t hessian;

    value = f.wsum(weight, x, u);
    EXPECT_EQ(value, 23);

    gradient.setZero();
    value = f.gradient(gradient, weight, x, u);
    Eigen::Vector<scalar_t, 3> gradient_g;
    gradient_g << 2, 4, 12;
    EXPECT_EQ(value, 23);
    EXPECT_EQ(gradient, gradient_g);

    gradient.setZero();
    hessian.setZero();
    value = f.hessian(gradient, hessian, weight, x, u);
    EXPECT_EQ(value, 23);
    EXPECT_EQ(gradient, gradient_g);
    Eigen::Matrix<scalar_t, 3, 3> hessian_g;
    hessian_g << 2, 0, 0, 0, 2, 0, 0, 0, 4;
    EXPECT_EQ(hessian, hessian_g);
}

/**************************
 * Test the RK4 integrator
 **************************/

/**
 * Continuous-time dynamics - double integrator
 */
template<typename scalar_t>
struct sys_t : public laopt::Differentiable<sys_t<scalar_t>, true>
{
    Eigen::Matrix<scalar_t, 2, 2> A{{0,1},{0,0}};
    Eigen::Matrix<scalar_t, 2, 1> B{{0},{1}};

    template<typename X, typename U, typename Scalar = typename Eigen::MatrixBase<X>::Scalar>
    EIGEN_STRONG_INLINE Eigen::Vector<Scalar, 2>
    function_impl( const Eigen::MatrixBase<X>& x, const Eigen::MatrixBase<U>& u) noexcept
    {
        return A * x + B * u;
    }
};

TEST(FunctionTest, RK4) {
    using scalar_t = double;
    sys_t<scalar_t> sys;
    using dsys_t = laopt::functions::RK4<sys_t<scalar_t>, scalar_t>;
    dsys_t dsys(sys, 0.1);

    // Compute the discrete-time system and its jacobian
    laopt::IndexedVector<Eigen::Vector<scalar_t, 2>> x;
    laopt::IndexedVector<Eigen::Vector<scalar_t, 1>> u;
    x << 1,2;
    u << 3;

    Eigen::Vector<scalar_t, 2> value;
    Eigen::Matrix<scalar_t, 2, 3> jacobian;

    // Eval only - doesn't compute jacobian
    value = dsys(x.cast_base(), u.cast_base());

    Eigen::Vector<scalar_t, 2> val;
    val << 1.215, 2.3;

    EXPECT_TRUE(val.isApprox(value, 1e-4));

    // Compute jacobian and evaluation
    value.setZero();
    jacobian.setZero();
    dsys.jacobian(value, jacobian, x, u);

    Eigen::Matrix<scalar_t, 2, 3> jac;
    jac << 1, 0.1, 0.005, 0, 1, 0.1;
    EXPECT_TRUE(val.isApprox(value, 1e-4));
    EXPECT_TRUE(jac.isApprox(jacobian, 1e-4));
}


/***********************************************
 * Test the BSMatrix interface to the functions
 ***********************************************/

TEST(FunctionTest, BSMatrixJacobain) {
    using scalar_t = double;
    VectorFunction<scalar_t> test;

    laopt::IndexedVector<Eigen::Vector<scalar_t, 2>> x;
    laopt::IndexedVector<Eigen::Vector<scalar_t, 1>> u;
    x << 1,2;
    u << 3;

    // Function to run
    auto f = [&](auto& value, auto& jacobian)
    {
        for(int i = 0; i < 5; i++)
        {
            x << i, 2 * i;
            u << 3 * i;
            test.jacobian(value(Eigen::seqN(i*2, Eigen::fix<2>)), jacobian(Eigen::seqN(i*2, Eigen::fix<2>), laopt::multiSeq_to_index(Eigen::seqN(i*2, Eigen::fix<2>),Eigen::seqN(10+i, Eigen::fix<1>))), x, u);
        }
    };

    laopt::BSMatrixDenseConstruction<scalar_t> construct_value;
    construct_value.resize(10,1);
    laopt::BSMatrixSparsity sparsity_jacobian(10, 15);
    f(construct_value, sparsity_jacobian); // Extract sparsity pattern

    Eigen::VectorX<scalar_t> value_buffer(construct_value.rows());
    laopt::BSMatrixDenseDeployment<scalar_t> value(value_buffer);

    auto jacobian_tape = sparsity_jacobian.makeBSTape(10, 15);
    value.resize(10,1);
    f(value, jacobian_tape); // Extract operation sequence

    auto BS_jacobian = jacobian_tape.template makeBSMatrix<scalar_t>();

    Eigen::SparseMatrix<scalar_t> S_jacobian;
    BS_jacobian.allocate_memory(S_jacobian);

    value.set_zero();
    BS_jacobian.set_zero();
    f(value, BS_jacobian);

    Eigen::MatrixX<scalar_t> value_g(10,1);
    value_g << 0, 0, 20, 29, 80, 116, 180, 261, 320, 464;
    EXPECT_TRUE(value_g.isApprox(value.value(), 1e-4));

    auto jacobian_g = triplet_to_sparse<double>(10,15,{{0,0,0},{1,0,0},{0,1,0},{1,1,0},{2,2,21},{3,2,32},{2,3,2},{3,3,4},{4,4,42},{5,4,64},{4,5,4},{5,5,8},{6,6,63},{7,6,96},{6,7,6},{7,7,12},{8,8,84},{9,8,128},{8,9,8},{9,9,16},{0,10,0},{1,10,0},{2,11,5},{3,11,6},{4,12,10},{5,12,12},{6,13,15},{7,13,18},{8,14,20},{9,14,24}});
    EXPECT_TRUE(jacobian_g.isApprox(S_jacobian, 1e-4));
}

TEST(FunctionTest, Identity) {

    // Discrete dynamics
    using scalar_t = double;
    laopt::functions::IDENTITY id;

    Eigen::Vector<scalar_t, 2> value;
    Eigen::Matrix<scalar_t, 2, 2> jacobian;

    laopt::IndexedVector<Eigen::Vector<scalar_t, 2>> x;
    x << 1, 2;

    jacobian.setZero();
    id.jacobian(value, jacobian, x);

    Eigen::MatrixX<scalar_t> value_g(2,1);
    value_g << 1, 2;

    Eigen::MatrixX<scalar_t> jacobian_g(2,2);
    jacobian_g << 1, 0, 0, 1;

    EXPECT_TRUE(value_g.isApprox(value, 1e-4));
    EXPECT_TRUE(jacobian_g.isApprox(jacobian, 1e-4));
}


/**************************
 * Test weighted sum
 **************************/

struct wsum_func_t : public laopt::Differentiable<wsum_func_t, true>
{
    Eigen::Matrix2d Q{{2,1},{1,4}};

    template<typename X, typename Z, typename Scalar = typename X::Scalar>
    EIGEN_STRONG_INLINE Eigen::Vector<Scalar, 2>
    function_impl(const Eigen::MatrixBase<X>& x, const Eigen::MatrixBase<Z>& z) noexcept
    {
        Eigen::Vector<Scalar, 2> out;
        out(0) = 0.5 * (x - z).dot(Q * x);
        out(1) = 2.0 * (z - x).dot(Q * x);
        return out;
    }
};

TEST(FunctionTest, WeightedSum)
{
    wsum_func_t func;

    using scalar_t = double;
    laopt::IndexedVector<Eigen::Vector<scalar_t,2>> x;
    laopt::IndexedVector<Eigen::Vector<scalar_t,2>> z;
    Eigen::Vector<scalar_t,2> weight;

    x << 1,2;
    z << 3,4;
    weight << 1,1;

    scalar_t value;
    Eigen::Vector<scalar_t,4> gradient;
    Eigen::Matrix<scalar_t,4,4> hessian;

    Eigen::Vector<scalar_t, 4> gradient_g;
    gradient_g << 3, 1.5, 6, 13.5;

    Eigen::Matrix<scalar_t, 4, 4> hessian_g;
    hessian_g << -6, -3, 3, 1.5, -3, -12, 1.5, 6, 3, 1.5, 0, 0, 1.5, 6, 0, 0;

    value = func.wsum(weight, x, z);
    EXPECT_EQ(value, 39);

    gradient.setZero();
    value = func.gradient(gradient, weight, x, z);
    EXPECT_EQ(value, 39);
    EXPECT_TRUE(gradient.isApprox(gradient_g, 1e-4));

    hessian.setZero();
    gradient.setZero();
    value = func.hessian(gradient, hessian, weight, x, z);
    EXPECT_EQ(value, 39);
    EXPECT_TRUE(gradient.isApprox(gradient_g, 1e-4));
    EXPECT_TRUE(hessian.isApprox(hessian_g, 1e-4));
}

} // end namespace
