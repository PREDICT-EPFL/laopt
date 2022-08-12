/**
 * Unit test for the construction and computation of LAMPC functions.
 */

#include <iostream>
#include <chrono>

#include "laopt/laopt.hpp"

#include "test_utils.hpp"
#include "gtest/gtest.h"

namespace {

template<typename scalar_t>
struct TestFunction : public laopt::Differentiable<TestFunction<scalar_t>, true>
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


/**
 * Compute the jacobian and hessian of a function
 */
TEST(FunctionTest, Jacobian) {
    using scalar_t = double;
    using test_t = TestFunction<scalar_t>;
    // using test_t = ScalarFunction<scalar_t>;
    test_t test;

    Eigen::Vector<scalar_t, 2> x;
    Eigen::Vector<scalar_t, 1> u;
    x << 2, 1;
    u << 3;

    Eigen::Vector<scalar_t, 2> value;
    Eigen::Matrix<scalar_t, 2, 3> jacobian;
    std::array<Eigen::Matrix<scalar_t, 3, 3>, 2> hessian;

    value = test(x, u);

    Eigen::Vector<scalar_t, 2> val;
    val << 38, 56;
    EXPECT_EQ(value, val);

    test.jacobian(value, jacobian, x, u);

    Eigen::Matrix<scalar_t, 2, 3> jac;
    jac << 21, 4, 10, 34, 8, 12;
    EXPECT_EQ(jacobian, jac);
}

/**
 * Confirm that we can compute the size of the output at compile time and that
 * we can work with functions that have scalar outputs.
 * 
 * Eigen swaps from matrix to scalar outputs for some common functions (e.g., dot)
 * and so it's likely that users would create functions that return scalars.
 */
TEST(FunctionTest, GetOutput)
{
    using scalar_t = double;

    using Arg1 = Eigen::Vector<scalar_t, 2>;
    using Arg2 = Eigen::Vector<scalar_t, 1>;

{
    int n_inputs = TestFunction<scalar_t>::FuncInfo<Arg1, Arg2>::n_inputs;
    int n_outputs = TestFunction<scalar_t>::FuncInfo<Arg1, Arg2>::n_outputs;
    EXPECT_EQ(n_inputs, 3);
    EXPECT_EQ(n_outputs, 2);
}

{
    int n_inputs = ScalarFunction<scalar_t>::FuncInfo<Arg1, Arg2>::n_inputs;
    int n_outputs = ScalarFunction<scalar_t>::FuncInfo<Arg1, Arg2>::n_outputs;
    EXPECT_EQ(n_inputs, 3);
    EXPECT_EQ(n_outputs, 1);
}

    ScalarFunction<scalar_t> f;
    Arg1 x;
    Arg2 u;
    x << 1, 2;
    u << 3;
    Eigen::Vector<scalar_t, 1> weight;
    weight << 1;

    using info = ScalarFunction<scalar_t>::FuncInfo<Arg1, Arg2>;
    info::return_t ret;
    info::scalar_t value;
    info::gradient_t gradient;
    info::jacobian_t jacobian;
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

/**
 * Speed comparison for Jacobian computation of small functions against 
 * direct EigenDiff calls
 */
template<typename InputX, typename InputY, typename Output>
void test_jacobian(const Eigen::MatrixBase<InputX> &x, const Eigen::MatrixBase<InputY> &y, Output &out)
{
    Eigen::Matrix<double, 2, 2> A{{1,2},{3,4}};
    Eigen::Matrix<double, 2, 1> B{{5},{6}};

    out = x(0) * (A * x + B * y);
}


template<typename Scalar_t = double, int NX = 3>
struct TestSpeed
{
    using Scalar = Scalar_t;
    /* second order derivative */
    using Derivatives = Eigen::Matrix<Scalar, NX, 1>;
    using ADScalar = Eigen::AutoDiffScalar<Derivatives>;
    using outerDerivatives = Eigen::Matrix<ADScalar, NX, 1>;
    using outerADScalar = Eigen::AutoDiffScalar<outerDerivatives>;
    using ADx_t = Eigen::Matrix<outerADScalar, 3, 1>;

    using Jacobian = Eigen::Matrix<Scalar, 2, 3>;
    using Value = Eigen::Vector<Scalar, 2>;
    using Hessian = Eigen::Matrix<Scalar, 3, 3>;

    template<typename InputX, typename InputY>
    void AutoDiff_hessian(const Eigen::MatrixBase<InputX>& x_in, const Eigen::MatrixBase<InputY>& y_in,
        Value& val,
        Jacobian& jacobian,
        std::array<Hessian,2>& hessian)
    {
        ADx_t x;

        /* initialize derivatives */
        for (int i = 0; i < NX; i++) {
            if(i<2) x(i).value().value() = x_in(i);
            else    x(i).value().value() = y_in(0);
            x(i).value().derivatives() = Derivatives::Unit(NX, i);
            x(i).derivatives() = Derivatives::Unit(NX, i);
            /* initialize hessian matrix to zero */
            for (int j = 0; j < NX; j++) {
                x(i).derivatives()(j).derivatives().setZero();
            }
        }

        using outerAD_t = Eigen::Vector<outerADScalar, 2>;
        outerAD_t out;
        test_jacobian(x(Eigen::seqN(0, Eigen::fix<2>)), x(Eigen::seqN(2, Eigen::fix<1>)), out);

        // Copy into buffers
        for(int i=0; i<2; i++)
        {
            val(i) = out[i].value().value();
            jacobian.row(i) = out[i].value().derivatives();
            for (int j = 0; j < 3; j++) {
                hessian[i].template middleRows<1>(j) = out[i].derivatives()(j).derivatives().transpose();
            }
        }
    }

    template<typename InputX, typename InputY, typename Weight, typename Gradient, typename Hessian>
    auto wsum_hessian(const Eigen::MatrixBase<InputX>& x_in, const Eigen::MatrixBase<InputY>& y_in,
        const Eigen::MatrixBase<Weight>& weight,
        Eigen::MatrixBase<Gradient>& gradient, Eigen::MatrixBase<Hessian>& hessian)
    {
        using scalar_t = typename Eigen::MatrixBase<InputX>::Scalar;
        Eigen::Vector<scalar_t, 2> value;
        Eigen::Matrix<scalar_t, 2, 3> jacobian;
        std::array<Eigen::Matrix<scalar_t, 3, 3>, 2> hessian_array;

        AutoDiff_hessian(x_in, y_in, value, jacobian, hessian_array);

        gradient = jacobian.transpose() * weight;
        hessian = weight(0) * hessian_array[0] + weight(1) * hessian_array[1];
        return weight.dot(value);
    }


    /* first order derivative */
    using ADx1_t = Eigen::Matrix<ADScalar, 2, 1>;
    using ADy1_t = Eigen::Matrix<ADScalar, 1, 1>;

    template<typename InputX, typename InputY>
    void AutoDiff_jacobian(Eigen::MatrixBase<InputX>& x_in, Eigen::MatrixBase<InputY>& y_in,
        Value& val,
        Jacobian& jacobian)
    {
        ADx1_t x;
        x = x_in;

        ADy1_t y;
        y = y_in;

        /* initialize derivatives */
        x[0].derivatives().array() = 0;
        x[1].derivatives().array() = 0;
        y[0].derivatives().array() = 0;
        for (int i=0; i<x.rows(); i++) {
            x[i].derivatives().coeffRef(i) = 1;
        }
        y[0].derivatives().coeffRef(2) = 1;

        Eigen::Vector<ADScalar, 2> out;
        test_jacobian(x, y, out);

        for(int i=0; i<2; i++)
        {
            val(i) = out[i].value();
            jacobian.row(i) = out[i].derivatives();
        }
    }
};


#ifdef NDEBUG
/**
 * Compare the speed of jacobian computation for small problems
 */
TEST(FunctionTest, Jacobian_Speed) {
    using scalar_t = double;
    TestSpeed<> direct;

    using test_t = TestFunction<scalar_t>;
    test_t test;

    Eigen::Vector<scalar_t, 2> x;
    Eigen::Vector<scalar_t, 1> u;
    x << 1,2;
    u << 3;

    Eigen::Vector<scalar_t,2> value;
    Eigen::Matrix<scalar_t, 2, 3> jacobian;

    std::cout.setf(std::ios::fixed);
    std::cout.setf(std::ios::showpoint);
    std::cout.precision(4);

    double acc = 0;
    std::size_t NUM_EXP = 100000000;
    auto start = std::chrono::steady_clock::now();
    for(size_t i = 0; i < NUM_EXP; ++i)
    {
        x(0) = (double) i;
        test.jacobian(value, jacobian, x, u);
        acc += value(0);
    }
    auto end = std::chrono::steady_clock::now();

    std::cout << "Time per jacobian (lampc ): "
      << std::setw(20) << std::chrono::duration_cast<std::chrono::nanoseconds>((end - start)).count()/(double)NUM_EXP
      << " ns" << "  (" << acc << ")" << std::endl;

    double lampc_jacobian_time = std::chrono::duration_cast<std::chrono::nanoseconds>((end - start)).count()/(double)NUM_EXP;

    acc = 0;
    start = std::chrono::steady_clock::now();
    for(size_t i = 0; i < NUM_EXP; ++i)
    {
        x(0) = (double) i;
        direct.AutoDiff_jacobian(x,u,value,jacobian);
        acc += value(0);
    }
    end = std::chrono::steady_clock::now();

    std::cout << "Time per jacobian (direct): "
      << std::setw(20) << std::chrono::duration_cast<std::chrono::nanoseconds>((end - start)).count()/(double)NUM_EXP
      << " ns" << "  (" << acc << ")" << std::endl;

    double direct_jacobian_time = std::chrono::duration_cast<std::chrono::nanoseconds>((end - start)).count()/(double)NUM_EXP;

    // Less than 30% overhead for LAMPC
    EXPECT_TRUE((lampc_jacobian_time - direct_jacobian_time) / direct_jacobian_time < 0.3);
}
#else

TEST(FunctionTest, Jacobian_Speed) {
    std::cout << "Skipping Jacobian_Speed test because we're in debug mode";
    std::cout << std::endl;
}

#endif

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

TEST(RK4Test, SimpleTest) {
    using scalar_t = double;
    sys_t<scalar_t> sys;
    using dsys_t = laopt::functions::RK4<sys_t<scalar_t>, scalar_t>;
    dsys_t dsys(sys, 0.1);

    // Compute the discrete-time system and its jacobian
    Eigen::Vector<scalar_t, 2> x;
    Eigen::Vector<scalar_t, 1> u;
    x << 1,2;
    u << 3;

    Eigen::Vector<scalar_t, 2> value;
    Eigen::Matrix<scalar_t, 2, 3> jacobian;

    // Eval only - doesn't compute jacobian
    value = dsys(x, u);

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
        for(int i = 0; i < 5; i++)
        {
            x(0) = i;
            x(1) = 2 * i;
            u(0) = 3 * i;
            test.jacobian(value(Eigen::seqN(i*2,2)), jacobian(Eigen::seqN(i*2,2), laopt::multiSeq_to_index(Eigen::seqN(i*2, Eigen::fix<2>),Eigen::seqN(10+i,Eigen::fix<1>))), x, u);
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

TEST(FunctionsTest, id) {

    // Discrete dynamics
    using scalar_t = double;
    laopt::functions::IDENTITY id;

    Eigen::Vector<scalar_t, 2> value;
    Eigen::Matrix<scalar_t, 2, 2> jacobian;

    Eigen::Vector<scalar_t, 2> x;
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

TEST(WeightedSum, Basic)
{
    wsum_func_t func;

    using scalar_t = double;
    Eigen::Vector<scalar_t,2> x;
    Eigen::Vector<scalar_t,2> z;
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


/**
 * Speed comparison for hessian computation against 
 * direct EigenDiff calls
 */
#ifdef NDEBUG
TEST(Functions, Hessian_TestSpeed)
{
    using scalar_t = double;
    TestSpeed<> direct;

    using test_t = TestFunction<scalar_t>;
    test_t test;

    Eigen::Vector<scalar_t, 2> x;
    Eigen::Vector<scalar_t, 1> u;
    Eigen::Vector<scalar_t, 2> weight;
    x << 1,2;
    u << 3;
    weight << 1,1;

    scalar_t value;
    Eigen::Vector<scalar_t,3> gradient;
    Eigen::Matrix<scalar_t,3,3> hessian;

    std::cout.setf(std::ios::fixed);
    std::cout.setf(std::ios::showpoint);
    std::cout.precision(4);

    double acc = 0;
    size_t NUM_EXP = 10000000;
    auto start = std::chrono::steady_clock::now();
    for(size_t i = 0; i < NUM_EXP; ++i)
    {
        x(0) = (double) i;
        gradient.setZero(); hessian.setZero();
        value = direct.wsum_hessian(x, u, weight, gradient, hessian);
        acc += value;
    }
    auto end = std::chrono::steady_clock::now();

    std::cout << "Time per weighted sum hessian (direct): "
      << std::setw(20) << std::chrono::duration_cast<std::chrono::nanoseconds>((end - start)).count()/(double)NUM_EXP
      << " ns" << "  (" << acc << ")" << std::endl;

    double direct_time = std::chrono::duration_cast<std::chrono::nanoseconds>((end - start)).count()/(double)NUM_EXP;

    acc = 0;
    start = std::chrono::steady_clock::now();
    for(size_t i = 0; i < NUM_EXP; ++i)
    {
        x(0) = i;
        gradient.setZero(); hessian.setZero();
        value = test.hessian(gradient, hessian, weight, x, u);
        acc += value;
    }
    end = std::chrono::steady_clock::now();

    std::cout << "Time per weighted sum hessian (lampc ): "
      << std::setw(20) << std::chrono::duration_cast<std::chrono::nanoseconds>((end - start)).count()/(double)NUM_EXP
      << " ns" << "  (" << acc << ")" << std::endl;

    double lampc_time = std::chrono::duration_cast<std::chrono::nanoseconds>((end - start)).count()/(double)NUM_EXP;

    EXPECT_TRUE((lampc_time - direct_time)/direct_time < 0.3); // Less than 30% overhead for lampc
}
#else

#include "colormod.hpp"
TEST(Functions, TestSpeed)
{
    std::cout << "Skipping weighted sum Hessian speeed test because we're in debug mode";
    std::cout << std::endl;
}

#endif

}
