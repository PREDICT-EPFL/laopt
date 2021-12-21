/**
 * Unit test for the construction and computation of LAMPC constraint sets.
 */

#include "lampc_function.hpp"
#include "lampc_constraints.hpp"

#include <iomanip>
#include <Eigen/Eigenvalues> 
#include <type_traits>


#include "gtest/gtest.h"

namespace {
using namespace lampc;

template<typename Scalar, std::size_t N, typename range>
struct ConTest;

template<typename Scalar, std::size_t N, std::size_t... ind>
struct ConTest<Scalar, N, std::integer_sequence<std::size_t, ind...>>
{
    struct param_t
    {
        Eigen::Matrix<Scalar, 2, 1> x0 {-1, -2};
        const Eigen::Matrix<Scalar, 2, 2> A {{1.0, 0.0}, {0.1, 1.0}};
        Eigen::Matrix<Scalar, 2, 1> B {0.1, 0.005};
        Eigen::Matrix<Scalar, 1, 1> ref {3};

        Eigen::Matrix<Scalar, 2, 1> q {1, 1e3}; // Stage-cost weights
        Eigen::Matrix<Scalar, 1, 1> r {1e-3}; // Stage-cost weights
    };


    FUNCTION(dynamics, Scalar, param_t, (xplus, 2), (x, 2), (u, 1))
    {
        xplus = p.A.template cast<T>() * x + x(0) * p.B.template cast<T>() * u;
    }

    FUNCTION(dynamics_eq, Scalar, param_t, (out, 2), (xplus, 2), (x, 2), (u, 1))
    {
        Eigen::Matrix<T, 2, 1> tmp;
        dynamics::template impl<T>(p, tmp, x, u);
        out = tmp - xplus;
    }

    FUNCTION(test_func, Scalar, param_t, (out, 3), (x, 2), (u, 1))
    {
        out << x(0)*x(1)+x(0)*x(0)*10.0,
            u(0)*x(0)*x(1)+u(0)*x(0)*x(0)*10.0,
            u(0);
    }


    template<int i>
    struct x : Variable<2> {static constexpr const char* name="x";};

    template<int i>
    struct u : Variable<1> {static constexpr const char* name="u";};

    using variables = VariableSet<Scalar, u<ind>..., x<ind>..., x<N>>;

    template<int i>
    struct con_dynamics : FunctionConstraint<dynamics_eq, x<i+1>, x<i>, u<i>> {};

    struct con_test : FunctionConstraint<test_func, x<1>, u<0>> {};

    using constraints = std::tuple<con_dynamics<ind>..., con_test>;

    using conset = ConstraintSet<variables, constraints>;
};


TEST(ConTest, EvaluateJacobian) {
    using Scalar = double;
    const std::size_t N = 5;

    using Test = ConTest<Scalar, N, std::make_integer_sequence<std::size_t, N>>;
    using param_t = Test::param_t;

    using conset = typename Test::conset;

    conset::variable_t x;
    conset::constraint_t con;	
    conset::jacobian_t J;	
    conset::hessian_t H;
    conset::constraint_t lambda;
    Test::param_t param;

    Scalar val = 1;
    for(auto& i: x)
    	i = val++;

    conset cons;
    cons.sparse_init_jacobian(J);
    cons.sparse_init_hessian(H);

    cons.eval(param, x, con);
	std::cout << "con = " << con.transpose() << std::endl;

	cons.jacobian(param, x, con, J);

	std::cout << "con = " << con.transpose() << std::endl;
    std::cout << "J = \n" << Eigen::MatrixX<Scalar>(J) << std::endl;
    std::cout << "H = \n" << Eigen::MatrixX<Scalar>(H) << std::endl;

    std::cout << "sparse_hessian_offsets = ";
    for(const auto x: cons.sparse_hessian_offsets)
	    std::cout << x << ", ";
	std::cout << "\n";

    lambda.array() = 1;
    Scalar value;
    conset::variable_t gradient;
    conset::hessian_t hessian;

    gradient.array() = 0;

    // cons.hessian(param, x, lambda, value, gradient, hessian);

    std::cout << "value = " << value << std::endl;
    std::cout << "gradient    = " << gradient.transpose() << std::endl;
    std::cout << "lambda' * J = " << lambda.transpose() * Eigen::MatrixX<Scalar>(J) << std::endl;
};


}