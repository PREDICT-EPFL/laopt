#include <iostream>
#include "laopt/laopt.hpp"
#include "laopt/solvers/sqp_solver.hpp"
#ifdef LAOPT_WITH_OSQP
#include "laopt/solvers/osqp_interface.hpp"
#endif
#ifdef LAOPT_WITH_PIQP
#include "laopt/solvers/piqp_interface.hpp"
#endif
#ifdef LAOPT_WITH_PROXQP
#include "laopt/solvers/proxqp_interface.hpp"
#endif
#ifdef LAOPT_WITH_QPSWIFT
#include "laopt/solvers/qpswift_interface.hpp"
#endif

#include "test_utils.hpp"
#include "gtest/gtest.h"

/**
 * min_x f(x) = (x2-2)^2 - x3 + x4
 *  s.t.
 *       0 = x1^2 + x2 - 1
 *       -1 <= x1 <= 1
 *       -1 <= x3 <= 1
 *       -1 <= x4 <= 1
 */
template<typename scalar_t_>
struct SimpleExample : public laopt::Differentiable<SimpleExample<scalar_t_>, laopt::TAGGED>
{
    using scalar_t = scalar_t_;

    template<size_t n>
    using Variable = laopt::Variable<scalar_t, n>;

    struct Cost {};
    template<typename X, typename Scalar = typename Eigen::MatrixBase<X>::Scalar>
    EIGEN_STRONG_INLINE Scalar
    function_impl(Cost, const Eigen::MatrixBase<X>& x) noexcept
    {
        return (x(1) - 2.0) * (x(1) - 2.0) - x(2) + x(3);
    }

    /**
     * Equalities
     *
     * x1^2 + x2 - 1
     */
    struct Equality {};
    template<typename X, typename Scalar = typename Eigen::MatrixBase<X>::Scalar>
    EIGEN_STRONG_INLINE Scalar
    function_impl(Equality, const Eigen::MatrixBase<X>& x) noexcept
    {
        return x(0) * x(0) + x(1) - 1.0;
    }

    Variable<4> x_var;

    SimpleExample() = default;

    template<typename Derived>
    void define_problem(laopt::OptProblem<Derived>& problem)
    {
        problem.add_variable(x_var);

        problem.add_obj(1.0 * this->function(Cost{}, x_var));

        problem.add_constr(-1 <= x_var({0, 2, 3}) <= 1);

        problem.add_constr(1.0f * this->function(Equality{}, x_var) == 0);
    }
};

template<template<typename, int...> class Solver>
static void generic_sqp_test()
{
    using scalar_t = double;

    using UserCode = SimpleExample<scalar_t>;
    UserCode my_problem;
    auto sparsity = laopt::generate_sparsity(my_problem);
    auto tape = laopt::generate_tape(my_problem, sparsity);

    std::cout << sparsity << std::endl;
    std::cout << tape << std::endl;

    using Problem = laopt::Problem<UserCode>;
    Problem prob(my_problem, tape);

    laopt::SQPSolver<Problem, Solver<Problem::scalar_t>> solver(prob);
    solver.settings().hessian_approximation = laopt::hessian_approximation_t::EXACT;
    solver.settings().verbose = true;

    // Set the initial primal variable
    my_problem.x_var << 0.5, 1.5, 0.0, 0.0;

    solver.solve();

    EXPECT_NEAR(my_problem.x_var(0), 0, 1e-4);
    EXPECT_NEAR(my_problem.x_var(1), 1, 1e-4);
    EXPECT_NEAR(my_problem.x_var(2), 1, 1e-4);
    EXPECT_NEAR(my_problem.x_var(3), -1, 1e-4);

    EXPECT_NEAR(solver.primal()(0), 0, 1e-4);
    EXPECT_NEAR(solver.primal()(1), 1, 1e-4);
    EXPECT_NEAR(solver.primal()(2), 1, 1e-4);
    EXPECT_NEAR(solver.primal()(3), -1, 1e-4);
    EXPECT_NEAR(solver.dual()(0), 2, 1e-4);
    EXPECT_NEAR(solver.dual_bounds()(0), 0, 1e-4);
    EXPECT_NEAR(solver.dual_bounds()(1), 0, 1e-4);
    EXPECT_NEAR(solver.dual_bounds()(2), 1, 1e-4);
    EXPECT_NEAR(solver.dual_bounds()(3), -1, 1e-4);
}

#ifdef LAOPT_WITH_OSQP
TEST(SQPTest, SimpleExampleOSQP)
{
    generic_sqp_test<laopt::OSQPSolver>();
}
#endif

#ifdef LAOPT_WITH_PIQP
TEST(SQPTest, SimpleExamplePIQP)
{
    generic_sqp_test<laopt::PIQPSolver>();
}
#endif

#ifdef LAOPT_WITH_PROXQP
TEST(SQPTest, SimpleExampleProxQP)
{
    generic_sqp_test<laopt::ProxQPSolver>();
}
#endif

#ifdef LAOPT_WITH_QPSWIFT
TEST(SQPTest, SimpleExampleQPSwift)
{
    generic_sqp_test<laopt::QPSwiftSolver>();
}
#endif
