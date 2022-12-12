#include <iostream>
#include "laopt/laopt.hpp"
#include "laopt/solvers/sqp_solver.hpp"
#ifdef LAOPT_WITH_OSQP
#include "laopt/solvers/osqp_interface.hpp"
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
 * min_x f(x) = (x2-2)^2
 *  s.t.
 *       0 = x1^2 + x2 - 1
 *       -1 <= x1 <= 1
 */
template<typename scalar_t_>
struct SimpleExample : public laopt::Differentiable<SimpleExample<scalar_t_>>
{
    using scalar_t = scalar_t_;

    template<size_t n>
    using Variable = laopt::Variable<scalar_t, n>;

    struct Cost {};
    template<typename X, typename Scalar = typename Eigen::MatrixBase<X>::Scalar>
    EIGEN_STRONG_INLINE Scalar
    function_impl(Cost, const Eigen::MatrixBase<X>& x1, const Eigen::MatrixBase<X>& x2) noexcept
    {
        return (x2(0) - 2.0) * (x2(0) - 2.0);
    }

    /**
     * Equalities
     *
     * x1^2 + x2 - 1
     */
    struct Equality {};
    template<typename X, typename Scalar = typename Eigen::MatrixBase<X>::Scalar>
    EIGEN_STRONG_INLINE Scalar
    function_impl(Equality, const Eigen::MatrixBase<X>& x1, const Eigen::MatrixBase<X>& x2) noexcept
    {
        return x1(0) * x1(0) + x2(0) - 1.0;
    }

    Variable<1> x1_var;
    Variable<1> x2_var;

    SimpleExample() = default;

    template<typename Problem>
    void define_problem(Problem& problem)
    {
        problem.add_variable(x1_var);
        problem.add_variable(x2_var);

        problem.add_obj(this->function(Cost{}, x1_var, x2_var));

        problem.add_constr(-1 <= x1_var <= 1);

        problem.add_constr(this->function(Equality{}, x1_var, x2_var) == 0);
    }
};

#ifdef LAOPT_WITH_OSQP
TEST(SQPTest, SimpleExampleOSQP)
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

    laopt::SQPSolver<Problem, laopt::OSQPSolver<Problem::scalar_t>> solver(prob);
    solver.settings().verbose = true;

    // Set the initial primal variable
    my_problem.x1_var << 0.5;
    my_problem.x2_var << 1.5;

    solver.solve();

    EXPECT_NEAR(my_problem.x1_var, 0, 1e-4);
    EXPECT_NEAR(my_problem.x2_var, 1, 1e-4);
}
#endif

#ifdef LAOPT_WITH_PROXQP
TEST(SQPTest, SimpleExampleProxQP)
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

    laopt::SQPSolver<Problem, laopt::ProxQPSolver<Problem::scalar_t>> solver(prob);
    solver.settings().verbose = true;

    // Set the initial primal variable
    my_problem.x1_var << 0.5;
    my_problem.x2_var << 1.5;

    solver.solve();

    EXPECT_NEAR(my_problem.x1_var, 0, 1e-4);
    EXPECT_NEAR(my_problem.x2_var, 1, 1e-4);
}
#endif

#ifdef LAOPT_WITH_QPSWIFT
TEST(SQPTest, SimpleExampleQPSwift)
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

    laopt::SQPSolver<Problem, laopt::QPSwiftSolver<Problem::scalar_t>> solver(prob);
    solver.settings().verbose = true;

    // Set the initial primal variable
    my_problem.x1_var << 0.5;
    my_problem.x2_var << 1.5;

    solver.solve();

    EXPECT_NEAR(my_problem.x1_var, 0, 1e-4);
    EXPECT_NEAR(my_problem.x2_var, 1, 1e-4);
}
#endif
