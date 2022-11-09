/**
 * Unit test for the computation of IpOpt problems.
 */

#include <iostream>
#include "laopt/laopt.hpp"
#include "laopt/ipopt_interface/ipopt_wrapper.hpp"

#include "test_utils.hpp"
#include "gtest/gtest.h"

using namespace Ipopt;

namespace
{

/**
 * Example used in the Ipopt C++ example
 * 
 * min_x f(x) = -(x2-2)^2
 *  s.t.
 *       0 = x1^2 + x2 - 1
 *       -1 <= x1 <= 1
 */
template<typename scalar_t_>
struct Ipopt_example : public laopt::Differentiable<Ipopt_example<scalar_t_>>
{
    using scalar_t = scalar_t_;

    template<size_t n>
    using Variable = laopt::Variable<scalar_t, n>;

    struct Cost {};
    template<typename X, typename Scalar = typename Eigen::MatrixBase<X>::Scalar>
    EIGEN_STRONG_INLINE Scalar
    function_impl(Cost, const Eigen::MatrixBase<X>& x1, const Eigen::MatrixBase<X>& x2) noexcept
    {
        return -(x2(0) - 2.0) * (x2(0) - 2.0);
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

    Ipopt_example() = default;

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


TEST(IpoptTest, IpoptExample)
{
    using scalar_t = double;

    using UserCode = Ipopt_example<scalar_t>;
    UserCode my_problem;
    auto sparsity = laopt::generate_sparsity(my_problem);
    auto tape = laopt::generate_tape(my_problem, sparsity);

    std::cout << sparsity << std::endl;
    std::cout << tape << std::endl;

    using Problem = laopt::Problem<UserCode>;
    Problem prob(my_problem, tape);

    laopt::IpoptWrapper<Problem> solver(prob);

    // Set the initial primal variable
    my_problem.x1_var << 0.5;
    my_problem.x2_var << 1.5;

    solver.solve();

    EXPECT_NEAR(my_problem.x1_var, 1, 1e-4);
    EXPECT_NEAR(my_problem.x2_var, 0, 1e-4);
};

/*
 * Solve problem 71 from the Hock-Schittkowsky test suite [6],
 * 
 * min x1*x4*(x1+x2+x3)+x3
 * s.t. x1*x2*x3*x4 >= 25
 *      x1^2 + x2^2 + x3^2 + x4^2 = 40
 *      1 <= x1,x2,x3,x4 <= 5
 */
template<typename _scalar_t>
struct Prob71 : public laopt::Differentiable<Prob71<_scalar_t>>
{
    using scalar_t = _scalar_t; // laopt::Problem assumes that the user code will contain a scalar_t

    /**
     * Obj = x1*x4*(x1+x2+x3)+x3
     */
    struct Obj {};
    template<typename X, typename Scalar = typename Eigen::MatrixBase<X>::Scalar>
    EIGEN_STRONG_INLINE Scalar
    function_impl(Obj, const Eigen::MatrixBase<X>& x) noexcept
    {
        return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2];
    }

    /**
     * x1*x2*x3*x4 >= 25
     * x1^2 + x2^2 + x3^2 + x4^2 = 40
     */
    struct Constraints {};
    template<typename X, typename Scalar = typename Eigen::MatrixBase<X>::Scalar>
    EIGEN_STRONG_INLINE Eigen::Vector<Scalar, 2>
    function_impl(Constraints, const Eigen::MatrixBase<X>& x) noexcept
    {
        Eigen::Vector<Scalar, 2> out;
        out << x[0] * x[1] * x[2] * x[3],
                x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3];
        return out;
    }

    // Define our variables
    laopt::Variable<scalar_t, 4> x_var;

    Prob71() = default;

    template<typename Problem>
    void define_problem(Problem& problem)
    {
        problem.add_variable(x_var);

        problem.add_obj(this->function(Obj{}, x_var));

        problem.add_constr(1 <= x_var <= 5);
        problem.add_constr(Eigen::Vector<scalar_t, 2>{25, 40} <= this->function(Constraints{}, x_var) <= Eigen::Vector<scalar_t, 2>{2e9, 40});
    }
};


/**
 * Symbolic values of the various elements we need
 */

// Objective function
template<typename Derived, typename scalar_t = typename Eigen::MatrixBase<Derived>::Scalar>
scalar_t f(const Eigen::MatrixBase<Derived>& x)
{
    scalar_t x1 = x[0];
    scalar_t x2 = x[1];
    scalar_t x3 = x[2];
    scalar_t x4 = x[3];

    return x1 * x4 * (x1 + x2 + x3) + x3;
}

// Gradient of objective function
template<typename Derived, typename scalar_t = typename Eigen::MatrixBase<Derived>::Scalar>
Eigen::Vector<scalar_t, 4> grad_f(const Eigen::MatrixBase<Derived>& x)
{
    scalar_t x1 = x[0];
    scalar_t x2 = x[1];
    scalar_t x3 = x[2];
    scalar_t x4 = x[3];
    Eigen::Vector<scalar_t, 4> gf;
    gf << x1 * x4 + x4 * (x1 + x2 + x3),
            x1 * x4,
            x1 * x4 + 1,
            x1 * (x1 + x2 + x3);
    return gf;
}

// Constraints
template<typename Derived, typename scalar_t = typename Eigen::MatrixBase<Derived>::Scalar>
Eigen::Vector<scalar_t, 2> g(const Eigen::MatrixBase<Derived>& x)
{
    scalar_t x1 = x[0];
    scalar_t x2 = x[1];
    scalar_t x3 = x[2];
    scalar_t x4 = x[3];
    Eigen::Vector<scalar_t, 2> g;
    g << x1 * x2 * x3 * x4,
         x1 * x1 + x2 * x2 + x3 * x3 + x4 * x4;
    return g;
}

// Jacobian of constraints
template<typename Derived, typename scalar_t = typename Eigen::MatrixBase<Derived>::Scalar>
Eigen::Matrix<scalar_t, 2, 4> jac_g(const Eigen::MatrixBase<Derived>& x)
{
    scalar_t x1 = x[0];
    scalar_t x2 = x[1];
    scalar_t x3 = x[2];
    scalar_t x4 = x[3];
    Eigen::Matrix<scalar_t, 2, 4> jac_g;
    jac_g << x2 * x3 * x4, x1 * x3 * x4, x1 * x2 * x4, x1 * x2 * x3,
             2 * x1,
             2 * x2,
             2 * x3,
             2 * x4;
    return jac_g;
}

// Hessian of lagrangian
template<typename Derived, typename L, typename scalar_t = typename Eigen::MatrixBase<Derived>::Scalar>
Eigen::Matrix<scalar_t, 4, 4>
hessian_l(const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<L>& dual, const scalar_t obj_factor)
{
    scalar_t x1 = x[0];
    scalar_t x2 = x[1];
    scalar_t x3 = x[2];
    scalar_t x4 = x[3];
    scalar_t l1 = dual[0];
    scalar_t l2 = dual[1];

    Eigen::Matrix<scalar_t, 4, 4> O;
    O << 2 * x4, x4, x4, 2 * x1 + x2 + x3,
         x4, 0, 0, x1,
         x4, 0, 0, x1,
         2 * x1 + x2 + x3, x1, x1, 0;

    Eigen::Matrix<scalar_t, 4, 4> L1;
    L1 << 0, x3 * x4, x2 * x4, x2 * x3,
          x3 * x4, 0, x1 * x4, x1 * x3,
          x2 * x4, x1 * x4, 0, x1 * x2,
          x2 * x3, x1 * x3, x1 * x2, 0;

    Eigen::Matrix<scalar_t, 4, 4> L2;
    L2 << 2, 0, 0, 0,
            0, 2, 0, 0,
            0, 0, 2, 0,
            0, 0, 0, 2;

    Eigen::Matrix<scalar_t, 4, 4> H;
    H = obj_factor * O + l1 * L1 + l2 * L2;

    return H;
}


TEST(IpoptTest, Prob71)
{
    using scalar_t = double;
    Prob71<scalar_t> prob71;
    using Problem = laopt::Problem<Prob71<scalar_t>>;
    Problem prob = laopt::generate(prob71);

    laopt::IpoptWrapper<Problem> solver(prob);

    prob71.x_var << 1, 5, 5, 1;

    solver.solve();

    // Print out the solution
    std::cout << std::setprecision(6) << std::defaultfloat;

    // Check the solution
    Eigen::Vector<scalar_t, 4> sol_primal{1, 4.743, 3.82115, 1.37941};
    EXPECT_TRUE(solver.primal().isApprox(sol_primal, 1e-4));

    Eigen::Vector<scalar_t, 2> sol_dual{-0.552294, 0.161469};
    EXPECT_TRUE(solver.dual().isApprox(sol_dual, 1e-4));

    // Compute the value and jacobian of the constraints at the optimal point
    laopt::ProblemMemory<scalar_t> mem(prob);

    // Check the computation of variable bounds
    prob.eval_variable_bounds(mem.variable_bounds);
    EXPECT_TRUE(mem.variable_bounds.lb.isApprox(Eigen::Vector<scalar_t, 4>{1, 1, 1, 1}, 1e-4));
    EXPECT_TRUE(mem.variable_bounds.ub.isApprox(Eigen::Vector<scalar_t, 4>{5, 5, 5, 5}, 1e-4));

    // Check the computation of the constraints
    Eigen::VectorX<scalar_t> var = solver.primal();
    Eigen::VectorX<scalar_t> dual = solver.dual();
    prob.set_decision_variable(var);

    prob.eval_constraints(laopt::Jacobian{}, mem.constraints);
    EXPECT_TRUE(mem.constraints.lb.isApprox(Eigen::Vector<scalar_t, 2>{25, 40}, 1e-4));
    EXPECT_TRUE(mem.constraints.ub.isApprox(Eigen::Vector<scalar_t, 2>{2e9, 40}, 1e-4));
    EXPECT_TRUE(mem.constraints.value.isApprox(g(prob71.x_var), 1e-4));
    EXPECT_TRUE(Eigen::MatrixX<scalar_t>(mem.constraints.jacobian).isApprox(jac_g(prob71.x_var), 1e-4));

    // Check the computation of the objective
    scalar_t obj = prob.eval_objective(laopt::Gradient{}, mem.objective);
    EXPECT_NEAR(obj, f(var), 1e-4);
    EXPECT_TRUE(mem.objective.gradient.isApprox(grad_f(var), 1e-4));

    // Check the computation of the lagrangian
    prob.eval_lagrangian(laopt::Hessian{}, 1.0, dual, mem.lagrangian);
    EXPECT_TRUE(Eigen::MatrixX<scalar_t>(mem.lagrangian.hessian).isApprox(hessian_l(var, dual, 1.0), 1e-4));
}

} // end namespace
