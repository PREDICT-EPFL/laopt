#ifndef LAMPC_MULTIPLE_SHOOTING_TRANSCRIPTION_HPP
#define LAMPC_MULTIPLE_SHOOTING_TRANSCRIPTION_HPP

#include <Eigen/Dense>
#include "laopt/lampc.hpp"

// Advanced user (level 2)

template<typename OCP, int N>
class MultipleShootingTranscription : public laopt::Differentiable<MultipleShootingTranscription<OCP, N>>
{
public:

    OCP &ocp;

    using scalar_t = typename OCP::scalar_t; // laopt::Problem assumes that the user code will contain a scalar_t

    // Continuous dynamics
    struct Sys {};
    template<typename X, typename U, typename Scalar = typename Eigen::MatrixBase<X>::Scalar>
    EIGEN_STRONG_INLINE Eigen::Vector<Scalar, OCP::NX>
    function_impl(Sys, const Eigen::MatrixBase<X>& x, const Eigen::MatrixBase<U>& u) noexcept
    {
        Eigen::Vector<Scalar, OCP::NX> x_dot;
        ocp.template dynamics_impl<Scalar>(x_dot, x, u);
        return x_dot;
    }

    // Discretized dynamics
    using dsys_t = laopt::functions::RK4<MultipleShootingTranscription<OCP, N>, scalar_t, Sys>;
    dsys_t dsys;

    struct StageCost {};
    template<typename X, typename U, typename Scalar = typename Eigen::MatrixBase<X>::Scalar>
    EIGEN_STRONG_INLINE Scalar
    function_impl(StageCost, const Eigen::MatrixBase<X>& x, const Eigen::MatrixBase<U>& u) noexcept
    {
        Scalar lagrange;
        ocp.template lagrange_term_impl<Scalar>(lagrange, x, u);
        return lagrange;
    }

    struct TerminalCost {};
    template<typename X, typename Scalar = typename Eigen::MatrixBase<X>::Scalar>
    EIGEN_STRONG_INLINE Scalar
    function_impl(TerminalCost, const Eigen::MatrixBase<X>& x) noexcept
    {
        Scalar mayer;
        ocp.template mayer_term_impl<Scalar>(mayer, x);
        return mayer;
    }
    laopt::Function<MultipleShootingTranscription<OCP, N>, TerminalCost> terminal_cost;

    template<size_t n>
    using Variable = laopt::Variable<scalar_t, n>;

    Eigen::Vector<scalar_t, N + 1> T;
    std::array<Variable<OCP::NX>, N + 1> X;
    std::array<Variable<OCP::NU>, N + 1> U;

    explicit MultipleShootingTranscription(OCP &ocp_) :
    ocp(ocp_),
    dsys(*this, ocp.tf / N),
    terminal_cost(*this)
    {
        for(int i = 0; i < N + 1; i++)
        {
            T(i) = i * ocp.tf / N;
        }
    };

    template<typename Problem>
    void define_problem(Problem& problem) {
        // define variables
        for (int i = 0; i < N + 1; i++)
        {
            problem.add_variable(X[i]);
            problem.add_variable(U[i]);
        }

        // add objective
        for (int i = 0; i < N; i++)
        {
            problem.add_obj(this->function(StageCost{}, X[i], U[i]));
        }
        problem.add_obj(terminal_cost(X[N]));

        // add box constraints
        for (int i = 0; i < N + 1; i++)
        {
            problem.add_constr(ocp.lbx <= X[i] <= ocp.ubx);
            problem.add_constr(ocp.lbu <= U[i] <= ocp.ubu);
        }

        // add initial state constraint
        problem.add_constr(X[0] == ocp.x0);

        // add dynamics constraints
        for (int i = 0; i < N; i++)
        {
            problem.add_constr(X[i + 1] - dsys(X[i], U[i]) == 0);
        }

        // add last input constraint
        problem.add_constr(U[N] - U[N - 1] == 0);
    }
};

#endif //LAMPC_MULTIPLE_SHOOTING_TRANSCRIPTION_HPP
