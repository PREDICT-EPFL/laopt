#ifndef LAMPC_MULTIPLE_SHOOTING_TRANSCRIPTION_HPP
#define LAMPC_MULTIPLE_SHOOTING_TRANSCRIPTION_HPP

#include <Eigen/Dense>
#include "lampc.hpp"

// Advanced user (level 2)

template<typename OCP, int N>
class MultipleShootingTranscription : public lampc::Differentiable<MultipleShootingTranscription<OCP, N>>
{
public:

    OCP &ocp;

    using scalar_t = typename OCP::scalar_t; // lampc::Problem assumes that the user code will contain a scalar_t

    // Continuous dynamics
    struct Sys {};
    template<typename OutValue, typename X, typename U, typename Scalar = typename Eigen::MatrixBase<X>::Scalar>
    EIGEN_STRONG_INLINE Eigen::Vector<Scalar, OCP::NX>
    function_impl(Sys, const Eigen::MatrixBase<X>& x, const Eigen::MatrixBase<U>& u) noexcept
    {
        Eigen::Vector<Scalar, OCP::NX> x_dot;
        ocp.template dynamics_impl<Scalar>(x_dot, x, u);
        return x_dot;
    }

    // Discretized dynamics
    using dsys_t = lampc::RK4<MultipleShootingTranscription<OCP, N>, scalar_t, Sys>;
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

    template<size_t n>
    using Variable = lampc::Variable<scalar_t, n>;

    Eigen::Vector<scalar_t, N + 1> T;
    std::array<Variable<OCP::NX>, N + 1> X;
    std::array<Variable<OCP::NU>, N + 1> U;

    // Functions we use to define the eq constraints for the dynamics
    lampc::EQ<dsys_t> dsys_eq;

    explicit MultipleShootingTranscription(OCP &ocp_) :
    ocp(ocp_),
    dsys(*this, ocp.tf / N),
    dsys_eq(dsys)
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
        for (int i=0; i < N; i++)
        {
            problem.add_obj(StageCost{}, X[i], U[i]);
        }
        problem.add_obj(TerminalCost{}, X[N]);

        // add box constraints
        for (int i = 0; i < N + 1; i++)
        {
            problem.add_constr(ocp.lbx <= X[i] <= 10);
            problem.add_constr(ocp.lbu <= U[i] <= ocp.ubu);
        }

        // add initial state constraint
        problem.add_constr(X[0] == ocp.x0);
//
//        // add dynamics constraints
//        for (int i = 0; i < N; i++)
//        {
//            problem.add_constr(dsys_eq(X[i + 1], X[i], U[i]) == 0);
//        }
//
//        // add last input constraint
//        problem.add_constr(U[N] == U[N - 1]);
    }
};

#endif //LAMPC_MULTIPLE_SHOOTING_TRANSCRIPTION_HPP
