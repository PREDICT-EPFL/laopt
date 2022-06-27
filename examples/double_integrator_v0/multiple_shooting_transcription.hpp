#ifndef LAMPC_MULTIPLE_SHOOTING_TRANSCRIPTION_HPP
#define LAMPC_MULTIPLE_SHOOTING_TRANSCRIPTION_HPP

#include <Eigen/Dense>
#include "lampc.hpp"

// Advanced user (level 2)

template<typename OCP, int N>
class MultipleShootingTranscription
{
public:

    OCP &ocp;

    using scalar_t = typename OCP::scalar_t; // lampc::Problem assumes that the user code will contain a scalar_t

    /**
     * Continuous-time dynamics - double integrator
     */
    struct sys_t
    {
        OCP &internal_ocp;
        explicit sys_t(OCP &p): internal_ocp(p) {}

        template<typename X, typename U, typename Scalar = typename Eigen::MatrixBase<X>::Scalar>
        EIGEN_STRONG_INLINE Eigen::Vector<Scalar, OCP::NX>
        impl(const Eigen::MatrixBase<X>& x, const Eigen::MatrixBase<U>& u) noexcept
        {
            Eigen::Vector<Scalar, OCP::NX> x_dot;
            internal_ocp.template dynamics_impl<Scalar>(x, u, x_dot);
            return x_dot;
        }
    } sys;

    // Discretized dynamics
    using dsys_t = lampc::functions::RK4<sys_t, scalar_t>;
    dsys_t dsys;

    struct stage_cost_t : public lampc::MakeDifferentiable<stage_cost_t>
    {
        OCP &internal_ocp;
        explicit stage_cost_t(OCP& p) : internal_ocp(p) {}

        template<typename X, typename U, typename Scalar = typename Eigen::MatrixBase<X>::Scalar>
        EIGEN_STRONG_INLINE Scalar
        impl(const Eigen::MatrixBase<X>& x, const Eigen::MatrixBase<U>& u) noexcept
        {
            Scalar lagrange;
            internal_ocp.template lagrange_term_impl<Scalar>(x, u, lagrange);
            return lagrange;
        }
    } stage_cost;

    struct terminal_cost_t : public lampc::MakeDifferentiable<terminal_cost_t>
    {
        OCP &internal_ocp;
        explicit terminal_cost_t(OCP& p) : internal_ocp(p) {}

        template<typename X, typename Scalar = typename Eigen::MatrixBase<X>::Scalar>
        EIGEN_STRONG_INLINE Scalar
        impl(const Eigen::MatrixBase<X>& x) noexcept
        {
            Scalar mayer;
            internal_ocp.template mayer_term_impl<Scalar>(x, mayer);
            return mayer;
        }
    } terminal_cost;

    template<size_t n>
    using Variable = lampc::Variable<scalar_t, n>;

    Eigen::Vector<scalar_t, N+1> T;
    std::array<Variable<OCP::NX>, N+1> X;
    std::array<Variable<OCP::NU>, N+1> U;

    // Functions we use to define the problem
    lampc::functions::eq<dsys_t> dsys_eq; // dsys(x,u) - xp

    lampc::functions::id id; // Identity

    explicit MultipleShootingTranscription(OCP &ocp_) :
    ocp(ocp_),
    sys(ocp),
    dsys(sys, ocp.tf / N),
    stage_cost(ocp),
    terminal_cost(ocp),
    dsys_eq(dsys)
    {
        for(int i = 0; i < N+1; i++) { T(i) = i * ocp.tf / N; }
    };

    template<typename OptProblem>
    void define_variables(OptProblem& problem)
    {
        for(int i = 0; i < N+1; i++)
        {
            problem.add_variable(X[i]);
            problem.add_variable(U[i]);
        }
    }

    template<typename Bounds>
    void eval_variable_bounds(Bounds& bnd)
    {
        // Add constraints on the variables
        for(auto& u : U) ocp.lbu <= bnd.add(u) <= ocp.ubu;
        for(auto& x : X) ocp.lbx <= bnd.add(x) <= ocp.ubx;
    }

    template<typename Constraints, typename Dtype>
    void eval_constraints(Dtype dtype, Constraints& con)
    {
        // Initial state
        con.add(dtype, id, X[0]) == ocp.x0;

        // Dynamics
        for(int i=0; i<N; i++) con.add(dtype, dsys_eq, X[i+1], X[i], U[i]) == 0;

        // Last input
        con.add(dtype, id, U[N]) == U[N-1];
    }

    template<typename Objective, typename Dtype>
    void eval_objective(Dtype dtype, Objective& obj)
    {
        for(int i=0; i<N; i++) { obj.add(dtype, stage_cost, X[i], U[i]); }
        obj.add(dtype, terminal_cost, X[N]);
    }
};

#endif //LAMPC_MULTIPLE_SHOOTING_TRANSCRIPTION_HPP
