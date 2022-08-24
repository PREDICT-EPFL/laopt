#ifndef LAOPT_MULTIPLESHOOTING_HPP
#define LAOPT_MULTIPLESHOOTING_HPP

// Advanced user (level 2)

#include <Eigen/Dense>
#include "laopt/laopt.hpp"

#define PRINT(x) \
//std::cout << __FUNCTION__ << x << std::endl // Comment this line in to activate PRINT function in the code

namespace transcription {

/*
 * Multiple Shooting
 * |     |     |     |    ...     |     |
 * 0     1     2     3    ...    N-1    N     Decision variable indices (initial condition + number of segments)
 * 0     1     2     3         N_segs-1       Segment indices of N_segs segments
 * */
template<typename ControlProblem, unsigned N_segs>
class MultipleShooting : public laopt::Differentiable<MultipleShooting<ControlProblem, N_segs>>
{
protected: // TODO ino1
    /* Mirror scalar type (from ControlProblem), define variable template with scalar type */
    using Scalar = typename ControlProblem::Scalar;
    template<int n>
    using variable_t = laopt::Variable<Scalar, n>;

    /* Instance of end user's ControlProblem */
    ControlProblem &controlProblem;

    /* Create discrete problem variables (define U_var with same length than X_var for easier data handling,
     * although last u will not be used */
    static const unsigned N = N_segs; // Last index of decision variables
    const double h{1.0 / N};
    Eigen::Vector<Scalar, N + 1> T; // Normalized time grid (0 ... 1)
    std::array<variable_t<ControlProblem::NX>, N + 1> X_var;
    std::array<variable_t<ControlProblem::NU>, N + 1> U_var;

public:
    /* Objective */
    struct StageCost {};
    template<typename x_t, typename u_t, typename scalar_t = typename Eigen::MatrixBase<x_t>::Scalar>
    EIGEN_STRONG_INLINE scalar_t
    function_impl(StageCost, const Eigen::MatrixBase<x_t> &x, const Eigen::MatrixBase<u_t> &u)
    {
        return h * controlProblem.template lagrange_term_impl<scalar_t>(x, u);
    }

    struct MayerCost {};
    template<typename x_t, typename scalar_t = typename Eigen::MatrixBase<x_t>::Scalar>
    EIGEN_STRONG_INLINE scalar_t
    function_impl(MayerCost, const Eigen::MatrixBase<x_t> &x)
    {
        return controlProblem.template mayer_term_impl<scalar_t>(x);
    }

    /* Dynamic constraints */
    struct DiscreteDynamics {};
    template<typename x_t, typename u_t, typename scalar_t = typename Eigen::MatrixBase<x_t>::Scalar>
    EIGEN_STRONG_INLINE Eigen::Vector<scalar_t, ControlProblem::NX>
    function_impl(DiscreteDynamics, const Eigen::MatrixBase<x_t> &x, const Eigen::MatrixBase<u_t> &u)
    {
        using state_t = typename x_t::PlainObject;
        state_t k1 = controlProblem.template dynamics_impl<scalar_t>(x, u);
        state_t k2 = controlProblem.template dynamics_impl<scalar_t>(x + h * 0.5 * k1, u);
        state_t k3 = controlProblem.template dynamics_impl<scalar_t>(x + h * 0.5 * k2, u);
        state_t k4 = controlProblem.template dynamics_impl<scalar_t>(x + h * k3, u);
        return x + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    }

public:
    explicit MultipleShooting(ControlProblem &ctrlProblem_) :
            controlProblem(ctrlProblem_) {}

    using scalar_t = typename ControlProblem::Scalar; // TODO: Change in laOPT to accept Scalar
    using TimeTrajectory = Eigen::Vector<Scalar, N + 1>;
    using StateTrajectory = Eigen::Matrix<Scalar, ControlProblem::NX, N + 1>;
    using InputTrajectory = Eigen::Matrix<Scalar, ControlProblem::NU, N + 1>;

    /* Set functions */
    template<int rows, typename Scalar = double>
    void set_X_guess(const Eigen::Matrix<Scalar, rows, 1> &x_guess)
    {
        for (unsigned i = 0; i < X_var.size(); i++) { X_var.at(i) << x_guess; }
    }
    template<int rows, int cols, typename Scalar = double>
    void set_X_guess(const Eigen::Matrix<Scalar, rows, cols> &X_guess)
    {
        for (unsigned i = 0; i < X_var.size(); i++) { X_var.at(i) << X_guess.col(i); }
    }

    /* Get functions */
    const TimeTrajectory &get_T_opt() const { return T; }
    StateTrajectory get_X_opt()
    {
        StateTrajectory X_opt;
        for (unsigned i = 0; i < X_var.size(); i++) { X_opt.col(i) << X_var.at(i); }
        return X_opt;
    }
    InputTrajectory get_U_opt()
    {
        InputTrajectory U_opt;
        for (unsigned i = 0; i < U_var.size(); i++) { U_opt.col(i) << U_var.at(i); }
        return U_opt;
    }

//protected: // TODO ino1 (would like to make this protected)
    template<typename OptProblem>
    void define_problem(OptProblem &optProblem)
    {
        PRINT("define_problem");
        /* Register variables */
        for (unsigned i = 0; i <= N; i++)
        {
            optProblem.add_variable(X_var[i]); // TODO eno1: Loop through array in add_variable() -> Do not want that
            optProblem.add_variable(U_var[i]);
        }

        /* Loop through grid points */
        for (unsigned i = 0; i < N; i++)
        {
            T(i) = i * controlProblem.tf / N;

            PRINT("T(" << i << ") = " << T(i));
            optProblem.add_obj(this->function(StageCost{}, X_var[i], U_var[i])); // TODO eno4: Allow expressions here
            optProblem.add_constr(X_var[i + 1] == this->function(DiscreteDynamics{}, X_var[i], U_var[i]));
        }

        /* Last grid point */
        T(N) = controlProblem.tf;
        PRINT("T(" << N << ") = " << T(N));
        optProblem.add_obj(this->function(MayerCost{}, X_var[N]));

        /* Box constraints */
        for (unsigned i = 0; i <= N; i++)
        {
            optProblem.add_constr(controlProblem.lbx <= X_var[i] <= controlProblem.ubx);
            optProblem.add_constr(controlProblem.lbu <= U_var[i] <= controlProblem.ubu);
        }

        /* Boundary constraints */
        optProblem.add_constr(controlProblem.x0_lb <= X_var[0] <= controlProblem.x0_ub);
        optProblem.add_constr(controlProblem.xf_lb <= X_var[N] <= controlProblem.xf_ub);

        /* Set last control equal second last for easier data handling */
        optProblem.add_constr(U_var[N] == U_var[N - 1]);
    }
};

} // namespace transcription

#endif //LAOPT_MULTIPLESHOOTING_HPP
