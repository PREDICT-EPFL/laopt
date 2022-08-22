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
 * |    |    |    |   ...    |    |
 * 0    1    2    3   ...   N-1   N     Decision variable indices (initial condition + number of segments)
 *   1    2    3                N       Number of segments
 * */
template<typename ControlProblem, unsigned N>
class MultipleShooting : public laopt::Differentiable<MultipleShooting<ControlProblem, N>>
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
    Eigen::Vector<Scalar, N + 1> T; // Fixed time
    std::array<variable_t<ControlProblem::NX>, N + 1> X_var;
    std::array<variable_t<ControlProblem::NU>, N + 1> U_var;
    double h{1.0 / N};

public:
    /* Objective */
    struct StageCost {};
    template<typename x_t, typename u_t, typename scalar_t = typename Eigen::MatrixBase<x_t>::Scalar>
    EIGEN_STRONG_INLINE scalar_t
    function_impl(StageCost, const Eigen::MatrixBase<x_t> &x, const Eigen::MatrixBase<u_t> &u)
    {
        scalar_t lagrange;
        controlProblem.template lagrange_term_impl<scalar_t>(lagrange, x, u);
        return h * lagrange;
    }

    struct MayerCost {};
    template<typename x_t, typename scalar_t = typename Eigen::MatrixBase<x_t>::Scalar>
    EIGEN_STRONG_INLINE scalar_t
    function_impl(MayerCost, const Eigen::MatrixBase<x_t> &x)
    {
        scalar_t mayer;
        controlProblem.template mayer_term_impl<scalar_t>(mayer, x);
        return mayer;
    }

    /* Dynamic constraints */
    struct DiscreteDynamics {};
    template<typename x_t, typename u_t, typename scalar_t = typename Eigen::MatrixBase<x_t>::Scalar>
    EIGEN_STRONG_INLINE Eigen::Vector<scalar_t, ControlProblem::NX>
    function_impl(DiscreteDynamics, const Eigen::MatrixBase<x_t> &x, const Eigen::MatrixBase<u_t> &u)
    {
        Eigen::Vector<scalar_t, ControlProblem::NX> x_dot;

        const double H = (controlProblem.tf - controlProblem.t0);

        using Vec = typename x_t::PlainObject;
        controlProblem.template dynamics_impl<scalar_t>(x_dot, x, u);
        Vec k1 = H * x_dot;
        controlProblem.template dynamics_impl<scalar_t>(x_dot, x + h * 0.5 * k1, u);
        Vec k2 = H * x_dot;
        controlProblem.template dynamics_impl<scalar_t>(x_dot, x + h * 0.5 * k2, u);
        Vec k3 = H * x_dot;
        controlProblem.template dynamics_impl<scalar_t>(x_dot, x + h * k3, u);
        Vec k4 = H * x_dot;
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
        for (unsigned i = 0; i < N + 1; i++)
        {
            optProblem.add_variable(X_var[i]); // TODO eno1: Loop through array in add_variable()
            optProblem.add_variable(U_var[i]);
        }

        /* Loop through discretization points */
        for (unsigned i = 0; i < N; i++)
        {
            T(i) = i * controlProblem.tf / N;

            PRINT("T(" << i << ") = " << T(i));
            optProblem.add_obj(this->function(StageCost{}, X_var[i], U_var[i])); // TODO eno4: Allow expressions here
            optProblem.add_constr(X_var[i + 1] == this->function(DiscreteDynamics{}, X_var[i], U_var[i]));
        }

        /* Last discretization point */
        T(N) = controlProblem.tf;
        PRINT("T(" << N << ") = " << T(N));
        optProblem.add_obj(this->function(MayerCost{}, X_var[N]));

        /* Set last control equal second last for easier data handling */
        optProblem.add_constr(U_var[N] == U_var[N - 1]);

        /* Box constraints */
        for (unsigned i = 0; i < N + 1; i++)
        {
            optProblem.add_constr(controlProblem.lbx <= X_var[i] <= controlProblem.ubx);
            optProblem.add_constr(controlProblem.lbu <= U_var[i] <= controlProblem.ubu);
        }

        /* Boundary constraints */
        optProblem.add_constr(controlProblem.x0_lb <= X_var[0] <= controlProblem.x0_ub);
        optProblem.add_constr(controlProblem.xf_lb <= X_var[N] <= controlProblem.xf_ub);
    }
};

} // namespace transcription

#endif //LAOPT_MULTIPLESHOOTING_HPP
