#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>

#include <laopt/laopt.hpp>
#include <laopt/solvers/piqp_interface.hpp>
#include <laopt/solvers/sqp_solver.hpp>
#include <laopt/tools/control_problem_base.hpp>
#include <laopt/tools/multiple_shooting.hpp>

class InvertedPendulum
    : public laopt_tools::ControlProblemBase<double, 2, 1>
{
public:
    template <typename X, typename U, typename P, typename T0,
              typename TF, typename Tau,
              typename Scalar = typename X::Scalar>
    Scalar lagrange_term_impl(
        const Eigen::MatrixBase<X>& x,
        const Eigen::MatrixBase<U>& u,
        const Eigen::MatrixBase<P>& p,
        const Eigen::MatrixBase<T0>& t0,
        const Eigen::MatrixBase<TF>& tf,
        const Tau& tau)
    {
        unused(p, t0, tf, tau);
        return 10.0 * x(0) * x(0) + u(0) * u(0);
    }

    template <typename XF, typename P, typename T0, typename TF,
              typename Scalar = typename XF::Scalar>
    Scalar mayer_term_impl(
        const Eigen::MatrixBase<XF>& xf,
        const Eigen::MatrixBase<P>& p,
        const Eigen::MatrixBase<T0>& t0,
        const Eigen::MatrixBase<TF>& tf)
    {
        unused(p, t0, tf);
        return 100.0 * xf(0) * xf(0);
    }

    template <typename X, typename U, typename P, typename T0,
              typename TF, typename Tau,
              typename Scalar = typename X::Scalar>
    state_t<Scalar> dynamics_impl(
        const Eigen::MatrixBase<X>& x,
        const Eigen::MatrixBase<U>& u,
        const Eigen::MatrixBase<P>& p,
        const Eigen::MatrixBase<T0>& t0,
        const Eigen::MatrixBase<TF>& tf,
        const Tau& tau)
    {
        unused(p, t0, tf, tau);

        const double g = 9.81;
        const double l = 0.5;
        const double m = 0.15;
        const double b = 0.1;

        const Scalar theta = x(0);
        const Scalar omega = x(1);
        const Scalar torque = u(0);

        state_t<Scalar> x_dot;
        x_dot << omega,
            (m * g * l * sin(theta) - b * omega + torque)
                / (m * l * l);
        return x_dot;
    }
};

int main(int argc, char** argv)
{
    const char* output_path = argc > 1 ? argv[1] : "trajectory.csv";

    using Ocp = InvertedPendulum;
    using Transcription = laopt_tools::MultipleShooting<Ocp, 40>;
    using Problem = laopt::Problem<Transcription>;
    using QPSolver = laopt::PIQPSolver<double>;

    auto ocp = std::make_shared<Ocp>();
    auto transcription = std::make_shared<Transcription>(ocp);
    auto problem = std::make_shared<Problem>(transcription);

    ocp->set_x0(Ocp::State{3.141592653589793, 0.0});
    ocp->set_tf(1.5);
    ocp->u_lb << -3.0;
    ocp->u_ub << 3.0;

    laopt::SQPSolver<Problem, QPSolver> solver(problem);
    solver.settings().verbose = true;
    solver.solve();

    const auto time = transcription->get_T_opt();
    const auto state = transcription->get_X_opt();
    const auto input = transcription->get_U_opt();

    std::ofstream csv(output_path);
    if (!csv)
    {
        throw std::runtime_error("could not open trajectory output file");
    }

    csv.precision(16);
    csv << "time,angle,angular_velocity,torque\n";
    for (Eigen::Index i = 0; i < time.size(); ++i)
    {
        // Multiple shooting has one fewer input than state nodes. Repeat the
        // final zero-order-held input at the final state node for plotting.
        const Eigen::Index input_index = std::min(i, input.cols() - 1);
        csv << time(i) << ','
            << state(0, i) << ','
            << state(1, i) << ','
            << input(0, input_index) << '\n';
    }

    std::cout << "wrote " << time.size() << " samples to " << output_path << '\n';
    std::cout << "terminal state: "
              << state.col(state.cols() - 1).transpose() << '\n';
}
