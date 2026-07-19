// #include "laopt/laopt.hpp"

#include "inverted_pendulum_simple_ocp.hpp"
#include "laopt/tools/multiple_shooting.hpp"
#include "laopt/tools/radau_collocation.hpp"
#include "laopt/solvers/ipopt_interface.hpp"
// #include "laopt/solvers/sqp_solver.hpp"
// #include "laopt/solvers/piqp_interface.hpp"

int main()
{
    const double Ts = 0.01;

    // Construction
    using Ocp = InvertedPendulumSimpleOcp;
    // using Transcription = laopt_tools::MultipleShooting<Ocp, 20>;
    using Transcription = laopt_tools::RadauCollocation<Ocp, 10, 3>;
    using OptProblem = laopt::Problem<Transcription>;
    using Solver = laopt::IpoptSolver<OptProblem>;
    // using Solver = laopt::SQPSolver<OptProblem, laopt::PIQPSolver<>>;

    auto ocp = std::make_shared<Ocp>();
    auto transcription = std::make_shared<Transcription>(ocp);
    auto opt_problem = std::make_shared<OptProblem>(transcription);
    Solver solver(opt_problem);

    ocp->set_tf( 1.5 );
    ocp->u_ub << +3;
    ocp->u_lb << -3;

    // Solve for initial state x0
    Ocp::State x0 = {M_PI, 0};
    ocp->set_x0(x0);
    Ipopt::ApplicationReturnStatus solve_status = solver.solve();

    // Obtain solution
    const Eigen::VectorXd T_opt = transcription->get_T_opt();
    const Eigen::MatrixXd X_opt = transcription->get_X_opt();
    const Eigen::MatrixXd U_opt = transcription->get_U_opt();
    const Eigen::MatrixXd TX_resampled = transcription->get_TX_resampled( Ts );
    const Eigen::MatrixXd TU_resampled = transcription->get_TU_resampled( Ts );

    std::cout << "T_opt:\n" << T_opt.transpose() << "\n"
              << "X_opt:\n" << X_opt << "\n"
              << "U_opt:\n" << U_opt << "\n";

    std::cout << "TX_resampled:\n" << TX_resampled << "\n";
    return 0;
}
