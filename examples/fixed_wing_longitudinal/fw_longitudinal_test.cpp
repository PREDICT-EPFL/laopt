#include <iostream>
#include <iomanip>

#include "LonOcpEigen.hpp"
#include "../double_integrator_v0/multiple_shooting_transcription.hpp"
#include "ipopt_interface.hpp"

int main()
{
    /* Choose OCP and Transcription */
    using Ocp = lon_ocp::LonFlightOCP;
    const int N = 30;
    using Transcription = MultipleShootingTranscription<Ocp, N>;

    /* Define specific Tape, LAMPC, and IPOPT problem types for the resulting NLP */
    using Tape = lampc::TapeInfo<Transcription>;
    using LaProblem = lampc::Problem<Transcription>;
    using IpoptProblem = lampc::Solver_IPOpt<LaProblem>;

    /* Construct OCP and transcription, optionally generate/store tape for that combination */
    Ocp ocp;
    ocp.model.load_params_from_yaml("eg4_xflr-Pvw-YR.yaml");
    ocp.objectives[lon_ocp::TrackAngle] = true;
//    ocp.objectives[lon_ocp::TrackVa] = true;
    ocp.objectives[lon_ocp::MinimizeControl] = true;
    ocp.tf = 2.0;

    Transcription transcription(ocp);
    Tape tape = lampc::generate_tape(transcription, lampc::generate_sparsity(transcription));

    /* Construct LAMPC and IPOPT problems for transcribed OCP using according tape, link decision variables between problems */
    LaProblem nlp(transcription, tape); // Tape is optional here and could also be generated internally
    SmartPtr<IpoptProblem> ipopt_nlp = new IpoptProblem(nlp);
    nlp.set_decision_variable(ipopt_nlp->init_primal);

    /* Create IPOPT solver, setup, and initialize */
    SmartPtr<IpoptApplication> ipopt_solver = IpoptApplicationFactory();
    // ipopt_solver->Options()->SetStringValue("hessian_approximation", "limited-memory");
    ipopt_solver->Options()->SetIntegerValue("print_level", 5);
    ApplicationReturnStatus ipopt_status = ipopt_solver->Initialize();
    if( ipopt_status != Solve_Succeeded ) { std::cout << std::endl << std::endl << "*** Error during initialization!" << std::endl; }

    /* Set initial state and solve the problem */
    ocp.x0 << ocp.model.get_default_initial_state();
    ipopt_status = ipopt_solver->OptimizeTNLP(ipopt_nlp);
    if( ipopt_status != Solve_Succeeded ) { std::cout << std::endl << std::endl << "*** Error during solution!" << std::endl; }

    /* Print out the solution */
    std::cout << std::endl << std::endl << std::endl << std::endl;
    std::cout << std::setprecision(2) << std::defaultfloat;

    Eigen::Matrix<Ocp::scalar_t, Ocp::NX, N> X;
    Eigen::Matrix<Ocp::scalar_t, Ocp::NU, N - 1> U;
    int i = 0; for(auto& x: transcription.X) X.col(i++) << x;
    i = 0; for(auto& u: transcription.U) U.col(i++) << u;

    std::cout << "X = \n" << X << std::endl;
    std::cout << "U = \n" << U << std::endl;
    std::cout << "obj = " << nlp.eval_objective(lampc::Eval(), ipopt_nlp->sol_primal) << std::endl;

    return 0;
}
