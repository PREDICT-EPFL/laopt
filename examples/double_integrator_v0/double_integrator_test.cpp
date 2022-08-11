#include <iostream>
#include <iomanip>

#include "double_integrator_ocp.hpp"
#include "multiple_shooting_transcription.hpp"
#include "ipopt_interface.hpp"

int main()
{
    /* Choose OCP and Transcription */
    using Ocp = DoubleIntegratorOCP;
    const int N = 30;
    using Transcription = MultipleShootingTranscription<Ocp, N>;

    /* Define specific Tape, LAMPC, and IPOPT problem types for the resulting NLP */
    using Tape = laopt::TapeInfo<Transcription>;
    using LaProblem = laopt::Problem<Transcription>;
    using IpoptProblem = laopt::Solver_IPOpt<LaProblem>;

    /* Construct OCP and transcription, optionally generate/store tape for that combination */
    Ocp ocp;
    Transcription transcription(ocp);
    Tape tape = laopt::generate_tape(transcription, laopt::generate_sparsity(transcription));

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
    ocp.x0 << 1, 1;
    ipopt_status = ipopt_solver->OptimizeTNLP(ipopt_nlp);
    if( ipopt_status != Solve_Succeeded ) { std::cout << std::endl << std::endl << "*** Error during solution!" << std::endl; }

    /* Print out the solution */
    std::cout << std::endl << std::endl << std::endl << std::endl;
    std::cout << std::setprecision(6) << std::defaultfloat;

    Eigen::Matrix<Ocp::scalar_t, Ocp::NX, transcription.X.size()> X;
    Eigen::Matrix<Ocp::scalar_t, Ocp::NU, transcription.U.size()> U;
    int i{0};
    i = 0; for(auto& x: transcription.X) X.col(i++) << x;
    i = 0; for(auto& u: transcription.U) U.col(i++) << u;

    std::cout << "T = \n" << transcription.T.transpose() << std::endl;
    std::cout << "X = \n" << X << std::endl;
    std::cout << "U = \n" << U << std::endl;
    std::cout << "obj = " << nlp.eval_objective(laopt::Eval(), ipopt_nlp->sol_primal) << std::endl;

    return 0;
}
