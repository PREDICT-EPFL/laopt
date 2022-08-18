#include <iostream>
#include <iomanip>

#include "double_integrator_ocp.hpp"
#include "multiple_shooting_transcription.hpp"
#include "MultipleShooting.hpp"
#include "laopt/ipopt_wrapper.hpp"

int main()
{
    /* Choose OCP and Transcription */
    using Ocp = DoubleIntegratorOcp;

    /* Construct OCP and set OCP-specific properties */
    Ocp ocp;

    ocp.ubu << 10;
    ocp.lbu << -3;

    ocp.x_ref << 1, 0;

    ocp.set_x0({0.1, 0.2});               // for demonstration, last setting counts
    ocp.x0_lb = ocp.x0_ub = {0.1, 0.2};   // for demonstration, last setting counts
    ocp.set_x0({-0.1, -0.2}, {0.1, 0.2}); // for demonstration, last setting counts

    /* Solve with Multiple Shooting transcription */
    {
        const int N = 20;
        using Transcription = transcription::MultipleShooting<Ocp, N>;

        /* Define specific Tape, LAMPC, and IPOPT problem types for the resulting NLP */
        using Tape = laopt::TapeInfo<Transcription>;
        using OptProblem = laopt::Problem<Transcription>;
//        using IpoptProblem = laopt::Solver_IPOpt<OptProblem>;
        using Solver = laopt::IpoptWrapper<OptProblem>;

        /* Construct transcription for OCP, optionally generate/store tape for that combination */
        Transcription transcription(ocp);
        Tape tape = laopt::generate_tape(transcription, laopt::generate_sparsity(transcription));

        /* Construct laOPT and IPOPT problems for transcribed OCP using according tape, link decision variables between problems */
        OptProblem opt_problem(transcription, tape); // Tape is optional here and could also be generated internally
        Solver solver(opt_problem);

        solver.solve();

        /* Print out the solution */
        std::cout << std::endl << std::endl << std::endl << std::endl;
        std::cout << std::setprecision(6) << std::defaultfloat;

        Transcription::StateTrajectory Xopt = transcription.get_Xopt();
        Transcription::InputTrajectory Uopt = transcription.get_Uopt();

        std::cout << "T = \n" << transcription.get_Topt().transpose() << std::endl;
        std::cout << "Xopt = \n" << Xopt << std::endl;
        std::cout << "Uopt = \n" << Uopt << std::endl;
        const double objective_eval = opt_problem.eval_objective(laopt::Eval(), solver.sol_primal);
        std::cout << "obj: " << objective_eval << "\n";
    }

    return 0;
}
