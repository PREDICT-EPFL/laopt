#include <iostream>
#include <iomanip>

#include "double_integrator_ocp.hpp"
#include "multiple_shooting_transcription.hpp"
#include "MultipleShooting.hpp"
#include "ipopt_interface.hpp"

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
        using Tape = lampc::TapeInfo<Transcription>;
        using OptProblem = lampc::Problem<Transcription>;
        using IpoptProblem = lampc::Solver_IPOpt<OptProblem>;

        /* Construct transcription for OCP, optionally generate/store tape for that combination */
        Transcription transcription(ocp);
        Tape tape = lampc::generate_tape(transcription, lampc::generate_sparsity(transcription));

        /* Construct LAMPC and IPOPT problems for transcribed OCP using according tape, link decision variables between problems */
        OptProblem opt_problem(transcription, tape); // Tape is optional here and could also be generated internally
        SmartPtr<IpoptProblem> ipopt_problem = new IpoptProblem(opt_problem);
        opt_problem.set_decision_variable(ipopt_problem->init_primal);

        /* Create IPOPT solver, setup, and initialize */
        SmartPtr<IpoptApplication> ipopt_solver = IpoptApplicationFactory();
        // ipopt_solver->Options()->SetStringValue("hessian_approximation", "limited-memory");
        ipopt_solver->Options()->SetIntegerValue("print_level", 5);
        ApplicationReturnStatus ipopt_status = ipopt_solver->Initialize();
        if (ipopt_status != Solve_Succeeded)
        {
            std::cout << std::endl << std::endl << "*** Error during initialization!" << std::endl;
        }

        /* Set initial state and solve the problem */
        ipopt_status = ipopt_solver->OptimizeTNLP(ipopt_problem);
        if (ipopt_status != Solve_Succeeded)
        {
            std::cout << std::endl << std::endl << "*** Error during solution!" << std::endl;
        }

        /* Print out the solution */
        std::cout << std::endl << std::endl << std::endl << std::endl;
        std::cout << std::setprecision(6) << std::defaultfloat;

        Transcription::StateTrajectory Xopt = transcription.get_Xopt();
        Transcription::InputTrajectory Uopt = transcription.get_Uopt();

        std::cout << "T = \n" << transcription.get_Topt().transpose() << std::endl;
        std::cout << "Xopt = \n" << Xopt << std::endl;
        std::cout << "Uopt = \n" << Uopt << std::endl;
        const double objective_eval = opt_problem.eval_objective(lampc::Eval(), ipopt_problem->sol_primal);
        std::cout << "obj: " << objective_eval << "\n";
    }

    return 0;
}
