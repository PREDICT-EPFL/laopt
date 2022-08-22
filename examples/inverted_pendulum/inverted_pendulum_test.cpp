#include <iostream>
#include <iomanip>
#include <chrono>

#include "inverted_pendulum_ocp.hpp"
#include "MultipleShooting.hpp"
#include "laopt/ipopt_wrapper.hpp"

int main()
{
    using namespace std::chrono;

    /* Choose OCP and Transcription */
    using Ocp = InvertedPendulumOcp;

    /* Construct OCP and set OCP-specific properties */
    Ocp ocp;

    ocp.ubu << 3;
    ocp.lbu << -3;

    ocp.tf = 1.5;

    ocp.angle_ref = 20.0 * M_PI / 180.0;
    ocp.mayer_multiplier = 0;

    ocp.set_x0({M_PI, 0});

    /* Solve with Multiple Shooting transcription */
    {
        std::cout << "Multiple Shooting\n";
        const int N = 30;
        using Transcription = transcription::MultipleShooting<Ocp, N>;

        /* Define specific Tape, laOPT, and IPOPT problem types for the resulting NLP */
        using Tape = laopt::TapeInfo<Transcription>;
        using OptProblem = laopt::Problem<Transcription>;
        using Solver = laopt::IpoptWrapper<OptProblem>;

        /* Construct transcription for OCP, optionally generate/store tape for that combination */
        Transcription transcription(ocp);
        Tape tape = laopt::generate_tape(transcription, laopt::generate_sparsity(transcription));

        /* Construct laOPT and IPOPT problems for transcribed OCP using according tape */
        OptProblem opt_problem(transcription, tape); // Tape is optional here and could also be generated internally
        Solver solver(opt_problem);

        const steady_clock::time_point tstart = steady_clock::now();
        solver.solve();
        const steady_clock::time_point tend = steady_clock::now();
        const long duration_us = duration_cast<microseconds>(tend - tstart).count();

//        solver.solve(); // Call second time to test repeatability

        /* Print out the solution */
        std::cout << "\n\n";
        std::cout << std::setprecision(4) << std::defaultfloat;

        Transcription::TimeTrajectory T_opt = transcription.get_T_opt().transpose();
        Transcription::StateTrajectory X_opt = transcription.get_X_opt();
        Transcription::InputTrajectory U_opt = transcription.get_U_opt();
        const double obj_eval = opt_problem.eval_objective(laopt::Eval(), solver.sol_primal());

        std::cout << "Comp. time: " << duration_us << " us, tf = " << T_opt(T_opt.size() - 1) << " s, obj = " << obj_eval << "\n";
        std::cout << "T = [" << transcription.get_T_opt().transpose() << "];\n";
        std::cout << "X_opt = [\n" << X_opt << "];\n";
        std::cout << "U_opt = [\n" << U_opt << "];\n";
        std::cout << "obj = " << obj_eval << ";\n";
    }

    return 0;
}
