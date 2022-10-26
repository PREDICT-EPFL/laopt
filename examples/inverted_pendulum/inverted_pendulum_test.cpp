#include <iostream>
#include <iomanip>
#include <chrono>

#include "inverted_pendulum_ocp.hpp"
#include "MultipleShooting.hpp"
#include "RadauCollocation.hpp"
#include "laopt/ipopt_wrapper.hpp"

#include "examples_helper.hpp"

int main()
{
    using namespace std::chrono;

    /* Choose OCP and Transcription */
    using Ocp = InvertedPendulumOcp;

    /* Construct OCP and set OCP-specific properties */
    Ocp ocp;

    ocp.tf_lb = 1.5;
    ocp.tf_ub = 2;
    ocp.w_tf = 5;

    ocp.ubu << 3;
    ocp.lbu << -3;

//    ocp.angle_ref = 20.0 * M_PI / 180.0;
//    ocp.mayer_multiplier = 0;

    ocp.set_x0({3.14, 0});

    /* Resampling test parameters */
    const double Ts_max = 0.02;
    const double t_test = 0.166;

    /* Solve with Multiple Shooting transcription */
    if (true)
    {
        std::cout << "Multiple Shooting\n";
        const int N = 20;
        using Transcription = transcription::MultipleShooting<Ocp, N>;

        /* Define specific Tape, laOPT, and IPOPT problem types for the resulting NLP */
        using Tape = laopt::TapeInfo<Transcription>;
        using OptProblem = laopt::Problem<Transcription>;
        using Solver = laopt::IpoptWrapper<OptProblem>;

        /* Construct transcription for OCP, optionally generate/store tape for that combination */
        Transcription transcription(ocp);
        Tape tape = laopt::generate_tape(transcription, laopt::generate_sparsity(transcription));
//        std::cout << "Constraints Jacobian sparsity:\n" << tape.constraints.jacobian.sparsity_structure << std::endl;
//        std::cout << "Objective Hessian sparsity:\n" << tape.objective.hessian.sparsity_structure << std::endl;

        /* Construct laOPT and IPOPT problems for transcribed OCP using according tape */
        OptProblem opt_problem(transcription, tape); // Tape is optional here and could also be generated internally
        Solver solver(opt_problem, /* print_level (default = 0) */ 5);

        const steady_clock::time_point t_start = steady_clock::now();
        solver.solve();
        const steady_clock::time_point t_end = steady_clock::now();
        const long duration_us = duration_cast<microseconds>(t_end - t_start).count();

        solver.solve(); // Call second time to test repeatability
        const steady_clock::time_point t_end2 = steady_clock::now();
        const long duration2_us = duration_cast<microseconds>(t_end2 - t_end).count();

        /* Print out the solution */
        print_solution(transcription, opt_problem, solver, duration_us, duration2_us);
        print_sampled_solution(transcription, Ts_max, t_test);
    }

    /* Solve with Radau Collocation transcription */
    if (true)
    {
        std::cout << "Radau Collocation\n";
        const int D_poly = 4;
        const int N_segs = 3;
        using Transcription = transcription::RadauCollocation<Ocp, N_segs, D_poly>;

        /* Define specific Tape, laOPT, and IPOPT problem types for the resulting NLP */
        using Tape = laopt::TapeInfo<Transcription>;
        using OptProblem = laopt::Problem<Transcription>;
        using Solver = laopt::IpoptWrapper<OptProblem>;

        /* Construct transcription for OCP, optionally generate/store tape for that combination */
        Transcription transcription(ocp);
        Tape tape = laopt::generate_tape(transcription, laopt::generate_sparsity(transcription));
//        std::cout << "Constraints Jacobian sparsity:\n" << tape.constraints.jacobian.sparsity_structure << std::endl;
//        std::cout << "Objective Hessian sparsity:\n" << tape.objective.hessian.sparsity_structure << std::endl;

        /* Construct laOPT and IPOPT problems for transcribed OCP using according tape */
        OptProblem opt_problem(transcription, tape); // Tape is optional here and could also be generated internally
        Solver solver(opt_problem, /* print_level (default = 0) */ 0);

        const steady_clock::time_point t_start = steady_clock::now();
        solver.solve();
        const steady_clock::time_point t_end = steady_clock::now();
        const long duration_us = duration_cast<microseconds>(t_end - t_start).count();

        solver.solve(); // Call second time to test repeatability
        const steady_clock::time_point t_end2 = steady_clock::now();
        const long duration2_us = duration_cast<microseconds>(t_end2 - t_end).count();

        /* Print out the solution */
        print_solution(transcription, opt_problem, solver, duration_us, duration2_us);
        print_sampled_solution(transcription, Ts_max, t_test);
    }

    return 0;
}