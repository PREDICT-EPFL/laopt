#include <iostream>
#include <chrono>

#include "laopt/laopt.hpp"

#include "inverted_pendulum_ocp.hpp"
#include "laopt/tools/multiple_shooting.hpp"
#include "laopt/tools/radau_collocation.hpp"
#ifdef LAOPT_WITH_IPOPT
#include "laopt/ipopt_interface/ipopt_wrapper.hpp"
#endif
#include "laopt/solvers/sqp_solver.hpp"
#ifdef LAOPT_WITH_PIQP
#include "laopt/solvers/piqp_interface.hpp"
#endif

#include "examples_helper.hpp"

int main()
{
    using namespace std::chrono;

    /* Choose OCP and Transcription */
    using Ocp = InvertedPendulumOcp;

    /* Construct OCP and set OCP-specific properties */
    Ocp ocp;

    ocp.tf_ub = 2;
    ocp.tf_lb = 1.5;
    ocp.w_tf = 3;

    ocp.angle_ref = 0.0 * M_PI / 180.0;
    // ref_offset
    ocp.opt_params_ub.ref_offset << 1;
    ocp.opt_params_lb.ref_offset << 0;
    // us
    ocp.opt_params_ub.us << 1;
    ocp.opt_params_lb.us << -1;

    ocp.u_ub << 3;
    ocp.u_lb << -3;

    ocp.set_x0({M_PI, 0});

    /* Resampling test parameters */
    const double Ts_max = 0.02;
    const double t_test = 0.166;

    auto solve_and_print = [&](auto& transcription, auto& opt_problem, auto& solver)
    {
        const steady_clock::time_point t_start = steady_clock::now();
        solver.solve();
        const steady_clock::time_point t_end = steady_clock::now();
        const long duration_us = duration_cast<microseconds>(t_end - t_start).count();

        solver.solve(); // Call second time to test repeatability
        const steady_clock::time_point t_end2 = steady_clock::now();
        const long duration2_us = duration_cast<microseconds>(t_end2 - t_end).count();

        /* Print out the solution */
        print_solution(transcription, opt_problem, duration_us, duration2_us);
        print_sampled_solution(transcription, Ts_max, t_test);

        /// Access optimization parameters by struct
        Ocp::OptParam opt_params = transcription.get_opt_params();
        std::cout << "opt_params: ref_offset: " << opt_params.ref_offset << ", us: " << opt_params.us << "\n\n";
    };

    /* Solve with Multiple Shooting transcription */
    if (true)
    {
        const int N = 20;
        using Transcription = laopt_tools::MultipleShooting<Ocp, N>;

        /* Define specific Tape and laOPT problem types for the resulting NLP */
        using Tape = laopt::TapeInfo<Transcription>;
        using OptProblem = laopt::Problem<Transcription>;

        /* Construct transcription for OCP, optionally generate/store tape for that combination */
        Transcription transcription(ocp);
        Tape tape = laopt::generate_tape(transcription, laopt::generate_sparsity(transcription));

        /* Construct laOPT problem for transcribed OCP using according tape */
        OptProblem opt_problem(transcription, tape); // Tape is optional here and could also be generated internally

#ifdef LAOPT_WITH_IPOPT
        {
            std::cout << "Multiple Shooting - Ipopt\n";

            using Solver = laopt::IpoptWrapper<OptProblem>;
            Solver solver(opt_problem);

            solve_and_print(transcription, opt_problem, solver);
        }
#endif

#ifdef LAOPT_WITH_PIQP
        {
            std::cout << "Multiple Shooting - SQP\n";

            using Solver = laopt::SQPSolver<OptProblem, laopt::PIQPSolver<OptProblem::scalar_t>>;
            Solver solver(opt_problem);
            solver.settings().verbose = true;
            solver.settings().hessian_approximation = laopt::hessian_approximation_t::EXACT_NO_CONSTRAINTS;

            solve_and_print(transcription, opt_problem, solver);
        }
#endif
    }

    /* Solve with Radau Collocation transcription */
    if (true)
    {
        const int D_poly = 4;
        const int N_segs = 3;
        using Transcription = laopt_tools::RadauCollocation<Ocp, N_segs, D_poly>;

        /* Define specific Tape and laOPT problem types for the resulting NLP */
        using Tape = laopt::TapeInfo<Transcription>;
        using OptProblem = laopt::Problem<Transcription>;

        /* Construct transcription for OCP, optionally generate/store tape for that combination */
        Transcription transcription(ocp);
        Tape tape = laopt::generate_tape(transcription, laopt::generate_sparsity(transcription));

        /* Construct laOPT problem for transcribed OCP using according tape */
        OptProblem opt_problem(transcription, tape); // Tape is optional here and could also be generated internally

#ifdef LAOPT_WITH_IPOPT
        {
            std::cout << "Radau Collocation - Ipopt\n";

            using Solver = laopt::IpoptWrapper<OptProblem>;
            Solver solver(opt_problem);

            solve_and_print(transcription, opt_problem, solver);
        }
#endif

#ifdef LAOPT_WITH_PIQP
        {
            std::cout << "Radau Collocation - SQP\n";

            using Solver = laopt::SQPSolver<OptProblem, laopt::PIQPSolver<OptProblem::scalar_t>>;
            Solver solver(opt_problem);
            solver.settings().verbose = true;
            solver.settings().hessian_approximation = laopt::hessian_approximation_t::EXACT_NO_CONSTRAINTS;

            solve_and_print(transcription, opt_problem, solver);
        }
#endif
    }

    return 0;
}
