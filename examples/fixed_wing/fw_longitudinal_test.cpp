#include <iostream>
#include <chrono>

#include "laopt/laopt.hpp"

#include "LonOcpEigen.hpp"
#include "laopt/tools/multiple_shooting.hpp"
#include "laopt/tools/radau_collocation.hpp"
#include "laopt/ipopt_interface/ipopt_wrapper.hpp"
#include "laopt/solvers/sqp_solver.hpp"
#include "laopt/solvers/osqp_interface.hpp"

#include "examples_helper.hpp"

int main()
{
    using namespace std::chrono;

    /* Choose OCP and Transcription */
    using Ocp = lon_ocp::LonFlightOCP;

    /* Construct and setup OCP */
    Ocp ocp;
    ocp.model.set_state_representation(flight_model::LongitudinalFlightPath);
    ocp.model.load_params_from_yaml("eg4_xflr-Pvw-YR.yaml");

    ocp.objectives[lon_ocp::TrackAngle] = true;
//    ocp.objectives[lon_ocp::TrackVa] = true;
    ocp.objectives[lon_ocp::MinimizeControl] = true;

    ocp.set_tf(1.5);

    ocp.pitch_ref = -20.0 * M_PI / 180.0;
    ocp.Va_ref = 11.0;

    ocp.mayer_multiplier = 10;
    ocp.W_pitch_err = 10;
    ocp.W_Va_err = 1;
    ocp.R.diagonal() << 1, 0.1;

    ocp.u_ub << 0.8 * ocp.model.u_physical_ubound(0), 0.001;
    ocp.u_lb << 0.8 * ocp.model.u_physical_lbound(0), 0;

    /* Set initial state */
    ocp.set_x0(ocp.model.get_default_initial_state());

    /* Resampling test parameters */
    const double Ts_max = 0.02;
    const double t_test = 0.166;

    auto solve_and_print = [&](auto& transcription, auto& opt_problem, auto& solver)
    {
        /* Set initial guess for state trajectory */
        transcription.set_X_guess(ocp.model.get_default_initial_state());
        std::cout << "X_guess = \n" << transcription.get_X_opt() << "\n";

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

        {
            std::cout << "Multiple Shooting - Ipopt\n";

            using Solver = laopt::IpoptWrapper<OptProblem>;
            Solver solver(opt_problem);

            solve_and_print(transcription, opt_problem, solver);
        }

        {
            std::cout << "Multiple Shooting - SQP\n";

            using Solver = laopt::SQPSolver<OptProblem, laopt::OSQPSolver<OptProblem::scalar_t>>;
            Solver solver(opt_problem);
            solver.settings().verbose = true;
            solver.settings().hessian_approximation = laopt::hessian_approximation_t::EXACT_NO_CONSTRAINTS;

            solve_and_print(transcription, opt_problem, solver);
        }
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

        {
            std::cout << "Radau Collocation - Ipopt\n";

            using Solver = laopt::IpoptWrapper<OptProblem>;
            Solver solver(opt_problem);

            solve_and_print(transcription, opt_problem, solver);
        }

        {
            std::cout << "Radau Collocation - SQP\n";

            using Solver = laopt::SQPSolver<OptProblem, laopt::OSQPSolver<OptProblem::scalar_t>>;
            Solver solver(opt_problem);
            solver.settings().verbose = true;
            solver.settings().hessian_approximation = laopt::hessian_approximation_t::EXACT_NO_CONSTRAINTS;

            solve_and_print(transcription, opt_problem, solver);
        }
    }
    return 0;
}