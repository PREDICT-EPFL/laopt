#include <iostream>
#include <iomanip>
#include <chrono>

#include "laopt/laopt.hpp"

#include "LonOcpEigen.hpp"
#include "laopt/tools/MultipleShooting.hpp"
#include "laopt/tools/RadauCollocation.hpp"
#include "laopt/ipopt_interface/ipopt_wrapper.hpp"

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

    /* Solve with Multiple Shooting transcription */
    if (true)
    {
        std::cout << "Multiple Shooting\n";
        const int N = 20;
        using Transcription = laopt_tools::MultipleShooting<Ocp, N>;

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
    }

    /* Solve with Radau Collocation transcription */
    if (true)
    {
        std::cout << "Radau Collocation\n";
        const int D_poly = 4;
        const int N_segs = 3;
        using Transcription = laopt_tools::RadauCollocation<Ocp, N_segs, D_poly>;

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
    }
    return 0;
}