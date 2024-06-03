#include <iostream>
#include <chrono>

#include "laopt/laopt.hpp"

#include "FixedWingOcpEigen.hpp"
#include "laopt/tools/multiple_shooting.hpp"
#include "laopt/tools/radau_collocation.hpp"
#ifdef LAOPT_WITH_IPOPT
#include "laopt/solvers/ipopt_interface.hpp"
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
    using Ocp = fixed_wing_ocp::FixedWingFlightOCP;

    /* Construct and setup OCP */
    std::shared_ptr<Ocp> ocp = std::make_shared<Ocp>();
    ocp->model.load_params_from_yaml("eg4_xflr-Pvw-YR.yaml");

    ocp->objectives[fixed_wing_ocp::TrackVa] = true;
    ocp->objectives[fixed_wing_ocp::MinimizeControl] = true;

    ocp->set_tf(1.5);

    ocp->Va_ref = 12.0;

    ocp->mayer_multiplier = 10;
    ocp->W_Va_err = 1;
    ocp->R.diagonal() << 0.1, 1, 1, 1;

    ocp->u_ub << 0.8 * ocp->model.u_physical_ubound;
    ocp->u_lb << 0.8 * ocp->model.u_physical_lbound;

    /* Set initial state */
    ocp->set_x0(ocp->model.get_default_initial_state());

    /* Resampling test parameters */
    const double Ts_max = 0.02;
    const double t_test = 0.166;

    auto solve_and_print = [&](auto& transcription, auto& opt_problem, auto& solver)
    {
        using Transcription = typename std::remove_reference<decltype(transcription)>::type::element_type;

        /* Set initial guess for state trajectory */
        typename Transcription::StateTrajectory X_guess;
        for (int i = 0; i < Transcription::StateTrajectory::ColsAtCompileTime; i++)
        {
            X_guess.col(i) = ocp->model.get_default_initial_state();
            X_guess(6, i) = i * 0.5 * (ocp->tf_lb + ocp->tf_ub) / Transcription::StateTrajectory::ColsAtCompileTime *
                            ocp->model.get_default_initial_state()(0);
        }
        transcription->set_X_guess(X_guess);
        std::cout << "X_guess = \n" << transcription->get_X_opt() << "\n";

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
        std::shared_ptr<Transcription> transcription = std::make_shared<Transcription>(ocp);
        Tape tape = laopt::generate_tape(transcription, laopt::generate_sparsity(transcription));

        /* Construct laOPT problem for transcribed OCP using according tape */
        std::shared_ptr<OptProblem> opt_problem = std::make_shared<OptProblem>(transcription, tape); // Tape is optional here and could also be generated internally

#ifdef LAOPT_WITH_IPOPT
        {
            std::cout << "Multiple Shooting - Ipopt\n";

            using Solver = laopt::IpoptSolver<OptProblem>;
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
            solver.settings().hessian_approximation = laopt::hessian_approximation_t::GAUSS_NEWTON;

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
        std::shared_ptr<Transcription> transcription = std::make_shared<Transcription>(ocp);
        Tape tape = laopt::generate_tape(transcription, laopt::generate_sparsity(transcription));

        /* Construct laOPT problem for transcribed OCP using according tape */
        std::shared_ptr<OptProblem> opt_problem = std::make_shared<OptProblem>(transcription, tape); // Tape is optional here and could also be generated internally

#ifdef LAOPT_WITH_IPOPT
        {
            std::cout << "Radau Collocation - Ipopt\n";

            using Solver = laopt::IpoptSolver<OptProblem>;
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
            solver.settings().hessian_approximation = laopt::hessian_approximation_t::GAUSS_NEWTON;

            solve_and_print(transcription, opt_problem, solver);
        }
#endif
    }
    return 0;
}