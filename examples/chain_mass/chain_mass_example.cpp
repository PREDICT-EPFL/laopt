#include <iostream>
#include <chrono>

#include "laopt/laopt.hpp"

#include "chain_mass_ocp.hpp"
#include "laopt/tools/multiple_shooting.hpp"
#include "laopt/tools/radau_collocation.hpp"
#ifdef LAOPT_WITH_IPOPT
#include "laopt/solvers/ipopt_interface.hpp"
#endif
#include "laopt/solvers/sqp_solver.hpp"
#ifdef LAOPT_WITH_PIQP
#include "laopt/solvers/piqp_interface.hpp"
//#include "laopt/solvers/hpipm_interface.hpp"
#endif

#include "examples_helper.hpp"

int main()
{
    using namespace std::chrono;

    /* Choose OCP and Transcription */
    using Ocp = ChainMassOcp<5>;

    /* Construct OCP and set OCP-specific properties */
    std::shared_ptr<Ocp> ocp = std::make_shared<Ocp>();

    ocp->set_tf(8.0);

    ocp->u_ub = Ocp::Input::Constant(1);
    ocp->u_lb = Ocp::Input::Constant(-1);

    Ocp::State x0 = Ocp::State::Zero();
    for (int i = 0; i < Ocp::M - 1; i++)
    {
        x0(3 * i) = 7.0 * (i + 1) / (Ocp::M - 1);
    }
    ocp->set_x0(x0);

    /* Resampling test parameters */
    const double Ts_max = 0.02;
    const double t_test = 0.166;

    auto solve_and_print = [&](auto& transcription, auto& opt_problem, auto& solver)
    {
        /* Set initial guess for state trajectory */
        transcription->set_X_guess(x0);
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

    auto solve_and_avg = [&](auto& transcription, auto& opt_problem, auto& solver)
    {
        int runs = 1000;
        long duration_us_total = 0;
        for (int i = 0; i < runs; i++) {
            transcription->set_X_guess(x0);
            const high_resolution_clock::time_point t_start = high_resolution_clock::now();
            solver.solve();
            const high_resolution_clock::time_point t_end = high_resolution_clock::now();
            const long duration_us = duration_cast<microseconds>(t_end - t_start).count();
            duration_us_total += duration_us;
            std::cout << "comp_time: " << (double) duration_us / 1e3 << " ms" << std::endl;
        }
        std::cout << "comp_time_avg: " << (double) duration_us_total / runs / 1e3 << " ms" << std::endl;
    };

    /* Solve with Multiple Shooting transcription */
    if (true)
    {
        const int N = 40;
//        const int N = 10;
        using Transcription = laopt_tools::MultipleShooting<Ocp, N, laopt::ERK4, laopt::EIGEN_ALL>;

        /* Define specific Tape and laOPT problem types for the resulting NLP */
        using Tape = laopt::TapeInfo<Transcription>;
//        using BSTape = laopt::BSTapeInfo<Transcription>;
        using OptProblem = laopt::Problem<Transcription>;
//        using BSOptProblem = laopt::BSProblem<Transcription>;

        /* Construct transcription for OCP, optionally generate/store tape for that combination */
        std::shared_ptr<Transcription> transcription = std::make_shared<Transcription>(ocp);
        Tape tape = laopt::generate_tape(transcription, laopt::generate_sparsity(transcription));
//        BSTape tape = laopt::generate_bs_tape(transcription, laopt::generate_sparsity(transcription));

        /* Construct laOPT problem for transcribed OCP using according tape */
        std::shared_ptr<OptProblem> opt_problem = std::make_shared<OptProblem>(transcription, tape); // Tape is optional here and could also be generated internally
//        std::shared_ptr<BSOptProblem> opt_problem = std::make_shared<BSOptProblem>(transcription, tape); // Tape is optional here and could also be generated internally

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
//            using Solver = laopt::SQPSolver<BSOptProblem, laopt::PIQPSolver<BSOptProblem::scalar_t>>;
//            using Solver = laopt::SQPSolver<BSOptProblem, laopt::HPIPMSolver>;
            Solver solver(opt_problem);
            solver.settings().verbose = false;
            solver.qp_solver().settings().verbose = false;
            solver.qp_solver().settings().elastic_mode = false;
            solver.settings().max_iter = 1;
            solver.settings().hessian_approximation = laopt::hessian_approximation_t::GAUSS_NEWTON;
//            solver.settings().globalization_strategy = laopt::globalization_t::LINE_SEARCH_FILTER;
            solver.settings().globalization_strategy = laopt::globalization_t::FULL_STEP;

            solve_and_avg(transcription, opt_problem, solver);
        }
#endif
    }

    return 0;
}
