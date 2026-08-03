#include <iostream>
#include <chrono>

#include "laopt/laopt.hpp"

#include "double_integrator_ocp.hpp"
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
    /** Select variants to build **/
    constexpr bool with_eigen_AD = true;
    constexpr bool with_casadi_AD = true;
    constexpr bool with_multiple_shooting = true;
    constexpr bool with_radau_collocation = false;
    constexpr bool with_ipopt = true;
    constexpr bool with_sqp_piqp = false;

    using namespace std::chrono;

    /* Choose OCP and Transcription */
    using Ocp = DoubleIntegratorOcp;

    /* Construct OCP and set OCP-specific properties */
    std::shared_ptr<Ocp> ocp = std::make_shared<Ocp>();

    ocp->set_tf(1.2);

    ocp->u_ub << 10;
    ocp->u_lb << -3;

    ocp->x_ref << 1, 0;

    ocp->set_x0({0.1, 0.2});               // for demonstration, last setting counts
    ocp->x0_lb = ocp->x0_ub = {0.1, 0.2};  // for demonstration, last setting counts
    ocp->x0_ub << 0.1, 0.2;                // for demonstration, last setting counts
    ocp->x0_lb << -0.1, -0.2;

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
    };

    if (with_eigen_AD)
    {
        std::cout << "---------------------------\n"
                     "Eigen autodiff (default)\n"
                     "---------------------------\n";

        /* Solve with Multiple Shooting transcription */
        if (with_multiple_shooting)
        {
            const int N = 20;
            using Transcription = laopt_tools::MultipleShooting<Ocp, N>;

            /* Define specific Tape, laOPT, and IPOPT problem types for the resulting NLP */
            using Tape = laopt::TapeInfo<Transcription>;
            using OptProblem = laopt::Problem<Transcription>;

            /* Construct transcription for OCP, optionally generate/store tape for that combination */
            std::shared_ptr<Transcription> transcription = std::make_shared<Transcription>(ocp);
            Tape tape = laopt::generate_tape(transcription, laopt::generate_sparsity(transcription));

            /* Construct laOPT problem for transcribed OCP using according tape (optional) */
            std::shared_ptr<OptProblem> opt_problem = std::make_shared<OptProblem>(transcription, tape);

            if (with_ipopt)
            {
#ifdef LAOPT_WITH_IPOPT
                std::cout << "Multiple Shooting - Ipopt\n";

                using Solver = laopt::IpoptSolver<OptProblem>;
                Solver solver(opt_problem);

                solve_and_print(transcription, opt_problem, solver);
#endif
            }
            if (with_sqp_piqp)
            {
#ifdef LAOPT_WITH_PIQP
                std::cout << "Multiple Shooting - SQP (PIQP)\n";

                using Solver = laopt::SQPSolver<OptProblem, laopt::PIQPSolver<OptProblem::scalar_t>>;
                Solver solver(opt_problem);
                solver.settings().verbose = true;
//                solver.settings().hessian_approximation = laopt::hessian_approximation_t::GAUSS_NEWTON;

                solve_and_print(transcription, opt_problem, solver);
#endif
            }
        }

        /* Solve with Radau Collocation transcription */
        if (with_radau_collocation)
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

            /* Construct laOPT problem for transcribed OCP using according tape (optional) */
            std::shared_ptr<OptProblem> opt_problem = std::make_shared<OptProblem>(transcription, tape);

            if (with_ipopt)
            {
#ifdef LAOPT_WITH_IPOPT
                std::cout << "Multiple Shooting - Ipopt\n";

                using Solver = laopt::IpoptSolver<OptProblem>;
                Solver solver(opt_problem);

                solve_and_print(transcription, opt_problem, solver);
#endif
            }
            if (with_sqp_piqp)
            {
#ifdef LAOPT_WITH_PIQP
                std::cout << "Multiple Shooting - SQP (PIQP)\n";

                using Solver = laopt::SQPSolver<OptProblem, laopt::PIQPSolver<OptProblem::scalar_t>>;
                Solver solver(opt_problem);
                solver.settings().verbose = true;
//                solver.settings().hessian_approximation = laopt::hessian_approximation_t::GAUSS_NEWTON;

                solve_and_print(transcription, opt_problem, solver);
#endif
            }
        }
    }

    if (with_casadi_AD)
    {
#ifdef LAOPT_WITH_CASADI
        std::cout << "---------------------------\n"
                     "CasADi autodiff\n"
                     "---------------------------\n";

        /* Solve with Multiple Shooting transcription */
        if (with_multiple_shooting)
        {
            const int N = 20;
            using Transcription = laopt_tools::MultipleShooting<Ocp, N, laopt::ERK4,
                    laopt::CASADI_ALL | laopt::CASADI_NO_JIT>;

            /* Define specific Tape, laOPT, and IPOPT problem types for the resulting NLP */
            using Tape = laopt::TapeInfo<Transcription>;
            using OptProblem = laopt::Problem<Transcription>;

            /* Construct transcription for OCP, optionally generate/store tape for that combination */
            std::shared_ptr<Transcription> transcription = std::make_shared<Transcription>(ocp);
            Tape tape = laopt::generate_tape(transcription, laopt::generate_sparsity(transcription));

            /* Construct laOPT problem for transcribed OCP using according tape (optional) */
            std::shared_ptr<OptProblem> opt_problem = std::make_shared<OptProblem>(transcription,
                                                                                   tape); // Tape is optional here and could also be generated internally

            if (with_ipopt)
            {
#ifdef LAOPT_WITH_IPOPT
                std::cout << "Multiple Shooting - Ipopt\n";

                using Solver = laopt::IpoptSolver<OptProblem>;
                Solver solver(opt_problem);

                solve_and_print(transcription, opt_problem, solver);
#endif
            }
            if (with_sqp_piqp)
            {
#ifdef LAOPT_WITH_PIQP
                std::cout << "Multiple Shooting - SQP (PIQP)\n";

                using Solver = laopt::SQPSolver<OptProblem, laopt::PIQPSolver<OptProblem::scalar_t>>;
                Solver solver(opt_problem);
                solver.settings().verbose = true;
//                solver.settings().hessian_approximation = laopt::hessian_approximation_t::GAUSS_NEWTON;

                solve_and_print(transcription, opt_problem, solver);
#endif
            }
        }

        /* Solve with Radau Collocation transcription */
        if (with_radau_collocation)
        {
            const int D_poly = 4;
            const int N_segs = 3;
            using Transcription = laopt_tools::RadauCollocation<Ocp, N_segs, D_poly,
                    laopt::CASADI_ALL | laopt::CASADI_NO_JIT>;

            /* Define specific Tape and laOPT problem types for the resulting NLP */
            using Tape = laopt::TapeInfo<Transcription>;
            using OptProblem = laopt::Problem<Transcription>;

            /* Construct transcription for OCP, optionally generate/store tape for that combination */
            std::shared_ptr<Transcription> transcription = std::make_shared<Transcription>(ocp);
            Tape tape = laopt::generate_tape(transcription, laopt::generate_sparsity(transcription));

            /* Construct laOPT problem for transcribed OCP using according tape (optional) */
            std::shared_ptr<OptProblem> opt_problem = std::make_shared<OptProblem>(transcription,
                                                                                   tape); // Tape is optional here and could also be generated internally

            if (with_ipopt)
            {
#ifdef LAOPT_WITH_IPOPT
                std::cout << "Multiple Shooting - Ipopt\n";

                using Solver = laopt::IpoptSolver<OptProblem>;
                Solver solver(opt_problem);

                solve_and_print(transcription, opt_problem, solver);
#endif
            }
            if (with_sqp_piqp)
            {
#ifdef LAOPT_WITH_PIQP
                std::cout << "Multiple Shooting - SQP (PIQP)\n";

                using Solver = laopt::SQPSolver<OptProblem, laopt::PIQPSolver<OptProblem::scalar_t>>;
                Solver solver(opt_problem);
                solver.settings().verbose = true;
//                solver.settings().hessian_approximation = laopt::hessian_approximation_t::GAUSS_NEWTON;

                solve_and_print(transcription, opt_problem, solver);
#endif
            }
        }
#endif
    }

    return 0;
}
