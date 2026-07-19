#include <iostream>
#include <fstream>
#include <chrono>
#include <filesystem>

#include "laopt/laopt.hpp"

#include "rocket_ocp.hpp"
#include "laopt/tools/multiple_shooting.hpp"

#include "laopt/solvers/ipopt_interface.hpp"

#include "laopt/solvers/sqp_solver.hpp"
#ifdef LAOPT_WITH_QPSWIFT
#include "laopt/solvers/qpswift_interface.hpp"
#endif
#ifdef LAOPT_WITH_PIQP
#include "laopt/solvers/piqp_interface.hpp"
#endif
#ifdef LAOPT_WITH_OSQP
#include "laopt/solvers/osqp_interface.hpp"
#endif
#ifdef LAOPT_WITH_PROXQP
#include "laopt/solvers/proxqp_interface.hpp"
#endif

#include "examples_helper.hpp"

int main()
{
    using namespace std::chrono;

    /* Choose OCP and Transcription */
    using Ocp = RocketOcp;

    /* Construct OCP and set OCP-specific properties */
    std::shared_ptr<Ocp> ocp = std::make_shared<Ocp>();

    ocp->set_tf(3.0);

    Ocp::State x0;
    x0.setZero();
    ocp->set_x0(x0);

    ocp->ref << 2, 2, 2, Ocp::deg2rad(135);

    /* Resampling test parameters */
    const double Ts_max = 0.02;
    const double t_test = 0.166;

    // const int N = 60;
    const int N = 10;
    using Transcription = laopt_tools::MultipleShooting<Ocp, N, laopt::ERK4, laopt::EIGEN_ALL>;

    /* Define specific Tape and laOPT problem types for the resulting NLP */
    using Tape = laopt::TapeInfo<Transcription>;
    using OptProblem = laopt::Problem<Transcription>;

    /* Construct transcription for OCP, optionally generate/store tape for that combination */
    std::shared_ptr<Transcription> transcription = std::make_shared<Transcription>(ocp);
    Tape tape = laopt::generate_tape(transcription, laopt::generate_sparsity(transcription));
    /* Construct laOPT problem for transcribed OCP using according tape */
    std::shared_ptr<OptProblem> opt_problem = std::make_shared<OptProblem>(transcription, tape); // Tape is optional here and could also be generated internally

    using IpoptSolver = laopt::IpoptSolver<OptProblem>;
    IpoptSolver ipopt_solver(opt_problem);
    ipopt_solver.set_print_level(5);

    steady_clock::time_point t_start = steady_clock::now();
    ipopt_solver.solve();
    steady_clock::time_point t_end = steady_clock::now();
    long duration_us = duration_cast<microseconds>(t_end - t_start).count();

    /* Print out the solution */
    print_solution(transcription, opt_problem, duration_us, duration_us);
    print_sampled_solution(transcription, Ts_max, t_test);

    using SQPSolver = laopt::SQPSolver<OptProblem, laopt::PIQPSolver<OptProblem::scalar_t>>;
    // using SQPSolver = laopt::SQPSolver<OptProblem, laopt::QPSwiftSolver<OptProblem::scalar_t>>;
    // using SQPSolver = laopt::SQPSolver<OptProblem, laopt::OSQPSolver<OptProblem::scalar_t>>;
    // using SQPSolver = laopt::SQPSolver<OptProblem, laopt::ProxQPSolver<OptProblem::scalar_t>>;
    SQPSolver sqp_solver(opt_problem);
    sqp_solver.settings().verbose = true;
    // sqp_solver.settings().globalization_strategy = laopt::globalization_t::LINE_SEARCH_L1;
    sqp_solver.settings().hessian_approximation = laopt::hessian_approximation_t::GAUSS_NEWTON;
    sqp_solver.settings().max_watchdog_steps = 0;
    // sqp_solver.settings().max_iter = 1;

    // sqp_solver.set_initial_primal(ipopt_solver.primal());
    // sqp_solver.set_initial_dual(ipopt_solver.dual());
    // sqp_solver.set_initial_dual_bounds(ipopt_solver.dual_bounds());

    auto integrate = [&](double h, auto x0_, auto u0_)
    {
        Ocp::Param p; p.setZero();
        Eigen::Vector<double, 1> t0, tf; t0(0) = 0; tf(0) = 0;// Unused in dynamics
        double tau{0}; // Unused in dynamics
        Ocp::State k1 = ocp->dynamics_impl(x0_, u0_, p, t0, tf, tau);
        Ocp::State k2 = ocp->dynamics_impl(x0_ + h * 0.5 * k1, u0_, p, t0, tf, tau);
        Ocp::State k3 = ocp->dynamics_impl(x0_ + h * 0.5 * k2, u0_, p, t0, tf, tau);
        Ocp::State k4 = ocp->dynamics_impl(x0_ + h * k3, u0_, p, t0, tf, tau);
        return static_cast<Ocp::State>(x0_ + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4));
    };

    // std::ofstream csv;
    // csv.open("rocket_sqp.csv");
    // csv << "solver,step,sqp_iter,qp_iter,time\n";
    // csv.open("rocket_sqp.csv", std::ios_base::app);

    double Ts = 1.0 / 20;

    for (int i = 0; i < 100; i++)
    {
        // std::__fs::filesystem::path tmp_folder = "/tmp/SQP_tmp";
        // std::__fs::filesystem::create_directory(tmp_folder);

        t_start = steady_clock::now();
        sqp_solver.solve();
        t_end = steady_clock::now();
        duration_us = duration_cast<microseconds>(t_end - t_start).count();
        std::cout << "Comp. time: " << duration_us / 1e3 << " ms\n";
        // csv << "piqp," << i << "," << sqp_solver.info().iter << "," << sqp_solver.info().qp_iter << "," << duration_us / 1e6 << "\n";
        // csv << "qpswift," << i << "," << sqp_solver.info().iter << "," << sqp_solver.info().qp_iter << "," << duration_us / 1e6 << "\n";
        // csv << "osqp," << i << "," << sqp_solver.info().iter << "," << sqp_solver.info().qp_iter << "," << duration_us / 1e6 << "\n";
        // csv << "proxqp," << i << "," << sqp_solver.info().iter << "," << sqp_solver.info().qp_iter << "," << duration_us / 1e6 << "\n";

        // std::__fs::filesystem::path to_folder = "/tmp/sqp_rocket/" + std::to_string(i);
        // std::__fs::filesystem::rename(tmp_folder, to_folder);

        x0 = integrate(Ts, x0, transcription->get_u_at(0));
        std::cout << "x: " << x0.transpose() << std::endl;

        Eigen::VectorX<Ocp::Scalar> primal = sqp_solver.primal();
        primal(Eigen::seq(0, Eigen::indexing::last - Ocp::NX - Ocp::NU)) = primal(Eigen::seq(Ocp::NX + Ocp::NU, Eigen::indexing::last));
        Eigen::VectorX<Ocp::Scalar> dual = sqp_solver.dual();
        dual(Eigen::seq(0, Eigen::indexing::last - 1)) = dual(Eigen::seq(1, Eigen::indexing::last));
        Eigen::VectorX<Ocp::Scalar> dual_bounds = sqp_solver.dual_bounds();
        dual_bounds(Eigen::seq(0, Eigen::indexing::last - 2)) = dual_bounds(Eigen::seq(2, Eigen::indexing::last));

        sqp_solver.set_initial_primal(primal);
        sqp_solver.set_initial_dual(dual);
        sqp_solver.set_initial_dual_bounds(dual_bounds);

        ocp->set_x0(x0);
    }

    return 0;
}
