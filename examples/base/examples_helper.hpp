#ifndef LAOPT_EXAMPLES_HELPER_HPP
#define LAOPT_EXAMPLES_HELPER_HPP

#include <iostream>

template<typename Transcription, typename OptProblem, typename Solver, typename Duration>
void print_solution(const Transcription &transcription, OptProblem &opt_problem, const Solver &solver,
                    const Duration &duration_us, const Duration &duration2_us)
{
    /* Print out the solution */
    std::cout << "\n\n";
    std::cout << std::setprecision(4) << std::defaultfloat;

    const Eigen::VectorXd T_opt = transcription.get_T_opt();
    const Eigen::MatrixXd X_opt = transcription.get_X_opt();
    const Eigen::MatrixXd U_opt = transcription.get_U_opt();
    double obj_eval = opt_problem.eval_objective(laopt::Eval());

    std::cout << "Comp. time (warm): " << duration_us / 1e3 << " (" << duration2_us / 1e3 << ") ms, tf = "
              << T_opt(T_opt.size() - 1) << " s, obj = " << obj_eval << "\n";
    std::cout << "MATLAB-copyable output:\n";
    std::cout << "T_opt = [\n" << T_opt.transpose() << "];\n";
    std::cout << "X_opt = [\n" << X_opt << "];\n";
    std::cout << "U_opt = [\n" << U_opt << "];\n";
    std::cout << "obj = " << obj_eval << ";\n";
    std::cout << "comp_time = " << duration_us / 1e6 << ";\n";
    std::cout << "\n";
}

template<typename Transcription, typename Scalar>
void print_sampled_solution(const Transcription &transcription,
                            const Scalar &Ts_max = 0.01, const Scalar &t_test = 0.166)
{
    const auto TXn = transcription.get_TX_resampled(Ts_max);
    const auto TUn = transcription.get_TU_resampled(Ts_max);
    const auto x_test = transcription.get_x_at(t_test);
    const auto u_test = transcription.get_u_at(t_test);

    std::cout << "Resampling at " << TXn(0, 1) - TXn(0, 0) << " s  (" << Ts_max << " max)\n";
    std::cout << "MATLAB-copyable output:\n";
    std::cout << "TXn = [\n" << TXn << "];\n";
    std::cout << "TUn = [\n" << TUn << "];\n";
    std::cout << "t_test = " << t_test << ";\n";
    std::cout << "x_test = [" << x_test << "];\n";
    std::cout << "u_test = [" << u_test << "];\n";
    std::cout << "\n";
}

#endif //LAOPT_EXAMPLES_HELPER_HPP