#ifndef LAOPT_EXAMPLES_HELPER_HPP
#define LAOPT_EXAMPLES_HELPER_HPP

#include <iostream>

template<typename Transcription, typename OptProblem, typename Solver, typename Duration>
void print_solution(const Transcription &transcription, const OptProblem &opt_problem, const Solver &solver,
                    const Duration &duration_us, const Duration &duration2_us)
{
    /* Print out the solution */
    std::cout << "\n\n";
    std::cout << std::setprecision(4) << std::defaultfloat;

    const Eigen::VectorXd T_opt = transcription.get_T_opt();
    const Eigen::MatrixXd X_opt = transcription.get_X_opt();
    const Eigen::MatrixXd U_opt = transcription.get_U_opt();
    Eigen::VectorX<typename Solver::Scalar> sol_primal = solver.sol_primal();
    double obj_eval = -1;// opt_problem.eval_objective(laopt::Eval(), sol_primal);

    std::cout << "Comp. time (warm): " << duration_us/1e3 << " (" << duration2_us/1e3 << ") ms, tf = " << T_opt(T_opt.size() - 1) << " s, obj = " << obj_eval << "\n";
    std::cout << "MATLAB-copyable output:\n";
    std::cout << "T_opt = [\n" << T_opt.transpose() << "];\n";
    std::cout << "X_opt = [\n" << X_opt << "];\n";
    std::cout << "U_opt = [\n" << U_opt << "];\n";
    std::cout << "obj = " << obj_eval << ";\n";
    std::cout << "comp_time = " << duration_us / 1e6 << ";\n";
    std::cout << "\n";
}

#endif //LAOPT_EXAMPLES_HELPER_HPP