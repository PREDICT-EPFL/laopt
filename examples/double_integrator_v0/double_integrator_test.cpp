#include <iostream>
#include <iomanip>

#include "double_integrator_ocp.hpp"
#include "multiple_shooting_transcription.hpp"
#include "ipopt_interface.hpp"

int main()
{
    using OCP = DoubleIntegratorOCP;
    OCP ocp;

    static const int N = 30;
    using Transcription = MultipleShootingTranscription<OCP, N>;
    Transcription transcription(ocp);

    lampc::TapeInfo<Transcription> tape = lampc::generate_tape(transcription, lampc::generate_sparsity(transcription));
    using Problem = lampc::Problem<Transcription>;
    Problem prob(transcription, tape);

    // Create the IPOpt solver
    SmartPtr<lampc::Solver_IPOpt<Problem>> mynlp = new lampc::Solver_IPOpt<Problem>(prob);
    SmartPtr<IpoptApplication> app = IpoptApplicationFactory();

    // app->Options()->SetStringValue("hessian_approximation", "limited-memory");
    app->Options()->SetIntegerValue("print_level", 5);

    // Initialize the IpoptApplication and process the options
    ApplicationReturnStatus status;
    status = app->Initialize();
    if( status != Solve_Succeeded )
    {
        std::cout << std::endl << std::endl << "*** Error during initialization!" << std::endl;
    }

    // Set the initial primal variable
    prob.set_decision_variable(mynlp->init_primal);
    ocp.x0 << 1, 1;

    // Solve the problem
    status = app->OptimizeTNLP(mynlp);

    std::cout << std::endl << std::endl << std::endl << std::endl;

    // Print out the solution
    std::cout << std::setprecision(2) << std::defaultfloat;

    Eigen::Matrix<OCP::scalar_t, 2, N> X;
    Eigen::Matrix<OCP::scalar_t, 1, N - 1> U;
    int i = 0; for(auto& x: transcription.X) X.col(i++) << x;
    i = 0; for(auto& u: transcription.U) U.col(i++) << u;

    std::cout << "X = \n" << X << std::endl;
    std::cout << "U = \n" << U << std::endl;
    std::cout << "obj = " << prob.eval_objective(lampc::Eval(), mynlp->sol_primal) << std::endl;

    return 0;
}
