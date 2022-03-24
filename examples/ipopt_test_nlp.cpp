/**
 * Use polyMPC and LACompiler to solve a simple QP
 */

// #define SEG(len,offset) template segment<len>(offset)

#include <iostream>
#include "lampc.hpp"
#include "ipopt_test_nlp_functions.hpp"
#include "ipopt_test.compiled.hpp"

#include <chrono>

#include "ipopt_interface.hpp"

int main()
{
    using param_t = ipopt_nlp_test::param_t;
    using myNLP = NLP_Ipopt<ipopt_nlp_test>;
    param_t param;

    Ipopt::SmartPtr<myNLP> mynlp = new myNLP(param);
    Ipopt::SmartPtr<Ipopt::IpoptApplication> app = new Ipopt::IpoptApplication();

    app->Options()->SetNumericValue("tol", 1e-7);
    // app->Options()->SetStringValue("mu_strategy", "adaptive");
    app->Options()->SetStringValue("output_file", "ipopt.out");
    // app->Options()->SetStringValue("hessian_approximation", "limited-memory");

    Ipopt::ApplicationReturnStatus status;
    status = app->Initialize();
    if( status != Ipopt::Solve_Succeeded )
    {
        std::cout << std::endl << std::endl << "*** Error during initialization!" << std::endl;
        return -1;
    }

    const std::size_t NUM_EXP = 1;
    auto start = std::chrono::steady_clock::now();
    for(int i = 0; i < NUM_EXP; ++i)
    {
        mynlp->x0 << 1,5,5,1;
        status = app->OptimizeTNLP(mynlp);
    }
    auto end = std::chrono::steady_clock::now();

    std::cout << "\n\n\n\n===================== IPOPT SOLUTION =====================\n";
    std::cout << "IPOPT time: "
        << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
        << " ms" << std::endl;
    std::cout << "x = " << "[" <<   
        ipopt_nlp_test::x1(mynlp->sol.primal) << ", " <<
        ipopt_nlp_test::x2(mynlp->sol.primal) << ", " <<
        ipopt_nlp_test::x3(mynlp->sol.primal) << ", " <<
        ipopt_nlp_test::x4(mynlp->sol.primal) << "]\n";

    return 1;
}
