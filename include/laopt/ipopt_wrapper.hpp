#ifndef LAOPT_IPOPT_WRAPPER_HPP
#define LAOPT_IPOPT_WRAPPER_HPP

#include "ipopt_interface.hpp"
#include "laopt/problem.hpp"

namespace laopt {

template<typename OptProblem>
class IpoptWrapper
{
    using IpoptProblem = laopt::Solver_IPOpt<OptProblem>;
public:
    explicit IpoptWrapper(OptProblem &opt_problem)
    {
        using namespace Ipopt;

        std::cout << "IpoptWrapper()\n";

        /* Create IPOPT problem and link decision variables to OptProblem */
        ipopt_problem = new IpoptProblem(opt_problem);
        opt_problem.set_decision_variable(ipopt_problem->init_primal);

        /* Create IPOPT application, setup, and initialize */
        ipopt_app = IpoptApplicationFactory();
        // ipopt_app->Options()->SetStringValue("hessian_approximation", "limited-memory");
        ipopt_app->Options()->SetIntegerValue("print_level", 5);
        ApplicationReturnStatus ipopt_status = ipopt_app->Initialize();
        if (ipopt_status != Solve_Succeeded) { std::cout << "\n\n*** Error during initialization!\n"; }
    }
    ~IpoptWrapper()
    {
        std::cout << "~IpoptWrapper()\n";
        double test = 1;
    }

    bool solve()
    {
        std::cout << "IpoptWrapper::solve()\n";

        using namespace Ipopt;

        ApplicationReturnStatus ipopt_status = ipopt_app->OptimizeTNLP(ipopt_problem);
        if (ipopt_status != Ipopt::Solve_Succeeded) { std::cout << "\n\n*** Error during solution!\n"; }
        return ipopt_status;
    }

public:
    Ipopt::SmartPtr<IpoptProblem> ipopt_problem;
    Ipopt::SmartPtr<Ipopt::IpoptApplication> ipopt_app;
};

}

#endif //LAOPT_IPOPT_WRAPPER_HPP
