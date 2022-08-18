#ifndef LAOPT_IPOPT_WRAPPER_HPP
#define LAOPT_IPOPT_WRAPPER_HPP

#include "ipopt_interface.hpp"
#include "laopt/problem.hpp"

namespace laopt {

template<typename OptProblem>
class IpoptWrapper
{
public:
    using IpoptProblem = laopt::Solver_IPOpt<OptProblem>;

    explicit IpoptWrapper(OptProblem &opt_problem)
    {
        using namespace Ipopt;

        /* Create IPOPT problem and link decision variables to OptProblem */
        problem = new IpoptProblem(opt_problem);
        opt_problem.set_decision_variable(problem->init_primal);

        /* Create IPOPT application, setup, and initialize */
        application = IpoptApplicationFactory();
        // ipopt_app->Options()->SetStringValue("hessian_approximation", "limited-memory");
        application->Options()->SetIntegerValue("print_level", 5);
        ApplicationReturnStatus ipopt_status = application->Initialize();
        if (ipopt_status != Solve_Succeeded) { std::cout << "\n\n*** Error during initialization!\n"; }
    }

    bool solve()
    {
        using namespace Ipopt;

        ApplicationReturnStatus ipopt_status = application->OptimizeTNLP(problem);
        if (ipopt_status != Ipopt::Solve_Succeeded) { std::cout << "\n\n*** Error during solution!\n"; }
        return ipopt_status;
    }

    IpoptProblem &operator()() { return *problem; }

protected:
    Ipopt::SmartPtr<IpoptProblem> problem;
    Ipopt::SmartPtr<Ipopt::IpoptApplication> application;
};

}

#endif //LAOPT_IPOPT_WRAPPER_HPP
