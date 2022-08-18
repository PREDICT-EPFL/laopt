#ifndef LAOPT_IPOPT_WRAPPER_HPP
#define LAOPT_IPOPT_WRAPPER_HPP

#include "ipopt_interface.hpp"
#include "laopt/problem.hpp"

namespace laopt {

template<typename OptProblem>
class IpoptWrapper : public Solver_IPOpt<OptProblem>
{
public:
    explicit IpoptWrapper(OptProblem &opt_problem) : Solver_IPOpt<OptProblem>(opt_problem)
    {
        using namespace Ipopt;

        /* Create IPOPT problem and link decision variables to OptProblem */
        opt_problem.set_decision_variable(this->init_primal);

        /* Create IPOPT application, setup, and initialize */
        ipopt_app = IpoptApplicationFactory();
        // ipopt_app->Options()->SetStringValue("hessian_approximation", "limited-memory");
        ipopt_app->Options()->SetIntegerValue("print_level", 5);
        ApplicationReturnStatus ipopt_status = ipopt_app->Initialize();
        if (ipopt_status != Solve_Succeeded) { std::cout << "\n\n*** Error during initialization!\n"; }
    }

    bool solve()
    {
        using namespace Ipopt;

        ApplicationReturnStatus ipopt_status = ipopt_app->OptimizeTNLP(this);
        if (ipopt_status != Ipopt::Solve_Succeeded) { std::cout << "\n\n*** Error during solution!\n"; }
        return ipopt_status;
    }

protected:
    Ipopt::SmartPtr<Ipopt::IpoptApplication> ipopt_app;
};

}

#endif //LAOPT_IPOPT_WRAPPER_HPP
