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
        ipopt_problem = new IpoptProblem(opt_problem);
        opt_problem.set_decision_variable(ipopt_problem->init_primal);

        /* Create IPOPT application, setup, and initialize */
        ipopt_application = IpoptApplicationFactory();
        // ipoptApp->Options()->SetStringValue("hessian_approximation", "limited-memory");
        ipopt_application->Options()->SetIntegerValue("print_level", 5);
        ApplicationReturnStatus ipopt_status = ipopt_application->Initialize();
        if (ipopt_status != Solve_Succeeded) { std::cout << "\n*** IpoptWrapper: Error during initialization!\n\n"; }
    }

    Ipopt::ApplicationReturnStatus solve() const
    {
        using namespace Ipopt;

        ApplicationReturnStatus ipopt_status = ipopt_application->OptimizeTNLP(ipopt_problem);
        if (ipopt_status != Ipopt::Solve_Succeeded) { std::cout << "\n*** IpoptWrapper: Error during solution!\n\n"; }
        return ipopt_status;
    }

    using Scalar = typename OptProblem::scalar_t;
    /* Setters */
    void set_initial_primal(const Eigen::VectorX<Scalar> &init_primal) { ipopt_problem->init_primal = init_primal; }
    void set_initial_dual(const Eigen::VectorX<Scalar> &init_dual) { ipopt_problem->init_dual = init_dual; }

    /* Getters */
    Eigen::VectorX<Scalar> &sol_primal() const { return ipopt_problem->sol_primal; }
    Eigen::VectorX<Scalar> &sol_dual() const { return ipopt_problem->sol_dual; }

protected:
    Ipopt::SmartPtr<IpoptProblem> ipopt_problem;
    Ipopt::SmartPtr<Ipopt::IpoptApplication> ipopt_application;
};

}

#endif //LAOPT_IPOPT_WRAPPER_HPP