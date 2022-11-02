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

    explicit IpoptWrapper(OptProblem &opt_problem, int print_level = 0)
    {
        using namespace Ipopt;

        /* Create IPOPT problem and link decision variables to OptProblem */
        ipopt_problem = new IpoptProblem(opt_problem);

        /* Create IPOPT application, setup, and initialize */
        ipopt_application = IpoptApplicationFactory();

        set_banner_message(true);
        set_print_level(print_level);
        set_tol(1e-3);

        ApplicationReturnStatus ipopt_status = ipopt_application->Initialize();
        if (ipopt_status != Solve_Succeeded) { std::cout << "\n*** IpoptWrapper: Error during initialization!\n\n"; }
    }
    IpoptWrapper(OptProblem &opt_problem, const Ipopt::OptionsList& options) : IpoptWrapper(opt_problem)
    {
        ipopt_application->Options() = Ipopt::SmartPtr<Ipopt::OptionsList>(new Ipopt::OptionsList(options));
    }

    void set_tol(double tol) { ipopt_application->Options()->SetNumericValue("tol", tol); }
    void set_max_iter(int max_iter) { ipopt_application->Options()->SetIntegerValue("max_iter", max_iter); }
    void set_banner_message(bool active) { ipopt_application->Options()->SetBoolValue("sb", active); }
    void set_print_level(int print_level) { ipopt_application->Options()->SetIntegerValue("print_level", print_level); }

    Ipopt::ApplicationReturnStatus solve() const
    {
        using namespace Ipopt;

        ApplicationReturnStatus ipopt_status = ipopt_application->OptimizeTNLP(ipopt_problem);
        if (ipopt_status != Ipopt::Solve_Succeeded) { std::cout << "\n*** IpoptWrapper: Error during solution!\n"
                                                                   "Error code "<< ipopt_status << '\n'; }
        return ipopt_status;
    }

    using Scalar = typename OptProblem::scalar_t;
    /* Setters */
    void set_initial_primal(const Eigen::VectorX<Scalar> &init_primal) { ipopt_problem->primal = init_primal; }
    void set_initial_dual(const Eigen::VectorX<Scalar> &init_dual) { ipopt_problem->dual = init_dual; }

    /* Getters */
    const Eigen::VectorX<Scalar> &primal() const { return ipopt_problem->primal; }
    const Eigen::VectorX<Scalar> &dual_lb() const { return ipopt_problem->dual_lb; }
    const Eigen::VectorX<Scalar> &dual_ub() const { return ipopt_problem->dual_ub; }
    const Eigen::VectorX<Scalar> &dual() const { return ipopt_problem->dual; }

protected:
    Ipopt::SmartPtr<IpoptProblem> ipopt_problem;
    Ipopt::SmartPtr<Ipopt::IpoptApplication> ipopt_application;
};

}

#endif //LAOPT_IPOPT_WRAPPER_HPP