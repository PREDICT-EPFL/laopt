#include "lampc.hpp"

#include "ipopt_interface.hpp"
#include "polympc_interface.hpp"

/***********************************************************
    Implement the hs071_nlp example from ipopt

    min   x1*x4*(x1 + x2 + x3)  +  x3
    s.t.  x1*x2*x3*x4                   >=  25
          x1**2 + x2**2 + x3**2 + x4**2  =  40
          1 <=  x1,x2,x3,x4  <= 5

 ***********************************************************/

template<typename scalar_t_>
struct Opt_t
{
    using scalar_t = scalar_t_;

    struct param_t
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    };

    struct ineq_
    {
        template<typename T>
        static EIGEN_STRONG_INLINE void eval(const param_t& p, Vec<T, 1> out, cVec<T, 4>& x) noexcept
        {
            out(0) = x(0) * x(1) * x(2) * x(3);
        }
    };
    using ineq = Jacobian<ineq_, scalar_t, param_t, 1, 4>;

    struct eq_
    {
        template<typename T>
        static EIGEN_STRONG_INLINE void eval(const param_t& p, Vec<T, 1> out, cVec<T, 4>& x) noexcept
        {
            out(0) = x.dot(x) - 40.0;
        }
    };
    using eq = Jacobian<eq_, scalar_t, param_t, 1, 4>;

    struct cost_
    {
        template<typename T>
        static EIGEN_STRONG_INLINE void eval(const param_t& p, Vec<T, 1> out, cVec<T, 4>& x) noexcept
        {
            out(0) = x(0) * x(3) * (x(0) + x(1) + x(2)) + x(2);
        }
    };
    using cost = Jacobian<cost_, scalar_t, param_t, 1, 4>;

    /*
        Constraint bounds
     */
    template<std::size_t len>
    struct bnd_zero
    {
        static EIGEN_STRONG_INLINE void eval(const param_t& p, const int iteration, 
                                             Vec<scalar_t, len> lb, Vec<scalar_t, len> ub) noexcept
        {
            lb.array() = 0;
            ub.array() = 0;
        }
    };

    struct bnd_ineq
    {
        static EIGEN_STRONG_INLINE void eval(const param_t& p, const int iteration, 
                                             Vec<scalar_t, 1> lb, Vec<scalar_t, 1> ub) noexcept
        {
            lb.array() = 25;
            ub.array() = 1e20;
        }
    };

    /*
        Variable bounds
     */
    struct bnd_x
    {
        static EIGEN_STRONG_INLINE void eval(const param_t& p, const int iteration, 
                                             Vec<scalar_t, 4> lb, Vec<scalar_t, 4> ub) noexcept
        {
            lb.array() = 1;
            ub.array() = 5;
        }
    };


    // Define variable accessors and ordering
    using x = var_t<bnd_x, 4, 1>;
    using variables = VariableList_t<scalar_t, x>;

    using inequalities = std::tuple
    <
        con_t<bnd_ineq, ineq, 1, iterator<x,0,0>>
    >;

    using equalities = std::tuple
    <
        con_t<bnd_zero<1>, eq, 1, iterator<x,0,0>>
    >;

    using objective = std::tuple
    <
        con_t<bnd_zero<1>, cost, 1, iterator<x,0,0>>
    >;

    // Create our NLP
    using problem_t = make_problem<variables, equalities, inequalities, objective>;
};



using scalar_t = double;
using problem_t = Opt_t<scalar_t>::problem_t;

/***********************************************************
    Implementation
 ***********************************************************/
int main()
{
    // {
    //     problem_t prob;
    //     problem_t::variable_vec var;
    //     problem_t::param_t param;
    //     var << 1,2,3,4;
    //     std::cout << "prob.objective(param, var) = " << prob.objective(param, var) << std::endl;

    //     problem_t::lagrangian_t::eq_dual_vec eq_dual;
    //     eq_dual = problem_t::lagrangian_t::eq_dual_vec::Ones();
    //     problem_t::lagrangian_t::ineq_dual_vec ineq_dual;
    //     ineq_dual = problem_t::lagrangian_t::ineq_dual_vec::Ones();
    //     problem_t::lagrangian_t::var_dual_vec var_dual;
    //     var_dual = problem_t::lagrangian_t::var_dual_vec::Ones();

    //     std::cout << "prob.lagrangian(param, var) = " << prob.lagrangian(param, var, eq_dual, ineq_dual, var_dual) << std::endl;


    //     problem_t::variable_vec lag_gradient;
    //     prob.lagrangian(param, var, eq_dual, ineq_dual, var_dual, lag_gradient);
    //     std::cout << "lagrangian gradient = " << lag_gradient.transpose() << std::endl;

    //     problem_t::obj_hessian_mat lag_hessian;
    //     prob.lagrangian(param, var, eq_dual, ineq_dual, var_dual, lag_gradient, lag_hessian);
    //     std::cout << "lagrangian gradient = " << lag_gradient.transpose() << std::endl;
    //     std::cout << "lagrangian hessian = \n" << lag_hessian.transpose() << std::endl;

    //     Eigen::SparseMatrix<scalar_t> lag_hessian_s(problem_t::num_variables, problem_t::num_variables);
    //     prob.lagrangian.initialize_sparse_hessian(lag_hessian_s);
    //     prob.lagrangian(param, var, eq_dual, ineq_dual, var_dual, lag_gradient, lag_hessian_s);
    //     std::cout << "lagrangian gradient = " << lag_gradient.transpose() << std::endl;
    //     std::cout << "lagrangian hessian = \n" << lag_hessian_s.transpose() << std::endl;

    // }


    {
        using prob_t = Opt_t<scalar_t>::problem_t;
        using Solver = SQPSolver<LAProblemBase<prob_t>>;

        Solver solver;
        solver.problem.setBounds(solver);

        Solver::nlp_variable_t x0, x;
        Solver::nlp_dual_t y0;

        x0 << 1, 5, 5, 1;
        y0.array() = 1;

        solver.settings().max_iter = 50;
        solver.settings().line_search_max_iter = 5;
        solver.solve(x0, y0);

        x = solver.primal_solution();

        std::cout << "iter = " << solver.info().iter << std::endl;
        std::cout << "Solution " << x.transpose() << std::endl;
    }


    {
        using prob_t = Opt_t<scalar_t>::problem_t;
        using myNLP = NLP_Ipopt<prob_t>;
        prob_t::param_t p;

        Ipopt::SmartPtr<myNLP> mynlp = new myNLP(p);
        Ipopt::SmartPtr<Ipopt::IpoptApplication> app = new Ipopt::IpoptApplication();

        app->Options()->SetNumericValue("tol", 1e-7);
        app->Options()->SetStringValue("mu_strategy", "adaptive");
        app->Options()->SetStringValue("output_file", "ipopt.out");
        // app->Options()->SetStringValue("hessian_approximation", "limited-memory");

        Ipopt::ApplicationReturnStatus status;
        status = app->Initialize();
        if( status != Ipopt::Solve_Succeeded )
        {
            std::cout << std::endl << std::endl << "*** Error during initialization!" << std::endl;
            return (int) status;
        }

        // Ask Ipopt to solve the problem
        mynlp->x0.array() = 0;
        std::cout << "x0 = " << mynlp->x0.transpose() << std::endl;
        status = app->OptimizeTNLP(mynlp);

        // std::cout << "solution = " << mynlp->sol.primal.transpose() << std::endl;

        myNLP::sol_t &sol = mynlp->sol;
        std::cout << "x = \n" << Opt_t<scalar_t>::x()(sol.primal).transpose() << std::endl;
    }

}