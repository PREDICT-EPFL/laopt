#include "lampc.hpp"
#include "ipopt_interface.hpp"

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

    using constraints = std::tuple
    <
        con_t<bnd_ineq, ineq, 1, iterator<x,0,0>>,
        con_t<bnd_zero<1>, eq, 1, iterator<x,0,0>>
    >;

    using objective = std::tuple
    <
        con_t<bnd_zero<1>, cost, 1, iterator<x,0,0>>
    >;

    // Create our NLP
    using problem_t = make_problem<variables, constraints, objective>;
};


/***********************************************************
    Implementation
 ***********************************************************/
int main()
{
    using scalar_t = double;

    // {
    //     using MatrixX = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;
    //     using Problem = Opt_t<scalar_t>::problem_t;
    //     Problem prob;

    //     using param_t = Opt_t<scalar_t>::param_t;
    //     param_t p;

    //     std::cout << "num_variables = " << prob.num_variables << std::endl;
    //     std::cout << "num_constraints = " << prob.num_constraints << std::endl;

    //     Problem::variable_t var;
    //     for(int i=0; i<Problem::num_variables; i++) var[i] = i;
    //     Problem::constraint_t con;
    //     for(int i=0; i<Problem::num_constraints; i++) con[i] = 0;

    //     Problem::constraint_t w;
    //     for(int i=0; i<Problem::num_constraints; i++) w[i] = i;

    //     Problem::constraint_jacobian_t J;
    //     J.setZero();

    //     Eigen::SparseMatrix<scalar_t> sJ(Problem::num_constraints, Problem::num_variables);
    //     prob.constraints.initialize_sparse_jacobian(sJ);

    //     Eigen::Matrix<scalar_t, Problem::num_variables, Problem::num_variables> H;
    //     H.setZero();

    //     Problem::constraint_t lb;
    //     Problem::constraint_t ub;
    //     lb.array() = -100.0;
    //     ub.array() = -100.0;

    //     Problem::variable_t x_lb; Problem::variable_t x_ub;
    //     x_lb.array() = -100.0; x_ub.array() = -100.0;

    //     std::cout << "\n\n";
    //     std::cout << "======================================\n";
    //     std::cout << "===Test computation of constraints ===\n";
    //     std::cout << "======================================\n";

    //     std::cout << "==> Computing value of constraints <==\n";
    //     prob.constraints(p, var, con);
    //     std::cout << "con = " << con.transpose() << std::endl;
    //     std::cout << "\n\n";

    //     std::cout << "==> Computing dense jacobian <==\n";
    //     prob.constraints(p, var, con, J);
    //     std::cout << "con = " << con.transpose() << std::endl;
    //     std::cout << "J = \n" << J << std::endl;
    //     std::cout << "\n\n";

    //     std::cout << "==> Computing sparse jacobian <==\n";
    //     prob.constraints(p, var, con, sJ);
    //     std::cout << "con = " << con.transpose() << std::endl;
    //     std::cout << "sJ = \n" << MatrixX(sJ) << std::endl;
    //     std::cout << "\n\n";

    //     std::cout << "==> Computing lower and upper bounds <==\n";
    //     prob.constraints.get_bounds(p, lb, ub);
    //     std::cout << "lb = " << lb.transpose() << " ub = " << ub.transpose() << std::endl;

    //     std::cout << "==> Computing variable bounds <==\n";
    //     prob.variables.get_bounds(p, x_lb, x_ub);
    //     std::cout << "x_lb = " << x_lb.transpose() << " x_ub = " << x_ub.transpose() << std::endl;

    //     std::cout << "\n\n";
    //     std::cout << "====================================\n";
    //     std::cout << "===Test computation of objective ===\n";
    //     std::cout << "====================================\n";

    //     for(int i=0; i<Problem::num_variables; i++) var[i] = 1;
    //     std::cout << "obj = " << prob.objective(p, var) << std::endl;

    //     Problem::variable_t grad;
    //     auto val = prob.objective(p, var, grad);
    //     std::cout << "obj = " << val << std::endl;
    //     std::cout << "gradient = " << grad.transpose() << std::endl;

    //     std::cout << "\n\n ---------- DENSE HESSIAN ----------\n\n";
    //     Problem::obj_hessian_t hessian;
    //     val = prob.objective(p, var, grad, hessian);
    //     std::cout << "obj = " << val << std::endl;
    //     std::cout << "gradient = " << grad.transpose() << std::endl;
    //     std::cout << "hessian = \n" << hessian << std::endl;

    //     std::cout << "\n\n ---------- SPARSE HESSIAN ----------\n\n";
    //     Eigen::SparseMatrix<scalar_t> s_hessian(Problem::num_variables, Problem::num_variables);
    //     prob.objective.initialize_sparse_hessian(s_hessian);

    //     val = prob.objective(p, var, grad, s_hessian);
    //     std::cout << "obj = " << val << std::endl;
    //     std::cout << "gradient = " << grad.transpose() << std::endl;
    //     std::cout << "hessian = \n" << MatrixX(s_hessian) << std::endl;

    //     std::cout << "\n\n";
    //     std::cout << "=====================================\n";
    //     std::cout << "===Test computation of lagrangian ===\n";
    //     std::cout << "=====================================\n";

    //     std::cout << "dual variables = " << prob.lagrangian.w.template tail<Problem::num_constraints>().transpose() << std::endl;

    //     Eigen::SparseMatrix<scalar_t> l_hessian(Problem::num_variables, Problem::num_variables);
    //     prob.lagrangian.initialize_sparse_hessian(l_hessian);

    //     val = prob.lagrangian(p, var, grad, l_hessian);
    //     std::cout << "obj = " << val << std::endl;
    //     std::cout << "gradient = " << grad.transpose() << std::endl;
    //     std::cout << "hessian = \n" << MatrixX(l_hessian) << std::endl;
    // }

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
    mynlp->x0 = {1, 5, 5, 1};
    std::cout << "x0 = " << mynlp->x0.transpose() << std::endl;
    status = app->OptimizeTNLP(mynlp);

    // std::cout << "solution = " << mynlp->sol.primal.transpose() << std::endl;

    myNLP::sol_t &sol = mynlp->sol;
    std::cout << "x = \n" << Opt_t<scalar_t>::x()(sol.primal).transpose() << std::endl;
}