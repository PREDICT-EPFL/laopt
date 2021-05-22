#include "lampc.hpp"
#include "ipopt_interface.hpp"
#include "polympc_interface.hpp"

#include <iomanip>
#include <Eigen/Eigenvalues> 
#include <type_traits>

/***********************************************************
    Code generated from Python or from user
 ***********************************************************/

template<typename scalar_t_>
struct Opt_t
{
    using scalar_t = scalar_t_;
    static constexpr auto INF = std::numeric_limits<scalar_t>::infinity();

    struct param_t
    {
        Eigen::Matrix<scalar_t, 2, 1> x0 = {-1, -2};
        const Eigen::Matrix<scalar_t, 2, 2> A = {{1.0, 0.0}, {0.1, 1.0}};
        Eigen::Matrix<scalar_t, 2, 1> B = {0.1, 0.005};
        Eigen::Matrix<scalar_t, 1, 1> ref = {3};

        Eigen::Matrix<scalar_t, 2, 1> q = {100, 1e3}; // Stage-cost weights
        Eigen::Matrix<scalar_t, 1, 1> r = {1}; // Stage-cost weights
    };

    struct func1_
    {
        template<typename T>
        static EIGEN_STRONG_INLINE void eval(const param_t& p, Vec<T, 2> out, cVec<T, 1>& var0_, cVec<T, 2>& var1_) noexcept
        {
            Eigen::Matrix<T, 2, 1> tmp1;
            dynamics_::template eval<T>(p, tmp1, p.x0.template cast<T>(), var0_);
            out = var1_ - tmp1;
        }
    };
    using func1 = Jacobian<func1_, scalar_t, param_t, 2, 1, 2>;

    struct initial_state_
    {
        template<typename T>
        static EIGEN_STRONG_INLINE void eval(const param_t& p, Vec<T, 2> out, cVec<T, 2>& x0) noexcept
        {
            out = x0 - p.x0.template cast<T>();
        }
    };
    using initial_state = Jacobian<initial_state_, scalar_t, param_t, 2, 2>;

    struct dynamics_
    {
        template<typename T>
        static EIGEN_STRONG_INLINE void eval(const param_t& p, Vec<T, 2> out, cVec<T, 2>& x, cVec<T, 1>& u) noexcept
        {
            out = p.A.template cast<T>() * x + p.B.template cast<T>() * u;
            // out(0) += x(0)*x(1)*u(0);
            // out(1) += x(0)*10*u(0);
        }
    };
    using dynamics = Jacobian<dynamics_, scalar_t, param_t, 2, 2, 1>;

    struct func3_
    {
        template<typename T>
        static EIGEN_STRONG_INLINE void eval(const param_t& p, Vec<T, 2> out, cVec<T, 1>& var0_, cVec<T, 2>& var1_, cVec<T, 2>& var2_) noexcept
        {
            Matrix<T, 2, 1> tmp2;
            dynamics_::template eval<T>(p, tmp2, var1_, var0_);
            out = var2_ - tmp2;
        }
    };
    using func3 = Jacobian<func3_, scalar_t, param_t, 2, 1, 2, 2>;

    struct func2_
    {
        template<typename T>
        static EIGEN_STRONG_INLINE void eval(const param_t& p, Vec<T, 2> out, cVec<T, 1>& var0_, cVec<T, 2>& var1_) noexcept
        {
            Matrix<T, 2, 1> tmp3;
            dynamics_::template eval<T>(p, tmp3, var1_, var0_);
            out = var1_ - tmp3;
        }
    };
    using func2 = Jacobian<func2_, scalar_t, param_t, 2, 1, 2>;

    struct stage_cost_
    {
        template<typename T>
        static EIGEN_STRONG_INLINE void eval(const param_t& p, Vec<T, 1> out, 
                cVec<T, 2>& x, cVec<T, 1>& u, cVec<T, 2>& xss, cVec<T, 1>& uss) noexcept
        {
            Eigen::Matrix<T, 2, 1> x_err = x - xss;
            Eigen::Matrix<T, 1, 1> u_err = u - uss;

            out(0) = x_err.cwiseProduct(p.q.template cast<T>()).dot(x_err) + u_err.cwiseProduct(p.r.template cast<T>()).dot(u_err);
        }
    };
    using stage_cost = Jacobian<stage_cost_, scalar_t, param_t, 1, 2, 1, 2, 1>;

    struct terminal_cost_
    {
        template<typename T>
        static EIGEN_STRONG_INLINE void eval(const param_t& p, Vec<T, 1> out, cVec<T, 2>& xss, cVec<T, 1>& uss) noexcept
        {
            Eigen::Matrix<T, 2, 1> x_err;
            x_err(0) = xss(0);
            x_err(1) = xss(1) - (p.ref.template cast<T>())(0);

            out(0) = 1e3*(x_err.cwiseProduct(p.q.template cast<T>()).dot(x_err) + uss.cwiseProduct(p.r.template cast<T>()).dot(uss));
        }
    };
    using terminal_cost = Jacobian<terminal_cost_, scalar_t, param_t, 1, 2, 1>;


    struct u_constraint_
    {
        template<typename T>
        static EIGEN_STRONG_INLINE void eval(const param_t& p, Vec<T, 1> out, cVec<T, 1>& u) noexcept
        {
            out = u;
        }
    };
    using u_constraint = Jacobian<u_constraint_, scalar_t, param_t, 1, 1>;


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

    struct bnd_u
    {
        static EIGEN_STRONG_INLINE void eval(const param_t& p, const int iteration, 
                                             Vec<scalar_t, 1> lb, Vec<scalar_t, 1> ub) noexcept
        {
            lb.array() = -1;
            ub.array() = 5*iteration;
        }
    };

    /*
        Variable bounds
     */
    struct xss_bnd
    {
        static EIGEN_STRONG_INLINE void eval(const param_t& p, const int iteration, 
                                             Vec<scalar_t, 2> lb, Vec<scalar_t, 2> ub) noexcept
        {
            lb.array() = -INF;
            ub.array() = INF;
        }
    };

    struct uss_bnd
    {
        static EIGEN_STRONG_INLINE void eval(const param_t& p, const int iteration, 
                                             Vec<scalar_t, 1> lb, Vec<scalar_t, 1> ub) noexcept
        {
            lb.array() = -INF;
            ub.array() = INF;
        }
    };

    struct x_bnd
    {
        static EIGEN_STRONG_INLINE void eval(const param_t& p, const int iteration, 
                                             Vec<scalar_t, 2> lb, Vec<scalar_t, 2> ub) noexcept
        {
            lb.array() = -5;
            ub.array() = 5;
        }
    };

    struct u_bnd
    {
        static EIGEN_STRONG_INLINE void eval(const param_t& p, const int iteration, 
                                             Vec<scalar_t, 1> lb, Vec<scalar_t, 1> ub) noexcept
        {
            lb.array() = -20;
            ub.array() = 20;
        }
    };

    // Define variable accessors and ordering
    static constexpr int N = 15;
    using xss = var_t<xss_bnd, 2, 1>;
    using uss = var_t<uss_bnd, 1, 1, xss>;
    using x   = var_t<x_bnd,   2, N+1, uss>;
    using u   = var_t<u_bnd,   1, N , x>;

    using variables = VariableList_t<scalar_t, xss, uss, x, u>;

    using equalities = std::tuple
    <
        // x(0) == x0
        con_t<bnd_zero<2>, initial_state, 1, iterator<x,0,0>>,

        // dynamics(u(i), x(i)) == x(i+1) for i in range(0, N-1)
        con_t<bnd_zero<2>, func3, N, iterator<u,0,1>, iterator<x,0,1>, iterator<x,1,1>>, 

        // dynamics(xss, uss) == xss
        con_t<bnd_zero<2>, func2, 1, iterator<uss>, iterator<xss>>
    >;

    using inequalities = std::tuple<
        con_t<uss_bnd, u_constraint, 1, iterator<uss>>
    >;

    using objective = std::tuple
    <
        // x(i)'*Q*x(i) + u(i)'*R*u(i) for i in range(0, N-1)
        con_t<bnd_zero<1>, stage_cost, N-2, iterator<x,0,1>, iterator<u,0,1>, iterator<xss,0,0>, iterator<uss,0,0>>,

        // (xss(0); xss(1)-ref)'*Q*(xss(0); xss(1)-ref) + uss*R*uss
        con_t<bnd_zero<1>, terminal_cost, 1, iterator<xss,0,0>, iterator<uss,0,0>>
    >;

    // Create our NLP
    using problem_t = make_problem<variables, equalities, inequalities, objective>;
};


/***********************************************************
    Implementation
 ***********************************************************/
int main()
{
    using scalar_t = double;

    // {
    //     using MatrixX = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;
    //     std::cout << "\n\n===Test direct computation of functions===\n";
        
    //     Opt::param_t p;

    //     Eigen::Matrix<scalar_t, 2, 1> x = {1, 2};
    //     Eigen::Matrix<scalar_t, 1, 1> u = {3};

    //     std::cout << "stage_cost::eval(p, x, u) = " << Opt::stage_cost::eval(p, x, u).transpose() << std::endl;
    //     std::cout << "\n\n";

    //     std::cout << "jac = stage_cost::jac(p, x, u)" << std::endl;
    //     auto jac = Opt::stage_cost::jac(p, x, u);
    //     std::cout << "jac.val = " << jac.val.transpose() << std::endl;
    //     std::cout << "jac.jacobian = \n" << jac.jacobian << std::endl;
    //     std::cout << "\n\n";

    //     std::cout << "hessian = stage_cost::hessian(p, x, u) = " << Opt::stage_cost::eval(p, x, u).transpose() << std::endl;
    //     auto hessian = Opt::stage_cost::hessian(p, x, u);
    //     std::cout << "hessian.val = " << hessian.val.transpose() << std::endl;
    //     std::cout << "hessian.jacobian = \n" << hessian.jacobian << std::endl;
    //     for(int i=0; i<Opt::stage_cost::num_outputs; i++)
    //     {
    //         std::cout << "hessian.hessian(" << i << ")\n";
    //         std::cout << hessian.hessian[i] << std::endl;
    //     }        
    // }


    // {
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

        const std::size_t NUM_EXP = 1;
        polympc::time_point start = polympc::get_time();
        for(int i = 0; i < NUM_EXP; ++i)
        {
            mynlp->x0.array() = 0;
            status = app->OptimizeTNLP(mynlp);
        }
        polympc::time_point stop = polympc::get_time();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        std::cout << "\n\n\n\n===================== IPOPT SOLUTION =====================\n";
        std::cout << "IPOPT time " << std::setprecision(9)
                  << static_cast<double>(duration.count()) / NUM_EXP << " [microseconds]" << "\n";
        myNLP::sol_t &sol = mynlp->sol;
        std::cout << "x = \n" << Opt_t<scalar_t>::x()(sol.primal).transpose() << std::endl;
        std::cout << "u = \n" << Opt_t<scalar_t>::u()(sol.primal).transpose() << std::endl;
        std::cout << "xss = \n" << Opt_t<scalar_t>::xss()(sol.primal) << std::endl;
        std::cout << "uss = \n" << Opt_t<scalar_t>::uss()(sol.primal) << std::endl;
        std::cout << "\n\n\n\n";
    }

    {
        using prob_t = Opt_t<scalar_t>::problem_t;
        using Solver = SQPSolver<LAProblemBase<prob_t>>;

        Solver solver;
        solver.problem.setBounds(solver);

        Solver::nlp_variable_t x0, x;
        Solver::nlp_dual_t y0;

        solver.settings().max_iter = 50;
        solver.settings().line_search_max_iter = 5;
        // solver.qp_settings().eps_abs = 1e-6;
        // solver.qp_settings().eps_rel = 1e-6;
        // solver.qp_settings().max_iter = 1000;

        // const Solver::parameter_t p;
        // const Solver::nlp_dual_t lam = y0;
        // Solver::nlp_variable_t cost_grad;
        // Solver::nlp_hessian_t lag_hessian;
        // Solver::nlp_jacobian_t A;
        // Solver::nlp_constraints_t b;

        // std::cout << type_name<decltype(A)>() << std::endl;
        // solver.linearisation(x, p, lam, cost_grad, lag_hessian, A, b);

        // std::cout << "A = \n" << A << std::endl;
        // std::cout << "b = " << b.transpose() << std::endl;

        // {
        //     scalar_t lagrangian;
        //     Solver::nlp_variable_t lag_gradient;
        //     Solver::nlp_hessian_t lag_hessian;
        //     Solver::nlp_variable_t cost_gradient;
        //     Solver::nlp_constraints_t g;
        //     Solver::nlp_jacobian_t jac_g;

        //     solver.problem.lagrangian_gradient_hessian(x, p, lam, lagrangian, lag_gradient, lag_hessian, cost_gradient, g, jac_g);

        //     // std::cout << "jac_g = \n" << jac_g << std::endl;
        //     // std::cout << "g = " << g.transpose() << std::endl;
        //     std::cout << "lag_hessian = \n" << lag_hessian << std::endl;

        //     std::cout << "eigenvalues(hessian) = ";
        //     Eigen::EigenSolver<Solver::nlp_hessian_t> eigensolver(lag_hessian);
        //     for (int i = 0; i < eigensolver.eigenvalues().rows(); i++) {
        //       double v = eigensolver.eigenvalues()(i).real();
        //       std::cout << v << ", ";
        //     }
        //     std::cout << std::endl;
        // }


        const std::size_t NUM_EXP = 1;
        polympc::time_point start = polympc::get_time();
        for(int i = 0; i < NUM_EXP; ++i)
        {
            x0.array() = 0;
            y0.array() = 0;
            solver.solve(x0, y0);
        }
        polympc::time_point stop = polympc::get_time();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        auto info = solver.info();
        std::cout << "---- solution status ----\n";
        std::cout << "iter = " << info.iter << std::endl;
        std::cout << "qp_solver_iter = " << info.qp_solver_iter << std::endl;
        // std::cout << "status = " << info.status << std::endl;


        x = solver.primal_solution();

        std::cout << "\n\n\n\n===================== POLYMPC SOLUTION =====================\n";
        std::cout << "polympc time " << std::setprecision(9)
                  << static_cast<double>(duration.count()) / NUM_EXP << " [microseconds]" << "\n";
        std::cout << "x = \n" << Opt_t<scalar_t>::x()(x).transpose() << std::endl;
        std::cout << "u = \n" << Opt_t<scalar_t>::u()(x).transpose() << std::endl;
        std::cout << "xss = \n" << Opt_t<scalar_t>::xss()(x) << std::endl;
        std::cout << "uss = \n" << Opt_t<scalar_t>::uss()(x) << std::endl;
        std::cout << "\n\n\n\n";
    }
}