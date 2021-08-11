#include "lampc.hpp"
#include "lampc_utility.hpp"

#include "ipopt_interface.hpp"
#include "polympc_interface.hpp"

#include <iomanip>
#include <Eigen/Eigenvalues> 
#include <type_traits>

#include <qpmad/solver.h>
#include "solvers/box_ADMM.hpp"

/***********************************************************
    Code generated from Python or from user
 ***********************************************************/

using namespace lampc;

template<typename scalar_t>
struct MyOpt
{
    static constexpr auto INF = lampc::INF<scalar_t>;

    struct param_t
    {
        Eigen::Matrix<scalar_t, 2, 1> x0 {-1, -2};
        const Eigen::Matrix<scalar_t, 2, 2> A {{1.0, 0.0}, {0.1, 1.0}};
        Eigen::Matrix<scalar_t, 2, 1> B {0.1, 0.005};
        Eigen::Matrix<scalar_t, 1, 1> ref {3};

        Eigen::Matrix<scalar_t, 2, 1> q {1, 1e3}; // Stage-cost weights
        Eigen::Matrix<scalar_t, 1, 1> r {1e-3}; // Stage-cost weights
    };


    FUNCTION(initial_state, (out, 2), (x0, 2))
    {
        out = x0 - p.x0.template cast<T>();
    }

    FUNCTION(dynamics, (xplus, 2), (x, 2), (u, 1))
    {
        xplus = p.A.template cast<T>() * x + p.B.template cast<T>() * u;
    }

    FUNCTION(dynamics_eq, (out, 2), (xplus, 2), (x, 2), (u, 1))
    {
        Matrix<T, 2, 1> tmp;
        dynamics::template impl<T>(p, tmp, x, u);
        out = tmp - xplus;
    }

    FUNCTION(steady_state, (out, 2), (xss, 2), (uss, 1))
    {
        Matrix<T, 2, 1> xplus;
        dynamics::template impl<T>(p, xplus, xss, uss);
        out = xss - xplus;
    }

    FUNCTION(stage_cost, (val, 1), (x, 2), (u, 1), (xss, 2), (uss, 1))
    {
        Eigen::Matrix<T, 2, 1> x_err = x - xss;
        Eigen::Matrix<T, 1, 1> u_err = u - uss;

        val(0) = x_err.cwiseProduct(p.q.template cast<T>()).dot(x_err) + u_err.cwiseProduct(p.r.template cast<T>()).dot(u_err);
    }

    FUNCTION(terminal_cost, (val, 1), (xss, 2), (uss, 1))
    {
        Eigen::Matrix<T, 2, 1> x_err;
        x_err(0) = xss(0);
        x_err(1) = xss(1) - (p.ref.template cast<T>())(0);

        val(0) = 1e3*(x_err.cwiseProduct(p.q.template cast<T>()).dot(x_err) + uss.cwiseProduct(p.r.template cast<T>()).dot(uss));
    }

    FUNCTION(u_constraint, (val, 1), (u, 1))
    {
        val = u;
    }

    // CONSTANT_FUNCTION(...)
    // {
    //     -> defines jacobian and hessian as zero
    // }

    // LINEAR_FUNCTION(...)
    // {
    //     -> defines hessian as zero
    // }

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

    static constexpr int N = 5;
    using variables = Make_Variables(scalar_t, param_t,
        (0, xss, (var_t<xss_bnd, 2, 1>)), 
        (1, uss, (var_t<uss_bnd, 1, 1>)), 
        (2,   x, (var_t<x_bnd, 2, N+1>)),
        (3,   u, (var_t<u_bnd, 1, N>)));


    using equalities = std::tuple
    <
        // x(0) == x0
        con_t<bnd_zero<2>, initial_state, 1, iterator<x,0,0>>,

        // // dynamics(u(i), x(i)) == x(i+1) for i in range(0, N-1)
        con_t<bnd_zero<2>, dynamics_eq, N, iterator<x,1,1>, iterator<x,0,1>, iterator<u,0,1>>, 

        // dynamics(xss, uss) == xss
        con_t<bnd_zero<2>, steady_state, 1, iterator<xss>, iterator<uss>>
    >;

    using inequalities = std::tuple<
        con_t<uss_bnd, u_constraint, 1, iterator<uss>>
    >;

    using objective = std::tuple
    <
        // // x(i)'*Q*x(i) + u(i)'*R*u(i) for i in range(0, N-1)
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

struct Bob
{
    using scalar_t = double;
    using opt = MyOpt<scalar_t>;
    using param_t = opt::param_t;
    using prob_t = opt::problem_t;

    void test_function_computation()
    {
        using MatrixX = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;
        std::cout << "\n\n===Test direct computation of functions===\n";
        
        param_t p;

        Eigen::Matrix<scalar_t, 2, 1> x {1, 2};
        Eigen::Matrix<scalar_t, 1, 1> u {3};
        Eigen::Matrix<scalar_t, 2, 1> xss {2, 3};
        Eigen::Matrix<scalar_t, 1, 1> uss {4};

        std::cout << "stage_cost::eval(p, x, u, xss, uss) = " << opt::stage_cost::eval(p, x, u, xss, uss).transpose() << std::endl;
        std::cout << "\n\n";

        std::cout << "jac = stage_cost::jac(p, x, u)" << std::endl;
        auto jac = opt::stage_cost::jac(p, x, u, xss, uss);
        std::cout << "jac.val = " << jac.val.transpose() << std::endl;
        std::cout << "jac.jacobian = \n" << jac.jacobian << std::endl;
        std::cout << "\n\n";

        std::cout << "hessian = stage_cost::hessian(p, x, u) = " << opt::stage_cost::eval(p, x, u, xss, uss).transpose() << std::endl;
        auto hessian = opt::stage_cost::hessian(p, x, u, xss, uss);
        std::cout << "hessian.val = " << hessian.val.transpose() << std::endl;
        std::cout << "hessian.jacobian = \n" << hessian.jacobian << std::endl;
        for(int i=0; i<opt::stage_cost::num_outputs; i++)
        {
            std::cout << "hessian.hessian(" << i << ")\n";
            std::cout << hessian.hessian[i] << std::endl;
        }        
    }


    void test_problem_evaluation()
    {
        using MatrixX = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;
        using Problem = prob_t;
        Problem prob;
        param_t p;

        std::cout << "num_variables = " << prob.num_variables << std::endl;
        std::cout << "num_constraints = " << prob.constraints.num_constraints << std::endl;

        Problem::variable_vec var;
        for(int i=0; i<Problem::num_variables; i++) var[i] = i;
        Problem::constraints_vec con;
        for(int i=0; i<Problem::constraints_t::num_constraints; i++) con[i] = 0;

        Problem::constraints_vec w;
        for(int i=0; i<Problem::num_constraints; i++) w[i] = i;

        Problem::constraints_jacobian_mat J;
        J.setZero();

        Eigen::SparseMatrix<scalar_t> sJ(Problem::num_constraints, Problem::num_variables);
        prob.constraints.initialize_sparse_jacobian(sJ);

        Eigen::Matrix<scalar_t, Problem::num_variables, Problem::num_variables> H;
        H.setZero();

        Problem::constraints_vec lb;
        Problem::constraints_vec ub;
        lb.array() = -100.0;
        ub.array() = -100.0;

        Problem::variable_vec x_lb; Problem::variable_vec x_ub;
        x_lb.array() = -100.0; x_ub.array() = -100.0;

        std::cout << "\n\n";
        std::cout << "======================================\n";
        std::cout << "===Test computation of constraints ===\n";
        std::cout << "======================================\n";

        std::cout << "==> Computing value of constraints <==\n";
        prob.constraints(p, var, con);
        std::cout << "con = " << con.transpose() << std::endl;
        std::cout << "\n\n";

        std::cout << "==> Computing dense jacobian <==\n";
        prob.constraints(p, var, con, J);
        std::cout << "con = " << con.transpose() << std::endl;
        std::cout << "J = \n" << J << std::endl;
        std::cout << "\n\n";

        std::cout << "==> Computing sparse jacobian <==\n";
        prob.constraints(p, var, con, sJ);
        std::cout << "con = " << con.transpose() << std::endl;
        std::cout << "sJ = \n" << MatrixX(sJ) << std::endl;
        std::cout << "\n\n";

        std::cout << "==> Computing lower and upper bounds <==\n";
        prob.constraints.get_bounds(p, lb, ub);
        std::cout << "lb = " << lb.transpose() << " ub = " << ub.transpose() << std::endl;

        std::cout << "==> Computing variable bounds <==\n";
        prob.variables.get_bounds(p, x_lb, x_ub);
        std::cout << "x_lb = " << x_lb.transpose() << " x_ub = " << x_ub.transpose() << std::endl;

        std::cout << "\n\n";
        std::cout << "====================================\n";
        std::cout << "===Test computation of objective ===\n";
        std::cout << "====================================\n";

        for(int i=0; i<Problem::num_variables; i++) var[i] = 1;
        std::cout << "obj = " << prob.objective(p, var) << std::endl;

        Problem::variable_vec grad;
        auto val = prob.objective(p, var, grad);
        std::cout << "obj = " << val << std::endl;
        std::cout << "gradient = " << grad.transpose() << std::endl;

        std::cout << "\n\n ---------- DENSE HESSIAN ----------\n\n";
        Problem::obj_hessian_mat hessian;
        val = prob.objective(p, var, grad, hessian);
        std::cout << "obj = " << val << std::endl;
        std::cout << "gradient = " << grad.transpose() << std::endl;
        std::cout << "hessian = \n" << hessian << std::endl;

        std::cout << "\n\n ---------- SPARSE HESSIAN ----------\n\n";
        Eigen::SparseMatrix<scalar_t> s_hessian(Problem::num_variables, Problem::num_variables);
        prob.objective.initialize_sparse_hessian(s_hessian);

        val = prob.objective(p, var, grad, s_hessian);
        std::cout << "obj = " << val << std::endl;
        std::cout << "gradient = " << grad.transpose() << std::endl;
        std::cout << "hessian = \n" << MatrixX(s_hessian) << std::endl;

        std::cout << "\n\n";
        std::cout << "=====================================\n";
        std::cout << "===Test computation of lagrangian ===\n";
        std::cout << "=====================================\n";

        Eigen::SparseMatrix<scalar_t> l_hessian(Problem::num_variables, Problem::num_variables);
        prob.lagrangian.initialize_sparse_hessian(l_hessian);

        Problem::lagrangian_t::eq_dual_vec eq_dual;
        Problem::lagrangian_t::ineq_dual_vec ineq_dual;
        Problem::lagrangian_t::var_dual_vec var_dual;

        eq_dual.array() = 1.0;
        ineq_dual.array() = 2.0;
        var_dual.array() = 3.0;

        val = prob.lagrangian(p, var, 1.0, eq_dual, ineq_dual, var_dual, grad, l_hessian);
        std::cout << "obj = " << val << std::endl;
        std::cout << "gradient = " << grad.transpose() << std::endl;
        std::cout << "hessian = \n" << MatrixX(l_hessian) << std::endl;
    }


    void solve_ipopt()
    {
        // using prob_t = Opt_t<scalar_t>::problem_t;
        using myNLP = NLP_Ipopt<opt::problem_t>;
        param_t p;

        Ipopt::SmartPtr<myNLP> mynlp = new myNLP(p);
        Ipopt::SmartPtr<Ipopt::IpoptApplication> app = new Ipopt::IpoptApplication();

        app->Options()->SetNumericValue("tol", 1e-7);
        app->Options()->SetStringValue("mu_strategy", "adaptive");
        app->Options()->SetStringValue("output_file", "ipopt.out");
        // app->Options()->SetStringValue("hessian_approximation", "limited-memory");

        Ipopt::ApplicationReturnStatus status;

        const std::size_t NUM_EXP = 1;
        polympc::time_point start = polympc::get_time();
        for(int i = 0; i < NUM_EXP; ++i)
        {
            status = app->Initialize();
            if( status != Ipopt::Solve_Succeeded )
            {
                std::cout << std::endl << std::endl << "*** Error during initialization!" << std::endl;
                return;
            }
            mynlp->x0.array() = 0;
            status = app->OptimizeTNLP(mynlp);
        }
        polympc::time_point stop = polympc::get_time();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        std::cout << "\n\n\n\n===================== IPOPT SOLUTION =====================\n";
        std::cout << "IPOPT time " << std::setprecision(9)
                  << static_cast<double>(duration.count()) / NUM_EXP << " [microseconds]" << "\n";
        myNLP::sol_t &sol = mynlp->sol;
        std::cout << "x = \n" << opt::x()(sol.primal).transpose() << std::endl;
        std::cout << "u = \n" << opt::u()(sol.primal).transpose() << std::endl;
        std::cout << "xss = \n" << opt::xss()(sol.primal) << std::endl;
        std::cout << "uss = \n" << opt::uss()(sol.primal) << std::endl;
        std::cout << "\n\n\n\n";
    }

    void solve_polympc()
    {        
        using prob_t = opt::problem_t;
        using Problem = LAProblemBase<prob_t>;
        using Solver = SQPSolver<LAProblemBase<prob_t>>;

        Solver solver;
        solver.problem.setBounds(solver);

        Solver::nlp_variable_t x0, x;
        Solver::nlp_dual_t y0;

        solver.settings().max_iter = 50;
        solver.settings().line_search_max_iter = 5;
        // solver.qp_settings().eps_abs = 1e-6;
        // solver.qp_settings().eps_rel = 1e-6;
        // solver.qp_settings().max_iter = 5000;

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
        //     Solver::nlp_variable_vec lag_gradient;
        //     Solver::nlp_hessian_t lag_hessian;
        //     Solver::nlp_variable_vec cost_gradient;
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



        x = solver.primal_solution();

        std::cout << "\n\n\n\n===================== POLYMPC SOLUTION =====================\n";

        auto info = solver.info();
        std::cout << "\n---- solution status ----\n";
        std::cout << "Size of the solver: " << sizeof (solver) << "\n";
        std::cout << "iter = " << info.iter << std::endl;
        std::cout << "qp_solver_iter = " << info.qp_solver_iter << std::endl;
        // std::cout << "status = " << info.status << std::endl;

        std::cout << "\npolympc time " << std::setprecision(9)
                  << static_cast<double>(duration.count()) / NUM_EXP << " [microseconds]" << "\n";
        std::cout << "x = \n" << opt::x()(x).transpose() << std::endl;
        std::cout << "u = \n" << opt::u()(x).transpose() << std::endl;
        std::cout << "xss = \n" << opt::xss()(x) << std::endl;
        std::cout << "uss = \n" << opt::uss()(x) << std::endl;
        std::cout << "\n\n\n\n";
    }

    void solve_qpmad()
    {
        using prob_t = opt::problem_t;
        prob_t::variable_vec x;
        x = prob_t::variable_vec::Zero();
        prob_t::obj_hessian_mat H;
        prob_t::obj_gradient_vec h;
        prob_t::variable_vec lb;
        prob_t::variable_vec ub;
        prob_t::constraints_jacobian_mat A;
        prob_t::constraints_vec Alb;
        prob_t::constraints_vec Aub;

        prob_t prob;
        prob_t::param_t param;

        prob.objective(param, x, h, H);
        H.diagonal().array() += 1e-6;

        // lb <= c(x) <= ub
        // lb <= A*(x - x0) + c(x0) <= ub
        // lb + A*x0 - c(x) <= A*x <= ub + A*x0 - c(x0)
        prob_t::constraints_vec con;
        prob.constraints(param, x, con, A);

        prob_t::constraints_vec nl_lb;
        prob_t::constraints_vec nl_ub;
        prob.constraints.get_bounds(param, nl_lb, nl_ub);
        Alb = nl_lb + A*x - con;
        Aub = nl_ub + A*x - con;

        prob.variables.get_bounds(param, lb, ub);

        qpmad::Solver solver;
        const std::size_t NUM_EXP = 1;
        qpmad::Solver::ReturnStatus status;
        polympc::time_point start = polympc::get_time();
        for(int i = 0; i < NUM_EXP; ++i)
        {
            status = solver.solve(x, H, h, lb, ub, A, Alb, Aub);
        }
        polympc::time_point stop = polympc::get_time();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        if (status != qpmad::Solver::OK)
        {
            std::cerr << "Error" << std::endl;
        }

        std::cout << "\n\n\n\n===================== QPMAD SOLUTION =====================\n";

        std::cout << "\nqpmad time " << std::setprecision(9)
                  << static_cast<double>(duration.count()) / NUM_EXP << " [microseconds]" << "\n";
        std::cout << "x = \n" << opt::x()(x).transpose() << std::endl;
        std::cout << "u = \n" << opt::u()(x).transpose() << std::endl;
        std::cout << "xss = \n" << opt::xss()(x) << std::endl;
        std::cout << "uss = \n" << opt::uss()(x) << std::endl;
        std::cout << "\n\n\n\n";

        // std::cout << "qpmad solution = " << x.transpose() << std::endl;
    }

    void solve_admm()
    {
        using prob_t = opt::problem_t;
        prob_t::variable_vec x0;
        x0 = prob_t::variable_vec::Zero();
        prob_t::obj_hessian_mat H;
        prob_t::obj_gradient_vec h;
        prob_t::variable_vec lb;
        prob_t::variable_vec ub;
        prob_t::constraints_jacobian_mat A;
        prob_t::constraints_vec Alb;
        prob_t::constraints_vec Aub;

        prob_t prob;
        prob_t::param_t param;

        prob.objective(param, x0, h, H);
        H.diagonal().array() += 1e-6;

        // lb <= c(x) <= ub
        // lb <= A*(x - x0) + c(x0) <= ub
        // lb + A*x0 - c(x) <= A*x <= ub + A*x0 - c(x0)
        prob_t::constraints_vec con;
        prob.constraints(param, x0, con, A);

        prob_t::constraints_vec nl_lb;
        prob_t::constraints_vec nl_ub;
        prob.constraints.get_bounds(param, nl_lb, nl_ub);

        Alb = nl_lb + A*x0 - con;
        Aub = nl_ub + A*x0 - con;

        prob.variables.get_bounds(param, lb, ub);

        using Solver = boxADMM<prob_t::num_variables, prob_t::constraints_t::num_constraints>;
        Solver solver;
        solver.m_x.array() = 0;
        solver.m_y.array() = 0;

        const std::size_t NUM_EXP = 1;
        status_t status;
        polympc::time_point start = polympc::get_time();
        for(int i = 0; i < NUM_EXP; ++i)
        {
            status = solver.solve(H, h, A, Alb, Aub, lb, ub);
        }
        polympc::time_point stop = polympc::get_time();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        // std::cout << "status = " << status << std::endl;

        auto x = solver.m_x;

        std::cout << "\n\n\n\n===================== ADMM SOLUTION =====================\n";

        std::cout << "\nadmm time " << std::setprecision(9)
                  << static_cast<double>(duration.count()) / NUM_EXP << " [microseconds]" << "\n";
        std::cout << "x = \n" << opt::x()(x).transpose() << std::endl;
        std::cout << "u = \n" << opt::u()(x).transpose() << std::endl;
        std::cout << "xss = \n" << opt::xss()(x) << std::endl;
        std::cout << "uss = \n" << opt::uss()(x) << std::endl;
        std::cout << "\n\n\n\n";


        // std::cout << "qpmad solution = " << x.transpose() << std::endl;
    }
};



int main()
{
    Bob bob;
    
    bob.test_function_computation();
    bob.test_problem_evaluation();

    bob.solve_ipopt();
    bob.solve_polympc();
    bob.solve_qpmad();
    bob.solve_admm();
}
