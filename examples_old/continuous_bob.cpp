#include "lampc.hpp"
#include "lampc_utility.hpp"

#include "ipopt_interface.hpp"
#include "polympc_interface.hpp"

#include "polynomials/ebyshev.hpp"
#include "polynomials/splines.hpp"

#include <iomanip>
#include <Eigen/Eigenvalues> 
#include <type_traits>

#include <qpmad/solver.h>
#include "solvers/box_ADMM.hpp"

using namespace lampc;


#define test_POLY_ORDER 7
#define test_NUM_SEG    1
#define test_NUM_EXP    1

/** benchmark the new collocation class */
using Polynomial = polympc::Chebyshev<test_POLY_ORDER, polympc::GAUSS_LOBATTO, double>;
using Approximation = polympc::Spline<Polynomial, test_NUM_SEG>;



template<typename scalar_t>
struct MyOpt
{
    enum
    {
        /** Collocation dimensions */
        NUM_NODES    = Approximation::NUM_NODES,
        POLY_ORDER   = Approximation::POLY_ORDER,
        NUM_SEGMENTS = Approximation::NUM_SEGMENTS,

        NX = 3,
        NU = 2
    };

    template<typename T>
    using state_t = Eigen::Matrix<T, NX, 1>;
    template<typename T>
    using control_t = Eigen::Matrix<T, NU, 1>;

    static constexpr auto INF = lampc::INF<scalar_t>;
    using time_t   = typename Eigen::Matrix<scalar_t, NUM_NODES, 1>;

    struct param_t
    {
        Eigen::DiagonalMatrix<scalar_t, 3> Q{1,1,1};
        Eigen::DiagonalMatrix<scalar_t, 2> R{1,1};
        Eigen::DiagonalMatrix<scalar_t, 3> QN{1,1,1};

        scalar_t d = 2.0;

        /** compute collocation parameters */
        const typename Approximation::diff_mat_t  m_D     = Approximation::compute_diff_matrix();
        const typename Approximation::nodes_t     m_nodes = Approximation::compute_nodes();
        const typename Approximation::q_weights_t m_quad_weights = Approximation::compute_int_weights();
        time_t time_nodes = time_t::Zero();

        Eigen::Matrix<scalar_t, NX, 1> x0{1, 2, 3};

        scalar_t t_start{0};
        scalar_t t_stop{5};
    };

    FUNCTION(dynamics, (xdot, NX), (x, NX), (u, NU), (pp, 2))
    {
        xdot(0) = u(0) * cos(x(2)) * cos(u(1)) + pp(0);
        xdot(1) = u(0) * sin(x(2)) * cos(u(1)) + pp(1);
        xdot(2) = u(0) * sin(u(1)) / T(p.d);
    }

    FUNCTION(colloc, (out, NX * NUM_NODES), (x, NX * NUM_NODES), (u, NU * NUM_NODES), (pp, 2))
    {
        const T t_scale = T((p.t_stop - p.t_start) / (2 * NUM_SEGMENTS));

        Eigen::Map<const Eigen::Matrix<T, NX, NUM_NODES>> X(x.data());
        Eigen::Map<const Eigen::Matrix<T, NU, NUM_NODES>> U(u.data());

        Eigen::Map<Eigen::Matrix<T, NUM_NODES, NX>> constraint(out.data());
        constraint = p.m_D.template cast<T>() * (X.transpose());

        Eigen::Matrix<T, NX, 1> tmp; tmp.setZero(); // Temporary to store the evaluation of the dynamics

        for(int k = 0; k < NUM_NODES; k++)
        {
            dynamics::template impl<T>(p, tmp, X.col(k), U.col(k), pp);
            constraint.row(k) -= (t_scale * tmp).transpose();
        }

        // std::cout << "constraint = " << constraint << std::endl;
    }

    FUNCTION(initial_state, (out, NX), (x, NX * NUM_NODES))
    {
        Eigen::Map<const Eigen::Matrix<T, NX, NUM_NODES>> X(x.data());
        out = X.template rightCols<1>() - p.x0.template cast<T>();
    }

    // FUNCTION(mayer_term_impl, (mayer, 1), (x, 3), (u, 2))
    // {
    //     Eigen::Matrix<T,3,3> Qm = Q.toDenseMatrix().template cast<T>();
    //     mayer << x.dot(Qm * x);
    // }

    FUNCTION(u_constraint, (val, NU * NUM_NODES), (u, NU * NUM_NODES))
    {
        val = u;
    }

    FUNCTION(stage_cost, (val, 1), (x, NX * NUM_NODES), (u, NU * NUM_NODES), (pp, 2))
    {
        val(0) = x.dot(x) + u.dot(u) + pp.dot(pp);
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
                                             Vec<scalar_t, NU * NUM_NODES> lb, Vec<scalar_t, NU * NUM_NODES> ub) noexcept
        {
            lb.array() = -1;
            ub.array() = 5*iteration;
        }
    };

    /*
        Variable bounds
     */
    struct x_bnd
    {
        static EIGEN_STRONG_INLINE void eval(const param_t& p, const int iteration, 
                                             Vec<scalar_t, NX * NUM_NODES> lb, Vec<scalar_t, NX * NUM_NODES> ub) noexcept
        {
            lb.array() = -5;
            ub.array() = 5;
        }
    };

    struct u_bnd
    {
        static EIGEN_STRONG_INLINE void eval(const param_t& p, const int iteration, 
                                             Vec<scalar_t, NU * NUM_NODES> lb, Vec<scalar_t, NU * NUM_NODES> ub) noexcept
        {
            lb.array() = -20;
            ub.array() = 20;
        }
    };

    struct pp_bnd
    {
        static EIGEN_STRONG_INLINE void eval(const param_t& p, const int iteration, 
                                             Vec<scalar_t, 2> lb, Vec<scalar_t, 2> ub) noexcept
        {
            lb.array() = -2;
            ub.array() = 2;
        }
    };

    using variables = Make_Variables(scalar_t, param_t,
        (0, x, (var_t<x_bnd, NX * NUM_NODES, 1>)),
        (1, u, (var_t<u_bnd, NU * NUM_NODES, 1>)),
        (2, pp, (var_t<pp_bnd, 2, 1>)));


    using equalities = std::tuple
    <
        // x(0) == x0
        con_t<bnd_zero<NX>, initial_state, 1, iterator<x,0,0>>,

        // dot x = f(x, u, p)
        con_t<bnd_zero<NX * NUM_NODES>, colloc, 1, iterator<x,0,0>, iterator<u,0,0>, iterator<pp,0,0>>
    >;

    using inequalities = std::tuple<
        con_t<u_bnd, u_constraint, 1, iterator<u,0,0>>
    >;

    using objective = std::tuple
    <
        // x(i)'*Q*x(i) + u(i)'*R*u(i) for i in range(0, N-1)
        con_t<bnd_zero<1>, stage_cost, 1, iterator<x,0,0>, iterator<u,0,0>, iterator<pp,0,0>>
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

        Eigen::Matrix<scalar_t, opt::NX * opt::NUM_NODES, 1> x;
        x.array() = 1.0;
        Eigen::Matrix<scalar_t, opt::NU * opt::NUM_NODES, 1> u;
        u.array() = 2.0;
        Eigen::Matrix<scalar_t, 2, 1> pp = {6, 7};

        std::cout << "stage_cost::eval(p, x, u, pp) = " << opt::stage_cost::eval(p, x, u, pp).transpose() << std::endl;
        std::cout << "\n\n";

        auto tmp = opt::colloc::eval(p, x, u, pp);
        Eigen::Map<const Eigen::Matrix<scalar_t, opt::NX, opt::NUM_NODES>> X_con(tmp.data());
        std::cout << "X_con = \n" << X_con << std::endl;

        auto J_ret = opt::colloc::jac(p, x, u, pp);
        std::cout << "J_ret.jacobian = \n" << J_ret.jacobian << std::endl;

        std::cout << "jac = stage_cost::jac(p, x, u)" << std::endl;
        auto jac = opt::stage_cost::jac(p, x, u, pp);
        std::cout << "jac.val = " << jac.val.transpose() << std::endl;
        std::cout << "jac.jacobian = \n" << jac.jacobian << std::endl;
        std::cout << "\n\n";

        std::cout << "hessian = stage_cost::hessian(p, x, u)\n";
        auto hessian = opt::stage_cost::hessian(p, x, u, pp);
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
        std::cout << "lb = " << lb.transpose() << "\nub = " << ub.transpose() << std::endl;

        std::cout << "==> Computing variable bounds <==\n";
        prob.variables.get_bounds(p, x_lb, x_ub);
        std::cout << "x_lb = " << x_lb.transpose() << "\nx_ub = " << x_ub.transpose() << std::endl;

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

        Eigen::Map<const Eigen::Matrix<scalar_t, opt::NX, opt::NUM_NODES>> X(opt::x()(sol.primal).data());
        Eigen::Map<const Eigen::Matrix<scalar_t, opt::NU, opt::NUM_NODES>> U(opt::u()(sol.primal).data());

        std::cout << "X = \n" << X << std::endl;
        std::cout << "U = \n" << U << std::endl;
        std::cout << "pp = " << opt::pp()(sol.primal) << std::endl;
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
        Eigen::Map<const Eigen::Matrix<scalar_t, opt::NX, opt::NUM_NODES>> X(opt::x()(x).data());
        Eigen::Map<const Eigen::Matrix<scalar_t, opt::NU, opt::NUM_NODES>> U(opt::u()(x).data());

        std::cout << "X = \n" << X << std::endl;
        std::cout << "U = \n" << U << std::endl;
        std::cout << "pp = " << opt::pp()(x) << std::endl;
        std::cout << "\n\n\n\n";
    }

    // void solve_qpmad()
    // {
    //     using prob_t = opt::problem_t;
    //     prob_t::variable_vec x;
    //     x = prob_t::variable_vec::Zero();
    //     prob_t::obj_hessian_mat H;
    //     prob_t::obj_gradient_vec h;
    //     prob_t::variable_vec lb;
    //     prob_t::variable_vec ub;
    //     prob_t::constraints_jacobian_mat A;
    //     prob_t::constraints_vec Alb;
    //     prob_t::constraints_vec Aub;

    //     prob_t prob;
    //     prob_t::param_t param;

    //     prob.objective(param, x, h, H);
    //     H.diagonal().array() += 1e-6;

    //     // lb <= c(x) <= ub
    //     // lb <= A*(x - x0) + c(x0) <= ub
    //     // lb + A*x0 - c(x) <= A*x <= ub + A*x0 - c(x0)
    //     prob_t::constraints_vec con;
    //     prob.constraints(param, x, con, A);

    //     prob_t::constraints_vec nl_lb;
    //     prob_t::constraints_vec nl_ub;
    //     prob.constraints.get_bounds(param, nl_lb, nl_ub);
    //     Alb = nl_lb + A*x - con;
    //     Aub = nl_ub + A*x - con;

    //     prob.variables.get_bounds(param, lb, ub);

    //     qpmad::Solver solver;
    //     const std::size_t NUM_EXP = 1;
    //     qpmad::Solver::ReturnStatus status;
    //     polympc::time_point start = polympc::get_time();
    //     for(int i = 0; i < NUM_EXP; ++i)
    //     {
    //         status = solver.solve(x, H, h, lb, ub, A, Alb, Aub);
    //     }
    //     polympc::time_point stop = polympc::get_time();
    //     auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    //     if (status != qpmad::Solver::OK)
    //     {
    //         std::cerr << "Error" << std::endl;
    //     }

    //     std::cout << "\n\n\n\n===================== QPMAD SOLUTION =====================\n";

    //     std::cout << "\nqpmad time " << std::setprecision(9)
    //               << static_cast<double>(duration.count()) / NUM_EXP << " [microseconds]" << "\n";
    //     std::cout << "x = \n" << opt::x()(x).transpose() << std::endl;
    //     std::cout << "u = \n" << opt::u()(x).transpose() << std::endl;
    //     std::cout << "xss = \n" << opt::xss()(x) << std::endl;
    //     std::cout << "uss = \n" << opt::uss()(x) << std::endl;
    //     std::cout << "\n\n\n\n";

    //     // std::cout << "qpmad solution = " << x.transpose() << std::endl;
    // }

    // void solve_admm()
    // {
    //     using prob_t = opt::problem_t;
    //     prob_t::variable_vec x0;
    //     x0 = prob_t::variable_vec::Zero();
    //     prob_t::obj_hessian_mat H;
    //     prob_t::obj_gradient_vec h;
    //     prob_t::variable_vec lb;
    //     prob_t::variable_vec ub;
    //     prob_t::constraints_jacobian_mat A;
    //     prob_t::constraints_vec Alb;
    //     prob_t::constraints_vec Aub;

    //     prob_t prob;
    //     prob_t::param_t param;

    //     prob.objective(param, x0, h, H);
    //     H.diagonal().array() += 1e-6;

    //     // lb <= c(x) <= ub
    //     // lb <= A*(x - x0) + c(x0) <= ub
    //     // lb + A*x0 - c(x) <= A*x <= ub + A*x0 - c(x0)
    //     prob_t::constraints_vec con;
    //     prob.constraints(param, x0, con, A);

    //     prob_t::constraints_vec nl_lb;
    //     prob_t::constraints_vec nl_ub;
    //     prob.constraints.get_bounds(param, nl_lb, nl_ub);

    //     Alb = nl_lb + A*x0 - con;
    //     Aub = nl_ub + A*x0 - con;

    //     prob.variables.get_bounds(param, lb, ub);

    //     using Solver = boxADMM<prob_t::num_variables, prob_t::constraints_t::num_constraints>;
    //     Solver solver;
    //     solver.m_x.array() = 0;
    //     solver.m_y.array() = 0;

    //     const std::size_t NUM_EXP = 1;
    //     status_t status;
    //     polympc::time_point start = polympc::get_time();
    //     for(int i = 0; i < NUM_EXP; ++i)
    //     {
    //         status = solver.solve(H, h, A, Alb, Aub, lb, ub);
    //     }
    //     polympc::time_point stop = polympc::get_time();
    //     auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    //     // std::cout << "status = " << status << std::endl;

    //     auto x = solver.m_x;

    //     std::cout << "\n\n\n\n===================== ADMM SOLUTION =====================\n";

    //     std::cout << "\nadmm time " << std::setprecision(9)
    //               << static_cast<double>(duration.count()) / NUM_EXP << " [microseconds]" << "\n";
    //     std::cout << "x = \n" << opt::x()(x).transpose() << std::endl;
    //     std::cout << "u = \n" << opt::u()(x).transpose() << std::endl;
    //     std::cout << "xss = \n" << opt::xss()(x) << std::endl;
    //     std::cout << "uss = \n" << opt::uss()(x) << std::endl;
    //     std::cout << "\n\n\n\n";


    //     // std::cout << "qpmad solution = " << x.transpose() << std::endl;
    // }
};



int main()
{
    Bob bob;

    // bob.test_function_computation();
    // bob.test_problem_evaluation();

    bob.solve_ipopt();
    bob.solve_polympc();
    // bob.solve_qpmad();
    // bob.solve_admm();
}
