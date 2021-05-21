#include "lampc.hpp"
#include "ipopt_interface.hpp"
#include "polympc_interface.hpp"

#include <iomanip>
#include <Eigen/Eigenvalues> 
#include <type_traits>

/***********************************************************
    Simple QP

    min  0.5 * (x - 10)^2 + 0.5 * y0^2 + 2 * y1^2
    s.t. 
         x + y(0) + y(1) = 1

         -1 <= x + y(0) <= 1
         -1 <= 3*x + y(0) + 2*y(1) <= 1

         1 <= x <= 5
         -2 <= y <= 6

 ***********************************************************/

template<typename scalar_t_>
struct Opt_t
{
    using scalar_t = scalar_t_;
    static constexpr auto INF = std::numeric_limits<scalar_t>::infinity();

    struct param_t
    {
    };

    struct g_ineq_
    {
        template<typename T>
        static EIGEN_STRONG_INLINE void eval(const param_t& p, Vec<T, 2> out, cVec<T, 1>& x, cVec<T, 2>& y) noexcept
        {
            out << x(0) + y(0), 
                   T(3) * x(0) + y(0) + T(2) * y(1);
        }
    };
    using g_ineq = Jacobian<g_ineq_, scalar_t, param_t, 2, 1, 2>;

    struct g_eq_
    {
        template<typename T>
        static EIGEN_STRONG_INLINE void eval(const param_t& p, Vec<T, 1> out, cVec<T, 1>& x, cVec<T, 2>& y) noexcept
        {
            out << x(0) + y(0) + y(1);
        }
    };
    using g_eq = Jacobian<g_eq_, scalar_t, param_t, 1, 1, 2>;

    struct cost_
    {
        template<typename T>
        static EIGEN_STRONG_INLINE void eval(const param_t& p, Vec<T, 1> out, cVec<T, 1>& x, cVec<T, 2>& y) noexcept
        {
            out(0) = T(0.5) * (x(0) - T(10.0))*(x(0) - T(10.0)) + T(0.5) * y(0) * y(0) + T(2.0) * y(1) * y(1);
        }
    };
    using cost = Jacobian<cost_, scalar_t, param_t, 1, 1, 2>;

    /*
        Constraint bounds
     */
    struct bnd_ineq
    {
        static EIGEN_STRONG_INLINE void eval(const param_t& p, const int iteration, 
                                             Vec<scalar_t, 2> lb, Vec<scalar_t, 2> ub) noexcept
        {
            lb.array() = -1;
            ub.array() = 1;
        }
    };

    struct bnd_eq
    {
        static EIGEN_STRONG_INLINE void eval(const param_t& p, const int iteration, 
                                             Vec<scalar_t, 1> lb, Vec<scalar_t, 1> ub) noexcept
        {
            lb.array() = 0;
            ub.array() = 0;
        }
    };


    /*
        Variable bounds
     */
    struct x_bnd
    {
        static EIGEN_STRONG_INLINE void eval(const param_t& p, const int iteration, 
                                             Vec<scalar_t, 1> lb, Vec<scalar_t, 1> ub) noexcept
        {
            lb.array() = 1;
            ub.array() = 5;
        }
    };

    struct y_bnd
    {
        static EIGEN_STRONG_INLINE void eval(const param_t& p, const int iteration, 
                                             Vec<scalar_t, 2> lb, Vec<scalar_t, 2> ub) noexcept
        {
            lb.array() = -2;
            ub.array() = 6;
        }
    };

    // Define variable accessors and ordering
    using x = var_t<x_bnd, 1, 1>;
    using y = var_t<y_bnd, 2, 1, x>;

    using variables = VariableList_t<scalar_t, x, y>;

    using equalities = std::tuple
    <
        con_t<bnd_eq, g_eq, 1, iterator<x,0,0>, iterator<y,0,0>>
    >;

    using inequalities = std::tuple<
        con_t<bnd_ineq, g_ineq, 1, iterator<x,0,0>, iterator<y,0,0>>
    >;

    using objective = std::tuple
    <
        con_t<bnd_eq, cost, 1, iterator<x,0,0>, iterator<y,0,0>>
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

    // {
    //     using prob_t = Opt_t<scalar_t>::problem_t;
    //     using myNLP = NLP_Ipopt<prob_t>;
    //     prob_t::param_t p;

    //     Ipopt::SmartPtr<myNLP> mynlp = new myNLP(p);
    //     Ipopt::SmartPtr<Ipopt::IpoptApplication> app = new Ipopt::IpoptApplication();

    //     app->Options()->SetNumericValue("tol", 1e-7);
    //     app->Options()->SetStringValue("mu_strategy", "adaptive");
    //     app->Options()->SetStringValue("output_file", "ipopt.out");
    //     // app->Options()->SetStringValue("hessian_approximation", "limited-memory");

    //     Ipopt::ApplicationReturnStatus status;
    //     status = app->Initialize();
    //     if( status != Ipopt::Solve_Succeeded )
    //     {
    //         std::cout << std::endl << std::endl << "*** Error during initialization!" << std::endl;
    //         return (int) status;
    //     }

    //     const std::size_t NUM_EXP = 1;
    //     polympc::time_point start = polympc::get_time();
    //     for(int i = 0; i < NUM_EXP; ++i)
    //     {
    //         mynlp->x0.array() = 0;
    //         status = app->OptimizeTNLP(mynlp);
    //     }
    //     polympc::time_point stop = polympc::get_time();
    //     auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    //     std::cout << "\n\n\n\n===================== IPOPT SOLUTION =====================\n";
    //     std::cout << "IPOPT time " << std::setprecision(9)
    //               << static_cast<double>(duration.count()) / NUM_EXP << " [microseconds]" << "\n";
    //     myNLP::sol_t &sol = mynlp->sol;
    //     std::cout << "x = \n" << Opt_t<scalar_t>::x()(sol).transpose() << std::endl;
    //     std::cout << "y = \n" << Opt_t<scalar_t>::y()(sol).transpose() << std::endl;
    //     std::cout << "\n\n\n\n";
    // }

    {
        using prob_t = Opt_t<scalar_t>::problem_t;
        using Solver = SQPSolver<LAProblemBase<prob_t>>;

        Solver solver;
        solver.problem.setBounds(solver);

        Solver::nlp_variable_t x0, x;
        Solver::nlp_dual_t y0;

        solver.settings().max_iter = 50;
        solver.settings().line_search_max_iter = 5;

    //     const Solver::parameter_t p;
    //     const Solver::nlp_dual_t lam = y0;
    //     Solver::nlp_variable_t m_H;
    //     Solver::nlp_hessian_t m_H;
    //     Solver::nlp_jacobian_t m_A;
    //     Solver::nlp_constraints_t m_al;

    //     std::cout << type_name<decltype(A)>() << std::endl;
    //     solver.linearisation(x, p, lam, m_H, m_H, m_A, m_al);

    //     m_al = -m_al;
    //     m_au =  m_al;
    //     m_al.template tail<NUM_INEQ>() += m_lbg;
    //     m_au.template tail<NUM_INEQ>() += m_ubg;
    //     m_lx.noalias() = m_lbx - m_x;
    //     m_ux.noalias() = m_ubx - m_x;


    // qp_status = m_qp_solver.solve(m_H, m_h, m_A, m_al, m_au, m_lx, m_ux);


    //     std::cout << "A = \n" << A << std::endl;
    //     std::cout << "b = " << b.transpose() << std::endl;

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

        const std::size_t NUM_EXP = 100;
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
        std::cout << "y = \n" << Opt_t<scalar_t>::y()(x).transpose() << std::endl;
        std::cout << "\n\n\n\n";
    }
}