#include "lampc.hpp"
#include "ipopt_interface.hpp"
#include "polympc_interface.hpp"

#include <iomanip>
#include <Eigen/Eigenvalues> 
#include <type_traits>

#include <qpmad/solver.h>

using namespace lampc;

/***********************************************************
    Simple QP

    min  0.5 * (x - 10)^2 + 0.5 * y0^2 + 2 * y1^2
    s.t. 
         x + y(0) + y(1) = 0

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

    // FUNCTION(g_ineq, (out, 2), (x, 1), (y, 2))
    // {
    //     out << x(0) + y(0), 
    //            3 * x(0) + y(0) + 2 * y(1);
    // }

    // FUNCTION(g_eq, (out, 1), (x, 1), (y, 2))
    // {
    //     out << x(0) + y(0) + y(1);
    // }

    // FUNCTION(cost, (out, 1), (x, 1), (y, 2))
    // {
    //     // out(0) = T(0.5) * (x(0) - T(10.0))*(x(0) - T(10.0)) + T(0.5) * y(0) * y(0) + T(2.0) * y(1) * y(1);
    //     out(0) = T(0.5) * (x(0) - T(10.0))*(x(0) - T(10.0)) + T(0.5) * y(0) * y(0) + T(2.0) * y(1) * y(1);
    // }

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
        Variables
     */
    struct x_info : var_size<1, 1> // Len, Number of variables
    {
        static EIGEN_STRONG_INLINE void bndFunc(const param_t& p, const int iteration, 
                                                Vec<scalar_t, len> lb, Vec<scalar_t, len> ub) noexcept
        {
            lb.array() = 1;
            ub.array() = 5;
        }
    };

    struct y_info : var_size<2, 1> // Len, Number of variables
    {
        static EIGEN_STRONG_INLINE void bndFunc(const param_t& p, const int iteration, 
                                                Vec<scalar_t, len> lb, Vec<scalar_t, len> ub) noexcept
        {
            lb.array() = -2;
            ub.array() = 6;
        }
    };
    
    // Defines
    // - variable_t
    // - num_vars
    // - offsets in var tuple
    using vars = make_variables<x_info, y_info>;
    using x = vars.get_var<0>;
    using y = vars.get_var<1>;

    template<int start=0, int step=0>
    using y = vars.itr<vars.get_by_index<1>, start, step>;




    // x overloads the same get functions as itr, but just returns x<0,0> for x<i,j> regardless of i and j

    eq_t<initial, 1, x>
    eq_t<dynamics, N, x::itr<1, 1>, x::itr<0, 1>, u::itr<0, 1>>


    // // using Opt = Make_Variables(scalar_t, param_t,
    // //     (0, x, (var_t<x_bnd, 1, 1>)), 
    // //     (1, y, (var_t<y_bnd, 2, 1>)));

    // // Builds a VariableList_t
    // // - computes total number of variables
    // // - computes offsets for each variable into variable list
    // // - builds a tuple of var_t's
    // using variables = typename make_variables<scalar_t, param_t,
    //         var_t<x_bnd, 1, 1>, // x
    //         var_t<y_bnd, 2, 1>  // y
    //     >::type;

    // // Convenience names for the variables
    // // TODO: Figure out how these can be iterators...
    // using x = typename variables::template get_by_index<0>;
    // using y = typename variables::template get_by_index<1>;

    // template<int start, int step=0>
    // using x_itr = typename variables::template itr<x, start, step>;

    // template<int start, int step=0>
    // using y_itr = typename variables::template itr<y, start, step>;

    // using equalities = std::tuple
    // <
    //     con_t<bnd_eq, g_eq, 1, x_itr<0,0>, y_itr<0,0>>
    // >;

    // using inequalities = std::tuple<
    //     con_t<bnd_ineq, g_ineq, 1, x_itr<0,0>, y_itr<0,0>>
    // >;

    // using objective = std::tuple
    // <
    //     con_t<bnd_eq, cost, 1, x_itr<0,0>, y_itr<0,0>>
    // >;

    // // Create our NLP
    // using problem_t = make_problem<variables, equalities, inequalities, objective, DENSE>;
};


/***********************************************************
 * Various methods to solve the QP
 ***********************************************************/
// template<typename scalar_t>
// void test_ipopt()
// {
//     using prob_t = typename Opt_t<scalar_t>::problem_t;
//     using myNLP = NLP_Ipopt<prob_t>;
//     typename prob_t::param_t p;

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
//         return;
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
//     std::cout << "x = \n" << typename Opt_t<scalar_t>::x()(mynlp->sol.primal).transpose() << std::endl;
//     std::cout << "y = \n" << typename Opt_t<scalar_t>::y()(mynlp->sol.primal).transpose() << std::endl;
//     std::cout << "\n\n\n\n";
// }

template<typename scalar_t>
void test_sqp()
{
    using prob_t = typename Opt_t<scalar_t>::problem_t;
    using Solver = SQPSolver<LAProblemBase<prob_t>>;

    Solver solver;
    solver.problem.setBounds(solver);

    typename Solver::nlp_variable_t x0{Solver::VAR_SIZE, 1}, x{Solver::VAR_SIZE, 1};
    typename Solver::nlp_dual_t y0{Solver::NUM_CONSTR, 1};

    solver.settings().max_iter = 1000;
    solver.settings().line_search_max_iter = 50;
    solver.qp_settings().eps_abs = 1e-6;
    solver.qp_settings().eps_rel = 1e-6;
    solver.qp_settings().max_iter = 1000;

    const std::size_t NUM_EXP = 1;
    polympc::time_point start = polympc::get_time();
    for(int i = 0; i < NUM_EXP; ++i)
    {
        x0.array() = 0;
        y0.array() = 0;
        std::cout << "x0.shape = [" << x0.rows() << ", " << x0.cols() << "]\n";
        std::cout << "y0.shape = [" << y0.rows() << ", " << y0.cols() << "]\n";
        solver.solve(x0, y0);
    }
    polympc::time_point stop = polympc::get_time();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    x = solver.primal_solution();

    std::cout << "\n\n\n\n===================== POLYMPC SOLUTION =====================\n";
    auto info = solver.info();
    std::cout << "---- solution status ----\n";
    std::cout << "iter = " << info.iter << std::endl;
    std::cout << "qp_solver_iter = " << info.qp_solver_iter << std::endl;
    // std::cout << "status = " << info.status << std::endl;
    std::cout << "polympc time " << std::setprecision(9)
              << static_cast<double>(duration.count()) / NUM_EXP << " [microseconds]" << "\n";
    std::cout << "x = \n" << typename Opt_t<scalar_t>::x()(x).transpose() << std::endl;
    std::cout << "y = \n" << typename Opt_t<scalar_t>::y()(x).transpose() << std::endl;
    std::cout << "\n\n\n\n";
}

// template<typename scalar_t>
// void test_qpmad()
// {
//     using prob_t = typename Opt_t<scalar_t>::problem_t;
//     typename prob_t::variable_vec x;
//     x << 0, 0, 0;
//     typename prob_t::obj_hessian_mat H;
//     typename prob_t::obj_gradient_vec h;
//     typename prob_t::variable_vec lb;
//     typename prob_t::variable_vec ub;
//     typename prob_t::constraints_jacobian_mat A;
//     typename prob_t::constraints_vec Alb;
//     typename prob_t::constraints_vec Aub;

//     prob_t prob;
//     typename prob_t::param_t param;

//     prob.objective(param, x, h, H);

//     // lb <= c(x) <= ub
//     // lb <= A*(x - x0) + c(x0) <= ub
//     // lb + A*x0 - c(x) <= A*x <= ub + A*x0 - c(x0)
//     typename prob_t::constraints_vec con;
//     prob.constraints(param, x, con, A);

//     typename prob_t::constraints_vec nl_lb;
//     typename prob_t::constraints_vec nl_ub;
//     prob.constraints.get_bounds(param, nl_lb, nl_ub);
//     Alb = nl_lb + A*x - con;
//     Aub = nl_ub + A*x - con;

//     prob.variables.get_bounds(param, lb, ub);

//     typename qpmad::Solver solver;

//     const std::size_t NUM_EXP = 1;
//     typename qpmad::Solver::ReturnStatus status;
//     typename polympc::time_point start = polympc::get_time();
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
//     std::cout << "qpmad time " << std::setprecision(9)
//               << static_cast<double>(duration.count()) / NUM_EXP << " [microseconds]" << "\n";
//     std::cout << "x = \n" << typename Opt_t<scalar_t>::x()(x).transpose() << std::endl;
//     std::cout << "y = \n" << typename Opt_t<scalar_t>::y()(x).transpose() << std::endl;
//     std::cout << "\n\n\n\n";

//     // prob_t::lagrangian_t::eq_dual_vec   eq_dual {1.2};
//     // prob_t::lagrangian_t::ineq_dual_vec ineq_dual {2.3, 3.4};
//     // prob_t::lagrangian_t::var_dual_vec  var_dual {4.5, 5.6, 6.7};

//     // std::cout << prob << std::endl;
//     // std::cout << "\n\n\n";
//     // prob.print_linearization(std::cout, param, x, eq_dual, ineq_dual, var_dual);
// }



/***********************************************************
    Implementation
 ***********************************************************/
int main()
{
    using scalar_t = double;
    using opt_t = Opt_t<scalar_t>;
    using prob_t = typename opt_t::problem_t;
    using var_types = typename prob_t::var_types;

    var_types::variable_vec var = var_types::init_variable_vec();    
    std::cout << "type(var) == " << type_name<decltype(var)>() << std::endl;

    var_types::constraints_vec con = var_types::init_constraints_vec();    
    std::cout << "type(con) == " << type_name<decltype(con)>() << std::endl;

    var_types::constraints_jacobian_mat J = var_types::init_constraints_jacobian_mat();
    std::cout << "type(J) == " << type_name<decltype(J)>() << std::endl;

    prob_t prob;
    prob_t::param_t p;

    prob.constraints(p, var, con);

    var[0] = 1;
    var[1] = 2;
    var[2] = 3;

    std::cout << "opt_t::x::get(var, 0) = " << opt_t::x::get(var, 0) << std::endl;
    user_var_t<opt_t::x> x(var);
    std::cout << "x(var, 0) = " << x() << std::endl;

    std::cout << "con = " << con.transpose() << std::endl;
    std::cout << "J = \n" << J << std::endl;

    // test_qpmad<scalar_t>();
    // test_sqp<scalar_t>();
    // test_ipopt<scalar_t>();
}
