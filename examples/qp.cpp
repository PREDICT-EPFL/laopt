/**
 * Use polyMPC and LACompiler to solve a simple QP
 */

// #define SEG(len,offset) template segment<len>(offset)

#include <iostream>
#include "lampc.hpp"
#include "qp_functions.hpp"
#include "qp.compiled.hpp"

// #include "qp_lib.hpp"
#include <chrono>

#include "ipopt_interface.hpp"

/**
 * Problem structure
 * 
 * min sum w_i f_i(x)
 * s.t. g_lb <= g(x) <= g_ub
 *        lb <=   x  <=   ub
 */

struct QP_bnds : public QP
{
  static void variable_bounds(param_t &param, Eigen::Ref<variable_t> lb, Eigen::Ref<variable_t> ub)
  {
    QP::variable_bounds(param, lb, ub);

    std::cout << "Called\n";
    std::cout << "param.x0 = " << param.x0.transpose() << std::endl;

    x(lb,0) = param.x0;
    x(ub,0) = param.x0;

    std::cout << "lb = " << x(lb,0).transpose() << std::endl;
    std::cout << "ub = " << x(ub,0).transpose() << std::endl;
  }
};

        // std::cout << "Called\n";
        // std::cout << "param.x0 = " << param.x0.transpose() << std::endl;
        // auto lb = Eigen::Map<variable_t>(x_l);
        // auto ub = Eigen::Map<variable_t>(x_u);
        // std::cout << "lb = " << Prob::x(lb,0).transpose() << std::endl;
        // std::cout << "ub = " << Prob::x(ub,0).transpose() << std::endl;

        // Prob::x(lb,0) = param.x0;
        // Prob::x(ub,0) = param.x0;

        // std::cout << "lb = " << Prob::x(lb,0).transpose() << std::endl;
        // std::cout << "ub = " << Prob::x(ub,0).transpose() << std::endl;



int main()
{

    using scalar_t = double;
    Eigen::Vector<scalar_t, 2> xp;
    Eigen::Vector<scalar_t, 2> x;
    Eigen::Vector<scalar_t, 1> u;
    MyFunctions<scalar_t>::param_t p;

    x.array() = 1;
    u.array() = 2;
    MyFunctions<scalar_t>::dynamics::template impl<scalar_t>(p, xp, x, u);

    std::cout << "xp = " << xp.transpose() << std::endl;

    auto E = MyFunctions<scalar_t>::dynamics::eval(p, x, u);
    auto J = MyFunctions<scalar_t>::dynamics::jac(p, x, u);

    std::cout << "J.val = " << J.val << std::endl;
    std::cout << "J.jacobian = \n" << J.jacobian << std::endl;

    // QP_bnds::variable_t x;
    // x.array() = 1.5;

    // QP_bnds::param_t param;
    // QP_bnds::constraints::out_t eq;
    // QP_bnds::constraints::jacobian_t J;

    // QP_bnds::xss(x).array() = 3;
    // QP_bnds::uss(x).array() = 3;

    // // Compute jacobian
    // QP_bnds::constraints::initialize_jacobian(J);
    // QP_bnds::constraints::eval(param, x, eq, J);

    // std::cout << "eq = " << eq.transpose() << std::endl;
    // std::cout << "J = \n" << Eigen::MatrixX<double>(J) << std::endl;

    // QP_bnds::objective::weight_t w;
    // w.array() = 1.2;
    // std::cout << "objective = " << QP_bnds::objective::eval(param, w, x) << std::endl;

    // QP_bnds::objective::gradient_t grad;
    // std::cout << "objective = " << QP_bnds::objective::eval(param, w, x, grad) << std::endl;
    // std::cout << "gradient  = " << grad.transpose() << std::endl;

    // QP_bnds::objective::hessian_t H;
    // QP_bnds::objective::initialize_hessian(H);
    // std::cout << "H = \n" << Eigen::MatrixX<double>(H) << std::endl;
    // std::cout << "objective = " << QP_bnds::objective::eval(param, w, x, grad, H) << std::endl;
    // std::cout << "H = \n" << Eigen::MatrixX<double>(H) << std::endl;


    // auto start = std::chrono::steady_clock::now();
    // long N = 100000;
    // for(int i=0; i<N; i++)
    //     QP_bnds::constraints::eval(param, x, eq, J);
    // auto end = std::chrono::steady_clock::now();
    // std::cout << "Jacobian time in nanoseconds: "
    //     << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / (double)N
    //     << " ns" << std::endl;

    // start = std::chrono::steady_clock::now();
    // for(int i=0; i<N; i++)
    //     QP_bnds::objective::eval(param, w, x, grad);
    // end = std::chrono::steady_clock::now();
    // std::cout << "Gradient time in nanoseconds: "
    //     << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / (double)N
    //     << " ns" << std::endl;

    // start = std::chrono::steady_clock::now();
    // for(int i=0; i<N; i++)
    //     QP_bnds::objective::eval(param, w, x, grad, H);
    // end = std::chrono::steady_clock::now();
    // std::cout << "Hessian time in nanoseconds: "
    //     << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / (double)N
    //     << " ns" << std::endl;


    // using myNLP = NLP_Ipopt<QP_bnds>;
    // param.x0[0] = 0;
    // param.x0[1] = 1;

    // Ipopt::SmartPtr<myNLP> mynlp = new myNLP(param);
    // Ipopt::SmartPtr<Ipopt::IpoptApplication> app = new Ipopt::IpoptApplication();

    // app->Options()->SetNumericValue("tol", 1e-7);
    // // app->Options()->SetStringValue("mu_strategy", "adaptive");
    // app->Options()->SetStringValue("output_file", "ipopt.out");
    // // app->Options()->SetStringValue("hessian_approximation", "limited-memory");

    // Ipopt::ApplicationReturnStatus status;
    // status = app->Initialize();
    // if( status != Ipopt::Solve_Succeeded )
    // {
    //     std::cout << std::endl << std::endl << "*** Error during initialization!" << std::endl;
    //     return -1;
    // }

    // const std::size_t NUM_EXP = 1;
    // start = std::chrono::steady_clock::now();
    // for(int i = 0; i < NUM_EXP; ++i)
    // {
    //     mynlp->x0.array() = 0;
    //     status = app->OptimizeTNLP(mynlp);
    // }
    // end = std::chrono::steady_clock::now();

    // std::cout << "\n\n\n\n===================== IPOPT SOLUTION =====================\n";
    // std::cout << "IPOPT time: "
    //     << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
    //     << " ms" << std::endl;
    // std::cout << "x = \n" << QP_bnds::x(mynlp->sol.primal).transpose() << std::endl;
    // std::cout << "u = \n" << QP_bnds::u(mynlp->sol.primal).transpose() << std::endl;
    // std::cout << "\n\n\n\n";

    return 1;
}
