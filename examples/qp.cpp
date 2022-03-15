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

// struct Prob : QP
// {
//   using param_t = QP::param_t;
//   using scalar_t = QP::scalar_t;
//   using variable_t = QP::variable_t;

//   using constraint_t = typename Prob::constraints_vec;
//   using constraint_jacobian_t = typename Prob::constraints_jacobian_mat;
//   using obj_gradient_t = typename Prob::obj_gradient_vec;
//   using obj_hessian_t = typename Prob::obj_hessian_mat;
//   using obj_t = typename Prob::obj_vec;
// };


/**
 * Problem structure
 * 
 * min sum w_i f_i(x)
 * s.t. g_lb <= g(x) <= g_ub
 *        lb <=   x  <=   ub
 */


int main()
{
    QP::variable_t x;
    x.array() = 1.5;

    QP::param_t param;
    QP::constraints::out_t eq;
    QP::constraints::jacobian_t J;

    QP::xss(x).array() = 3;
    QP::uss(x).array() = 3;

    // Compute jacobian
    QP::constraints::initialize_jacobian(J);
    QP::constraints::eval(param, x, eq, J);

    std::cout << "eq = " << eq.transpose() << std::endl;
    std::cout << "J = \n" << Eigen::MatrixX<double>(J) << std::endl;

    QP::objective::weight_t w;
    w.array() = 1.2;
    std::cout << "objective = " << QP::objective::eval(param, w, x) << std::endl;

    QP::objective::gradient_t grad;
    std::cout << "objective = " << QP::objective::eval(param, w, x, grad) << std::endl;
    std::cout << "gradient  = " << grad.transpose() << std::endl;

    QP::objective::hessian_t H;
    QP::objective::initialize_hessian(H);
    std::cout << "H = \n" << Eigen::MatrixX<double>(H) << std::endl;
    std::cout << "objective = " << QP::objective::eval(param, w, x, grad, H) << std::endl;
    std::cout << "H = \n" << Eigen::MatrixX<double>(H) << std::endl;


    auto start = std::chrono::steady_clock::now();
    long N = 1000000;
    for(int i=0; i<N; i++)
        QP::constraints::eval(param, x, eq, J);
    auto end = std::chrono::steady_clock::now();
    std::cout << "Jacobian time in nanoseconds: "
        << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / (double)N
        << " ns" << std::endl;

    for(int i=0; i<N; i++)
        QP::objective::eval(param, w, x, grad);
    end = std::chrono::steady_clock::now();
    std::cout << "Gradient time in nanoseconds: "
        << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / (double)N
        << " ns" << std::endl;

    for(int i=0; i<N; i++)
        QP::objective::eval(param, w, x, grad, H);
    end = std::chrono::steady_clock::now();
    std::cout << "Hessian time in nanoseconds: "
        << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / (double)N
        << " ns" << std::endl;

    return 1;
}
