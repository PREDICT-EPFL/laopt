/**
 * Use polyMPC and LACompiler to solve a simple QP
 */
// #include "lampc.hpp"

// #define SEG(len,offset) template segment<len>(offset)

// #include "qp_functions.hpp"
// #include "qp.compiled.hpp"

#include "qp_lib.hpp"

int main()
{
    QP::variable_t x;
    x.array() = 1;

    QP::param_t param;
    QP::equalities::out_t eq;
    QP::equalities::jacobian_t J;

    solve_qp(param, x, eq, J);

    std::cout << "eq = " << eq.transpose() << std::endl;

    param.x0[0] = 100;
    x.array() = 0;
    solve_qp(param, x, eq, J);
    std::cout << "eq = " << eq.transpose() << std::endl;

    std::cout << "J = \n" << Eigen::MatrixX<double>(J) << std::endl;

    QP::objective::weight_t w;
    w.array() = 1;
    x.array() = 1;
    std::cout << "objective = " << QP::objective::eval(param, w, x) << std::endl;

    QP::variable_t z;
    z.array() = 0;
    QP::x(z, 2)[0] = 1;
    QP::x(z, 2)[1] = 2;

    std::cout << "z = " << z.transpose() << std::endl;
    std::cout << "QP::x = \n" << QP::x(z) << std::endl;

    QP::objective::gradient_t grad;
    w.array() = 1;
    x.array() = 1;
    QP::xss(x).array() = 10;
    QP::uss(x).array() = 10;
    std::cout << "objective = " << QP::objective::eval(param, w, x, grad) << std::endl;
    std::cout << "gradient  = " << grad.transpose() << std::endl;

    std::cout << "x = \n" << QP::x(x) << std::endl;
    std::cout << "u = \n" << QP::u(x) << std::endl;
    std::cout << "xss = \n" << QP::xss(x) << std::endl;
    std::cout << "uss = \n" << QP::uss(x) << std::endl;

    // QP::equalities::hessian_t H;
    // QP::equalities::initialize_hessian(H);
    // std::cout << "H = \n" << Eigen::MatrixX<double>(H) << std::endl;

    return 1;
}
