/**
 * Use polyMPC and LACompiler to solve a simple QP
 */
#include "lampc.hpp"

#include "qp_functions.hpp"
#include "qp.compiled.hpp"

int main()
{
    QP::variables_t x;
    x.array() = 0;

    QP::param_t param;

    QP::equalities_t eq;
    QP::equalities_t::output_t con_eq;

    eq.eval_jacobian(param, x, con_eq);
    std::cout << "con_eq = " << con_eq.transpose() << std::endl;
    std::cout << "eq.jacobian = \n" << Eigen::MatrixX<double>(eq.jacobian) << std::endl;

    return 1;
}
