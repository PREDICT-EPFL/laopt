/**
 * Use polyMPC and LACompiler to solve a simple QP
 */
#include "qp_lib.hpp"

void solve_qp(QP::param_t param, 
              QP::equalities::variable_t &x, // Input
              QP::equalities::out_t &eq, // Output
              QP::equalities::jacobian_t &J)
{
    QP::equalities::eval(param, x, eq);
    QP::equalities::initialize_jacobian(J);
    QP::equalities::eval(param, x, eq, J);
}
