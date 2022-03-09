/**
 * Use polyMPC and LACompiler to solve a simple QP
 */
#include "qp_lib.hpp"

void solve_qp(QP::param_t &param, 
              QP::variable_t &x, // Input
              QP::equalities::out_t &eq, // Output
              QP::equalities::jacobian_t &J)
{
    QP::equalities::eval(param, x, eq, J);
}

void solve_hessian(QP::param_t &param, 
              QP::variable_t &x, // Input
              QP::equalities::out_t &eq, // Output
              QP::equalities::jacobian_t &J,
              QP::equalities::hessian_t &H)
{
    // QP::equalities::eval(param, x, eq, J, H);
}


void init_qp(QP::equalities::jacobian_t &J)
{
    QP::equalities::initialize_jacobian(J);
}
