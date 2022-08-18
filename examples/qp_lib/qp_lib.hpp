#ifndef QP_LIB_HPP
#define QP_LIB_HPP

#include "lampc.hpp"

#include "qp_functions.hpp"
#include "qp.compiled.hpp"

void solve_qp(QP::param_t &param, 
              QP::variable_t &x, // Input
              QP::equalities::out_t &eq, // Output
              QP::equalities::jacobian_t &J);
void solve_hessian(QP::param_t &param, 
              QP::variable_t &x, // Input
              QP::equalities::out_t &eq, // Output
              QP::equalities::jacobian_t &J,
              QP::equalities::hessian_t &H);
void init_qp(QP::equalities::jacobian_t &J);
#endif