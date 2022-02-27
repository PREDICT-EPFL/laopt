#ifndef QP_LIB_HPP
#define QP_LIB_HPP

#include "lampc.hpp"

// #define SEG(len,offset) template segment<len>(offset)

#include "qp_functions.hpp"
#include "qp.compiled.hpp"

void solve_qp(QP::param_t param, 
              QP::equalities::variable_t &x, // Input
              QP::equalities::out_t &eq, // Output
              QP::equalities::jacobian_t &J);

#endif