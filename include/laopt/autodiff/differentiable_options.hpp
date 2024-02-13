#ifndef LAOPT_DIFFERENTIABLE_OPTIONS_HPP
#define LAOPT_DIFFERENTIABLE_OPTIONS_HPP

namespace laopt
{

enum DiffOptions
{
    TAGLESS         = 0x00,
    TAGGED          = 0x01,
    EIGEN_ALL       = 0x00,
    CASADI_JACOBIAN = 0x02,
    CASADI_HESSIAN  = 0x04,
    CASADI_ALL      = CASADI_JACOBIAN | CASADI_HESSIAN,
};

struct DefaultTag {};

} // namespace laopt

#endif //LAOPT_DIFFERENTIABLE_OPTIONS_HPP
