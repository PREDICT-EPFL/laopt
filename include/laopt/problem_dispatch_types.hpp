#ifndef LAOPT_PROBLEM_DISPATCH_TYPES_HPP
#define LAOPT_PROBLEM_DISPATCH_TYPES_HPP

namespace laopt
{

/**
 * Tags to dispatch the relevant problem function evaluations
 */
struct Eval {};
struct Jacobian {};
struct Gradient {};
struct Hessian {};

} // namespace laopt

#endif // LAOPT_PROBLEM_DISPATCH_TYPES_HPP
