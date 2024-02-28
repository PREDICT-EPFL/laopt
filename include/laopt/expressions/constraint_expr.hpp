#ifndef LAOPT_CONSTRAINT_EXPR_HPP
#define LAOPT_CONSTRAINT_EXPR_HPP

#include <Eigen/Dense>
#include "laopt/expressions/expr_base.hpp"

namespace laopt {

/**
 * This helper struct has a member constant value equal to true if T is arithmetic, a Eigen matrix and not an ExprBase.
 */
template<typename T>
struct is_constant_non_expr : std::integral_constant<bool,
                                                     (std::is_arithmetic<T>::value ||
                                                      std::is_base_of<Eigen::MatrixBase<T>, T>::value) &&
                                                     !std::is_base_of<ExprBase<T>, T>::value> {};

/**
 * This helper struct has a member constant value equal to true if T is an constraint expression
 * with simple IndexedVector's as expressions, i.e., it's simple variable ineq or eq constraints.
 */
template<typename T>
struct is_variable_constraint_expr : std::integral_constant<bool, false> {};

/**
 * ConstraintExpr is the base expression type of constraint expressions.
 */
template<typename Derived>
class ConstraintExpr
{
public:
    Derived& derived()
    {
        return static_cast<Derived&>(*this);
    }

    const Derived& derived() const
    {
        return static_cast<const Derived&>(*this);
    }
};

} // namespace laopt

#endif //LAOPT_CONSTRAINT_EXPR_HPP
