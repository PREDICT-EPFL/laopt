#ifndef LAOPT_BOUNDED_EXPR_HPP
#define LAOPT_BOUNDED_EXPR_HPP

#include "Eigen/Dense"
#include "constraint_expr.hpp"
#include "ineq_constraint_expr.hpp"
#include "expr_evaluator.hpp"

namespace laopt {

/**
 * A ConstraintExpr represents a bounded expression of the form lb <= expr <= ub.
 * lb and ub can not be expressions.
 */
template<typename DerivedLb, typename Derived, typename DerivedUb>
class BoundedExpr : public ConstraintExpr<BoundedExpr<DerivedLb, Derived, DerivedUb>>
{
public:
    const DerivedLb& lb;
    const Derived& expr;
    const DerivedUb& ub;

    static constexpr int n_inputs = Derived::n_inputs;
    static constexpr int n_outputs = Derived::n_outputs;
    using Scalar = typename Derived::Scalar;

    explicit BoundedExpr(const DerivedLb& lb, const Derived& expr, const DerivedUb& ub) : lb(lb), expr(expr), ub(ub) {}

    EIGEN_STRONG_INLINE const Eigen::Vector<int, Derived::n_inputs> indices() const
    {
        return expr.indices();
    }
};

template<typename DerivedLb, typename Derived, typename DerivedUb>
struct is_variable_constraint_expr<BoundedExpr<DerivedLb, IndexedVector<Derived>, DerivedUb>> : std::integral_constant<bool, true> {};

template<typename DerivedLb, typename Derived, typename DerivedUb>
typename std::enable_if<is_constant_non_expr<DerivedLb>::value &&
                        std::is_base_of<ExprBase<Derived>, Derived>::value &&
                        is_constant_non_expr<DerivedUb>::value,
                        BoundedExpr<DerivedLb, Derived, DerivedUb>>::type
operator<=(const IneqConstraintExpr<DerivedLb, Derived>& ineq, const DerivedUb& ub)
{
    return BoundedExpr<DerivedLb, Derived, DerivedUb>(ineq.lhs, ineq.rhs, ub);
}

template<typename DerivedLb, typename Derived, typename DerivedUb>
typename std::enable_if<is_constant_non_expr<DerivedLb>::value &&
                        std::is_base_of<ExprBase<Derived>, Derived>::value &&
                        is_constant_non_expr<DerivedUb>::value,
                        BoundedExpr<DerivedLb, Derived, DerivedUb>>::type
operator<=(const DerivedLb& lb, const IneqConstraintExpr<Derived, DerivedUb>& ineq)
{
    return BoundedExpr<DerivedLb, Derived, DerivedUb>(lb, ineq.lhs, ineq.rhs);
}

template<typename DerivedLb, typename Derived, typename DerivedUb>
typename std::enable_if<is_constant_non_expr<DerivedLb>::value &&
                        std::is_base_of<ExprBase<Derived>, Derived>::value &&
                        is_constant_non_expr<DerivedUb>::value,
                        BoundedExpr<DerivedLb, Derived, DerivedUb>>::type
operator>=(const IneqConstraintExpr<Derived, DerivedUb>& ineq, const DerivedLb& lb)
{
    return BoundedExpr<DerivedLb, Derived, DerivedUb>(lb, ineq.lhs, ineq.rhs);
}

template<typename DerivedLb, typename Derived, typename DerivedUb>
typename std::enable_if<is_constant_non_expr<DerivedLb>::value &&
                        std::is_base_of<ExprBase<Derived>, Derived>::value &&
                        is_constant_non_expr<DerivedUb>::value,
                        BoundedExpr<DerivedLb, Derived, DerivedUb>>::type
operator>=(const DerivedUb& ub, const IneqConstraintExpr<DerivedLb, Derived>& ineq)
{
    return BoundedExpr<DerivedLb, Derived, DerivedUb>(ineq.lhs, ineq.rhs, ub);
}

template<typename DerivedLhs, typename DerivedRhs>
typename std::enable_if<is_constant_non_expr<DerivedRhs>::value, BoundedExpr<DerivedRhs, DerivedLhs, DerivedRhs>>::type
operator==(const ExprBase<DerivedLhs>& lhs, const DerivedRhs& rhs)
{
    return BoundedExpr<DerivedRhs, DerivedLhs, DerivedRhs>(rhs, lhs.derived(), rhs);
}

template<typename DerivedLhs, typename DerivedRhs>
typename std::enable_if<is_constant_non_expr<DerivedLhs>::value, BoundedExpr<DerivedLhs, DerivedRhs, DerivedLhs>>::type
operator==(const DerivedLhs& lhs, const ExprBase<DerivedRhs>& rhs)
{
    return BoundedExpr<DerivedLhs, DerivedRhs, DerivedLhs>(lhs, rhs.derived(), lhs);
}

template<typename DerivedLb, typename Derived, typename DerivedUb>
struct ExprEvaluator<BoundedExpr<DerivedLb, Derived, DerivedUb>>
{
    static EIGEN_STRONG_INLINE auto
    function(const BoundedExpr<DerivedLb, Derived, DerivedUb>& bounded_expr)
    {
        return ExprEvaluator<Derived>::function(bounded_expr.expr);
    }

    template<typename OutJacobian>
    static EIGEN_STRONG_INLINE void
    jacobian(const BoundedExpr<DerivedLb, Derived, DerivedUb>& bounded_expr, OutJacobian&& out_jacobian)
    {
        ExprEvaluator<Derived>::jacobian(bounded_expr.expr, std::forward<OutJacobian>(out_jacobian));
    }

    template<typename Weight>
    static EIGEN_STRONG_INLINE auto
    wsum(const BoundedExpr<DerivedLb, Derived, DerivedUb>& bounded_expr, const Weight& weight)
    {
        return ExprEvaluator<Derived>::wsum(bounded_expr.expr, weight);
    }

    template<typename OutGradient, typename Weight>
    static EIGEN_STRONG_INLINE void
    gradient(const BoundedExpr<DerivedLb, Derived, DerivedUb>& bounded_expr, OutGradient&& out_gradient, const Weight& weight)
    {
        ExprEvaluator<Derived>::gradient(bounded_expr.expr, std::forward<OutGradient>(out_gradient), weight);
    }

    template<typename OutHessian, typename Weight>
    static EIGEN_STRONG_INLINE void
    hessian(const BoundedExpr<DerivedLb, Derived, DerivedUb>& bounded_expr, OutHessian&& out_hessian, const Weight& weight)
    {
        ExprEvaluator<Derived>::hessian(bounded_expr.expr, std::forward<OutHessian>(out_hessian), weight);
    }

    static EIGEN_STRONG_INLINE auto
    lower_bound(const BoundedExpr<DerivedLb, Derived, DerivedUb>& bounded_expr)
    {
        return bounded_expr.lb;
    }

    static EIGEN_STRONG_INLINE auto
    upper_bound(const BoundedExpr<DerivedLb, Derived, DerivedUb>& bounded_expr)
    {
        return bounded_expr.ub;
    }
};

} // namespace laopt

#endif //LAOPT_BOUNDED_EXPR_HPP
