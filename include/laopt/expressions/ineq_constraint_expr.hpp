#ifndef LAOPT_INEQ_CONSTRAINT_EXPR_HPP
#define LAOPT_INEQ_CONSTRAINT_EXPR_HPP

#include "Eigen/Dense"
#include "constraint_expr.hpp"
#include "expr_evaluator.hpp"
#include "sub_expr.hpp"

namespace laopt {

/**
 * A IneqConstraintExpr represents a constraint of the form lhs <= rhs.
 */
template<typename DerivedLhs, typename DerivedRhs>
class IneqConstraintExpr : public ConstraintExpr<IneqConstraintExpr<DerivedLhs, DerivedRhs>>
{
public:
    const DerivedLhs& lhs;
    const DerivedRhs& rhs;

    using ExprType = typename std::conditional<std::is_base_of<ExprBase<DerivedLhs>, DerivedLhs>::value && std::is_base_of<ExprBase<DerivedRhs>, DerivedRhs>::value,
                                               SubExpr<DerivedLhs, DerivedRhs>,
                                               typename std::conditional<std::is_base_of<ExprBase<DerivedLhs>, DerivedLhs>::value,
                                                                         DerivedLhs,
                                                                         DerivedRhs>::type>::type;
    static constexpr int n_inputs = ExprType::n_inputs;
    static constexpr int n_outputs = ExprType::n_outputs;
    using Scalar = typename ExprType::Scalar;

    explicit IneqConstraintExpr(const DerivedLhs& lhs, const DerivedRhs& rhs) : lhs(lhs), rhs(rhs) {}

    template<typename DerivedLhs_, typename DerivedRhs_>
    EIGEN_STRONG_INLINE const typename std::enable_if<is_constant_non_expr<DerivedRhs_>::value, Eigen::Vector<int, n_inputs>>::type
    indices_impl(const ExprBase<DerivedLhs_>& lhs_, const DerivedRhs_& rhs_) const
    {
        return lhs_.derived().indices();
    }

    template<typename DerivedLhs_, typename DerivedRhs_>
    EIGEN_STRONG_INLINE const typename std::enable_if<is_constant_non_expr<DerivedLhs_>::value, Eigen::Vector<int, n_inputs>>::type
    indices_impl(const DerivedLhs_& lhs_, const ExprBase<DerivedRhs_>& rhs_) const
    {
        return rhs_.derived().indices();
    }

    template<typename DerivedLhs_, typename DerivedRhs_>
    EIGEN_STRONG_INLINE const Eigen::Vector<int, n_inputs>
    indices_impl(const ExprBase<DerivedLhs_>& lhs_, const ExprBase<DerivedRhs_>& rhs_) const
    {
        return SubExpr<DerivedLhs_, DerivedRhs_>(lhs_.derived(), rhs_.derived()).indices();
    }

    EIGEN_STRONG_INLINE const Eigen::Vector<int, n_inputs> indices() const
    {
        return indices_impl(lhs, rhs);
    }
};

template<typename DerivedLhs, typename DerivedRhs>
struct is_variable_constraint_expr<IneqConstraintExpr<IndexedVector<DerivedLhs>, DerivedRhs>> : std::integral_constant<bool, is_constant_non_expr<DerivedRhs>::value> {};

template<typename DerivedLhs, typename DerivedRhs>
struct is_variable_constraint_expr<IneqConstraintExpr<DerivedLhs, IndexedVector<DerivedRhs>>> : std::integral_constant<bool, is_constant_non_expr<DerivedLhs>::value> {};

template<typename DerivedLhs, typename DerivedRhs>
IneqConstraintExpr<DerivedLhs, DerivedRhs>
operator<=(const ExprBase<DerivedLhs>& lhs, const ExprBase<DerivedRhs>& rhs)
{
    return IneqConstraintExpr<DerivedLhs, DerivedRhs>(lhs.derived(), rhs.derived());
}

template<typename DerivedLhs, typename DerivedRhs>
typename std::enable_if<is_constant_non_expr<DerivedRhs>::value, IneqConstraintExpr<DerivedLhs, DerivedRhs>>::type
operator<=(const ExprBase<DerivedLhs>& lhs, const DerivedRhs& rhs)
{
    return IneqConstraintExpr<DerivedLhs, DerivedRhs>(lhs.derived(), rhs);
}

template<typename DerivedLhs, typename DerivedRhs>
typename std::enable_if<is_constant_non_expr<DerivedLhs>::value, IneqConstraintExpr<DerivedLhs, DerivedRhs>>::type
operator<=(const DerivedLhs& lhs, const ExprBase<DerivedRhs>& rhs)
{
    return IneqConstraintExpr<DerivedLhs, DerivedRhs>(lhs, rhs.derived());
}

template<typename DerivedLhs, typename DerivedRhs>
IneqConstraintExpr<DerivedRhs, DerivedLhs>
operator>=(const ExprBase<DerivedLhs>& lhs, const ExprBase<DerivedRhs>& rhs)
{
    return IneqConstraintExpr<DerivedRhs, DerivedLhs>(rhs.derived(), lhs.derived());
}

template<typename DerivedLhs, typename DerivedRhs>
typename std::enable_if<is_constant_non_expr<DerivedRhs>::value, IneqConstraintExpr<DerivedRhs, DerivedLhs>>::type
operator>=(const ExprBase<DerivedLhs>& lhs, const DerivedRhs& rhs)
{
    return IneqConstraintExpr<DerivedRhs, DerivedLhs>(rhs, lhs.derived());
}

template<typename DerivedLhs, typename DerivedRhs>
typename std::enable_if<is_constant_non_expr<DerivedLhs>::value, IneqConstraintExpr<DerivedRhs, DerivedLhs>>::type
operator>=(const DerivedLhs& lhs, const ExprBase<DerivedRhs>& rhs)
{
    return IneqConstraintExpr<DerivedRhs, DerivedLhs>(rhs.derived(), lhs);
}

template<typename DerivedLhs, typename DerivedRhs>
struct ExprEvaluator<IneqConstraintExpr<DerivedLhs, DerivedRhs>,
                     typename std::enable_if<std::is_base_of<ExprBase<DerivedLhs>, DerivedLhs>::value &&
                                             is_constant_non_expr<DerivedRhs>::value>::type>
{
    static EIGEN_STRONG_INLINE auto
    function(const IneqConstraintExpr<DerivedLhs, DerivedRhs>& ineq)
    {
        return ExprEvaluator<DerivedLhs>::function(ineq.lhs);
    }

    template<typename OutJacobian>
    static EIGEN_STRONG_INLINE void
    jacobian(const IneqConstraintExpr<DerivedLhs, DerivedRhs>& ineq, OutJacobian&& out_jacobian)
    {
        ExprEvaluator<DerivedLhs>::jacobian(ineq.lhs, std::forward<OutJacobian>(out_jacobian));
    }

    template<typename Weight>
    static EIGEN_STRONG_INLINE auto
    wsum(const IneqConstraintExpr<DerivedLhs, DerivedRhs>& ineq, const Weight& weight)
    {
        return ExprEvaluator<DerivedLhs>::wsum(ineq.lhs, weight);
    }

    template<typename OutGradient, typename Weight>
    static EIGEN_STRONG_INLINE void
    gradient(const IneqConstraintExpr<DerivedLhs, DerivedRhs>& ineq, OutGradient&& out_gradient, const Weight& weight)
    {
        ExprEvaluator<DerivedLhs>::gradient(ineq.lhs, std::forward<OutGradient>(out_gradient), weight);
    }

    template<typename OutHessian, typename Weight>
    static EIGEN_STRONG_INLINE void
    hessian(const IneqConstraintExpr<DerivedLhs, DerivedRhs>& ineq, OutHessian&& out_hessian, const Weight& weight)
    {
        ExprEvaluator<DerivedLhs>::hessian(ineq.lhs, std::forward<OutHessian>(out_hessian), weight);
    }

    static EIGEN_STRONG_INLINE auto
    lower_bound(const IneqConstraintExpr<DerivedLhs, DerivedRhs>& ineq)
    {
        return -std::numeric_limits<typename DerivedLhs::Scalar>::infinity();
    }

    static EIGEN_STRONG_INLINE auto
    upper_bound(const IneqConstraintExpr<DerivedLhs, DerivedRhs>& ineq)
    {
        return ineq.rhs;
    }
};

template<typename DerivedLhs, typename DerivedRhs>
struct ExprEvaluator<IneqConstraintExpr<DerivedLhs, DerivedRhs>,
                     typename std::enable_if<std::is_base_of<ExprBase<DerivedRhs>, DerivedRhs>::value &&
                                             is_constant_non_expr<DerivedLhs>::value>::type>
{
    static EIGEN_STRONG_INLINE auto
    function(const IneqConstraintExpr<DerivedLhs, DerivedRhs>& ineq)
    {
        return ExprEvaluator<DerivedRhs>::function(ineq.rhs);
    }

    template<typename OutJacobian>
    static EIGEN_STRONG_INLINE void
    jacobian(const IneqConstraintExpr<DerivedLhs, DerivedRhs>& ineq, OutJacobian&& out_jacobian)
    {
        ExprEvaluator<DerivedRhs>::jacobian(ineq.rhs, std::forward<OutJacobian>(out_jacobian));
    }

    template<typename Weight>
    static EIGEN_STRONG_INLINE auto
    wsum(const IneqConstraintExpr<DerivedLhs, DerivedRhs>& ineq, const Weight& weight)
    {
        return ExprEvaluator<DerivedRhs>::wsum(ineq.rhs, weight);
    }

    template<typename OutGradient, typename Weight>
    static EIGEN_STRONG_INLINE void
    gradient(const IneqConstraintExpr<DerivedLhs, DerivedRhs>& ineq, OutGradient&& out_gradient, const Weight& weight)
    {
        ExprEvaluator<DerivedRhs>::gradient(ineq.rhs, std::forward<OutGradient>(out_gradient), weight);
    }

    template<typename OutHessian, typename Weight>
    static EIGEN_STRONG_INLINE void
    hessian(const IneqConstraintExpr<DerivedLhs, DerivedRhs>& ineq, OutHessian&& out_hessian, const Weight& weight)
    {
        ExprEvaluator<DerivedRhs>::hessian(ineq.rhs, std::forward<OutHessian>(out_hessian), weight);
    }

    static EIGEN_STRONG_INLINE auto
    lower_bound(const IneqConstraintExpr<DerivedLhs, DerivedRhs>& ineq)
    {
        return ineq.lhs;
    }

    static EIGEN_STRONG_INLINE auto
    upper_bound(const IneqConstraintExpr<DerivedLhs, DerivedRhs>& ineq)
    {
        return std::numeric_limits<typename DerivedRhs::Scalar>::infinity();
    }
};

template<typename DerivedLhs, typename DerivedRhs>
struct ExprEvaluator<IneqConstraintExpr<DerivedLhs, DerivedRhs>,
                     typename std::enable_if<std::is_base_of<ExprBase<DerivedLhs>, DerivedLhs>::value &&
                                             std::is_base_of<ExprBase<DerivedRhs>, DerivedRhs>::value>::type>
{
    static EIGEN_STRONG_INLINE auto
    function(const IneqConstraintExpr<DerivedLhs, DerivedRhs>& ineq)
    {
        SubExpr<DerivedLhs, DerivedRhs> sub_expr(ineq.lhs, ineq.rhs);
        return ExprEvaluator<SubExpr<DerivedLhs, DerivedRhs>>::function(sub_expr);
    }

    template<typename OutJacobian>
    static EIGEN_STRONG_INLINE void
    jacobian(const IneqConstraintExpr<DerivedLhs, DerivedRhs>& ineq, OutJacobian&& out_jacobian)
    {
        SubExpr<DerivedLhs, DerivedRhs> sub_expr(ineq.lhs, ineq.rhs);
        ExprEvaluator<SubExpr<DerivedLhs, DerivedRhs>>::jacobian(sub_expr, std::forward<OutJacobian>(out_jacobian));
    }

    template<typename Weight>
    static EIGEN_STRONG_INLINE auto
    wsum(const IneqConstraintExpr<DerivedLhs, DerivedRhs>& ineq, const Weight& weight)
    {
        SubExpr<DerivedLhs, DerivedRhs> sub_expr(ineq.lhs, ineq.rhs);
        return ExprEvaluator<SubExpr<DerivedLhs, DerivedRhs>>::wsum(sub_expr, weight);
    }

    template<typename OutGradient, typename Weight>
    static EIGEN_STRONG_INLINE void
    gradient(const IneqConstraintExpr<DerivedLhs, DerivedRhs>& ineq, OutGradient&& out_gradient, const Weight& weight)
    {
        SubExpr<DerivedLhs, DerivedRhs> sub_expr(ineq.lhs, ineq.rhs);
        ExprEvaluator<SubExpr<DerivedLhs, DerivedRhs>>::gradient(sub_expr, std::forward<OutGradient>(out_gradient), weight);
    }

    template<typename OutHessian, typename Weight>
    static EIGEN_STRONG_INLINE void
    hessian(const IneqConstraintExpr<DerivedLhs, DerivedRhs>& ineq, OutHessian&& out_hessian, const Weight& weight)
    {
        SubExpr<DerivedLhs, DerivedRhs> sub_expr(ineq.lhs, ineq.rhs);
        ExprEvaluator<SubExpr<DerivedLhs, DerivedRhs>>::hessian(sub_expr, std::forward<OutHessian>(out_hessian), weight);
    }

    static EIGEN_STRONG_INLINE auto
    lower_bound(const IneqConstraintExpr<DerivedLhs, DerivedRhs>& ineq)
    {
        return -std::numeric_limits<typename DerivedLhs::Scalar>::infinity();
    }

    static EIGEN_STRONG_INLINE auto
    upper_bound(const IneqConstraintExpr<DerivedLhs, DerivedRhs>& ineq)
    {
        return 0;
    }
};

} // namespace laopt

#endif // LAOPT_INEQ_CONSTRAINT_EXPR_HPP
