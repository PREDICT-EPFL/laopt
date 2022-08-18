#ifndef LAOPT_EQ_CONSTRAINT_EXPR_HPP
#define LAOPT_EQ_CONSTRAINT_EXPR_HPP

#include "Eigen/Dense"
#include "constraint_expr.hpp"
#include "expr_evaluator.hpp"
#include "sub_expr.hpp"
#include "../indexed_vector.hpp"

namespace laopt {
/**
 * A EqConstraintExpr represents a constraint of the form lhs == rhs.
 */
template<typename DerivedLhs, typename DerivedRhs>
class EqConstraintExpr : public ConstraintExpr<EqConstraintExpr<DerivedLhs, DerivedRhs>>
{
public:
    const DerivedLhs& lhs;
    const DerivedRhs& rhs;

    using ExprType = typename std::conditional<std::is_base_of<BaseExpr<DerivedLhs>, DerivedLhs>::value && std::is_base_of<BaseExpr<DerivedRhs>, DerivedRhs>::value,
                                               SubExpr<DerivedLhs, DerivedRhs>,
                                               typename std::conditional<std::is_base_of<BaseExpr<DerivedLhs>, DerivedLhs>::value,
                                                                         DerivedLhs,
                                                                         DerivedRhs>::type>::type;
    static constexpr int n_inputs = ExprType::n_inputs;
    static constexpr int n_outputs = ExprType::n_outputs;
    using Scalar = typename ExprType::Scalar;

    explicit EqConstraintExpr(const DerivedLhs& lhs, const DerivedRhs& rhs) : lhs(lhs), rhs(rhs) {}

    template<typename DerivedLhs_, typename DerivedRhs_>
    EIGEN_STRONG_INLINE const typename std::enable_if<is_constant_non_expr<DerivedRhs_>::value, Eigen::Vector<int, n_inputs>>::type
    indices_impl(const BaseExpr<DerivedLhs_>& lhs_, const DerivedRhs_& rhs_) const
    {
        return lhs_.derived().indices();
    }

    template<typename DerivedLhs_, typename DerivedRhs_>
    EIGEN_STRONG_INLINE const typename std::enable_if<is_constant_non_expr<DerivedLhs_>::value, Eigen::Vector<int, n_inputs>>::type
    indices_impl(const DerivedLhs_& lhs_, const BaseExpr<DerivedRhs_>& rhs_) const
    {
        return rhs_.derived().indices();
    }

    template<typename DerivedLhs_, typename DerivedRhs_>
    EIGEN_STRONG_INLINE const Eigen::Vector<int, n_inputs>
    indices_impl(const BaseExpr<DerivedLhs_>& lhs_, const BaseExpr<DerivedRhs_>& rhs_) const
    {
        return SubExpr<DerivedLhs_, DerivedRhs_>(lhs_.derived(), rhs_.derived()).indices();
    }

    EIGEN_STRONG_INLINE const Eigen::Vector<int, n_inputs> indices() const
    {
        return indices_impl(lhs, rhs);
    }
};

template<typename DerivedLhs, typename DerivedRhs>
EqConstraintExpr<DerivedLhs, DerivedRhs>
operator==(const BaseExpr<DerivedLhs>& lhs, const BaseExpr<DerivedRhs>& rhs)
{
    return EqConstraintExpr<DerivedLhs, DerivedRhs>(lhs.derived(), rhs.derived());
}

// we need this special case to be not ambiguous with Eigen
template<typename DerivedLhs, typename DerivedRhs>
EqConstraintExpr<IndexedVector<DerivedLhs>, IndexedVector<DerivedRhs>>
operator==(const IndexedVector<DerivedLhs>& lhs, const IndexedVector<DerivedRhs>& rhs)
{
    return EqConstraintExpr<IndexedVector<DerivedLhs>, IndexedVector<DerivedRhs>>(lhs, rhs);
}

template<typename DerivedLhs, typename DerivedRhs>
struct ExprEvaluator<EqConstraintExpr<DerivedLhs, DerivedRhs>,
                     typename std::enable_if<std::is_base_of<BaseExpr<DerivedLhs>, DerivedLhs>::value &&
                                             is_constant_non_expr<DerivedRhs>::value>::type>
{
    static EIGEN_STRONG_INLINE auto
    function(const EqConstraintExpr<DerivedLhs, DerivedRhs>& eq)
    {
        return ExprEvaluator<DerivedLhs>::function(eq.lhs);
    }

    template<typename OutValue, typename OutJacobian>
    static EIGEN_STRONG_INLINE void
    jacobian(const EqConstraintExpr<DerivedLhs, DerivedRhs>& eq, OutValue&& out_value, OutJacobian&& out_jacobian)
    {
        ExprEvaluator<DerivedLhs>::jacobian(eq.lhs, std::forward<OutValue>(out_value), std::forward<OutJacobian>(out_jacobian));
    }

    template<typename Weight>
    static EIGEN_STRONG_INLINE auto
    wsum(const EqConstraintExpr<DerivedLhs, DerivedRhs>& eq, const Weight& weight)
    {
        return ExprEvaluator<DerivedLhs>::wsum(eq.lhs, weight);
    }

    template<typename OutGradient, typename Weight>
    static EIGEN_STRONG_INLINE auto
    gradient(const EqConstraintExpr<DerivedLhs, DerivedRhs>& eq, OutGradient&& out_gradient, const Weight& weight)
    {
        return ExprEvaluator<DerivedLhs>::gradient(eq.lhs, std::forward<OutGradient>(out_gradient), weight);
    }

    template<typename OutGradient, typename OutHessian, typename Weight>
    static EIGEN_STRONG_INLINE auto
    hessian(const EqConstraintExpr<DerivedLhs, DerivedRhs>& eq, OutGradient&& out_gradient, OutHessian&& out_hessian, const Weight& weight)
    {
        return ExprEvaluator<DerivedLhs>::hessian(eq.lhs, std::forward<OutGradient>(out_gradient), std::forward<OutHessian>(out_hessian), weight);
    }

    static EIGEN_STRONG_INLINE auto
    lower_bound(const EqConstraintExpr<DerivedLhs, DerivedRhs>& eq)
    {
        return eq.rhs;
    }

    static EIGEN_STRONG_INLINE auto
    upper_bound(const EqConstraintExpr<DerivedLhs, DerivedRhs>& eq)
    {
        return eq.rhs;
    }
};

template<typename DerivedLhs, typename DerivedRhs>
struct ExprEvaluator<EqConstraintExpr<DerivedLhs, DerivedRhs>,
                     typename std::enable_if<std::is_base_of<BaseExpr<DerivedRhs>, DerivedRhs>::value &&
                                             is_constant_non_expr<DerivedLhs>::value>::type>
{
    static EIGEN_STRONG_INLINE auto
    function(const EqConstraintExpr<DerivedLhs, DerivedRhs>& eq)
    {
        return ExprEvaluator<DerivedLhs>::function(eq.rhs);
    }

    template<typename OutValue, typename OutJacobian>
    static EIGEN_STRONG_INLINE void
    jacobian(const EqConstraintExpr<DerivedLhs, DerivedRhs>& eq, OutValue&& out_value, OutJacobian&& out_jacobian)
    {
        ExprEvaluator<DerivedLhs>::jacobian(eq.rhs, std::forward<OutValue>(out_value), std::forward<OutJacobian>(out_jacobian));
    }

    template<typename Weight>
    static EIGEN_STRONG_INLINE auto
    wsum(const EqConstraintExpr<DerivedLhs, DerivedRhs>& eq, const Weight& weight)
    {
        return ExprEvaluator<DerivedLhs>::wsum(eq.rhs, weight);
    }

    template<typename OutGradient, typename Weight>
    static EIGEN_STRONG_INLINE auto
    gradient(const EqConstraintExpr<DerivedLhs, DerivedRhs>& eq, OutGradient&& out_gradient, const Weight& weight)
    {
        return ExprEvaluator<DerivedLhs>::gradient(eq.rhs, std::forward<OutGradient>(out_gradient), weight);
    }

    template<typename OutGradient, typename OutHessian, typename Weight>
    static EIGEN_STRONG_INLINE auto
    hessian(const EqConstraintExpr<DerivedLhs, DerivedRhs>& eq, OutGradient&& out_gradient, OutHessian&& out_hessian, const Weight& weight)
    {
        return ExprEvaluator<DerivedLhs>::hessian(eq.rhs, std::forward<OutGradient>(out_gradient), std::forward<OutHessian>(out_hessian), weight);
    }

    static EIGEN_STRONG_INLINE auto
    lower_bound(const EqConstraintExpr<DerivedLhs, DerivedRhs>& eq)
    {
        return eq.lhs;
    }

    static EIGEN_STRONG_INLINE auto
    upper_bound(const EqConstraintExpr<DerivedLhs, DerivedRhs>& eq)
    {
        return eq.lhs;
    }
};

template<typename DerivedLhs, typename DerivedRhs>
struct ExprEvaluator<EqConstraintExpr<DerivedLhs, DerivedRhs>,
                     typename std::enable_if<std::is_base_of<BaseExpr<DerivedLhs>, DerivedLhs>::value &&
                                             std::is_base_of<BaseExpr<DerivedRhs>, DerivedRhs>::value>::type>
{
    static EIGEN_STRONG_INLINE auto
    function(const EqConstraintExpr<DerivedLhs, DerivedRhs>& eq)
    {
        SubExpr<DerivedLhs, DerivedRhs> sub_expr(eq.lhs, eq.rhs);
        return ExprEvaluator<SubExpr<DerivedLhs, DerivedRhs>>::function(sub_expr);
    }

    template<typename OutValue, typename OutJacobian>
    static EIGEN_STRONG_INLINE void
    jacobian(const EqConstraintExpr<DerivedLhs, DerivedRhs>& eq, OutValue&& out_value, OutJacobian&& out_jacobian)
    {
        SubExpr<DerivedLhs, DerivedRhs> sub_expr(eq.lhs, eq.rhs);
        ExprEvaluator<SubExpr<DerivedLhs, DerivedRhs>>::jacobian(sub_expr, std::forward<OutValue>(out_value), std::forward<OutJacobian>(out_jacobian));
    }

    template<typename Weight>
    static EIGEN_STRONG_INLINE auto
    wsum(const EqConstraintExpr<DerivedLhs, DerivedRhs>& eq, const Weight& weight)
    {
        SubExpr<DerivedLhs, DerivedRhs> sub_expr(eq.lhs, eq.rhs);
        return ExprEvaluator<SubExpr<DerivedLhs, DerivedRhs>>::wsum(sub_expr, weight);
    }

    template<typename OutGradient, typename Weight>
    static EIGEN_STRONG_INLINE auto
    gradient(const EqConstraintExpr<DerivedLhs, DerivedRhs>& eq, OutGradient&& out_gradient, const Weight& weight)
    {
        SubExpr<DerivedLhs, DerivedRhs> sub_expr(eq.lhs, eq.rhs);
        return ExprEvaluator<SubExpr<DerivedLhs, DerivedRhs>>::gradient(sub_expr, std::forward<OutGradient>(out_gradient), weight);
    }

    template<typename OutGradient, typename OutHessian, typename Weight>
    static EIGEN_STRONG_INLINE auto
    hessian(const EqConstraintExpr<DerivedLhs, DerivedRhs>& eq, OutGradient&& out_gradient, OutHessian&& out_hessian, const Weight& weight)
    {
        SubExpr<DerivedLhs, DerivedRhs> sub_expr(eq.lhs, eq.rhs);
        return ExprEvaluator<SubExpr<DerivedLhs, DerivedRhs>>::hessian(sub_expr, std::forward<OutGradient>(out_gradient), std::forward<OutHessian>(out_hessian), weight);
    }

    static EIGEN_STRONG_INLINE auto
    lower_bound(const EqConstraintExpr<DerivedLhs, DerivedRhs>& eq)
    {
        return 0;
    }

    static EIGEN_STRONG_INLINE auto
    upper_bound(const EqConstraintExpr<DerivedLhs, DerivedRhs>& eq)
    {
        return 0;
    }
};

} // namespace laopt

#endif // LAOPT_EQ_CONSTRAINT_EXPR_HPP
