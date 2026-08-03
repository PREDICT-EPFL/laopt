#ifndef LAOPT_EQ_CONSTRAINT_EXPR_HPP
#define LAOPT_EQ_CONSTRAINT_EXPR_HPP

#include <Eigen/Dense>
#include "laopt/expressions/constraint_expr.hpp"
#include "laopt/expressions/expr_evaluator.hpp"
#include "laopt/expressions/sub_expr.hpp"
#include "laopt/variable_map.hpp"

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

    using ExprType = typename std::conditional<(std::is_base_of<ExprBase<DerivedLhs>, DerivedLhs>::value || is_variable<DerivedLhs>::value) &&
                                               (std::is_base_of<ExprBase<DerivedRhs>, DerivedRhs>::value || is_variable<DerivedRhs>::value),
                                               SubExpr<DerivedLhs, DerivedRhs>,
                                               typename std::conditional<std::is_base_of<ExprBase<DerivedLhs>, DerivedLhs>::value ||
                                                                         is_variable<DerivedLhs>::value,
                                                                         DerivedLhs,
                                                                         DerivedRhs>::type>::type;
    enum {
        RowsAtCompileTime = ExprType::RowsAtCompileTime,
        ColsAtCompileTime = 1
    };
    using Scalar = typename ExprType::Scalar;

    explicit EqConstraintExpr(const DerivedLhs& lhs, const DerivedRhs& rhs) : lhs(lhs), rhs(rhs) {}
};

// exp == exp
template<typename DerivedLhs, typename DerivedRhs>
EqConstraintExpr<DerivedLhs, DerivedRhs>
operator==(const ExprBase<DerivedLhs>& lhs, const ExprBase<DerivedRhs>& rhs)
{
    return EqConstraintExpr<DerivedLhs, DerivedRhs>(lhs.derived(), rhs.derived());
}

// exp == var
template<typename DerivedLhs, typename DerivedRhs>
typename std::enable_if<is_variable<DerivedRhs>::value, EqConstraintExpr<DerivedLhs, DerivedRhs>>::type
operator==(const ExprBase<DerivedLhs>& lhs, const DerivedRhs& rhs)
{
    return EqConstraintExpr<DerivedLhs, DerivedRhs>(lhs.derived(), rhs);
}

// var == exp
template<typename DerivedLhs, typename DerivedRhs>
typename std::enable_if<is_variable<DerivedLhs>::value, EqConstraintExpr<DerivedLhs, DerivedRhs>>::type
operator==(const DerivedLhs& lhs, const ExprBase<DerivedRhs>& rhs)
{
    return EqConstraintExpr<DerivedLhs, DerivedRhs>(lhs, rhs.derived());
}

// var == var
template<typename DerivedLhs, typename DerivedRhs>
typename std::enable_if<is_variable<DerivedLhs>::value && is_variable<DerivedRhs>::value, EqConstraintExpr<DerivedLhs, DerivedRhs>>::type
operator==(const DerivedLhs& lhs, const DerivedRhs& rhs)
{
    return EqConstraintExpr<DerivedLhs, DerivedRhs>(lhs, rhs);
}

template<typename DerivedLhs, typename DerivedRhs>
struct ExprEvaluator<EqConstraintExpr<DerivedLhs, DerivedRhs>,
                     typename std::enable_if<(std::is_base_of<ExprBase<DerivedLhs>, DerivedLhs>::value || is_variable<DerivedLhs>::value) &&
                                             is_constant_non_expr<DerivedRhs>::value>::type>
{
    static EIGEN_STRONG_INLINE auto
    function(const EqConstraintExpr<DerivedLhs, DerivedRhs>& eq)
    {
        return ExprEvaluator<DerivedLhs>::function(eq.lhs);
    }

    template<typename OutJacobian, typename AScalar>
    static EIGEN_STRONG_INLINE void
    jacobian(const EqConstraintExpr<DerivedLhs, DerivedRhs>& eq, OutJacobian&& out_jacobian, const AScalar& alpha)
    {
        ExprEvaluator<DerivedLhs>::jacobian(eq.lhs, std::forward<OutJacobian>(out_jacobian), alpha);
    }

    template<typename Weight>
    static EIGEN_STRONG_INLINE auto
    wsum(const EqConstraintExpr<DerivedLhs, DerivedRhs>& eq, const Weight& weight)
    {
        return ExprEvaluator<DerivedLhs>::wsum(eq.lhs, weight);
    }

    template<typename OutGradient, typename Weight>
    static EIGEN_STRONG_INLINE void
    gradient(const EqConstraintExpr<DerivedLhs, DerivedRhs>& eq, OutGradient&& out_gradient, const Weight& weight)
    {
        ExprEvaluator<DerivedLhs>::gradient(eq.lhs, std::forward<OutGradient>(out_gradient), weight);
    }

    template<typename OutHessian, typename Weight>
    static EIGEN_STRONG_INLINE void
    hessian(const EqConstraintExpr<DerivedLhs, DerivedRhs>& eq, OutHessian&& out_hessian, const Weight& weight)
    {
        ExprEvaluator<DerivedLhs>::hessian(eq.lhs, std::forward<OutHessian>(out_hessian), weight);
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
                     typename std::enable_if<(std::is_base_of<ExprBase<DerivedRhs>, DerivedRhs>::value || is_variable<DerivedRhs>::value) &&
                                             is_constant_non_expr<DerivedLhs>::value>::type>
{
    static EIGEN_STRONG_INLINE auto
    function(const EqConstraintExpr<DerivedLhs, DerivedRhs>& eq)
    {
        return ExprEvaluator<DerivedLhs>::function(eq.rhs);
    }

    template<typename OutJacobian, typename AScalar>
    static EIGEN_STRONG_INLINE void
    jacobian(const EqConstraintExpr<DerivedLhs, DerivedRhs>& eq, OutJacobian&& out_jacobian, const AScalar& alpha)
    {
        ExprEvaluator<DerivedLhs>::jacobian(eq.rhs, std::forward<OutJacobian>(out_jacobian), alpha);
    }

    template<typename Weight>
    static EIGEN_STRONG_INLINE auto
    wsum(const EqConstraintExpr<DerivedLhs, DerivedRhs>& eq, const Weight& weight)
    {
        return ExprEvaluator<DerivedLhs>::wsum(eq.rhs, weight);
    }

    template<typename OutGradient, typename Weight>
    static EIGEN_STRONG_INLINE void
    gradient(const EqConstraintExpr<DerivedLhs, DerivedRhs>& eq, OutGradient&& out_gradient, const Weight& weight)
    {
        ExprEvaluator<DerivedLhs>::gradient(eq.rhs, std::forward<OutGradient>(out_gradient), weight);
    }

    template<typename OutHessian, typename Weight>
    static EIGEN_STRONG_INLINE void
    hessian(const EqConstraintExpr<DerivedLhs, DerivedRhs>& eq, OutHessian&& out_hessian, const Weight& weight)
    {
        ExprEvaluator<DerivedLhs>::hessian(eq.rhs, std::forward<OutHessian>(out_hessian), weight);
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
                     typename std::enable_if<(std::is_base_of<ExprBase<DerivedLhs>, DerivedLhs>::value || is_variable<DerivedLhs>::value) &&
                                             (std::is_base_of<ExprBase<DerivedRhs>, DerivedRhs>::value || is_variable<DerivedRhs>::value)>::type>
{
    static EIGEN_STRONG_INLINE auto
    function(const EqConstraintExpr<DerivedLhs, DerivedRhs>& eq)
    {
        SubExpr<DerivedLhs, DerivedRhs> sub_expr(eq.lhs, eq.rhs);
        return ExprEvaluator<SubExpr<DerivedLhs, DerivedRhs>>::function(sub_expr);
    }

    template<typename OutJacobian, typename AScalar>
    static EIGEN_STRONG_INLINE void
    jacobian(const EqConstraintExpr<DerivedLhs, DerivedRhs>& eq, OutJacobian&& out_jacobian, const AScalar& alpha)
    {
        SubExpr<DerivedLhs, DerivedRhs> sub_expr(eq.lhs, eq.rhs);
        ExprEvaluator<SubExpr<DerivedLhs, DerivedRhs>>::jacobian(sub_expr, std::forward<OutJacobian>(out_jacobian), alpha);
    }

    template<typename Weight>
    static EIGEN_STRONG_INLINE auto
    wsum(const EqConstraintExpr<DerivedLhs, DerivedRhs>& eq, const Weight& weight)
    {
        SubExpr<DerivedLhs, DerivedRhs> sub_expr(eq.lhs, eq.rhs);
        return ExprEvaluator<SubExpr<DerivedLhs, DerivedRhs>>::wsum(sub_expr, weight);
    }

    template<typename OutGradient, typename Weight>
    static EIGEN_STRONG_INLINE void
    gradient(const EqConstraintExpr<DerivedLhs, DerivedRhs>& eq, OutGradient&& out_gradient, const Weight& weight)
    {
        SubExpr<DerivedLhs, DerivedRhs> sub_expr(eq.lhs, eq.rhs);
        ExprEvaluator<SubExpr<DerivedLhs, DerivedRhs>>::gradient(sub_expr, std::forward<OutGradient>(out_gradient), weight);
    }

    template<typename OutHessian, typename Weight>
    static EIGEN_STRONG_INLINE void
    hessian(const EqConstraintExpr<DerivedLhs, DerivedRhs>& eq, OutHessian&& out_hessian, const Weight& weight)
    {
        SubExpr<DerivedLhs, DerivedRhs> sub_expr(eq.lhs, eq.rhs);
        ExprEvaluator<SubExpr<DerivedLhs, DerivedRhs>>::hessian(sub_expr, std::forward<OutHessian>(out_hessian), weight);
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
