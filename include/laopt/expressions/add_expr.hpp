#ifndef LAOPT_ADD_EXPR_HPP
#define LAOPT_ADD_EXPR_HPP

#include <Eigen/Dense>
#include "laopt/expressions/expr_base.hpp"
#include "laopt/expressions/expr_evaluator.hpp"

namespace laopt {

template<typename DerivedLhs, typename DerivedRhs>
class AddExpr : public ExprBase<AddExpr<DerivedLhs, DerivedRhs>>
{
public:
    const DerivedLhs lhs;
    const DerivedRhs rhs;

    static_assert((int) DerivedLhs::RowsAtCompileTime == (int) DerivedRhs::RowsAtCompileTime, "Output dimension of expressions must be the same");
    enum {
        RowsAtCompileTime = DerivedLhs::RowsAtCompileTime,
        ColsAtCompileTime = 1
    };
    using Scalar = typename DerivedLhs::Scalar;

    explicit AddExpr(const DerivedLhs& lhs, const DerivedRhs& rhs) : lhs(lhs), rhs(rhs) {}
};

template<typename DerivedLhs, typename DerivedRhs>
AddExpr<DerivedLhs, DerivedRhs> operator+(const ExprBase<DerivedLhs>& lhs, const ExprBase<DerivedRhs>& rhs)
{
    return AddExpr<DerivedLhs, DerivedRhs>(lhs.derived(), rhs.derived());
}

template<typename DerivedLhs, typename DerivedRhs>
typename std::enable_if<is_variable<DerivedRhs>::value, AddExpr<DerivedLhs, DerivedRhs>>::type
operator+(const ExprBase<DerivedLhs>& lhs, const DerivedRhs& rhs)
{
    return AddExpr<DerivedLhs, DerivedRhs>(lhs.derived(), rhs);
}

template<typename DerivedLhs, typename DerivedRhs>
typename std::enable_if<is_variable<DerivedLhs>::value, AddExpr<DerivedLhs, DerivedRhs>>::type
operator+(const DerivedLhs& lhs, const ExprBase<DerivedRhs>& rhs)
{
    return AddExpr<DerivedLhs, DerivedRhs>(lhs, rhs.derived());
}

template<typename DerivedLhs, typename DerivedRhs>
typename std::enable_if<is_variable<DerivedLhs>::value && is_variable<DerivedRhs>::value, AddExpr<DerivedLhs, DerivedRhs>>::type
operator+(const DerivedLhs& lhs, const DerivedRhs& rhs)
{
    return AddExpr<DerivedLhs, DerivedRhs>(lhs, rhs);
}

template<typename DerivedLhs, typename DerivedRhs>
struct ExprEvaluator<AddExpr<DerivedLhs, DerivedRhs>>
{
    static EIGEN_STRONG_INLINE auto
    function(const AddExpr<DerivedLhs, DerivedRhs>& expr) -> decltype(ExprEvaluator<DerivedLhs>::function(expr.lhs))
    {
        return ExprEvaluator<DerivedLhs>::function(expr.lhs) + ExprEvaluator<DerivedRhs>::function(expr.rhs);
    }

    template<typename OutJacobian, typename AScalar>
    static EIGEN_STRONG_INLINE void
    jacobian(const AddExpr<DerivedLhs, DerivedRhs>& expr, OutJacobian&& out_jacobian, const AScalar& alpha)
    {
        ExprEvaluator<DerivedLhs>::jacobian(expr.lhs, out_jacobian, alpha);
        ExprEvaluator<DerivedRhs>::jacobian(expr.rhs, out_jacobian, alpha);
    }

    template<typename Weight>
    static EIGEN_STRONG_INLINE auto
    wsum(const AddExpr<DerivedLhs, DerivedRhs>& expr, const Weight& weight)
    {
        return ExprEvaluator<DerivedLhs>::wsum(expr.lhs, weight) + ExprEvaluator<DerivedRhs>::wsum(expr.rhs, weight);
    }

    template<typename OutGradient, typename Weight>
    static EIGEN_STRONG_INLINE void
    gradient(const AddExpr<DerivedLhs, DerivedRhs>& expr, OutGradient&& out_gradient, const Weight& weight)
    {
        ExprEvaluator<DerivedLhs>::gradient(expr.lhs, out_gradient, weight);
        ExprEvaluator<DerivedRhs>::gradient(expr.rhs, out_gradient, weight);
    }

    template<typename OutHessian, typename Weight>
    static EIGEN_STRONG_INLINE void
    hessian(const AddExpr<DerivedLhs, DerivedRhs>& expr, OutHessian&& out_hessian, const Weight& weight)
    {
        ExprEvaluator<DerivedLhs>::hessian(expr.lhs, out_hessian, weight);
        ExprEvaluator<DerivedRhs>::hessian(expr.rhs, out_hessian, weight);
    }
};

} // namespace laopt

#endif // LAOPT_ADD_EXPR_HPP
