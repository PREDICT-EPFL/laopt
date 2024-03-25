#ifndef LAOPT_SCALAR_EXPR_HPP
#define LAOPT_SCALAR_EXPR_HPP

#include <Eigen/Dense>
#include "laopt/expressions/expr_base.hpp"
#include "laopt/expressions/expr_evaluator.hpp"

namespace laopt {

template<typename ScalarType, typename Derived>
class ScalarExpr : public ExprBase<ScalarExpr<ScalarType, Derived>>
{
public:
    const ScalarType scalar;
    const Derived expr;

    static constexpr int n_inputs = Derived::n_inputs;
    static constexpr int n_outputs = Derived::n_outputs;
    using Scalar = typename Derived::Scalar;

    explicit ScalarExpr(const ScalarType& scalar, const Derived& expr) : scalar(scalar), expr(expr) {}

    EIGEN_STRONG_INLINE const Eigen::Vector<int, n_inputs> indices() const
    {
        return expr.indices();
    }
};

template<typename ScalarType, typename Derived>
typename std::enable_if<std::is_arithmetic<ScalarType>::value, ScalarExpr<ScalarType, Derived>>::type
operator*(const ScalarType& scalar, const ExprBase<Derived>& expr)
{
    return ScalarExpr<ScalarType, Derived>(scalar, expr.derived());
}

template<typename ScalarType, typename Derived>
typename std::enable_if<std::is_arithmetic<ScalarType>::value, ScalarExpr<ScalarType, Derived>>::type
operator*(const ExprBase<Derived>& expr, const ScalarType& scalar)
{
    return ScalarExpr<ScalarType, Derived>(scalar, expr.derived());
}

// we need this special case to be not ambiguous with Eigen
template<typename ScalarType, typename Derived>
typename std::enable_if<std::is_arithmetic<ScalarType>::value, ScalarExpr<ScalarType, IndexedVector<Derived>>>::type
operator*(const ScalarType& scalar, const IndexedVector<Derived>& expr)
{
    return ScalarExpr<ScalarType, IndexedVector<Derived>>(scalar, expr);
}

// we need this special case to be not ambiguous with Eigen
template<typename ScalarType, typename Derived>
typename std::enable_if<std::is_arithmetic<ScalarType>::value, ScalarExpr<ScalarType, IndexedVector<Derived>>>::type
operator*(const IndexedVector<Derived>& expr, const ScalarType& scalar)
{
    return ScalarExpr<ScalarType, IndexedVector<Derived>>(scalar, expr);
}

template<typename ScalarType, typename Derived>
struct ExprEvaluator<ScalarExpr<ScalarType, Derived>>
{
    static EIGEN_STRONG_INLINE auto
    function(const ScalarExpr<ScalarType, Derived>& expr) -> decltype(ExprEvaluator<Derived>::function(expr.expr))
    {
        return expr.scalar * ExprEvaluator<Derived>::function(expr.expr);
    }

    template<typename OutJacobian, typename AScalar>
    static EIGEN_STRONG_INLINE void
    jacobian(const ScalarExpr<ScalarType, Derived>& expr, OutJacobian&& out_jacobian, const AScalar& alpha)
    {
        ExprEvaluator<Derived>::jacobian(expr.expr, out_jacobian, expr.scalar * alpha);
    }

    template<typename Weight>
    static EIGEN_STRONG_INLINE auto
    wsum(const ScalarExpr<ScalarType, Derived>& expr, const Weight& weight)
    {
        return expr.scalar * ExprEvaluator<Derived>::wsum(expr.expr, weight);
    }

    template<typename OutGradient, typename Weight>
    static EIGEN_STRONG_INLINE void
    gradient(const ScalarExpr<ScalarType, Derived>& expr, OutGradient&& out_gradient, const Weight& weight)
    {
        ExprEvaluator<Derived>::gradient(expr.expr, out_gradient, expr.scalar * weight);
    }

    template<typename OutHessian, typename Weight>
    static EIGEN_STRONG_INLINE void
    hessian(const ScalarExpr<ScalarType, Derived>& expr, OutHessian&& out_hessian, const Weight& weight)
    {
        ExprEvaluator<Derived>::hessian(expr.expr, out_hessian, expr.scalar * weight);
    }
};

} // namespace laopt

#endif // LAOPT_SCALAR_EXPR_HPP
