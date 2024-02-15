#ifndef LAOPT_SCALAR_EXPR_HPP
#define LAOPT_SCALAR_EXPR_HPP

#include <Eigen/Dense>

#include "expr_base.hpp"
#include "expr_evaluator.hpp"

namespace laopt {

template<typename ScalarType, typename Derived>
class ScalarExpr : public ExprBase<ScalarExpr<ScalarType, Derived>>
{
public:
    const ScalarType& scalar;
    const Derived& expr;

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

    template<typename OutJacobian>
    static EIGEN_STRONG_INLINE void
    jacobian(const ScalarExpr<ScalarType, Derived>& expr, OutJacobian&& out_jacobian)
    {
        Eigen::Matrix<typename Derived::Scalar, Derived::n_outputs, Derived::n_inputs> out_jacobian_expr;
        out_jacobian_expr.setZero();
        ExprEvaluator<Derived>::jacobian(expr.expr, out_jacobian_expr);

        out_jacobian(Eigen::all, Eigen::lastN(Derived::n_inputs)) += expr.scalar * out_jacobian_expr;
    }

    static EIGEN_STRONG_INLINE void
    jacobian(const ScalarExpr<ScalarType, Derived>& expr, BSMatrixSparsity&& out_jacobian)
    {
        jacobian_sparsity(expr, std::forward<BSMatrixSparsity>(out_jacobian));
    }

    template<typename SparsityNullMat>
    static EIGEN_STRONG_INLINE void
    jacobian(const ScalarExpr<ScalarType, Derived>& expr, BSSliceSparsity<BSMatrixSparsity, SparsityNullMat>&& out_jacobian)
    {
        jacobian_sparsity(expr, std::forward<BSSliceSparsity<BSMatrixSparsity, SparsityNullMat>>(out_jacobian));
    }

    template<typename SparsityNullMat>
    static EIGEN_STRONG_INLINE void
    jacobian_sparsity(const ScalarExpr<ScalarType, Derived>& expr, BSSliceSparsity<BSMatrixSparsity, SparsityNullMat>&& out_jacobian)
    {
        ExprEvaluator<Derived>::jacobian(expr.expr, out_jacobian(Eigen::all, Eigen::seqN(0, Eigen::fix<Derived::n_inputs>)));
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
        Eigen::Matrix<typename Derived::Scalar, Derived::n_inputs, 1> out_gradient_expr;
        out_gradient_expr.setZero();

        ExprEvaluator<Derived>::gradient(expr.expr, out_gradient_expr, weight);
        out_gradient(Eigen::lastN(Derived::n_inputs)) += expr.scalar * out_gradient_expr;
    }

    template<typename OutHessian, typename Weight>
    static EIGEN_STRONG_INLINE void
    hessian(const ScalarExpr<ScalarType, Derived>& expr, OutHessian&& out_hessian, const Weight& weight)
    {
        Eigen::Matrix<typename Derived::Scalar, Derived::n_inputs, Derived::n_inputs> out_hessian_expr;
        out_hessian_expr.setZero();
        ExprEvaluator<Derived>::hessian(expr.expr, out_hessian_expr, weight);

        out_hessian(Eigen::lastN(Derived::n_inputs), Eigen::lastN(Derived::n_inputs)) += expr.scalar * out_hessian_expr;
    }

    template<typename Weight>
    static EIGEN_STRONG_INLINE void
    hessian(const ScalarExpr<ScalarType, Derived>& expr, BSMatrixSparsity&& out_hessian, const Weight& weight)
    {
        hessian_sparsity(expr, std::forward<BSMatrixSparsity>(out_hessian), weight);
    }

    template<typename SparsityNullMat, typename Weight>
    static EIGEN_STRONG_INLINE void
    hessian(const ScalarExpr<ScalarType, Derived>& expr, BSSliceSparsity<BSMatrixSparsity, SparsityNullMat>&& out_hessian, const Weight& weight)
    {
        hessian_sparsity(expr, std::forward<BSSliceSparsity<BSMatrixSparsity, SparsityNullMat>>(out_hessian), weight);
    }

    template<typename SparsityNullMat, typename Weight>
    static EIGEN_STRONG_INLINE void
    hessian_sparsity(const ScalarExpr<ScalarType, Derived>& expr, BSSliceSparsity<BSMatrixSparsity, SparsityNullMat>&& out_hessian, const Weight& weight)
    {
        ExprEvaluator<Derived>::hessian(expr.expr, out_hessian(Eigen::seqN(0, Eigen::fix<Derived::n_inputs>), Eigen::seqN(0, Eigen::fix<Derived::n_inputs>)), weight);
    }
};

} // namespace laopt

#endif // LAOPT_SCALAR_EXPR_HPP
