#ifndef LAOPT_SUB_EXPR_HPP
#define LAOPT_SUB_EXPR_HPP

#include <Eigen/Dense>
#include "laopt/expressions/expr_base.hpp"
#include "laopt/expressions/expr_evaluator.hpp"
#include "laopt/indexed_vector.hpp"

namespace laopt {

template<typename DerivedLhs, typename DerivedRhs>
class SubExpr : public ExprBase<SubExpr<DerivedLhs, DerivedRhs>>
{
public:
    const DerivedLhs& lhs;
    const DerivedRhs& rhs;

    static_assert(DerivedLhs::n_outputs == DerivedRhs::n_outputs, "Output dimension of expressions must be the same");
    static constexpr int n_inputs = DerivedLhs::n_inputs + DerivedRhs::n_inputs;
    static constexpr int n_outputs = DerivedLhs::n_outputs;
    using Scalar = typename DerivedLhs::Scalar;

    explicit SubExpr(const DerivedLhs& lhs, const DerivedRhs& rhs) : lhs(lhs), rhs(rhs) {}

    EIGEN_STRONG_INLINE const Eigen::Vector<int, n_inputs> indices() const
    {
        return concatenate_indices(lhs.indices(), rhs.indices());
    }
};

template<typename DerivedLhs, typename DerivedRhs>
SubExpr<DerivedLhs, DerivedRhs> operator-(const ExprBase<DerivedLhs>& lhs, const ExprBase<DerivedRhs>& rhs)
{
    return SubExpr<DerivedLhs, DerivedRhs>(lhs.derived(), rhs.derived());
}

// we need this special case to be not ambiguous with Eigen
template<typename DerivedLhs, typename DerivedRhs>
SubExpr<IndexedVector<DerivedLhs>, IndexedVector<DerivedRhs>> operator-(const IndexedVector<DerivedLhs>& lhs, const IndexedVector<DerivedRhs>& rhs)
{
    return SubExpr<IndexedVector<DerivedLhs>, IndexedVector<DerivedRhs>>(lhs, rhs);
}

template<typename DerivedLhs, typename DerivedRhs>
struct ExprEvaluator<SubExpr<DerivedLhs, DerivedRhs>>
{
    static EIGEN_STRONG_INLINE auto
    function(const SubExpr<DerivedLhs, DerivedRhs>& expr) -> decltype(ExprEvaluator<DerivedLhs>::function(expr.lhs))
    {
        return ExprEvaluator<DerivedLhs>::function(expr.lhs) - ExprEvaluator<DerivedRhs>::function(expr.rhs);
    }

    template<typename OutJacobian, typename AScalar>
    static EIGEN_STRONG_INLINE void
    jacobian(const SubExpr<DerivedLhs, DerivedRhs>& expr, OutJacobian&& out_jacobian, const AScalar& alpha)
    {
        ExprEvaluator<DerivedLhs>::jacobian(expr.lhs, out_jacobian(Eigen::all, Eigen::seqN(0, Eigen::fix<DerivedLhs::n_inputs>)), alpha);
        ExprEvaluator<DerivedRhs>::jacobian(expr.rhs, out_jacobian(Eigen::all, Eigen::lastN(DerivedRhs::n_inputs)), -alpha);
    }

    template<typename AScalar>
    static EIGEN_STRONG_INLINE void
    jacobian(const SubExpr<DerivedLhs, DerivedRhs>& expr, BSMatrixSparsity&& out_jacobian, const AScalar& alpha)
    {
        jacobian_sparsity(expr, std::forward<BSMatrixSparsity>(out_jacobian), alpha);
    }

    template<typename SparsityNullMat, typename AScalar>
    static EIGEN_STRONG_INLINE void
    jacobian(const SubExpr<DerivedLhs, DerivedRhs>& expr, BSSliceSparsity<BSMatrixSparsity, SparsityNullMat>&& out_jacobian, const AScalar& alpha)
    {
        jacobian_sparsity(expr, std::forward<BSSliceSparsity<BSMatrixSparsity, SparsityNullMat>>(out_jacobian), alpha);
    }

    template<typename SparsityNullMat, typename AScalar>
    static EIGEN_STRONG_INLINE void
    jacobian_sparsity(const SubExpr<DerivedLhs, DerivedRhs>& expr, BSSliceSparsity<BSMatrixSparsity, SparsityNullMat>&& out_jacobian, const AScalar& alpha)
    {
        ExprEvaluator<DerivedLhs>::jacobian(expr.lhs, out_jacobian(Eigen::all, Eigen::seqN(0, Eigen::fix<DerivedLhs::n_inputs>)), alpha);
        ExprEvaluator<DerivedRhs>::jacobian(expr.rhs, out_jacobian(Eigen::all, Eigen::lastN(Eigen::fix<DerivedRhs::n_inputs>)), -alpha);
    }

    template<typename Weight>
    static EIGEN_STRONG_INLINE auto
    wsum(const SubExpr<DerivedLhs, DerivedRhs>& expr, const Weight& weight)
    {
        return ExprEvaluator<DerivedLhs>::wsum(expr.lhs, weight) - ExprEvaluator<DerivedRhs>::wsum(expr.rhs, weight);
    }

    template<typename OutGradient, typename Weight>
    static EIGEN_STRONG_INLINE void
    gradient(const SubExpr<DerivedLhs, DerivedRhs>& expr, OutGradient&& out_gradient, const Weight& weight)
    {
        ExprEvaluator<DerivedLhs>::gradient(expr.lhs, out_gradient(Eigen::seqN(0, Eigen::fix<DerivedLhs::n_inputs>)), weight);
        ExprEvaluator<DerivedRhs>::gradient(expr.rhs, out_gradient(Eigen::lastN(DerivedRhs::n_inputs)), -weight);
    }

    template<typename OutHessian, typename Weight>
    static EIGEN_STRONG_INLINE void
    hessian(const SubExpr<DerivedLhs, DerivedRhs>& expr, OutHessian&& out_hessian, const Weight& weight)
    {
        ExprEvaluator<DerivedLhs>::hessian(expr.lhs, out_hessian(Eigen::seqN(0, Eigen::fix<DerivedLhs::n_inputs>), Eigen::seqN(0, Eigen::fix<DerivedLhs::n_inputs>)), weight);
        ExprEvaluator<DerivedRhs>::hessian(expr.rhs, out_hessian(Eigen::lastN(DerivedRhs::n_inputs), Eigen::lastN(DerivedRhs::n_inputs)), -weight);
    }

    template<typename Weight>
    static EIGEN_STRONG_INLINE void
    hessian(const SubExpr<DerivedLhs, DerivedRhs>& expr, BSMatrixSparsity&& out_hessian, const Weight& weight)
    {
        hessian_sparsity(expr, std::forward<BSMatrixSparsity>(out_hessian), weight);
    }

    template<typename SparsityNullMat, typename Weight>
    static EIGEN_STRONG_INLINE void
    hessian(const SubExpr<DerivedLhs, DerivedRhs>& expr, BSSliceSparsity<BSMatrixSparsity, SparsityNullMat>&& out_hessian, const Weight& weight)
    {
        hessian_sparsity(expr, std::forward<BSSliceSparsity<BSMatrixSparsity, SparsityNullMat>>(out_hessian), weight);
    }

    template<typename SparsityNullMat, typename Weight>
    static EIGEN_STRONG_INLINE void
    hessian_sparsity(const SubExpr<DerivedLhs, DerivedRhs>& expr, BSSliceSparsity<BSMatrixSparsity, SparsityNullMat>&& out_hessian, const Weight& weight)
    {
        ExprEvaluator<DerivedLhs>::hessian(expr.lhs, out_hessian(Eigen::seqN(0, Eigen::fix<DerivedLhs::n_inputs>), Eigen::seqN(0, Eigen::fix<DerivedLhs::n_inputs>)), weight);
        ExprEvaluator<DerivedRhs>::hessian(expr.rhs, out_hessian(Eigen::lastN(DerivedRhs::n_inputs), Eigen::lastN(DerivedRhs::n_inputs)), -weight);
    }
};

template<typename DerivedLhs, typename DerivedRhs>
struct ExprEvaluator<SubExpr<DerivedLhs, IndexedVector<DerivedRhs>>>
{
    static EIGEN_STRONG_INLINE auto
    function(const SubExpr<DerivedLhs, IndexedVector<DerivedRhs>>& expr) -> decltype(ExprEvaluator<DerivedLhs>::function(expr.lhs))
    {
        common_functions::IDENTITY id;
        return ExprEvaluator<DerivedLhs>::function(expr.lhs) - id.function(expr.rhs.cast_base());
    }

    template<typename OutJacobian, typename AScalar>
    static EIGEN_STRONG_INLINE void
    jacobian(const SubExpr<DerivedLhs, IndexedVector<DerivedRhs>>& expr, OutJacobian&& out_jacobian, const AScalar& alpha)
    {
        ExprEvaluator<DerivedLhs>::jacobian(expr.lhs, out_jacobian(Eigen::all, Eigen::seqN(0, Eigen::fix<DerivedLhs::n_inputs>)), alpha);
        common_functions::IDENTITY id;
        id.jacobian(out_jacobian(Eigen::all, Eigen::lastN(IndexedVector<DerivedRhs>::n_inputs)), -alpha, expr.rhs);
    }

    template<typename Weight>
    static EIGEN_STRONG_INLINE auto
    wsum(const SubExpr<DerivedLhs, IndexedVector<DerivedRhs>>& expr, const Weight& weight)
    {
        common_functions::IDENTITY id;
        return ExprEvaluator<DerivedLhs>::wsum(expr.lhs, weight) - id.wsum(weight, expr.rhs);
    }

    template<typename OutGradient, typename Weight>
    static EIGEN_STRONG_INLINE void
    gradient(const SubExpr<DerivedLhs, IndexedVector<DerivedRhs>>& expr, OutGradient&& out_gradient, const Weight& weight)
    {
        common_functions::IDENTITY id;
        ExprEvaluator<DerivedLhs>::gradient(expr.lhs, out_gradient(Eigen::seqN(0, Eigen::fix<DerivedLhs::n_inputs>)), weight);
        id.gradient(out_gradient(Eigen::lastN(IndexedVector<DerivedRhs>::n_inputs)), -weight, expr.rhs);
    }

    template<typename OutHessian, typename Weight>
    static EIGEN_STRONG_INLINE void
    hessian(const SubExpr<DerivedLhs, IndexedVector<DerivedRhs>>& expr, OutHessian&& out_hessian, const Weight& weight)
    {
        common_functions::IDENTITY id;
        ExprEvaluator<DerivedLhs>::hessian(expr.lhs, out_hessian(Eigen::seqN(0, Eigen::fix<DerivedLhs::n_inputs>), Eigen::seqN(0, Eigen::fix<DerivedLhs::n_inputs>)), weight);
        id.hessian(out_hessian(Eigen::lastN(IndexedVector<DerivedRhs>::n_inputs), Eigen::lastN(IndexedVector<DerivedRhs>::n_inputs)), -weight, expr.rhs);
    }
};

} // namespace laopt

#endif // LAOPT_SUB_EXPR_HPP
