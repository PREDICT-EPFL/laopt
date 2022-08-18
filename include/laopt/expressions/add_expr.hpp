#ifndef LAOPT_ADD_EXPR_HPP
#define LAOPT_ADD_EXPR_HPP

#include <Eigen/Dense>

#include "base_expr.hpp"
#include "expr_evaluator.hpp"
#include "../indexed_vector.hpp"

namespace laopt {

template<typename DerivedLhs, typename DerivedRhs>
class AddExpr : public BaseExpr<AddExpr<DerivedLhs, DerivedRhs>>
{
public:
    const DerivedLhs& lhs;
    const DerivedRhs& rhs;

    static_assert(DerivedLhs::n_outputs == DerivedRhs::n_outputs, "Output dimension of expressions must be the same");
    static constexpr int n_inputs = DerivedLhs::n_inputs + DerivedRhs::n_inputs;
    static constexpr int n_outputs = DerivedLhs::n_outputs;
    using Scalar = typename DerivedLhs::Scalar;

    explicit AddExpr(const DerivedLhs& lhs, const DerivedRhs& rhs) : lhs(lhs), rhs(rhs) {}

    EIGEN_STRONG_INLINE const Eigen::Vector<int, n_inputs> indices() const
    {
        return concatenate_indices(lhs.indices(), rhs.indices());
    }
};

template<typename DerivedLhs, typename DerivedRhs>
AddExpr<DerivedLhs, DerivedRhs> operator+(const BaseExpr<DerivedLhs>& lhs, const BaseExpr<DerivedRhs>& rhs)
{
    return AddExpr<DerivedLhs, DerivedRhs>(lhs.derived(), rhs.derived());
}

// we need this special case to be not ambiguous with Eigen
template<typename DerivedLhs, typename DerivedRhs>
AddExpr<IndexedVector<DerivedLhs>, IndexedVector<DerivedRhs>> operator+(const IndexedVector<DerivedLhs>& lhs, const IndexedVector<DerivedRhs>& rhs)
{
    return AddExpr<IndexedVector<DerivedLhs>, IndexedVector<DerivedRhs>>(lhs, rhs);
}

template<typename DerivedLhs, typename DerivedRhs>
struct ExprEvaluator<AddExpr<DerivedLhs, DerivedRhs>>
{
    static EIGEN_STRONG_INLINE auto
    function(const AddExpr<DerivedLhs, DerivedRhs>& expr) -> decltype(ExprEvaluator<DerivedLhs>::function(expr.lhs))
    {
        return ExprEvaluator<DerivedLhs>::function(expr.lhs) + ExprEvaluator<DerivedRhs>::function(expr.rhs);
    }

    template<typename OutValue, typename OutJacobian>
    static EIGEN_STRONG_INLINE void
    jacobian(const AddExpr<DerivedLhs, DerivedRhs>& expr, OutValue&& out_value, OutJacobian&& out_jacobian)
    {
        ExprEvaluator<DerivedLhs>::jacobian(expr.lhs, out_value, out_jacobian(Eigen::all, Eigen::seqN(0, DerivedLhs::n_inputs)));
        ExprEvaluator<DerivedRhs>::jacobian(expr.rhs, out_value, out_jacobian(Eigen::all, Eigen::lastN(DerivedRhs::n_inputs)));
    }

    template<typename Weight>
    static EIGEN_STRONG_INLINE auto
    wsum(const AddExpr<DerivedLhs, DerivedRhs>& expr, const Weight& weight)
    {
        return ExprEvaluator<DerivedLhs>::wsum(expr.lhs, weight) + ExprEvaluator<DerivedRhs>::wsum(expr.rhs, weight);
    }

    template<typename OutGradient, typename Weight>
    static EIGEN_STRONG_INLINE auto
    gradient(const AddExpr<DerivedLhs, DerivedRhs>& expr, OutGradient&& out_gradient, const Weight& weight)
    {
        return ExprEvaluator<DerivedLhs>::gradient(expr.lhs, out_gradient(Eigen::seqN(0, DerivedLhs::n_inputs)), weight)
               + ExprEvaluator<DerivedRhs>::gradient(expr.rhs, out_gradient(Eigen::lastN(DerivedRhs::n_inputs)), weight);
    }

    template<typename OutGradient, typename OutHessian, typename Weight>
    static EIGEN_STRONG_INLINE auto
    hessian(const AddExpr<DerivedLhs, DerivedRhs>& expr, OutGradient&& out_gradient, OutHessian&& out_hessian, const Weight& weight)
    {
        return ExprEvaluator<DerivedLhs>::hessian(expr.lhs, out_gradient(Eigen::seqN(0, DerivedLhs::n_inputs)), out_hessian(Eigen::seqN(0, DerivedLhs::n_inputs), Eigen::seqN(0, DerivedLhs::n_inputs)), weight)
               + ExprEvaluator<DerivedRhs>::hessian(expr.rhs, out_gradient(Eigen::seqN(0, DerivedRhs::n_inputs)), out_hessian(Eigen::seqN(0, DerivedRhs::n_inputs), Eigen::seqN(0, DerivedRhs::n_inputs)), weight);
    }
};

} // namespace laopt

#endif // LAOPT_ADD_EXPR_HPP
