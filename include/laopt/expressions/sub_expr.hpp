#ifndef LAOPT_SUB_EXPR_HPP
#define LAOPT_SUB_EXPR_HPP

#include <Eigen/Dense>

#include "base_expr.hpp"
#include "expr_evaluator.hpp"
#include "../indexed_vector.hpp"

namespace laopt {

template<typename DerivedLhs, typename DerivedRhs>
class SubExpr : public BaseExpr<SubExpr<DerivedLhs, DerivedRhs>>
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
SubExpr<DerivedLhs, DerivedRhs> operator-(const BaseExpr<DerivedLhs>& lhs, const BaseExpr<DerivedRhs>& rhs)
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

    template<typename OutValue, typename OutJacobian>
    static EIGEN_STRONG_INLINE void
    jacobian(const SubExpr<DerivedLhs, DerivedRhs>& expr, OutValue&& out_value, OutJacobian&& out_jacobian)
    {
        ExprEvaluator<DerivedLhs>::jacobian(expr.lhs, out_value, out_jacobian(Eigen::all, Eigen::seqN(0, Eigen::fix<DerivedLhs::n_inputs>)));

        Eigen::Matrix<typename DerivedRhs::Scalar, DerivedRhs::n_outputs, 1> out_value_rhs;
        out_value_rhs.setZero();
        Eigen::Matrix<typename DerivedRhs::Scalar, DerivedRhs::n_outputs, DerivedRhs::n_inputs> out_jacobian_rhs;
        out_jacobian_rhs.setZero();
        ExprEvaluator<DerivedRhs>::jacobian(expr.rhs, out_value_rhs, out_jacobian_rhs);

        out_value -= out_value_rhs;
        out_jacobian(Eigen::all, Eigen::lastN(DerivedRhs::n_inputs)) -= out_jacobian_rhs;
    }

    template<typename Weight>
    static EIGEN_STRONG_INLINE auto
    wsum(const SubExpr<DerivedLhs, DerivedRhs>& expr, const Weight& weight)
    {
        return ExprEvaluator<DerivedLhs>::wsum(expr.lhs, weight) - ExprEvaluator<DerivedRhs>::wsum(expr.rhs, weight);
    }

    template<typename OutGradient, typename Weight>
    static EIGEN_STRONG_INLINE auto
    gradient(const SubExpr<DerivedLhs, DerivedRhs>& expr, OutGradient&& out_gradient, const Weight& weight)
    {
        auto value = ExprEvaluator<DerivedLhs>::gradient(expr.lhs, out_gradient(Eigen::seqN(0, Eigen::fix<DerivedLhs::n_inputs>)), weight);

        Eigen::Matrix<typename DerivedRhs::Scalar, DerivedRhs::n_inputs, 1> out_gradient_rhs;
        out_gradient_rhs.setZero();

        value -= ExprEvaluator<DerivedRhs>::gradient(expr.rhs, out_gradient_rhs, weight);
        out_gradient(Eigen::lastN(DerivedRhs::n_inputs)) -= out_gradient_rhs;

        return value;
    }

    template<typename OutGradient, typename OutHessian, typename Weight>
    static EIGEN_STRONG_INLINE auto
    hessian(const SubExpr<DerivedLhs, DerivedRhs>& expr, OutGradient&& out_gradient, OutHessian&& out_hessian, const Weight& weight)
    {
        auto value = ExprEvaluator<DerivedLhs>::hessian(expr.lhs, out_gradient(Eigen::seqN(0, Eigen::fix<DerivedLhs::n_inputs>)), out_hessian(Eigen::seqN(0, Eigen::fix<DerivedLhs::n_inputs>), Eigen::seqN(0, Eigen::fix<DerivedLhs::n_inputs>)), weight);

        Eigen::Matrix<typename DerivedRhs::Scalar, DerivedRhs::n_inputs, 1> out_gradient_rhs;
        out_gradient_rhs.setZero();
        Eigen::Matrix<typename DerivedRhs::Scalar, DerivedRhs::n_inputs, DerivedRhs::n_inputs> out_hessian_rhs;
        out_hessian_rhs.setZero();

        value -= ExprEvaluator<DerivedRhs>::hessian(expr.rhs, out_gradient_rhs, out_hessian_rhs, weight);
        out_gradient(Eigen::lastN(DerivedRhs::n_inputs)) -= out_gradient_rhs;
        out_hessian(Eigen::lastN(DerivedRhs::n_inputs), Eigen::lastN(DerivedRhs::n_inputs)) -= out_hessian_rhs;

        return value;
    }
};

template<typename DerivedLhs, typename DerivedRhs>
struct ExprEvaluator<SubExpr<DerivedLhs, IndexedVector<DerivedRhs>>>
{
    static EIGEN_STRONG_INLINE auto
    function(const SubExpr<DerivedLhs, IndexedVector<DerivedRhs>>& expr) -> decltype(ExprEvaluator<DerivedLhs>::function(expr.lhs))
    {
        functions::IDENTITY id(-1);
        return ExprEvaluator<DerivedLhs>::function(expr.lhs) + id.function(expr.rhs.cast_base());
    }

    template<typename OutValue, typename OutJacobian>
    static EIGEN_STRONG_INLINE void
    jacobian(const SubExpr<DerivedLhs, IndexedVector<DerivedRhs>>& expr, OutValue&& out_value, OutJacobian&& out_jacobian)
    {
        ExprEvaluator<DerivedLhs>::jacobian(expr.lhs, out_value, out_jacobian(Eigen::all, Eigen::seqN(0, Eigen::fix<DerivedLhs::n_inputs>)));
        functions::IDENTITY id(-1);
        id.jacobian(out_value,
                    out_jacobian(Eigen::all, Eigen::lastN(IndexedVector<DerivedRhs>::n_inputs)),
                    expr.rhs.cast_base());
    }

    template<typename Weight>
    static EIGEN_STRONG_INLINE auto
    wsum(const SubExpr<DerivedLhs, IndexedVector<DerivedRhs>>& expr, const Weight& weight)
    {
        functions::IDENTITY id(-1);
        return ExprEvaluator<DerivedLhs>::wsum(expr.lhs, weight) + id.wsum(weight, expr.rhs.cast_base());
    }

    template<typename OutGradient, typename Weight>
    static EIGEN_STRONG_INLINE auto
    gradient(const SubExpr<DerivedLhs, IndexedVector<DerivedRhs>>& expr, OutGradient&& out_gradient, const Weight& weight)
    {
        functions::IDENTITY id(-1);
        return ExprEvaluator<DerivedLhs>::gradient(expr.lhs, out_gradient(Eigen::seqN(0, Eigen::fix<DerivedLhs::n_inputs>)), weight)
               + id.gradient(out_gradient(Eigen::lastN(IndexedVector<DerivedRhs>::n_inputs)), weight, expr.rhs.cast_base());
    }

    template<typename OutGradient, typename OutHessian, typename Weight>
    static EIGEN_STRONG_INLINE auto
    hessian(const SubExpr<DerivedLhs, IndexedVector<DerivedRhs>>& expr, OutGradient&& out_gradient, OutHessian&& out_hessian, const Weight& weight)
    {
        functions::IDENTITY id(-1);
        return ExprEvaluator<DerivedLhs>::hessian(expr.lhs, out_gradient(Eigen::seqN(0, Eigen::fix<DerivedLhs::n_inputs>)), out_hessian(Eigen::seqN(0, Eigen::fix<DerivedLhs::n_inputs>), Eigen::seqN(0, Eigen::fix<DerivedLhs::n_inputs>)), weight)
               + id.hessian(out_gradient(Eigen::lastN(IndexedVector<DerivedRhs>::n_inputs)), out_hessian(Eigen::lastN(IndexedVector<DerivedRhs>::n_inputs), Eigen::lastN(IndexedVector<DerivedRhs>::n_inputs)), weight, expr.rhs.cast_base());
    }
};

} // namespace laopt

#endif // LAOPT_SUB_EXPR_HPP
