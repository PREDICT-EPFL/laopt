#ifndef LAOPT_EXPR_EVALUATOR_HPP
#define LAOPT_EXPR_EVALUATOR_HPP

#include "laopt/indexed_vector.hpp"
#include "laopt/common_functions.hpp"

namespace laopt {

// ExprEvaluator forward declaration
template<typename Derived, typename EnableIf = void>
struct ExprEvaluator;

template<typename Derived>
struct ExprEvaluator<IndexedVector<Derived>>
{
    static EIGEN_STRONG_INLINE auto
    function(const IndexedVector<Derived>& indexed_vector)
    {
        common_functions::IDENTITY id;
        return id.function(indexed_vector.cast_base());
    }

    template<typename OutJacobian, typename AScalar>
    static EIGEN_STRONG_INLINE void
    jacobian(const IndexedVector<Derived>& indexed_vector, OutJacobian&& out_jacobian, const AScalar& alpha)
    {
        common_functions::IDENTITY id;
        id.jacobian(out_jacobian,
                    alpha,
                    indexed_vector);
    }

    template<typename Weight>
    static EIGEN_STRONG_INLINE auto
    wsum(const IndexedVector<Derived>& indexed_vector, const Weight& weight)
    {
        common_functions::IDENTITY id;
        return id.wsum(weight,
                       indexed_vector);
    }

    template<typename OutGradient, typename Weight>
    static EIGEN_STRONG_INLINE void
    gradient(const IndexedVector<Derived>& indexed_vector, OutGradient&& out_gradient, const Weight& weight)
    {
        common_functions::IDENTITY id;
        id.gradient(out_gradient,
                    weight,
                    indexed_vector);
    }

    template<typename OutHessian, typename Weight>
    static EIGEN_STRONG_INLINE void
    hessian(const IndexedVector<Derived>& indexed_vector, OutHessian&& out_hessian, const Weight& weight)
    {
        common_functions::IDENTITY id;
        id.hessian(out_hessian,
                   weight,
                   indexed_vector);
    }
};

} // namespace laopt

#endif // LAOPT_EXPR_EVALUATOR_HPP
