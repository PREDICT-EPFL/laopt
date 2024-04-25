#ifndef LAOPT_EXPR_EVALUATOR_HPP
#define LAOPT_EXPR_EVALUATOR_HPP

#include "laopt/expressions/fwd.hpp"
#include "laopt/variable_map.hpp"
#include "laopt/differentiable_functions/identity.hpp"

namespace laopt {

template<typename MatrixType, int MapOptions, typename StrideType>
struct ExprEvaluator<VariableMap<MatrixType, MapOptions, StrideType>>
{
    static EIGEN_STRONG_INLINE auto
    function(const VariableMap<MatrixType, MapOptions, StrideType>& variable)
    {
        Identity id;
        return id.function(variable);
    }

    template<typename OutJacobian, typename AScalar>
    static EIGEN_STRONG_INLINE void
    jacobian(const VariableMap<MatrixType, MapOptions, StrideType>& variable, OutJacobian&& out_jacobian, const AScalar& alpha)
    {
        Identity id;
        id.jacobian(out_jacobian,
                    alpha,
                    variable);
    }

    template<typename Weight>
    static EIGEN_STRONG_INLINE auto
    wsum(const VariableMap<MatrixType, MapOptions, StrideType>& variable, const Weight& weight)
    {
        Identity id;
        return id.wsum(weight,
                       variable);
    }

    template<typename OutGradient, typename Weight>
    static EIGEN_STRONG_INLINE void
    gradient(const VariableMap<MatrixType, MapOptions, StrideType>& variable, OutGradient&& out_gradient, const Weight& weight)
    {
        Identity id;
        id.gradient(out_gradient,
                    weight,
                    variable);
    }

    template<typename OutHessian, typename Weight>
    static EIGEN_STRONG_INLINE void
    hessian(const VariableMap<MatrixType, MapOptions, StrideType>& variable, OutHessian&& out_hessian, const Weight& weight)
    {
        Identity id;
        id.hessian(out_hessian,
                   weight,
                   variable);
    }
};

} // namespace laopt

#endif // LAOPT_EXPR_EVALUATOR_HPP
