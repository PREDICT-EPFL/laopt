#ifndef LAOPT_EXPR_EVALUATOR_HPP
#define LAOPT_EXPR_EVALUATOR_HPP

#include "laopt/expressions/fwd.hpp"
#include "laopt/variable_map.hpp"
#include "laopt/differentiable_functions/identity.hpp"

namespace laopt {

template<typename T>
struct ExprEvaluator<T, typename std::enable_if<is_variable<T>::value>::type>
{
    static EIGEN_STRONG_INLINE auto
    function(const T& variable)
    {
        Identity id;
        return id.function(variable);
    }

    template<typename OutJacobian, typename AScalar>
    static EIGEN_STRONG_INLINE void
    jacobian(const T& variable, OutJacobian&& out_jacobian, const AScalar& alpha)
    {
        Identity id;
        id.jacobian(out_jacobian,
                    alpha,
                    variable);
    }

    template<typename Weight>
    static EIGEN_STRONG_INLINE auto
    wsum(const T& variable, const Weight& weight)
    {
        Identity id;
        return id.wsum(weight,
                       variable);
    }

    template<typename OutGradient, typename Weight>
    static EIGEN_STRONG_INLINE void
    gradient(const T& variable, OutGradient&& out_gradient, const Weight& weight)
    {
        Identity id;
        id.gradient(out_gradient,
                    weight,
                    variable);
    }

    template<typename OutHessian, typename Weight>
    static EIGEN_STRONG_INLINE void
    hessian(const T& variable, OutHessian&& out_hessian, const Weight& weight)
    {
        Identity id;
        id.hessian(out_hessian,
                   weight,
                   variable);
    }
};

} // namespace laopt

#endif // LAOPT_EXPR_EVALUATOR_HPP
