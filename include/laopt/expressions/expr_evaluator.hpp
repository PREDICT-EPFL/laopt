#ifndef LAOPT_EXPR_EVALUATOR_HPP
#define LAOPT_EXPR_EVALUATOR_HPP

#include "../indexed_vector.hpp"
#include "../functions.hpp"

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
        functions::IDENTITY id;
        return id.function(indexed_vector.cast_base());
    }

    template<typename OutValue, typename OutJacobian>
    static EIGEN_STRONG_INLINE void
    jacobian(const IndexedVector<Derived>& indexed_vector, OutValue&& out_value, OutJacobian&& out_jacobian)
    {
        functions::IDENTITY id;
        id.jacobian(out_value,
                    out_jacobian,
                    indexed_vector.cast_base());
    }

    template<typename Weight>
    static EIGEN_STRONG_INLINE auto
    wsum(const IndexedVector<Derived>& indexed_vector, const Weight& weight)
    {
        functions::IDENTITY id;
        return id.wsum(weight,
                       indexed_vector.cast_base());
    }

    template<typename OutGradient, typename Weight>
    static EIGEN_STRONG_INLINE auto
    gradient(const IndexedVector<Derived>& indexed_vector, OutGradient&& out_gradient, const Weight& weight)
    {
        functions::IDENTITY id;
        return id.gradient(out_gradient,
                           weight,
                           indexed_vector.cast_base());
    }

    template<typename OutGradient, typename OutHessian, typename Weight>
    static EIGEN_STRONG_INLINE auto
    hessian(const IndexedVector<Derived>& indexed_vector, OutGradient&& out_gradient, OutHessian&& out_hessian, const Weight& weight)
    {
        functions::IDENTITY id;
        return id.hessian(out_gradient,
                          out_hessian,
                          weight,
                          indexed_vector.cast_base());
    }
};

template<typename Function, typename Tag, typename Info, typename Capture>
struct ExprEvaluator<FunctionCapture<Function, Tag, Info, Capture>>
{
    static EIGEN_STRONG_INLINE auto
    function(const FunctionCapture<Function, Tag, Info, Capture>& function_capture)
    {
        return function_capture.capture([&](auto&&... vars) {
            return function_capture.func.function(Tag{},
                                                  vars.cast_base()...);
        });
    }

    template<typename OutValue, typename OutJacobian>
    static EIGEN_STRONG_INLINE void
    jacobian(const FunctionCapture<Function, Tag, Info, Capture>& function_capture, OutValue&& out_value, OutJacobian&& out_jacobian)
    {
        function_capture.capture([&](auto&&... vars) {
            function_capture.func.jacobian(Tag{},
                                           out_value,
                                           out_jacobian,
                                           vars.cast_base()...);
        });
    }

    template<typename Weight>
    static EIGEN_STRONG_INLINE auto
    wsum(const FunctionCapture<Function, Tag, Info, Capture>& function_capture, const Weight& weight)
    {
        return function_capture.capture([&](auto&&... vars) {
            return function_capture.func.wsum(Tag{},
                                              weight,
                                              vars.cast_base()...);
        });
    }

    template<typename OutGradient, typename Weight>
    static EIGEN_STRONG_INLINE auto
    gradient(const FunctionCapture<Function, Tag, Info, Capture>& function_capture, OutGradient&& out_gradient, const Weight& weight)
    {
        return function_capture.capture([&](auto&&... vars) {
            return function_capture.func.gradient(Tag{},
                                                  out_gradient,
                                                  weight,
                                                  vars.cast_base()...);
        });
    }

    template<typename OutGradient, typename OutHessian, typename Weight>
    static EIGEN_STRONG_INLINE auto
    hessian(const FunctionCapture<Function, Tag, Info, Capture> function_capture, OutGradient&& out_gradient, OutHessian&& out_hessian, const Weight& weight)
    {
        return function_capture.capture([&](auto&&... vars) {
            return function_capture.func.hessian(Tag{},
                                                 out_gradient,
                                                 out_hessian,
                                                 weight,
                                                 vars.cast_base()...);
        });
    }
};

} // namespace laopt

#endif // LAOPT_EXPR_EVALUATOR_HPP
