#ifndef LAOPT_FUNCTION_CAPTURE_HPP
#define LAOPT_FUNCTION_CAPTURE_HPP

#include <Eigen/Dense>
#include "laopt/expressions/expr_base.hpp"
#include "laopt/expressions/expr_evaluator.hpp"

namespace laopt
{

/*
 * This struct captures a function call.
 * Capture is a lambda which takes another lambda as input which is then called with the original parameters.
 */
template<typename Derived, typename Tag, typename Info, typename Capture>
class FunctionCapture : public ExprBase<FunctionCapture<Derived, Tag, Info, Capture>>
{
public:
    Derived& func;
    Capture capture;

    enum {
        RowsAtCompileTime = Info::n_outputs,
        ColsAtCompileTime = 1
    };
    using Scalar = typename Info::scalar_t;

    explicit FunctionCapture(Derived& func, const Capture& capture) : func(func), capture(capture) {}

    EIGEN_STRONG_INLINE Eigen::Vector<int, Info::n_inputs> indices() const
    {
        return capture([&](auto&&... vars) {
            return concatenate_indices(get_indices(vars, is_variable<decltype(vars)>{})...);
        });
    }

private:
    // We only consider IndexedVectors for AD. All other types don't contribute.
    template<typename T>
    EIGEN_STRONG_INLINE Eigen::Vector<int, T::RowsAtCompileTime>
    get_indices(const T& var, std::true_type) const
    {
        return variable_indices(var);
    }

    template<typename T>
    EIGEN_STRONG_INLINE Eigen::Vector<int, 0>
    get_indices(const T& var, std::false_type) const
    {
        return {};
    }
};

template<typename Function, typename Tag, typename Info, typename Capture>
struct ExprEvaluator<FunctionCapture<Function, Tag, Info, Capture>, typename std::enable_if<Info::is_return_expr::value>::type>
{
public:
    static EIGEN_STRONG_INLINE auto
    function(const FunctionCapture<Function, Tag, Info, Capture>& function_capture)
    {
        return function_capture.capture([&](auto&&... vars) {
            return function_capture.func.function(Tag{},
                                                  vars...);
        });
    }

    template<typename OutJacobian, typename AScalar>
    static EIGEN_STRONG_INLINE void
    jacobian(const FunctionCapture<Function, Tag, Info, Capture>& function_capture, OutJacobian&& out_jacobian, const AScalar& alpha)
    {
        function_capture.capture([&](auto&&... vars) {
            function_capture.func.jacobian(Tag{},
                                           out_jacobian,
                                           alpha,
                                           vars...);
        });
    }

    template<typename Weight>
    static EIGEN_STRONG_INLINE auto
    wsum(const FunctionCapture<Function, Tag, Info, Capture>& function_capture, const Weight& weight)
    {
        return function_capture.capture([&](auto&&... vars) {
            return function_capture.func.wsum(Tag{},
                                              weight,
                                              vars...);
        });
    }

    template<typename OutGradient, typename Weight>
    static EIGEN_STRONG_INLINE void
    gradient(const FunctionCapture<Function, Tag, Info, Capture>& function_capture, OutGradient&& out_gradient, const Weight& weight)
    {
        return function_capture.capture([&](auto&&... vars) {
            function_capture.func.gradient(Tag{},
                                           out_gradient,
                                           weight,
                                           vars...);
        });
    }

    template<typename OutHessian, typename Weight>
    static EIGEN_STRONG_INLINE void
    hessian(const FunctionCapture<Function, Tag, Info, Capture> function_capture, OutHessian&& out_hessian, const Weight& weight)
    {
        return function_capture.capture([&](auto&&... vars) {
            function_capture.func.hessian(Tag{},
                                          out_hessian,
                                          weight,
                                          vars...);
        });
    }
};

template<typename Function, typename Tag, typename Info, typename Capture>
struct ExprEvaluator<FunctionCapture<Function, Tag, Info, Capture>, typename std::enable_if<!Info::is_return_expr::value>::type>
{
public:
    static EIGEN_STRONG_INLINE auto
    function(const FunctionCapture<Function, Tag, Info, Capture>& function_capture)
    {
        return function_capture.capture([&](auto&&... vars) {
            return function_capture.func.function(Tag{},
                                                  vars...);
        });
    }

    template<typename OutJacobian, typename AScalar>
    static EIGEN_STRONG_INLINE void
    jacobian(const FunctionCapture<Function, Tag, Info, Capture>& function_capture, OutJacobian&& out_jacobian, const AScalar& alpha)
    {
        function_capture.capture([&](auto&&... vars) {
            function_capture.func.jacobian(Tag{},
                                           out_jacobian(Eigen::all, function_capture.indices()),
                                           alpha,
                                           vars...);
        });
    }

    template<typename Weight>
    static EIGEN_STRONG_INLINE auto
    wsum(const FunctionCapture<Function, Tag, Info, Capture>& function_capture, const Weight& weight)
    {
        return function_capture.capture([&](auto&&... vars) {
            return function_capture.func.wsum(Tag{},
                                              weight,
                                              vars...);
        });
    }

    template<typename OutGradient, typename Weight>
    static EIGEN_STRONG_INLINE void
    gradient(const FunctionCapture<Function, Tag, Info, Capture>& function_capture, OutGradient&& out_gradient, const Weight& weight)
    {
        return function_capture.capture([&](auto&&... vars) {
            function_capture.func.gradient(Tag{},
                                           out_gradient(function_capture.indices()),
                                           weight,
                                           vars...);
        });
    }

    template<typename OutHessian, typename Weight>
    static EIGEN_STRONG_INLINE void
    hessian(const FunctionCapture<Function, Tag, Info, Capture> function_capture, OutHessian&& out_hessian, const Weight& weight)
    {
        return function_capture.capture([&](auto&&... vars) {
            auto indices = function_capture.indices();
            function_capture.func.hessian(Tag{},
                                          out_hessian(indices, indices),
                                          weight,
                                          vars...);
        });
    }
};

} // namespace laopt

#endif // LAOPT_FUNCTION_CAPTURE_HPP
