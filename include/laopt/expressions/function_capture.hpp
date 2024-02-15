#ifndef LAOPT_FUNCTION_CAPTURE_HPP
#define LAOPT_FUNCTION_CAPTURE_HPP

#include "Eigen/Dense"
#include "expr_base.hpp"
#include "expr_evaluator.hpp"

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

    static constexpr int n_inputs = Info::n_inputs;
    static constexpr int n_outputs = Info::n_outputs;
    using Scalar = typename Info::scalar_t;

    explicit FunctionCapture(Derived& func, const Capture& capture) : func(func), capture(capture) {}

    EIGEN_STRONG_INLINE Eigen::Vector<int, n_inputs> indices() const
    {
        return capture([&](auto&&... vars) {
            return concatenate_indices(get_indices(vars)...);
        });
    }

private:
    // We only consider IndexedVectors for AD. All other types don't contribute.
    template<typename Base>
    Eigen::Vector<int, IndexedVector<Base>::n_inputs> get_indices(const IndexedVector<Base>& var) const
    {
        return var.indices();
    }

    template<typename T>
    Eigen::Vector<int, 0> get_indices(T) const
    {
        return {};
    }
};

template<typename Function, typename Tag, typename Info, typename Capture>
struct ExprEvaluator<FunctionCapture<Function, Tag, Info, Capture>>
{
private:
    template<typename Base>
    static EIGEN_STRONG_INLINE const Base& cast_variable_to_base(const IndexedVector<Base>& arg)
    {
        return arg.cast_base();
    }

    template<typename T>
    static EIGEN_STRONG_INLINE const T& cast_variable_to_base(const T& arg)
    {
        return arg;
    }

public:
    static EIGEN_STRONG_INLINE auto
    function(const FunctionCapture<Function, Tag, Info, Capture>& function_capture)
    {
        return function_capture.capture([&](auto&&... vars) {
            return function_capture.func.function(Tag{},
                                                  cast_variable_to_base(vars)...);
        });
    }

    template<typename OutJacobian>
    static EIGEN_STRONG_INLINE void
    jacobian(const FunctionCapture<Function, Tag, Info, Capture>& function_capture, OutJacobian&& out_jacobian)
    {
        function_capture.capture([&](auto&&... vars) {
            function_capture.func.jacobian(Tag{},
                                           out_jacobian,
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

} // namespace laopt

#endif // LAOPT_FUNCTION_CAPTURE_HPP
