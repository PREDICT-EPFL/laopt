#ifndef LAOPT_FUNCTION_CAPTURE_HPP
#define LAOPT_FUNCTION_CAPTURE_HPP

#include "Eigen/Dense"
#include "expr_base.hpp"

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

} // namespace laopt

#endif // LAOPT_FUNCTION_CAPTURE_HPP
