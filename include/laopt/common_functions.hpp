#ifndef LAOPT_COMMON_FUNCTIONS_HPP
#define LAOPT_COMMON_FUNCTIONS_HPP

#include "laopt/autodiff/differentiable.hpp"

namespace laopt {

namespace common_functions {

template<typename F, typename Scalar, typename Tag = DefaultTag, int Options = TAGLESS>
class RK4 : public Differentiable<RK4<F, Scalar, Tag, Options>, Options>
{
    F& f;
    Scalar h;

public:
    explicit RK4(F& f, Scalar step_size) : f(f), h(step_size) {}

    template<typename X, typename... Params>
    EIGEN_STRONG_INLINE typename X::PlainObject
    function_impl(const Eigen::MatrixBase<X>& x, const Eigen::MatrixBase<Params>&... params) noexcept
    {
        using Vec = typename X::PlainObject;
        Vec k1 = f.function(Tag{}, x,          params...);
        Vec k2 = f.function(Tag{}, x+h*0.5*k1, params...);
        Vec k3 = f.function(Tag{}, x+h*0.5*k2, params...);
        Vec k4 = f.function(Tag{}, x+h*k3,     params...);
        return x + h/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4);
    }
};

} // namespace common_functions

} // namespace laopt

#endif // LAOPT_COMMON_FUNCTIONS_HPP
