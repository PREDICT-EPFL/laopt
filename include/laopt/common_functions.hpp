#ifndef LAOPT_COMMON_FUNCTIONS_HPP
#define LAOPT_COMMON_FUNCTIONS_HPP

#include "autodiff/differentiable.hpp"

namespace laopt {

namespace common_functions {

class IDENTITY : public Differentiable<IDENTITY, TAGLESS>
{
    const double multiplier;

public:
    IDENTITY() : multiplier(1) {}
    explicit IDENTITY(double multiplier) : multiplier(multiplier) {}

    template<typename X>
    EIGEN_STRONG_INLINE typename X::PlainObject
    function_impl(const Eigen::MatrixBase<X>& x) noexcept
    {
        return multiplier * x;
    }

    template<typename OutJacobian, typename X>
    EIGEN_STRONG_INLINE void
    jacobian_impl(OutJacobian& jac, const Eigen::MatrixBase<X>& x) noexcept
    {
        using scalar_t = typename Eigen::MatrixBase<X>::Scalar;
        for(int i = 0; i < x.rows(); i++)
        {
            jac(Eigen::seqN(i, Eigen::fix<1>), Eigen::seqN(i, Eigen::fix<1>)) = Eigen::Matrix<scalar_t, 1, 1>::Constant(multiplier);
        }
    }

    template <typename Weight, typename X, typename scalar_t = typename Eigen::MatrixBase<Weight>::Scalar>
    EIGEN_STRONG_INLINE scalar_t
    wsum_impl(const Eigen::MatrixBase<Weight>& weight, const Eigen::MatrixBase<X>& x) noexcept
    {
        return multiplier * weight.dot(x);
    }

    template <typename Weight, typename OutGradient, typename X, typename scalar_t = typename Eigen::MatrixBase<Weight>::Scalar>
    EIGEN_STRONG_INLINE void
    gradient_impl(OutGradient& out_gradient, const Eigen::MatrixBase<Weight>& weight, const Eigen::MatrixBase<X>& x) noexcept
    {
        out_gradient += multiplier * weight;
    }

    template <typename Weight, typename OutHessian, typename X, typename scalar_t = typename Eigen::MatrixBase<Weight>::Scalar>
    EIGEN_STRONG_INLINE void
    hessian_impl(OutHessian&, const Eigen::MatrixBase<Weight>& weight, const Eigen::MatrixBase<X>& x) noexcept
    {
        // Hessian is zero, i.e., we don't set any values
    }
};

template<typename F, typename Scalar, typename Tag = DefaultTag>
class RK4 : public Differentiable<RK4<F, Scalar, Tag>, TAGLESS>
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
        Vec k1 = f(Tag{}, x,          params...);
        Vec k2 = f(Tag{}, x+h*0.5*k1, params...);
        Vec k3 = f(Tag{}, x+h*0.5*k2, params...);
        Vec k4 = f(Tag{}, x+h*k3,     params...);
        return x + h/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4);
    }
};

} // namespace common_functions

} // namespace laopt

#endif // LAOPT_COMMON_FUNCTIONS_HPP
