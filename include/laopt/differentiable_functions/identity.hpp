#ifndef LAOPT_IDENTITY_HPP
#define LAOPT_IDENTITY_HPP

#include "laopt/autodiff/differentiable.hpp"

namespace laopt {

class Identity : public Differentiable<Identity>
{
public:
    template<typename X>
    EIGEN_STRONG_INLINE typename X::PlainObject
    function_impl(const Eigen::MatrixBase<X>& x) noexcept
    {
        return x;
    }

    template<typename OutJacobian, typename AScalar, typename X>
    EIGEN_STRONG_INLINE void
    jacobian_impl(OutJacobian& jac, const AScalar& alpha, const Eigen::MatrixBase<X>& x) noexcept
    {
        jac.diagonal() += X::Constant(alpha);
    }

    template <typename Weight, typename X, typename scalar_t = typename Eigen::MatrixBase<Weight>::Scalar>
    EIGEN_STRONG_INLINE scalar_t
    wsum_impl(const Eigen::MatrixBase<Weight>& weight, const Eigen::MatrixBase<X>& x) noexcept
    {
        return weight.dot(x);
    }

    template <typename Weight, typename OutGradient, typename X, typename scalar_t = typename Eigen::MatrixBase<Weight>::Scalar>
    EIGEN_STRONG_INLINE void
    gradient_impl(OutGradient& out_gradient, const Eigen::MatrixBase<Weight>& weight, const Eigen::MatrixBase<X>& x) noexcept
    {
        out_gradient += weight;
    }

    template <typename Weight, typename OutHessian, typename X, typename scalar_t = typename Eigen::MatrixBase<Weight>::Scalar>
    EIGEN_STRONG_INLINE void
    hessian_impl(OutHessian&, const Eigen::MatrixBase<Weight>& weight, const Eigen::MatrixBase<X>& x) noexcept
    {
        // Hessian is zero, i.e., we don't set any values
    }
};

} // namespace laopt

#endif //LAOPT_IDENTITY_HPP
