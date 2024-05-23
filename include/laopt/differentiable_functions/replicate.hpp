#ifndef LAOPT_REPLICATE_HPP
#define LAOPT_REPLICATE_HPP

#include "laopt/autodiff/differentiable.hpp"

namespace laopt {

template<int N>
class Replicate : public Differentiable<Replicate<N>>
{
public:
    template<typename X>
    EIGEN_STRONG_INLINE typename Eigen::Vector<typename X::Scalar, N * X::RowsAtCompileTime>
    function_impl(const Eigen::MatrixBase<X>& x) noexcept
    {
        return x.template replicate<N, 1>();
    }

    template<typename OutJacobian, typename AScalar, typename X>
    EIGEN_STRONG_INLINE void
    jacobian_impl(OutJacobian& jac, const AScalar& alpha, const Eigen::MatrixBase<X>& x) noexcept
    {
        for (int i = 0; i < N; i++) {
            jac(Eigen::seqN(i * X::RowsAtCompileTime, Eigen::fix<X::RowsAtCompileTime>), Eigen::all).diagonal() += X::Constant(alpha);
        }
    }

    template <typename Weight, typename X, typename scalar_t = typename Eigen::MatrixBase<Weight>::Scalar>
    EIGEN_STRONG_INLINE scalar_t
    wsum_impl(const Eigen::MatrixBase<Weight>& weight, const Eigen::MatrixBase<X>& x) noexcept
    {
        return weight.dot(x.template replicate<N, 1>());
    }

    template <typename Weight, typename OutGradient, typename X, typename scalar_t = typename Eigen::MatrixBase<Weight>::Scalar>
    EIGEN_STRONG_INLINE void
    gradient_impl(OutGradient& out_gradient, const Eigen::MatrixBase<Weight>& weight, const Eigen::MatrixBase<X>& x) noexcept
    {
        out_gradient += Eigen::Reshaped<const Weight, X::RowsAtCompileTime, N, Eigen::ColMajor>(weight.derived()).rowwise().sum();
    }

    template <typename Weight, typename OutHessian, typename X, typename scalar_t = typename Eigen::MatrixBase<Weight>::Scalar>
    EIGEN_STRONG_INLINE void
    hessian_impl(OutHessian&, const Eigen::MatrixBase<Weight>& weight, const Eigen::MatrixBase<X>& x) noexcept
    {
        // Hessian is zero, i.e., we don't set any values
    }
};

} // namespace laopt

#endif //LAOPT_REPLICATE_HPP
