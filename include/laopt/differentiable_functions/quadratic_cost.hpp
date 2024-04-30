#ifndef LAOPT_QUADRATIC_COST_HPP
#define LAOPT_QUADRATIC_COST_HPP

#include "laopt/autodiff/differentiable.hpp"

namespace laopt {

template<typename MatrixType>
class QuadraticCost : public Differentiable<QuadraticCost<MatrixType>>
{
public:
    static_assert(MatrixType::RowsAtCompileTime == MatrixType::ColsAtCompileTime, "Matrix type is not quadratic");
    static_assert(MatrixType::RowsAtCompileTime >= 0 && MatrixType::ColsAtCompileTime >= 0, "Matrix type can't be dynamic");
    using VectorType = Eigen::Vector<typename MatrixType::Scalar, MatrixType::RowsAtCompileTime>;

    MatrixType Q;
    VectorType x_ref = VectorType::Zero();

    template<typename X, typename scalar_t = typename Eigen::MatrixBase<X>::Scalar>
    EIGEN_STRONG_INLINE scalar_t
    function_impl(const Eigen::MatrixBase<X> &x)
    {
        return (x - x_ref).dot(Q * (x - x_ref));
    }

    template<typename OutJacobian, typename AScalar, typename X>
    EIGEN_STRONG_INLINE void
    jacobian_impl(OutJacobian& jac,
                  const AScalar& alpha,
                  const Eigen::MatrixBase<X>& x) noexcept
    {
        jac += alpha * 2 * Q * (x - x_ref);
    }

    template <typename Weight, typename X, typename scalar_t = typename Eigen::MatrixBase<Weight>::Scalar>
    EIGEN_STRONG_INLINE scalar_t
    wsum_impl(const Eigen::MatrixBase<Weight>& weight,
              const Eigen::MatrixBase<X>& x) noexcept
    {
        return weight(0) * function_impl(x);
    }

    template <typename Weight, typename OutGradient, typename X>
    EIGEN_STRONG_INLINE void
    gradient_impl(OutGradient& out_gradient,
                  const Eigen::MatrixBase<Weight>& weight,
                  const Eigen::MatrixBase<X>& x) noexcept
    {
        out_gradient += weight(0) * 2 * Q * (x - x_ref);
    }

    template <typename Weight, typename OutHessian, typename X>
    EIGEN_STRONG_INLINE void
    hessian_impl(OutHessian& out_hessian,
                 const Eigen::MatrixBase<Weight>& weight,
                 const Eigen::MatrixBase<X>& x) noexcept
    {
        out_hessian += weight(0) * 2 * Q;
    }
};

} // namespace laopt

#endif //LAOPT_QUADRATIC_COST_HPP
