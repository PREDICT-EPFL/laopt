#ifndef LAOPT_QUADRATIC_COST_HPP
#define LAOPT_QUADRATIC_COST_HPP

#include "laopt/autodiff/differentiable.hpp"

namespace laopt {

enum QuadraticCostOpt
{
    Diagonal = 0x0,
    Full     = 0x1
};

template<typename Scalar, int NX, int Options = Diagonal>
class QuadraticCost : public Differentiable<QuadraticCost<Scalar, NX, Options>>
{
public:
    template<int Dim>
    using MatrixType = typename std::conditional<(Options & Full) != 0, Eigen::Matrix<Scalar, Dim, Dim>, Eigen::DiagonalMatrix<Scalar, Dim>>::type;

    MatrixType<NX> Q;

    Eigen::Vector<Scalar, NX> x_ref = Eigen::Vector<Scalar, NX>::Zero();

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
