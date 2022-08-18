#ifndef LAOPT_EIGEN_AUTODIFF_FIX_HPP
#define LAOPT_EIGEN_AUTODIFF_FIX_HPP

namespace Eigen {

template<typename DerType, typename BinOp>
struct ScalarBinaryOpTraits<AutoDiffScalar<DerType>, typename DerType::Scalar::Scalar, BinOp> {
typedef AutoDiffScalar<DerType> ReturnType;
};

template<typename DerType, typename BinOp>
struct ScalarBinaryOpTraits<typename DerType::Scalar::Scalar, AutoDiffScalar<DerType>, BinOp>
{
typedef AutoDiffScalar<DerType> ReturnType;
};

} // end namespace Eigen

#endif // LAOPT_EIGEN_AUTODIFF_FIX_HPP
