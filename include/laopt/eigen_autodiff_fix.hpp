#ifndef LAOPT_EIGEN_AUTODIFF_FIX_HPP
#define LAOPT_EIGEN_AUTODIFF_FIX_HPP

namespace Eigen {

// Allow 2nd derivative without casting
template<typename DerType, typename BinOp>
struct ScalarBinaryOpTraits<AutoDiffScalar<DerType>, typename DerType::Scalar::Scalar, BinOp>
{
typedef AutoDiffScalar<DerType> ReturnType;
};

template<typename DerType, typename BinOp>
struct ScalarBinaryOpTraits<typename DerType::Scalar::Scalar, AutoDiffScalar<DerType>, BinOp>
{
typedef AutoDiffScalar<DerType> ReturnType;
};

// Helper so we can use a comma in a c macro
#define LAOPT_COMMA ,

// Special expansion of EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY for abs to work with 2nd derivative
template<typename DerType, int... MatrixArgs>
inline const Eigen::AutoDiffScalar<
    EIGEN_EXPR_BINARYOP_SCALAR_RETURN_TYPE(typename Eigen::internal::remove_all<Eigen::Matrix<Eigen::AutoDiffScalar<DerType>LAOPT_COMMA MatrixArgs...>>::type,
                                           typename Eigen::internal::traits<typename Eigen::internal::remove_all<Eigen::Matrix<Eigen::AutoDiffScalar<DerType>LAOPT_COMMA MatrixArgs...>>::type>::Scalar, product)>
abs(const Eigen::AutoDiffScalar<Eigen::Matrix<Eigen::AutoDiffScalar<DerType>, MatrixArgs...>>& x)
{
    using namespace Eigen;
    typedef typename Eigen::internal::traits<typename Eigen::internal::remove_all<Eigen::Matrix<Eigen::AutoDiffScalar<DerType>, MatrixArgs...>>::type>::Scalar Scalar;
    EIGEN_UNUSED_VARIABLE(sizeof(Scalar));
    using std::abs;
    return Eigen::MakeAutoDiffScalar(abs(x.value()), x.derivatives() * (x.value() < Scalar(0) ? Scalar(-1) : Scalar(1)));
}

#undef LAOPT_COMMA

} // end namespace Eigen

#endif // LAOPT_EIGEN_AUTODIFF_FIX_HPP
