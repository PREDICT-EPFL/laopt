#ifndef LAOPT_EIGEN_COMPAT_HPP
#define LAOPT_EIGEN_COMPAT_HPP

#include <Eigen/Core>

#if EIGEN_MAJOR_VERSION >= 5

namespace Eigen
{
static constexpr Eigen::internal::all_t all;

template <typename SizeType, typename IncrType>
auto lastN(SizeType size, IncrType incr) -> decltype(seqN(Eigen::placeholders::last - (size - fix<1>()) * incr, size, incr)) {
    return seqN(Eigen::placeholders::last - (size - fix<1>()) * incr, size, incr);
}

template <typename SizeType>
auto lastN(SizeType size) -> decltype(seqN(Eigen::placeholders::last + fix<1>() - size, size)) {
    return seqN(Eigen::placeholders::last + fix<1>() - size, size);
}
}

#endif

#endif //LAOPT_EIGEN_COMPAT_HPP
