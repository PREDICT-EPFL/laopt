#ifndef LAOPT_DIFFERENTIABLE_OPTIONS_HPP
#define LAOPT_DIFFERENTIABLE_OPTIONS_HPP

namespace laopt
{

enum DiffOptions
{
    TAGLESS         = 0x00,
    TAGGED          = 0x01,
    EIGEN_ALL       = 0x00,
    CASADI_JACOBIAN = 0x02,
    CASADI_HESSIAN  = 0x04,
    CASADI_NO_JIT   = 0x08,
    CASADI_ALL      = CASADI_JACOBIAN | CASADI_HESSIAN,
};

struct DefaultTag {};

template<typename T>
static auto test_tag_eigen(int) -> decltype(typename T::UseEigen{}, std::true_type{});
template<typename T>
static auto test_tag_eigen(long) -> std::false_type;
template<typename T>
struct has_tag_eigen : decltype(test_tag_eigen<T>(0)){};

template<typename T>
static auto test_tag_casadi(int) -> decltype(typename T::UseCasadi{}, std::true_type{});
template<typename T>
static auto test_tag_casadi(long) -> std::false_type;
template<typename T>
struct has_tag_casadi : decltype(test_tag_casadi<T>(0)){};

template<typename T>
struct has_tag_override : std::conditional<has_tag_eigen<T>() || has_tag_casadi<T>(), std::true_type, std::false_type>
{
    static_assert(!(has_tag_eigen<T>() && has_tag_casadi<T>()), "Only Eigen or Casadi can be used for AD");
};

} // namespace laopt

#endif //LAOPT_DIFFERENTIABLE_OPTIONS_HPP
