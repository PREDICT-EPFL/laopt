#ifndef LAOPT_DIFFERENTIABLE_HPP
#define LAOPT_DIFFERENTIABLE_HPP

#include <Eigen/Dense>

#include "../utility.hpp"
#include "../expressions/function_capture.hpp"
#include "../indexed_vector.hpp"
#include "differentiable_options.hpp"
#include "differentiable_eigen.hpp"
#include "differentiable_casadi.hpp"

namespace laopt
{

template<typename Derived, int Options>
class DifferentiableBase : public DifferentiableBaseEigen<Derived, Options>,
                           public DifferentiableBaseCasadi<Derived, Options>
{
public:
    template<typename... Args>
    EIGEN_STRONG_INLINE auto
    operator()(Args&&... args) noexcept
    {
        return static_cast<Derived*>(this)->function(std::forward<Args>(args)...);
    }

protected:
    using DifferentiableBaseEigen<Derived, Options>::jacobian_impl_autodiff_eval;
    using DifferentiableBaseEigen<Derived, Options>::jacobian_impl_autodiff_sparsity;
    using DifferentiableBaseEigen<Derived, Options>::hessian_impl_autodiff_eval;
    using DifferentiableBaseEigen<Derived, Options>::hessian_impl_autodiff_sparsity;

    using DifferentiableBaseCasadi<Derived, Options>::jacobian_impl_autodiff_eval;
    using DifferentiableBaseCasadi<Derived, Options>::jacobian_impl_autodiff_sparsity;
    using DifferentiableBaseCasadi<Derived, Options>::hessian_impl_autodiff_eval;
    using DifferentiableBaseCasadi<Derived, Options>::hessian_impl_autodiff_sparsity;

    template<typename Tag, typename OutJacobian, typename... Args>
    EIGEN_STRONG_INLINE void
    jacobian_impl_autodiff(Tag&& tag, OutJacobian& out_jacobian, const Args&... args) noexcept
    {
        this->jacobian_impl_autodiff_eval(std::forward<Tag>(tag), out_jacobian, args...);
    }

    template<typename Tag, typename... Args>
    EIGEN_STRONG_INLINE void
    jacobian_impl_autodiff(Tag&& tag, BSMatrixSparsity& out_jacobian, const Args&... args) noexcept
    {
        this->jacobian_impl_autodiff_sparsity(std::forward<Tag>(tag), out_jacobian, args...);
    }
    template<typename Tag, typename SparsityNullMat, typename... Args>
    EIGEN_STRONG_INLINE void
    jacobian_impl_autodiff(Tag&& tag, BSSliceSparsity<BSMatrixSparsity, SparsityNullMat>& out_jacobian, const Args&... args) noexcept
    {
        this->jacobian_impl_autodiff_sparsity(std::forward<Tag>(tag), out_jacobian, args...);
    }

    template<typename Tag, typename Weight, typename OutHessian, typename... Args>
    EIGEN_STRONG_INLINE void
    hessian_impl_autodiff(Tag&& tag,
                          OutHessian& out_hessian,
                          const Eigen::MatrixBase<Weight>& weight,
                          const Args&... args) noexcept
    {
        this->hessian_impl_autodiff_eval(std::forward<Tag>(tag), out_hessian, weight, args...);
    }

    template<typename Tag, typename Weight, typename... Args>
    EIGEN_STRONG_INLINE void
    hessian_impl_autodiff(Tag&& tag,
                          BSMatrixSparsity& out_hessian,
                          const Eigen::MatrixBase<Weight>& weight,
                          const Args&... args) noexcept
    {
        this->hessian_impl_autodiff_sparsity(std::forward<Tag>(tag), out_hessian, weight, args...);
    }

    template<typename Tag, typename Weight, typename SparsityNullMat, typename... Args>
    EIGEN_STRONG_INLINE void
    hessian_impl_autodiff(Tag&& tag,
                          BSSliceSparsity<BSMatrixSparsity, SparsityNullMat>& out_hessian,
                          const Eigen::MatrixBase<Weight>& weight,
                          const Args&... args) noexcept
    {
        this->hessian_impl_autodiff_sparsity(std::forward<Tag>(tag), out_hessian, weight, args...);
    }

    // returns the value w'*f(x)
    template<typename Tag, typename Weight, typename... Args, typename scalar_t = typename Eigen::MatrixBase<Weight>::Scalar>
    inline scalar_t
    wsum_impl_autodiff(Tag&& tag,
                       const Eigen::MatrixBase<Weight>& weight,
                       const Args&... args) noexcept
    {
        using Info = typename Derived::template FuncInfo<Tag, Args...>;
        Eigen::Vector<scalar_t, Info::n_outputs> value = static_cast<Derived*>(this)->function(std::forward<Tag>(tag), cast_variable_to_base(args)...);
        return weight.dot(value);
    }

    template<typename Base>
    inline const Base& cast_variable_to_base(const IndexedVector<Base>& arg)
    {
        return arg.cast_base();
    }

    template<typename T>
    inline const T& cast_variable_to_base(const T& arg)
    {
        return arg;
    }

    // returns the value w'*f(x) and *adds* the gradient to gradient
    template<typename Tag, typename Weight, typename Gradient, typename... Args>
    inline void
    gradient_impl_autodiff(Tag&& tag,
                           Gradient& out_gradient,
                           const Eigen::MatrixBase<Weight>& weight,
                           const Args&... args) noexcept
    {
        using Info = typename Derived::template FuncInfo<Tag, Args...>;
        using Scalar = typename Info::scalar_t;

        // Call the (possibly overloaded) jacobian
        Eigen::Matrix<Scalar, Info::n_outputs, Info::n_inputs> jacobian;
        jacobian.setZero();
        static_cast<Derived*>(this)->jacobian(std::forward<Tag>(tag), jacobian, args...);
        out_gradient += jacobian.transpose() * weight;
    }
};

template<typename T, typename... Args>
static auto test_user_jacobian(int) -> decltype(std::declval<T>().jacobian_impl(std::declval<Args>()...), std::true_type{});
template<typename T, typename... Args>
static auto test_user_jacobian(long) -> std::false_type;
template<typename T, typename... Args>
struct has_user_jacobian : decltype(test_user_jacobian<T, Args...>(0)){};

template<typename T, typename... Args>
static auto test_user_wsum(int) -> decltype(std::declval<T>().wsum_impl(std::declval<Args>()...), std::true_type{});
template<typename T, typename... Args>
static auto test_user_wsum(long) -> std::false_type;
template<typename T, typename... Args>
struct has_user_wsum : decltype(test_user_wsum<T, Args...>(0)){};

template<typename T, typename... Args>
static auto test_user_gradient(int) -> decltype(std::declval<T>().gradient_impl(std::declval<Args>()...), std::true_type{});
template<typename T, typename... Args>
static auto test_user_gradient(long) -> std::false_type;
template<typename T, typename... Args>
struct has_user_gradient : decltype(test_user_gradient<T, Args...>(0)){};

template<typename T, typename... Args>
static auto test_user_hessian(int) -> decltype(std::declval<T>().hessian_impl(std::declval<Args>()...), std::true_type{});
template<typename T, typename... Args>
static auto test_user_hessian(long) -> std::false_type;
template<typename T, typename... Args>
struct has_user_hessian : decltype(test_user_hessian<T, Args...>(0)){};

template<typename Derived, int Options = TAGLESS, bool Tagged = (Options & TAGGED) != 0>
class Differentiable;

template<typename Derived, int Options>
class Differentiable<Derived, Options, true> : public DifferentiableBase<Differentiable<Derived, Options, true>, Options>
{
public:

    template<typename Tag, typename... Vars>
    struct FuncInfo
    {
        using raw_return_t = decltype(std::declval<Derived>().function_impl(Tag{}, std::declval<Vars>()...));

        static constexpr int n_inputs = meta::sum_template<meta::variable_info<Vars>::size...>();
        static constexpr int n_outputs = meta::matrix_info<raw_return_t>::RowsAtCompileTime;

        using scalar_t = typename meta::matrix_info<raw_return_t>::Scalar;

        using return_t = Eigen::Vector<scalar_t, n_outputs>;
        using jacobian_t = Eigen::Matrix<scalar_t, n_outputs, n_inputs>;
        using gradient_t = Eigen::Vector<scalar_t, n_inputs>;
        using hessian_t = Eigen::Matrix<scalar_t, n_inputs, n_inputs>;

        using is_return_matrix = typename meta::matrix_info<raw_return_t>::is_matrix_t;

        static_assert(n_outputs >= 0 && n_inputs >= 0, "The function cannot have a dynamic size.");
    };

private:

    template<typename Tag, typename... Args>
    EIGEN_STRONG_INLINE typename FuncInfo<Tag, Args...>::return_t
    call_function_impl_helper(meta::IsScalar, Tag&& tag, const Args&... args) noexcept
    {
        using F = FuncInfo<Tag, Args...>;
        Eigen::Matrix<typename F::scalar_t, 1, 1> out_value;
        out_value(0) = static_cast<Derived*>(this)->function_impl(std::forward<Tag>(tag), args...);
        return out_value;
    }

    template<typename Tag, typename... Args>
    EIGEN_STRONG_INLINE typename FuncInfo<Tag, Args...>::return_t
    call_function_impl_helper(meta::IsMatrix, Tag&& tag, const Args&... args) noexcept
    {
        return static_cast<Derived*>(this)->function_impl(std::forward<Tag>(tag), args...);
    }

    // captures the function call for variables
    template<typename Tag, typename... Args>
    EIGEN_STRONG_INLINE auto
    call_function_impl(std::true_type, Tag, const Args&... args) noexcept
    {
        auto capture = [&](auto f)
        {
            return f(args...);
        };
        using Info = FuncInfo<Tag, Args...>;
        return FunctionCapture<Derived, Tag, Info, decltype(capture)>(*static_cast<Derived*>(this), capture);
    }

    template<typename Tag, typename... Args>
    EIGEN_STRONG_INLINE typename FuncInfo<Tag, Args...>::return_t
    call_function_impl(std::false_type, Tag&& tag, const Args&... args) noexcept
    {
        using F = FuncInfo<Tag, Args...>;
        return call_function_impl_helper(typename F::is_return_matrix(), std::forward<Tag>(tag), args...);
    }

public:

    // user specified function
    template<typename Tag, typename... Args>
    EIGEN_STRONG_INLINE auto
    function(Tag&& tag, const Args&... args) noexcept
    {
        return call_function_impl(meta::contains_variable<Args...>{}, std::forward<Tag>(tag), args...);
    }

    // user specified jacobian code
    template <typename Tag, typename OutJacobian, typename... Args>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_jacobian<Derived, Tag, OutJacobian&, Args...>() == true, void>::type
    jacobian(Tag&& tag, OutJacobian&& out_jacobian, const Args&... args) noexcept
    {
        static_cast<Derived*>(this)->jacobian_impl(std::forward<Tag>(tag), out_jacobian, args...);
    }

    // default internal jacobian code
    template <typename Tag, typename OutJacobian, typename... Args>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_jacobian<Derived, Tag, OutJacobian&, Args...>() == false, void>::type
    jacobian(Tag&& tag, OutJacobian&& out_jacobian, const Args&... args) noexcept
    {
        this->jacobian_impl_autodiff(std::forward<Tag>(tag), out_jacobian, args...);
    }

    // user specified wsum code
    template <typename Tag, typename Weight, typename... Args, typename scalar_t = typename Eigen::MatrixBase<Weight>::Scalar>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_wsum<Derived, Tag, Eigen::MatrixBase<Weight>, Args...>() == true, scalar_t>::type
    wsum(Tag&& tag, const Eigen::MatrixBase<Weight>& weight, const Args&... args) noexcept
    {
        return static_cast<Derived*>(this)->wsum_impl(std::forward<Tag>(tag), weight, args...);
    }

    // default internal wsum code
    template <typename Tag, typename Weight, typename... Args, typename scalar_t = typename Eigen::MatrixBase<Weight>::Scalar>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_wsum<Derived, Tag, Eigen::MatrixBase<Weight>, Args...>() == false, scalar_t>::type
    wsum(Tag&& tag, const Eigen::MatrixBase<Weight>& weight, const Args&... args) noexcept
    {
        return this->wsum_impl_autodiff(std::forward<Tag>(tag), weight, args...);
    }

    // user specified gradient code
    template <typename Tag, typename Weight, typename OutGradient, typename... Args>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_gradient<Derived, Tag, OutGradient&, Eigen::MatrixBase<Weight>, Args...>() == true, void>::type
    gradient(Tag&& tag, OutGradient&& out_gradient, const Eigen::MatrixBase<Weight>& weight, const Args&... args) noexcept
    {
        static_cast<Derived*>(this)->gradient_impl(std::forward<Tag>(tag), out_gradient, weight, args...);
    }

    // default internal gradient code
    template <typename Tag, typename Weight, typename OutGradient, typename... Args>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_gradient<Derived, Tag, OutGradient&, Eigen::MatrixBase<Weight>, Args...>() == false, void>::type
    gradient(Tag&& tag, OutGradient&& out_gradient, const Eigen::MatrixBase<Weight>& weight, const Args&... args) noexcept
    {
        this->gradient_impl_autodiff(std::forward<Tag>(tag), out_gradient, weight, args...);
    }

    // user specified hessian code
    template <typename Tag, typename Weight, typename OutHessian, typename... Args>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_hessian<Derived, Tag, OutHessian&, Eigen::MatrixBase<Weight>, Args...>() == true, void>::type
    hessian(Tag&& tag, OutHessian&& out_hessian, const Eigen::MatrixBase<Weight>& weight, const Args&... args) noexcept
    {
        static_cast<Derived*>(this)->hessian_impl(std::forward<Tag>(tag), out_hessian, weight, args...);
    }

    // default internal hessian code
    template <typename Tag, typename Weight, typename OutHessian, typename... Args>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_hessian<Derived, Tag, OutHessian&, Eigen::MatrixBase<Weight>, Args...>() == false, void>::type
    hessian(Tag&& tag, OutHessian&& out_hessian, const Eigen::MatrixBase<Weight>& weight, const Args&... args) noexcept
    {
        this->hessian_impl_autodiff(std::forward<Tag>(tag), out_hessian, weight, args...);
    }
};

template<typename Derived, int Options>
class Differentiable<Derived, Options, false> : public DifferentiableBase<Differentiable<Derived, Options, false>, Options>
{
public:

    template<typename Tag, typename... Vars>
    struct FuncInfo
    {
        using raw_return_t = decltype(std::declval<Derived>().function_impl(std::declval<Vars>()...));

        static constexpr int n_inputs = meta::sum_template<meta::variable_info<Vars>::size...>();
        static constexpr int n_outputs = meta::matrix_info<raw_return_t>::RowsAtCompileTime;

        using scalar_t = typename meta::matrix_info<raw_return_t>::Scalar;

        using return_t = Eigen::Vector<scalar_t, n_outputs>;
        using jacobian_t = Eigen::Matrix<scalar_t, n_outputs, n_inputs>;
        using gradient_t = Eigen::Vector<scalar_t, n_inputs>;
        using hessian_t = Eigen::Matrix<scalar_t, n_inputs, n_inputs>;

        using is_return_matrix = typename meta::matrix_info<raw_return_t>::is_matrix_t;

        static_assert(n_outputs >= 0 && n_inputs >= 0, "The function cannot have a dynamic size.");
    };

private:

    template<typename... Args>
    EIGEN_STRONG_INLINE typename FuncInfo<DefaultTag, Args...>::return_t
    call_function_impl_helper(meta::IsScalar, const Args&... args) noexcept
    {
        using F = FuncInfo<DefaultTag, Args...>;
        Eigen::Matrix<typename F::scalar_t, 1, 1> out_value;
        out_value(0) = static_cast<Derived*>(this)->function_impl(args...);
        return out_value;
    }

    template<typename... Args>
    EIGEN_STRONG_INLINE typename FuncInfo<DefaultTag, Args...>::return_t
    call_function_impl_helper(meta::IsMatrix, const Args&... args) noexcept
    {
        return static_cast<Derived*>(this)->function_impl(args...);
    }

    // captures the function call for variables
    template<typename... Args>
    EIGEN_STRONG_INLINE auto
    call_function_impl(std::true_type, const Args&... args) noexcept
    {
        auto capture = [&](auto f)
        {
            return f(args...);
        };
        using Info = FuncInfo<DefaultTag, Args...>;
        return FunctionCapture<Derived, DefaultTag, Info, decltype(capture)>(*static_cast<Derived*>(this), capture);
    }

    template<typename... Args>
    EIGEN_STRONG_INLINE typename FuncInfo<DefaultTag, Args...>::return_t
    call_function_impl(std::false_type, const Args&... args) noexcept
    {
        using F = FuncInfo<DefaultTag, Args...>;
        return call_function_impl_helper(typename F::is_return_matrix(), args...);
    }

public:

    // case with no arguments
    EIGEN_STRONG_INLINE auto
    function() noexcept
    {
        return call_function_impl(std::false_type{});
    }

    // user specified function code
    template<typename FirstArg, typename... Args>
    EIGEN_STRONG_INLINE auto
    function(const FirstArg& first_arg, const Args&... args) noexcept
    {
        return call_function_impl(meta::contains_variable<FirstArg, Args...>{}, first_arg, args...);
    }

    // define DefaultTag for function
    template<typename... Args>
    EIGEN_STRONG_INLINE auto
    function(DefaultTag, const Args&... args) noexcept
    {
        return function(args...);
    }

    // user specified jacobian code
    template <typename OutJacobian, typename... Args>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_jacobian<Derived, OutJacobian&, Args...>() == true, void>::type
    jacobian(OutJacobian&& out_jacobian, const Args&... args) noexcept
    {
        static_cast<Derived*>(this)->jacobian_impl(out_jacobian, args...);
    }

    // default internal jacobian code
    template <typename OutValue, typename OutJacobian, typename... Args>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_jacobian<Derived, OutValue&, OutJacobian&, Args...>() == false, void>::type
    jacobian(OutValue&& out_value, OutJacobian&& out_jacobian, const Args&... args) noexcept
    {
        this->jacobian_impl_autodiff(DefaultTag{}, out_value, out_jacobian, args...);
    }

    // define DefaultTag for jacobian
    template <typename OutValue, typename OutJacobian, typename... Args>
    EIGEN_STRONG_INLINE void
    jacobian(DefaultTag, OutValue&& out_value, OutJacobian&& out_jacobian, const Args&... args) noexcept
    {
        jacobian(std::forward<OutValue>(out_value), std::forward<OutJacobian>(out_jacobian), args...);
    }

    // user specified wsum code
    template <typename Weight, typename... Args, typename scalar_t = typename Eigen::MatrixBase<Weight>::Scalar>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_wsum<Derived, Eigen::MatrixBase<Weight>, Args...>() == true, scalar_t>::type
    wsum(const Eigen::MatrixBase<Weight>& weight, const Args&... args) noexcept
    {
        return static_cast<Derived*>(this)->wsum_impl(weight, args...);
    }

    // default internal wsum code
    template <typename Weight, typename... Args, typename scalar_t = typename Eigen::MatrixBase<Weight>::Scalar>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_wsum<Derived, Eigen::MatrixBase<Weight>, Args...>() == false, scalar_t>::type
    wsum(const Eigen::MatrixBase<Weight>& weight, const Args&... args) noexcept
    {
        return this->wsum_impl_autodiff(DefaultTag{}, weight, args...);
    }

    // define DefaultTag for wsum
    template <typename Weight, typename... Args, typename scalar_t = typename Eigen::MatrixBase<Weight>::Scalar>
    EIGEN_STRONG_INLINE scalar_t
    wsum(DefaultTag, const Eigen::MatrixBase<Weight>& weight, const Args&... args) noexcept
    {
        return wsum(weight, args...);
    }

    // user specified gradient code
    template <typename Weight, typename OutGradient, typename... Args>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_gradient<Derived, OutGradient&, Eigen::MatrixBase<Weight>, Args...>() == true, void>::type
    gradient(OutGradient&& out_gradient, const Eigen::MatrixBase<Weight>& weight, const Args&... args) noexcept
    {
        static_cast<Derived*>(this)->gradient_impl(out_gradient, weight, args...);
    }

    // default internal gradient code
    template <typename Weight, typename OutGradient, typename... Args>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_gradient<Derived, OutGradient&, Eigen::MatrixBase<Weight>, Args...>() == false, void>::type
    gradient(OutGradient&& out_gradient, const Eigen::MatrixBase<Weight>& weight, const Args&... args) noexcept
    {
        this->gradient_impl_autodiff(DefaultTag{}, out_gradient, weight, args...);
    }

    // define DefaultTag for gradient
    template <typename Weight, typename OutGradient, typename... Args>
    EIGEN_STRONG_INLINE void
    gradient(DefaultTag, OutGradient&& out_gradient, const Eigen::MatrixBase<Weight>& weight, const Args&... args) noexcept
    {
        gradient(std::forward<OutGradient>(out_gradient), weight, args...);
    }

    // user specified hessian code
    template <typename Weight, typename OutHessian, typename... Args>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_hessian<Derived, OutHessian&, Eigen::MatrixBase<Weight>, Args...>() == true, void>::type
    hessian(OutHessian&& out_hessian, const Eigen::MatrixBase<Weight>& weight, const Args&... args) noexcept
    {
        static_cast<Derived*>(this)->hessian_impl(out_hessian, weight, args...);
    }

    // default internal hessian code
    template <typename Weight, typename OutHessian, typename... Args>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_hessian<Derived, OutHessian&, Eigen::MatrixBase<Weight>, Args...>() == false, void>::type
    hessian(OutHessian&& out_hessian, const Eigen::MatrixBase<Weight>& weight, const Args&... args) noexcept
    {
        this->hessian_impl_autodiff(DefaultTag{}, out_hessian, weight, args...);
    }

    // define DefaultTag for hessian
    template <typename Weight, typename OutHessian, typename... Args>
    EIGEN_STRONG_INLINE void
    hessian(DefaultTag, OutHessian&& out_hessian, const Eigen::MatrixBase<Weight>& weight, const Args&... args) noexcept
    {
        hessian(std::forward<OutHessian>(out_hessian), weight, args...);
    }
};

} // namespace laopt

#endif // LAOPT_DIFFERENTIABLE_HPP