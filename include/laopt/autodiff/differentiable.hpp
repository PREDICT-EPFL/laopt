#ifndef LAOPT_DIFFERENTIABLE_HPP
#define LAOPT_DIFFERENTIABLE_HPP

#include <Eigen/Dense>

#include "laopt/utility.hpp"
#include "laopt/expressions/fwd.hpp"
#include "laopt/indexed_vector.hpp"
#include "laopt/autodiff/differentiable_options.hpp"
#include "laopt/autodiff/differentiable_eigen.hpp"
#include "laopt/autodiff/differentiable_casadi.hpp"

namespace laopt
{

template<typename Derived>
class DifferentiableBase {};

template<typename Derived, int Options>
class DifferentiableImpl : public DifferentiableImplEigen<Derived, Options>,
                           public DifferentiableImplCasadi<Derived, Options>
{
protected:
    using DifferentiableImplEigen<Derived, Options>::jacobian_impl_autodiff_eval;
    using DifferentiableImplEigen<Derived, Options>::jacobian_impl_autodiff_sparsity;
    using DifferentiableImplEigen<Derived, Options>::hessian_impl_autodiff_eval;
    using DifferentiableImplEigen<Derived, Options>::hessian_impl_autodiff_sparsity;

    using DifferentiableImplCasadi<Derived, Options>::jacobian_impl_autodiff_eval;
    using DifferentiableImplCasadi<Derived, Options>::jacobian_impl_autodiff_sparsity;
    using DifferentiableImplCasadi<Derived, Options>::hessian_impl_autodiff_eval;
    using DifferentiableImplCasadi<Derived, Options>::hessian_impl_autodiff_sparsity;

    template<typename Tag, typename OutJacobian, typename AScalar, typename... Args>
    EIGEN_STRONG_INLINE void
    jacobian_impl_autodiff(const Tag& tag, OutJacobian& out_jacobian, const AScalar& alpha, const Args&... args) noexcept
    {
        using Info = typename Derived::template FuncInfo<Tag, Args...>;
        if (Info::n_inputs > 0 && Info::n_outputs > 0) {
            this->jacobian_impl_autodiff_eval(tag, out_jacobian, alpha, args...);
        }
    }

    template<typename Tag, typename AScalar, typename... Args>
    EIGEN_STRONG_INLINE void
    jacobian_impl_autodiff(const Tag& tag, BSMatrixSparsity& out_jacobian, const AScalar& alpha, const Args&... args) noexcept
    {
        using Info = typename Derived::template FuncInfo<Tag, Args...>;
        if (Info::n_inputs > 0 && Info::n_outputs > 0) {
            this->jacobian_impl_autodiff_sparsity(tag, out_jacobian, alpha, args...);
        }
    }
    template<typename Tag, typename AScalar, typename SparsityNullMat, typename... Args>
    EIGEN_STRONG_INLINE void
    jacobian_impl_autodiff(const Tag& tag, BSSliceSparsity<BSMatrixSparsity, SparsityNullMat>& out_jacobian, const AScalar& alpha, const Args&... args) noexcept
    {
        using Info = typename Derived::template FuncInfo<Tag, Args...>;
        if (Info::n_inputs > 0 && Info::n_outputs > 0) {
            this->jacobian_impl_autodiff_sparsity(tag, out_jacobian, alpha, args...);
        }
    }

    template<typename Tag, typename OutHessian, typename Weight, typename... Args>
    EIGEN_STRONG_INLINE void
    hessian_impl_autodiff(const Tag& tag,
                          OutHessian& out_hessian,
                          const Eigen::MatrixBase<Weight>& weight,
                          const Args&... args) noexcept
    {
        using Info = typename Derived::template FuncInfo<Tag, Args...>;
        if (Info::n_inputs > 0 && Info::n_outputs > 0) {
            this->hessian_impl_autodiff_eval(tag, out_hessian, weight, args...);
        }
    }

    template<typename Tag, typename Weight, typename... Args>
    EIGEN_STRONG_INLINE void
    hessian_impl_autodiff(const Tag& tag,
                          BSMatrixSparsity& out_hessian,
                          const Eigen::MatrixBase<Weight>& weight,
                          const Args&... args) noexcept
    {
        using Info = typename Derived::template FuncInfo<Tag, Args...>;
        if (Info::n_inputs > 0 && Info::n_outputs > 0) {
            this->hessian_impl_autodiff_sparsity(tag, out_hessian, weight, args...);
        }
    }

    template<typename Tag, typename SparsityNullMat, typename Weight, typename... Args>
    EIGEN_STRONG_INLINE void
    hessian_impl_autodiff(const Tag& tag,
                          BSSliceSparsity<BSMatrixSparsity, SparsityNullMat>& out_hessian,
                          const Eigen::MatrixBase<Weight>& weight,
                          const Args&... args) noexcept
    {
        using Info = typename Derived::template FuncInfo<Tag, Args...>;
        if (Info::n_inputs > 0 && Info::n_outputs > 0) {
            this->hessian_impl_autodiff_sparsity(tag, out_hessian, weight, args...);
        }
    }

    // returns the value w'*f(x)
    template<typename Tag, typename Weight, typename... Args, typename scalar_t = typename Eigen::MatrixBase<Weight>::Scalar>
    inline scalar_t
    wsum_impl_autodiff(const Tag& tag,
                       const Eigen::MatrixBase<Weight>& weight,
                       const Args&... args) noexcept
    {
        using Info = typename Derived::template FuncInfo<Tag, Args...>;
        Eigen::Vector<scalar_t, Info::n_outputs> value = to_matrix_type(static_cast<Derived*>(this)->function(tag, args...));
        return weight.dot(value);
    }

    // returns the value w'*f(x) and *adds* the gradient to gradient
    template<typename Tag, typename Gradient, typename Weight, typename... Args>
    inline void
    gradient_impl_autodiff(const Tag& tag,
                           Gradient& out_gradient,
                           const Eigen::MatrixBase<Weight>& weight,
                           const Args&... args) noexcept
    {
        using Info = typename Derived::template FuncInfo<Tag, Args...>;
        using Scalar = typename Info::scalar_t;

        // Call the (possibly overloaded) jacobian
        Eigen::Matrix<Scalar, Info::n_outputs, Info::n_inputs> jacobian;
        jacobian.setZero();
        static_cast<Derived*>(this)->jacobian(tag, jacobian, 1, args...);
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
class Differentiable<Derived, Options, true> : public DifferentiableBase<Derived>,
                                               public DifferentiableImpl<Differentiable<Derived, Options, true>, Options>
{
public:

    template<typename Tag, typename... Vars>
    struct FuncInfo
    {
    private:
        template<typename Derived_ = Derived>
        static auto get_raw_return_t(int) -> decltype(std::declval<Derived_>().function_impl(Tag{}, std::declval<Vars>()...));
        template<typename Derived_ = Derived>
        static auto get_raw_return_t(long) -> std::false_type;
        template<typename raw_return_t>
        static auto get_expr_return_t(int) -> decltype(ExprEvaluator<raw_return_t>::function(std::declval<raw_return_t>()));
        template<typename raw_return_t>
        static auto get_expr_return_t(long) -> raw_return_t;

    public:
        using raw_return_t = decltype(get_raw_return_t<>(0));
        using is_return_expr = typename std::is_base_of<ExprBase<raw_return_t>, raw_return_t>;
        using return_t = decltype(get_expr_return_t<raw_return_t>(0));

        using scalar_t = typename meta::matrix_info<return_t>::Scalar;

        static constexpr int n_inputs = meta::sum_template<meta::variable_info<Vars>::size...>();
        static constexpr int n_outputs = meta::matrix_info<return_t>::RowsAtCompileTime;

        using jacobian_t = Eigen::Matrix<scalar_t, n_outputs, n_inputs>;
        using gradient_t = Eigen::Vector<scalar_t, n_inputs>;
        using hessian_t = Eigen::Matrix<scalar_t, n_inputs, n_inputs>;

        static_assert(n_outputs >= 0 && n_inputs >= 0, "The function cannot have a dynamic size.");
    };

private:

    template<typename Tag, typename... Args>
    EIGEN_STRONG_INLINE auto
    call_function_impl_expr(std::true_type, const Tag& tag, const Args&... args) noexcept
    {
        return ExprEvaluator<typename FuncInfo<Tag, Args...>::raw_return_t>::function(static_cast<Derived*>(this)->function_impl(tag, args...));
    }

    template<typename Tag, typename... Args>
    EIGEN_STRONG_INLINE auto
    call_function_impl_expr(std::false_type, const Tag& tag, const Args&... args) noexcept
    {
        return static_cast<Derived*>(this)->function_impl(tag, args...);
    }

    // captures the function call for variables
    template<typename Tag, typename... Args>
    EIGEN_STRONG_INLINE auto
    call_function_impl_capture(Tag, const Args&... args) noexcept
    {
        auto capture = [&](auto f)
        {
            return f(args...);
        };
        using Info = FuncInfo<Tag, Args...>;
        return FunctionCapture<Derived, Tag, Info, decltype(capture)>(*static_cast<Derived*>(this), capture);
    }

public:
    template<typename Tag, typename... Args>
    EIGEN_STRONG_INLINE Derived&
    operator()(const Tag& tag, const Args&... args) noexcept
    {
        return call_function_impl_capture(tag, args...);
    }

    template<typename Tag, typename... Args>
    EIGEN_STRONG_INLINE auto
    expression(const Tag& tag, const Args&... args) noexcept
    {
        return call_function_impl_capture(tag, args...);
    }

    // user specified function
    template<typename Tag, typename... Args>
    EIGEN_STRONG_INLINE auto
    function(const Tag& tag, const Args&... args) noexcept
    {
        using F = FuncInfo<Tag, Args...>;
        return call_function_impl_expr(typename F::is_return_expr(), tag, args...);
    }

    // user specified jacobian code
    template <typename Tag, typename OutJacobian, typename AScalar, typename... Args>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_jacobian<Derived, Tag, OutJacobian&, AScalar, Args...>() == true, void>::type
    jacobian(const Tag& tag, OutJacobian&& out_jacobian, const AScalar& alpha, const Args&... args) noexcept
    {
        static_cast<Derived*>(this)->jacobian_impl(tag, out_jacobian, alpha, args...);
    }

    // delegate to differentiable jacobian code
    template <typename Tag, typename OutJacobian, typename AScalar, typename... Args>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_jacobian<Derived, Tag, OutJacobian&, AScalar, Args...>() == false
                                                && FuncInfo<Tag, Args...>::is_return_expr::value == true, void>::type
    jacobian(const Tag& tag, OutJacobian&& out_jacobian, const AScalar& alpha, const Args&... args) noexcept
    {
        ExprEvaluator<typename FuncInfo<Tag, Args...>::raw_return_t>::jacobian(static_cast<Derived*>(this)->function_impl(tag, args...), out_jacobian, alpha);
    }

    // default internal jacobian code
    template <typename Tag, typename OutJacobian, typename AScalar, typename... Args>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_jacobian<Derived, Tag, OutJacobian&, AScalar, Args...>() == false
                                                && FuncInfo<Tag, Args...>::is_return_expr::value == false, void>::type
    jacobian(const Tag& tag, OutJacobian&& out_jacobian, const AScalar& alpha, const Args&... args) noexcept
    {
        this->jacobian_impl_autodiff(tag, out_jacobian, alpha, args...);
    }

    // user specified wsum code
    template <typename Tag, typename Weight, typename... Args, typename scalar_t = typename Eigen::MatrixBase<Weight>::Scalar>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_wsum<Derived, Tag, Eigen::MatrixBase<Weight>, Args...>() == true, scalar_t>::type
    wsum(const Tag& tag, const Eigen::MatrixBase<Weight>& weight, const Args&... args) noexcept
    {
        return static_cast<Derived*>(this)->wsum_impl(tag, weight, args...);
    }

    // delegate to differentiable wsum code
    template <typename Tag, typename Weight, typename... Args, typename scalar_t = typename Eigen::MatrixBase<Weight>::Scalar>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_wsum<Derived, Tag, Eigen::MatrixBase<Weight>, Args...>() == false
                                                && FuncInfo<Tag, Args...>::is_return_expr::value == true, scalar_t>::type
    wsum(const Tag& tag, const Eigen::MatrixBase<Weight>& weight, const Args&... args) noexcept
    {
        return ExprEvaluator<typename FuncInfo<Tag, Args...>::raw_return_t>::wsum(static_cast<Derived*>(this)->function_impl(tag, args...), weight);
    }

    // default internal wsum code
    template <typename Tag, typename Weight, typename... Args, typename scalar_t = typename Eigen::MatrixBase<Weight>::Scalar>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_wsum<Derived, Tag, Eigen::MatrixBase<Weight>, Args...>() == false
                                                && FuncInfo<Tag, Args...>::is_return_expr::value == false, scalar_t>::type
    wsum(const Tag& tag, const Eigen::MatrixBase<Weight>& weight, const Args&... args) noexcept
    {
        return this->wsum_impl_autodiff(tag, weight, args...);
    }

    // user specified gradient code
    template <typename Tag, typename OutGradient, typename Weight, typename... Args>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_gradient<Derived, Tag, OutGradient&, Eigen::MatrixBase<Weight>, Args...>() == true, void>::type
    gradient(const Tag& tag, OutGradient&& out_gradient, const Eigen::MatrixBase<Weight>& weight, const Args&... args) noexcept
    {
        static_cast<Derived*>(this)->gradient_impl(tag, out_gradient, weight, args...);
    }

    // delegate to differentiable gradient code
    template <typename Tag, typename OutGradient, typename Weight, typename... Args>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_gradient<Derived, Tag, OutGradient&, Eigen::MatrixBase<Weight>, Args...>() == false
                                                && FuncInfo<Tag, Args...>::is_return_expr::value == true, void>::type
    gradient(const Tag& tag, OutGradient&& out_gradient, const Eigen::MatrixBase<Weight>& weight, const Args&... args) noexcept
    {
        ExprEvaluator<typename FuncInfo<Tag, Args...>::raw_return_t>::gradient(static_cast<Derived*>(this)->function_impl(tag, args...), out_gradient, weight);
    }

    // default internal gradient code
    template <typename Tag, typename OutGradient, typename Weight, typename... Args>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_gradient<Derived, Tag, OutGradient&, Eigen::MatrixBase<Weight>, Args...>() == false
                                                && FuncInfo<Tag, Args...>::is_return_expr::value == false, void>::type
    gradient(const Tag& tag, OutGradient&& out_gradient, const Eigen::MatrixBase<Weight>& weight, const Args&... args) noexcept
    {
        this->gradient_impl_autodiff(tag, out_gradient, weight, args...);
    }

    // user specified hessian code
    template <typename Tag, typename OutHessian, typename Weight, typename... Args>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_hessian<Derived, Tag, OutHessian&, Eigen::MatrixBase<Weight>, Args...>() == true, void>::type
    hessian(const Tag& tag, OutHessian&& out_hessian, const Eigen::MatrixBase<Weight>& weight, const Args&... args) noexcept
    {
        static_cast<Derived*>(this)->hessian_impl(tag, out_hessian, weight, args...);
    }

    // delegate to differentiable hessian code
    template <typename Tag, typename OutHessian, typename Weight, typename... Args>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_hessian<Derived, Tag, OutHessian&, Eigen::MatrixBase<Weight>, Args...>() == false
                                                && FuncInfo<Tag, Args...>::is_return_expr::value == true, void>::type
    hessian(const Tag& tag, OutHessian&& out_hessian, const Eigen::MatrixBase<Weight>& weight, const Args&... args) noexcept
    {
        ExprEvaluator<typename FuncInfo<Tag, Args...>::raw_return_t>::hessian(static_cast<Derived*>(this)->function_impl(tag, args...), out_hessian, weight);
    }

    // default internal hessian code
    template <typename Tag, typename OutHessian, typename Weight, typename... Args>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_hessian<Derived, Tag, OutHessian&, Eigen::MatrixBase<Weight>, Args...>() == false
                                                && FuncInfo<Tag, Args...>::is_return_expr::value == false, void>::type
    hessian(const Tag& tag, OutHessian&& out_hessian, const Eigen::MatrixBase<Weight>& weight, const Args&... args) noexcept
    {
        this->hessian_impl_autodiff(tag, out_hessian, weight, args...);
    }
};

template<typename Derived, int Options>
class Differentiable<Derived, Options, false> : public DifferentiableBase<Derived>,
                                                public DifferentiableImpl<Differentiable<Derived, Options, false>, Options>
{
public:

    template<typename Tag, typename... Vars>
    struct FuncInfo
    {
    private:
        template<typename Derived_ = Derived>
        static auto get_raw_return_t(int) -> decltype(std::declval<Derived_>().function_impl(std::declval<Vars>()...));
        template<typename Derived_ = Derived>
        static auto get_raw_return_t(long) -> std::false_type;
        template<typename raw_return_t>
        static auto get_expr_return_t(int) -> decltype(ExprEvaluator<raw_return_t>::function(std::declval<raw_return_t>()));
        template<typename raw_return_t>
        static auto get_expr_return_t(long) -> raw_return_t;

    public:
        using raw_return_t = decltype(get_raw_return_t<>(0));
        using is_return_expr = typename std::is_base_of<ExprBase<raw_return_t>, raw_return_t>;
        using return_t = decltype(get_expr_return_t<raw_return_t>(0));

        using scalar_t = typename meta::matrix_info<return_t>::Scalar;

        static constexpr int n_inputs = meta::sum_template<meta::variable_info<Vars>::size...>();
        static constexpr int n_outputs = meta::matrix_info<return_t>::RowsAtCompileTime;

        using jacobian_t = Eigen::Matrix<scalar_t, n_outputs, n_inputs>;
        using gradient_t = Eigen::Vector<scalar_t, n_inputs>;
        using hessian_t = Eigen::Matrix<scalar_t, n_inputs, n_inputs>;

        static_assert(n_outputs >= 0 && n_inputs >= 0, "The function cannot have a dynamic size.");
    };

private:

    template<typename... Args>
    EIGEN_STRONG_INLINE auto
    call_function_impl_expr(std::true_type, const Args&... args) noexcept
    {
        return ExprEvaluator<typename FuncInfo<DefaultTag, Args...>::raw_return_t>::function(static_cast<Derived*>(this)->function_impl(args...));
    }

    template<typename... Args>
    EIGEN_STRONG_INLINE auto
    call_function_impl_expr(std::false_type, const Args&... args) noexcept
    {
        return static_cast<Derived*>(this)->function_impl(args...);
    }

    // captures the function call for variables
    template<typename... Args>
    EIGEN_STRONG_INLINE auto
    call_function_impl_capture(const Args&... args) noexcept
    {
        auto capture = [&](auto f)
        {
            return f(args...);
        };
        using Info = FuncInfo<DefaultTag, Args...>;
        return FunctionCapture<Derived, DefaultTag, Info, decltype(capture)>(*static_cast<Derived*>(this), capture);
    }

public:

    template<typename... Args>
    EIGEN_STRONG_INLINE auto
    operator()(const Args&... args) noexcept
    {
        return call_function_impl_capture(args...);
    }

    template<typename... Args>
    EIGEN_STRONG_INLINE auto
    expression(const Args&... args) noexcept
    {
        return call_function_impl_capture(args...);
    }

    // user specified function code
    template<typename... Args>
    EIGEN_STRONG_INLINE auto
    function(const Args&... args) noexcept
    {
        using F = FuncInfo<DefaultTag, Args...>;
        return call_function_impl_expr(typename F::is_return_expr(), args...);
    }

    // define DefaultTag for function
    template<typename... Args>
    EIGEN_STRONG_INLINE auto
    function(DefaultTag, const Args&... args) noexcept
    {
        return function(args...);
    }

    // user specified jacobian code
    template <typename OutJacobian, typename AScalar, typename... Args>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_jacobian<Derived, OutJacobian&, AScalar, Args...>() == true, void>::type
    jacobian(OutJacobian&& out_jacobian, const AScalar& alpha, const Args&... args) noexcept
    {
        static_cast<Derived*>(this)->jacobian_impl(out_jacobian, alpha, args...);
    }

    // delegate to differentiable jacobian code
    template <typename OutJacobian, typename AScalar, typename... Args>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_jacobian<Derived, OutJacobian&, AScalar, Args...>() == false
                                                && FuncInfo<DefaultTag, Args...>::is_return_expr::value == true, void>::type
    jacobian(OutJacobian&& out_jacobian, const AScalar& alpha, const Args&... args) noexcept
    {
        ExprEvaluator<typename FuncInfo<DefaultTag, Args...>::raw_return_t>::jacobian(static_cast<Derived*>(this)->function_impl(args...), out_jacobian, alpha);
    }

    // default internal jacobian code
    template <typename OutJacobian, typename AScalar, typename... Args>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_jacobian<Derived, OutJacobian&, AScalar, Args...>() == false
                                                && FuncInfo<DefaultTag, Args...>::is_return_expr::value == false, void>::type
    jacobian(OutJacobian&& out_jacobian, const AScalar& alpha, const Args&... args) noexcept
    {
        this->jacobian_impl_autodiff(DefaultTag{}, out_jacobian, alpha, args...);
    }

    // define DefaultTag for jacobian
    template <typename OutJacobian, typename AScalar, typename... Args>
    EIGEN_STRONG_INLINE void
    jacobian(DefaultTag, OutJacobian&& out_jacobian, const AScalar& alpha, const Args&... args) noexcept
    {
        jacobian(std::forward<OutJacobian>(out_jacobian), alpha, args...);
    }

    // user specified wsum code
    template <typename Weight, typename... Args, typename scalar_t = typename Eigen::MatrixBase<Weight>::Scalar>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_wsum<Derived, Eigen::MatrixBase<Weight>, Args...>() == true, scalar_t>::type
    wsum(const Eigen::MatrixBase<Weight>& weight, const Args&... args) noexcept
    {
        return static_cast<Derived*>(this)->wsum_impl(weight, args...);
    }

    // delegate to differentiable wsum code
    template <typename Weight, typename... Args, typename scalar_t = typename Eigen::MatrixBase<Weight>::Scalar>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_wsum<Derived, Eigen::MatrixBase<Weight>, Args...>() == false
                                                && FuncInfo<DefaultTag, Args...>::is_return_expr::value == true, scalar_t>::type
    wsum(const Eigen::MatrixBase<Weight>& weight, const Args&... args) noexcept
    {
        return ExprEvaluator<typename FuncInfo<DefaultTag, Args...>::raw_return_t>::wsum(static_cast<Derived*>(this)->function_impl(args...), weight);
    }

    // default internal wsum code
    template <typename Weight, typename... Args, typename scalar_t = typename Eigen::MatrixBase<Weight>::Scalar>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_wsum<Derived, Eigen::MatrixBase<Weight>, Args...>() == false
                                                && FuncInfo<DefaultTag, Args...>::is_return_expr::value == false, scalar_t>::type
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
    template <typename OutGradient, typename Weight, typename... Args>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_gradient<Derived, OutGradient&, Eigen::MatrixBase<Weight>, Args...>() == true, void>::type
    gradient(OutGradient&& out_gradient, const Eigen::MatrixBase<Weight>& weight, const Args&... args) noexcept
    {
        static_cast<Derived*>(this)->gradient_impl(out_gradient, weight, args...);
    }

    // delegate to differentiable gradient code
    template <typename OutGradient, typename Weight, typename... Args>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_gradient<Derived, OutGradient&, Eigen::MatrixBase<Weight>, Args...>() == false
                                                && FuncInfo<DefaultTag, Args...>::is_return_expr::value == true, void>::type
    gradient(OutGradient&& out_gradient, const Eigen::MatrixBase<Weight>& weight, const Args&... args) noexcept
    {
        ExprEvaluator<typename FuncInfo<DefaultTag, Args...>::raw_return_t>::gradient(static_cast<Derived*>(this)->function_impl(args...), out_gradient, weight);
    }

    // default internal gradient code
    template <typename OutGradient, typename Weight, typename... Args>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_gradient<Derived, OutGradient&, Eigen::MatrixBase<Weight>, Args...>() == false
                                                && FuncInfo<DefaultTag, Args...>::is_return_expr::value == false, void>::type
    gradient(OutGradient&& out_gradient, const Eigen::MatrixBase<Weight>& weight, const Args&... args) noexcept
    {
        this->gradient_impl_autodiff(DefaultTag{}, out_gradient, weight, args...);
    }

    // define DefaultTag for gradient
    template <typename OutGradient, typename Weight, typename... Args>
    EIGEN_STRONG_INLINE void
    gradient(DefaultTag, OutGradient&& out_gradient, const Eigen::MatrixBase<Weight>& weight, const Args&... args) noexcept
    {
        gradient(std::forward<OutGradient>(out_gradient), weight, args...);
    }

    // user specified hessian code
    template <typename OutHessian, typename Weight, typename... Args>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_hessian<Derived, OutHessian&, Eigen::MatrixBase<Weight>, Args...>() == true, void>::type
    hessian(OutHessian&& out_hessian, const Eigen::MatrixBase<Weight>& weight, const Args&... args) noexcept
    {
        static_cast<Derived*>(this)->hessian_impl(out_hessian, weight, args...);
    }

    // delegate to differentiable hessian code
    template <typename OutHessian, typename Weight, typename... Args>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_hessian<Derived, OutHessian&, Eigen::MatrixBase<Weight>, Args...>() == false
                                                && FuncInfo<DefaultTag, Args...>::is_return_expr::value == true, void>::type
    hessian(OutHessian&& out_hessian, const Eigen::MatrixBase<Weight>& weight, const Args&... args) noexcept
    {
        ExprEvaluator<typename FuncInfo<DefaultTag, Args...>::raw_return_t>::hessian(static_cast<Derived*>(this)->function_impl(args...), out_hessian, weight);
    }

    // default internal hessian code
    template <typename OutHessian, typename Weight, typename... Args>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_hessian<Derived, OutHessian&, Eigen::MatrixBase<Weight>, Args...>() == false
                                                && FuncInfo<DefaultTag, Args...>::is_return_expr::value == false, void>::type
    hessian(OutHessian&& out_hessian, const Eigen::MatrixBase<Weight>& weight, const Args&... args) noexcept
    {
        this->hessian_impl_autodiff(DefaultTag{}, out_hessian, weight, args...);
    }

    // define DefaultTag for hessian
    template <typename OutHessian, typename Weight, typename... Args>
    EIGEN_STRONG_INLINE void
    hessian(DefaultTag, OutHessian&& out_hessian, const Eigen::MatrixBase<Weight>& weight, const Args&... args) noexcept
    {
        hessian(std::forward<OutHessian>(out_hessian), weight, args...);
    }
};

} // namespace laopt

#endif // LAOPT_DIFFERENTIABLE_HPP