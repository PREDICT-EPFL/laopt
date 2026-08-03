#ifndef LAOPT_DIFFERENTIABLE_HPP
#define LAOPT_DIFFERENTIABLE_HPP

#include <Eigen/Dense>

#include "laopt/utility.hpp"
#include "laopt/expressions/fwd.hpp"
#include "laopt/variable_map.hpp"
#include "laopt/autodiff/differentiable_options.hpp"
#include "laopt/autodiff/differentiable_eigen.hpp"
#ifdef LAOPT_WITH_CASADI
#include "laopt/autodiff/differentiable_casadi.hpp"
#endif

namespace laopt
{

template<typename Derived>
class DifferentiableBase {};

template<typename Derived, int Options>
class DifferentiableImpl : public DifferentiableImplEigen<Derived, Options>
#ifdef LAOPT_WITH_CASADI
        , public DifferentiableImplCasadi<Derived, Options>
#endif
{
protected:
    using DifferentiableImplEigen<Derived, Options>::jacobian_impl_autodiff_eval;
    using DifferentiableImplEigen<Derived, Options>::jacobian_impl_autodiff_sparsity;
    using DifferentiableImplEigen<Derived, Options>::hessian_impl_autodiff_eval;
    using DifferentiableImplEigen<Derived, Options>::hessian_impl_autodiff_sparsity;

#ifdef LAOPT_WITH_CASADI
    using DifferentiableImplCasadi<Derived, Options>::jacobian_impl_autodiff_eval;
    using DifferentiableImplCasadi<Derived, Options>::jacobian_impl_autodiff_sparsity;
    using DifferentiableImplCasadi<Derived, Options>::hessian_impl_autodiff_eval;
    using DifferentiableImplCasadi<Derived, Options>::hessian_impl_autodiff_sparsity;
#else
    static_assert((CASADI_ALL & Options) == 0, "Casadi support is not enabled, build laOPT with WITH_CASADI=ON");
#endif

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

    template<typename T>
    EIGEN_STRONG_INLINE typename std::enable_if<is_variable<T>::value, Eigen::Vector<typename T::Scalar, T::RowsAtCompileTime>>::type
    eval_arg(const Eigen::MatrixBase<T>& x) noexcept
    {
        return Eigen::Vector<typename T::Scalar, T::RowsAtCompileTime>(x);
    }

    template<typename T>
    EIGEN_STRONG_INLINE typename std::enable_if<!is_variable<T>::value, const T&>::type
    eval_arg(const T& x) noexcept
    {
        return x;
    }

    template<typename OrigArg, typename T>
    EIGEN_STRONG_INLINE typename std::enable_if<is_variable<OrigArg>::value, laopt::VariableMap<T>>::type
    arg_to_variable(T&& x) noexcept
    {
        static_assert(std::is_same<T, Eigen::Vector<typename OrigArg::Scalar, OrigArg::RowsAtCompileTime>>::value, "wrong type");
        return laopt::VariableMap<T>(x.data());
    }

    template<typename OrigArg, typename T>
    EIGEN_STRONG_INLINE typename std::enable_if<!is_variable<OrigArg>::value, const T&>::type
    arg_to_variable(const T& x) noexcept
    {
        return x;
    }

    template<typename T>
    EIGEN_STRONG_INLINE typename std::enable_if<is_variable<T>::value, int>::type
    set_arg_offset(T& x, int offset) noexcept
    {
        x.index_offset() = offset;
        return offset + T::RowsAtCompileTime;
    }

    template<typename T>
    EIGEN_STRONG_INLINE typename std::enable_if<!is_variable<T>::value, int>::type
    set_arg_offset(const T& x, int offset) noexcept
    {
        return offset;
    }

    template<typename Func, typename... Args>
    EIGEN_STRONG_INLINE void set_variable_offset_and_call(const Func& func, Args&&... args)
    {
        // set correct offsets
        int offset = 0;
        (void) std::initializer_list<int>{
            (
                offset = set_arg_offset(args, offset),
                0
            )...
        };
        func(args...);
    }

    template<typename Func, typename... Args>
    EIGEN_STRONG_INLINE void reset_args_and_call(const Func& func, const Args&... args)
    {
        set_variable_offset_and_call(func, arg_to_variable<Args>(eval_arg(args))...);
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

    template<typename Tag, typename... Args>
    struct FuncInfo
    {
    private:
        template<typename Derived_ = Derived>
        static auto get_raw_return_t(int) -> decltype(std::declval<Derived_>().function_impl(Tag{}, std::declval<Args>()...));
        template<typename Derived_ = Derived>
        static auto get_raw_return_t(long) -> std::false_type;
        template<typename raw_return_t>
        static auto get_expr_return_t(int) -> decltype(ExprEvaluator<raw_return_t>::function(std::declval<raw_return_t>()));
        template<typename raw_return_t>
        static auto get_expr_return_t(long) -> raw_return_t;

        // We only consider IndexedVectors for AD. All other types don't contribute.
        template<typename T>
        EIGEN_STRONG_INLINE Eigen::Vector<int, T::RowsAtCompileTime>
        static get_indices(const T& var, std::true_type)
        {
            return variable_indices(var);
        }

        template<typename T>
        EIGEN_STRONG_INLINE Eigen::Vector<int, 0>
        static get_indices(const T& var, std::false_type)
        {
            return {};
        }

    public:
        using raw_return_t = decltype(get_raw_return_t<>(0));
        using is_return_expr = typename std::is_base_of<ExprBase<raw_return_t>, raw_return_t>;
        using return_t = decltype(get_expr_return_t<raw_return_t>(0));

        using scalar_t = typename meta::matrix_info<return_t>::Scalar;

        static constexpr int n_inputs = meta::sum_template<variable_info<Args>::size...>();
        static constexpr int n_outputs = meta::matrix_info<return_t>::RowsAtCompileTime;

        using jacobian_t = Eigen::Matrix<scalar_t, n_outputs, n_inputs>;
        using gradient_t = Eigen::Vector<scalar_t, n_inputs>;
        using hessian_t = Eigen::Matrix<scalar_t, n_inputs, n_inputs>;

        static_assert(n_outputs >= 0 && n_inputs >= 0, "The function cannot have a dynamic size.");

        EIGEN_STRONG_INLINE Eigen::Vector<int, FuncInfo<Tag, Args...>::n_inputs>
        static indices(const Args&... args)
        {
            return concatenate_indices(get_indices(args, is_variable<decltype(args)>{})...);
        }
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
    EIGEN_STRONG_INLINE auto
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
        using return_t = typename F::return_t;
        static_assert(!std::is_base_of<Eigen::EigenBase<return_t>, return_t>::value || std::is_base_of<Eigen::PlainObjectBase<return_t>, return_t>::value,
                      "Your function_impl is returning a Eigen expression. Make sure the return type is a PlainObject type (e.g. .eval() the expression).");
        return call_function_impl_expr(typename F::is_return_expr(), tag, args...);
    }

    // user specified jacobian code
    template <typename Tag, typename OutJacobian, typename AScalar, typename... Args>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_jacobian<Derived, Tag, OutJacobian&, AScalar, Args...>() == true, void>::type
    jacobian(const Tag& tag, OutJacobian&& out_jacobian, const AScalar& alpha, const Args&... args) noexcept
    {
        auto&& out_jacobian_ = out_jacobian(Eigen::indexing::all, FuncInfo<Tag, Args...>::indices(args...));
        this->reset_args_and_call([&](auto&&... reset_args) {
            static_cast<Derived*>(this)->jacobian_impl(tag, out_jacobian_, alpha, reset_args...);
        }, args...);
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
        auto&& out_jacobian_ = out_jacobian(Eigen::indexing::all, FuncInfo<Tag, Args...>::indices(args...));
        this->reset_args_and_call([&](auto&&... reset_args) {
            this->jacobian_impl_autodiff(tag, out_jacobian_, alpha, reset_args...);
        }, args...);
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
        auto&& out_gradient_ = out_gradient(FuncInfo<Tag, Args...>::indices(args...));
        this->reset_args_and_call([&](auto&&... reset_args) {
            static_cast<Derived*>(this)->gradient_impl(tag, out_gradient_, weight, reset_args...);
        }, args...);
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
        auto&& out_gradient_ = out_gradient(FuncInfo<Tag, Args...>::indices(args...));
        this->reset_args_and_call([&](auto&&... reset_args) {
            this->gradient_impl_autodiff(tag, out_gradient_, weight, reset_args...);
        }, args...);
    }

    // user specified hessian code
    template <typename Tag, typename OutHessian, typename Weight, typename... Args>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_hessian<Derived, Tag, OutHessian&, Eigen::MatrixBase<Weight>, Args...>() == true, void>::type
    hessian(const Tag& tag, OutHessian&& out_hessian, const Eigen::MatrixBase<Weight>& weight, const Args&... args) noexcept
    {
        auto&& ind = FuncInfo<Tag, Args...>::indices(args...);
        auto&& out_hessian_ = out_hessian(ind, ind);
        this->reset_args_and_call([&](auto&&... reset_args) {
            static_cast<Derived*>(this)->hessian_impl(tag, out_hessian_, weight, reset_args...);
        }, args...);
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
        auto&& ind = FuncInfo<Tag, Args...>::indices(args...);
        auto&& out_hessian_ = out_hessian(ind, ind);
        this->reset_args_and_call([&](auto&&... reset_args) {
            this->hessian_impl_autodiff(tag, out_hessian_, weight, reset_args...);
        }, args...);
    }
};

template<typename Derived, int Options>
class Differentiable<Derived, Options, false> : public DifferentiableBase<Derived>,
                                                public DifferentiableImpl<Differentiable<Derived, Options, false>, Options>
{
public:

    template<typename Tag, typename... Args>
    struct FuncInfo
    {
    private:
        template<typename Derived_ = Derived>
        static auto get_raw_return_t(int) -> decltype(std::declval<Derived_>().function_impl(std::declval<Args>()...));
        template<typename Derived_ = Derived>
        static auto get_raw_return_t(long) -> std::false_type;
        template<typename raw_return_t>
        static auto get_expr_return_t(int) -> decltype(ExprEvaluator<raw_return_t>::function(std::declval<raw_return_t>()));
        template<typename raw_return_t>
        static auto get_expr_return_t(long) -> raw_return_t;

        // We only consider IndexedVectors for AD. All other types don't contribute.
        template<typename T>
        EIGEN_STRONG_INLINE Eigen::Vector<int, T::RowsAtCompileTime>
        static get_indices(const T& var, std::true_type)
        {
            return variable_indices(var);
        }

        template<typename T>
        EIGEN_STRONG_INLINE Eigen::Vector<int, 0>
        static get_indices(const T& var, std::false_type)
        {
            return {};
        }

    public:
        using raw_return_t = decltype(get_raw_return_t<>(0));
        using is_return_expr = typename std::is_base_of<ExprBase<raw_return_t>, raw_return_t>;
        using return_t = decltype(get_expr_return_t<raw_return_t>(0));

        using scalar_t = typename meta::matrix_info<return_t>::Scalar;

        static constexpr int n_inputs = meta::sum_template<variable_info<Args>::size...>();
        static constexpr int n_outputs = meta::matrix_info<return_t>::RowsAtCompileTime;

        using jacobian_t = Eigen::Matrix<scalar_t, n_outputs, n_inputs>;
        using gradient_t = Eigen::Vector<scalar_t, n_inputs>;
        using hessian_t = Eigen::Matrix<scalar_t, n_inputs, n_inputs>;

        static_assert(n_outputs >= 0 && n_inputs >= 0, "The function cannot have a dynamic size.");

        EIGEN_STRONG_INLINE Eigen::Vector<int, FuncInfo<Tag, Args...>::n_inputs>
        static indices(const Args&... args)
        {
            return concatenate_indices(get_indices(args, is_variable<decltype(args)>{})...);
        }
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
        using return_t = typename F::return_t;
        static_assert(!std::is_base_of<Eigen::EigenBase<return_t>, return_t>::value || std::is_base_of<Eigen::PlainObjectBase<return_t>, return_t>::value,
                      "Your function_impl is returning a Eigen expression. Make sure the return type is a PlainObject type (e.g. .eval() the expression).");
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
        auto&& out_jacobian_ = out_jacobian(Eigen::indexing::all, FuncInfo<DefaultTag, Args...>::indices(args...));
        this->reset_args_and_call([&](auto&&... reset_args) {
            static_cast<Derived*>(this)->jacobian_impl(out_jacobian_, alpha, reset_args...);
        }, args...);
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
        auto&& out_jacobian_ = out_jacobian(Eigen::indexing::all, FuncInfo<DefaultTag, Args...>::indices(args...));
        this->reset_args_and_call([&](auto&&... reset_args) {
            this->jacobian_impl_autodiff(DefaultTag{}, out_jacobian_, alpha, reset_args...);
        }, args...);
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
        auto&& out_gradient_ = out_gradient(FuncInfo<DefaultTag, Args...>::indices(args...));
        this->reset_args_and_call([&](auto&&... reset_args) {
            static_cast<Derived*>(this)->gradient_impl(out_gradient_, weight, reset_args...);
        }, args...);
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
        auto&& out_gradient_ = out_gradient(FuncInfo<DefaultTag, Args...>::indices(args...));
        this->reset_args_and_call([&](auto&&... reset_args) {
            this->gradient_impl_autodiff(DefaultTag{}, out_gradient_, weight, reset_args...);
        }, args...);
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
        auto&& ind = FuncInfo<DefaultTag, Args...>::indices(args...);
        auto&& out_hessian_ = out_hessian(ind, ind);
        this->reset_args_and_call([&](auto&&... reset_args) {
            static_cast<Derived*>(this)->hessian_impl(out_hessian_, weight, reset_args...);
        }, args...);
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
        auto&& ind = FuncInfo<DefaultTag, Args...>::indices(args...);
        auto&& out_hessian_ = out_hessian(ind, ind);
        this->reset_args_and_call([&](auto&&... reset_args) {
            this->hessian_impl_autodiff(DefaultTag{}, out_hessian_, weight, reset_args...);
        }, args...);
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