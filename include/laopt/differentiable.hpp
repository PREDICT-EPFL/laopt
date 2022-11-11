#ifndef LAOPT_DIFFERENTIABLE_HPP
#define LAOPT_DIFFERENTIABLE_HPP

#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>
#include "eigen_autodiff_fix.hpp"

#include "utility.hpp"
#include "expressions/expr_base.hpp"
#include "indexed_vector.hpp"

namespace laopt
{

struct DefaultTag {};

/*
 * This struct captures a function call.
 * Capture is a lambda which takes another lambda as input which is then called with the original parameters.
 */
template<typename Derived, typename Tag, typename Info, typename Capture>
class FunctionCapture : public ExprBase<FunctionCapture<Derived, Tag, Info, Capture>>
{
public:
    Derived& func;
    Capture capture;

    static constexpr int n_inputs = Info::n_inputs;
    static constexpr int n_outputs = Info::n_outputs;
    using Scalar = typename Info::scalar_t;

    explicit FunctionCapture(Derived& func, const Capture& capture) : func(func), capture(capture) {}

    EIGEN_STRONG_INLINE Eigen::Vector<int, n_inputs> indices() const
    {
        return capture([&](auto&&... vars) {
            return concatenate_indices(get_indices(vars)...);
        });
    }

private:
    // We only consider IndexedVectors for AD. All other types don't contribute.
    template<typename Base>
    Eigen::Vector<int, IndexedVector<Base>::n_inputs> get_indices(const IndexedVector<Base>& var) const
    {
        return var.indices();
    }

    template<typename T>
    Eigen::Vector<int, 0> get_indices(T) const
    {
        return {};
    }
};


template<typename Derived>
class DifferentiableBase
{
public:
    template<typename... Args>
    EIGEN_STRONG_INLINE auto
    operator()(Args&&... args) noexcept
    {
        return static_cast<Derived*>(this)->function(std::forward<Args>(args)...);
    }

protected:
    //
    // Eigen Autodiff Implementations
    //


    /**
     * Class that provides implementations of jacobian_impl, hessian_impl, gradient_impl
     * using the Eigen AutoDiff tool
     */

    // Compute the jacobian with eigen autodiff
    template<typename Tag, typename OutJacobian, typename... Args>
    EIGEN_STRONG_INLINE void
    jacobian_impl_autodiff(
            Tag&& tag, // Function to call
            OutJacobian& out_jacobian, // Outputs
            const Args&... args) noexcept // Function arguments
    {
        using Info = typename Derived::template FuncInfo<Tag, Args...>;

        // First order derivative
        using AD_scalar = Eigen::AutoDiffScalar<Eigen::Vector<typename Info::scalar_t, Info::n_inputs>>;
        using AD_Output = Eigen::Vector<AD_scalar, Info::n_outputs>;

        // Convert the arguments to AD variables, and call the function
        AD_Output out = seed_and_call(std::forward<Tag>(tag), make_ad<AD_scalar>(args)...);

        // Copy out into output variables
        for(int i = 0; i < out.rows(); i++)
        {
            out_jacobian(i, Eigen::all) += out[i].derivatives().transpose();
        }
    }

    // Take a vector input and return an AD version of the vector
    template<typename AD_scalar, typename Base>
    static EIGEN_STRONG_INLINE auto
    make_ad(const IndexedVector<Base>& x) noexcept
    {
        constexpr size_t n = Base::RowsAtCompileTime;
        Eigen::Vector<AD_scalar, n> y;
        y = x;
        for (int i = 0; i < y.rows(); i++) {
            y[i].derivatives().setZero();
        }
        return y;
    }

    template<typename AD_scalar, typename T>
    static EIGEN_STRONG_INLINE const T&
    make_ad(const T& x) noexcept
    {
        return x;
    }

    template<typename Tag, typename... Args>
    EIGEN_STRONG_INLINE auto
    seed_and_call(Tag&& tag, Args&&... args) noexcept
    {
        // Set derivative equal to identity
        int offset = 0;
        (void) std::initializer_list<int>{
            (
                offset = ad_seed(args, offset), // Set to unit vectors
                0
            )...
        };

        return static_cast<Derived*>(this)->function(std::forward<Tag>(tag), std::forward<Args>(args)...);
    }

    // Sets the input derivatives to the identity.
    // Assumes that the derivative matrix is initially zero
    template<typename X, int n>
    static EIGEN_STRONG_INLINE int
    ad_seed(Eigen::Vector<Eigen::AutoDiffScalar<X>, n>& x, int offset) noexcept
    {
        for (int i = 0; i < x.rows(); i++)
        {
            x[i].derivatives().coeffRef(i + offset) = 1;
        }
        return offset + x.rows();
    }

    template<typename T>
    static EIGEN_STRONG_INLINE int
    ad_seed(const T, int offset) noexcept
    {
        return offset;
    }

    /**
     * Returns the value w'*f(x).
     * out_gradient += gradient(w'*f(x))
     * out_hessian += hessian(w'*f(x))
     */
    template<typename Tag, typename Weight, typename OutHessian, typename... Args,
            typename scalar_t = typename Eigen::MatrixBase<Weight>::Scalar>
    EIGEN_STRONG_INLINE void
    hessian_impl_autodiff(Tag&& tag,
                          OutHessian& out_hessian,
                          const Eigen::MatrixBase<Weight>& weight,
                          const Args&... args) noexcept
    {
        using Info = typename Derived::template FuncInfo<Tag, Args...>;

        // Second order derivative
        using AD_scalar = Eigen::AutoDiffScalar<Eigen::Vector<scalar_t, Info::n_inputs>>;
        // only calculate one row of the hessian at once to reduce memory requirements
        using outerDerivatives = Eigen::Vector<AD_scalar, 1>;
        using outerADScalar = Eigen::AutoDiffScalar<outerDerivatives>;
        using outerAD_t = Eigen::Vector<outerADScalar, Info::n_outputs>;

        outerAD_t out;
        for (size_t i = 0; i < Info::n_inputs; i++) {
            // Convert to AD variables for the inputs and call our function for ith row of hessian
            out = seed_and_call2(i, std::forward<Tag>(tag), make_ad2<outerADScalar>(args)...);
            for(size_t j = 0; j < Info::n_outputs; j++) {
                out_hessian(i, Eigen::all) += weight(j) * out[j].derivatives()(0).derivatives().transpose();
            }
        }
    }

    // Take a vector input and return an AD version of the vector
    template<typename outerADScalar, typename Base>
    EIGEN_STRONG_INLINE auto
    make_ad2(const IndexedVector<Base>& x) noexcept
    {
        constexpr size_t n = Base::RowsAtCompileTime;
        Eigen::Vector<outerADScalar, n> y;
        // y = x;
        for (size_t i = 0; i < n; i++) {
            y(i).value().value() = x.cast_base()(i);
            y(i).value().derivatives().setZero();
            y(i).derivatives().setZero();
            y(i).derivatives()(0).derivatives().setZero();
        }
        return y;
    }

    template<typename outerADScalar, typename T>
    EIGEN_STRONG_INLINE const T&
    make_ad2(const T& x) noexcept
    {
        return x;
    }

    template<typename Tag, typename... Args>
    EIGEN_STRONG_INLINE auto seed_and_call2(int outer_index, Tag&& tag, Args&&... args) noexcept
    {
        // Set derivative equal to identity
        int inner_offset = 0;
        (void) std::initializer_list<int>{
            (
                inner_offset = ad_seed2(args, outer_index, inner_offset), // Set to unit vectors
                0
            )...
        };

        // Call our function
        return static_cast<Derived*>(this)->function(std::forward<Tag>(tag), std::forward<Args>(args)...);
    }

    // Sets the input derivatives to the identity.
    // Assumes that the derivative matrix is initially zero
    template <typename X, int n>
    EIGEN_STRONG_INLINE int
    ad_seed2(Eigen::Vector<Eigen::AutoDiffScalar<X>, n>& x, int outer_index, int inner_offset) noexcept
    {
        for (int i = 0; i < x.rows(); i++)
        {
            x(i).value().derivatives().coeffRef(i + inner_offset) = 1;
            // Seed only the relevant value for the current row of the hessian
            if (i + inner_offset == outer_index) {
                x(i).derivatives().coeffRef(0) = 1;
            }
        }

        return inner_offset + x.rows();
    }

    template<typename T>
    static EIGEN_STRONG_INLINE int
    ad_seed2(const T, int outer_index, int inner_offset) noexcept
    {
        return inner_offset;
    }

    /**
      * Returns the value w'*f(x)
      */
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

    /**
     * Returns the value w'*f(x) and *adds* the gradient to gradient
     */
    template<typename Tag, typename Weight, typename Gradient, typename... Args,
            typename scalar_t = typename Eigen::MatrixBase<Weight>::Scalar>
    inline void
    gradient_impl_autodiff(Tag&& tag,
                           Gradient& out_gradient,
                           const Eigen::MatrixBase<Weight>& weight,
                           const Args&... args) noexcept
    {
        using Info = typename Derived::template FuncInfo<Tag, Args...>;

        // Call the (possibly overloaded) jacobian
        Eigen::Matrix<scalar_t, Info::n_outputs, Info::n_inputs> jacobian;
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

template<typename Derived, bool tagless = false>
class Differentiable;

template<typename Derived>
class Differentiable<Derived, false> : public DifferentiableBase<Differentiable<Derived, false>>
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

        static_assert(n_outputs > 0 && n_inputs >= 0, "The function cannot have a dynamic size.");
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

template<typename Derived>
class Differentiable<Derived, true> : public DifferentiableBase<Differentiable<Derived, true>>
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

        static_assert(n_outputs > 0 && n_inputs >= 0, "The function cannot have a dynamic size.");
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