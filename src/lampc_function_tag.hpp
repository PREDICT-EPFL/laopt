#ifndef __LAMPC__FUNCTION_TAG_HPP
#define __LAMPC__FUNCTION_TAG_HPP

namespace lampc
{

inline namespace tags
{

struct DefaultTag {};

}; // namespace tags

/**
 * Base class (CRTP) that provides the functions:
 * - function
 * - jacobian 
 * - hessian
 * - gradient
 * 
 * Tag-dispatching is used to determine which function_impl, jacobian_impl, etc 
 * to call in the Derived class to call the user-defined functions.
 * 
 * The types jacobian, hessian, etc in the Tag are used to dispatch
 * to specific function types to implement these functions. 
 * 
 * Eigen_autodiff is included as a differentiation method by default, and others
 * can be added by the user when inheriting from Differentiable
 */
template<typename Derived, bool tagless = false>
struct Differentiable
{
    // user specified function code with tag
    template<typename Tag, typename OutValue, typename... Args, typename Dummy = void>
    EIGEN_STRONG_INLINE typename std::enable_if<!tagless, Dummy>::type
    function(Tag&& tag, OutValue&& outvalue, const Eigen::MatrixBase<Args>&... args) noexcept
    {
        using Scalar = lampc::meta::get_scalar_t<Args...>;
        static_cast<Derived*>(this)->template function_impl<Scalar>(
                std::forward<Tag>(tag),
                std::forward<OutValue>(outvalue),
                args...);
    }

    // user specified function code without tag
    template<typename OutValue, typename... Args, typename Dummy = void>
    EIGEN_STRONG_INLINE typename std::enable_if<tagless, Dummy>::type
    function(OutValue&& outvalue, const Eigen::MatrixBase<Args>&... args) noexcept
    {
        using Scalar = lampc::meta::get_scalar_t<Args...>;
        static_cast<Derived*>(this)->template function_impl<Scalar>(
                std::forward<OutValue>(outvalue),
                args...);
    }

    template<typename... Args, typename Dummy = void>
    EIGEN_STRONG_INLINE typename std::enable_if<tagless, Dummy>::type
    function(DefaultTag, Args&&... args) noexcept
    {
        function(std::forward<Args>(args)...);
    }

    template<typename... Args>
    EIGEN_STRONG_INLINE void
    operator()(Args&&... args) noexcept
    {
        function(std::forward<Args>(args)...);
    }

    /**
    * We call the jacobian function specified in the tag.
    *
    * If it's USER, then we cast to Derived and call the user's function.
    * Otherwise we call the type specified in the tag, under the assumption that
    * it's in the scope of this class.
    *
    * Better would be if we had a way to bring all jacobian_impl calls into the scope
    * of Derived, in which case we wouldn't need these enable_if's
    */

    template<typename T, typename... Args>
    static auto test_user_jacobian(int) -> decltype(std::declval<T>().jacobian_impl(std::declval<Args>()...), std::true_type{});
    template<typename T, typename... Args>
    static auto test_user_jacobian(long) -> std::false_type;
    template<typename T, typename... Args>
    struct has_user_jacobian : decltype(test_user_jacobian<T, Args...>(0)){};

    // user specified jacobian code with tag
    template <typename Tag, typename OutValue, typename OutJacobian, typename... Args>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_jacobian<Derived, Tag, OutValue, OutJacobian, Eigen::MatrixBase<Args>...>() == true && !tagless, void>::type
    jacobian(Tag&& tag, OutValue&& outvalue, OutJacobian&& outjacobian, const Eigen::MatrixBase<Args>&... args)
    {
        static_cast<Derived*>(this)->jacobian_impl(std::forward<Tag>(tag), std::forward<OutValue>(outvalue), std::forward<OutJacobian>(outjacobian), args...);
    }

    // user specified jacobian code without tag
    template <typename OutValue, typename OutJacobian, typename... Args>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_jacobian<Derived, OutValue, OutJacobian, Eigen::MatrixBase<Args>...>() == true && tagless, void>::type
    jacobian(OutValue&& outvalue, OutJacobian&& outjacobian, const Eigen::MatrixBase<Args>&... args)
    {
        static_cast<Derived*>(this)->jacobian_impl(std::forward<OutValue>(outvalue), std::forward<OutJacobian>(outjacobian), args...);
    }

    // default internal jacobian code
    template <typename Tag, typename OutValue, typename OutJacobian, typename... Args>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_jacobian<Derived, Tag, OutValue, OutJacobian, Eigen::MatrixBase<Args>...>() == false && !tagless, void>::type
    jacobian(Tag&& tag, OutValue&& outvalue, OutJacobian&& outjacobian, const Eigen::MatrixBase<Args>&... args)
    {
        jacobian_impl_autodiff(std::forward<Tag>(tag), std::forward<OutValue>(outvalue), std::forward<OutJacobian>(outjacobian), args...);
    }

    // delegate jacobian without tag
    template <typename OutValue, typename OutJacobian, typename... Args>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_jacobian<Derived, OutValue, OutJacobian, Eigen::MatrixBase<Args>...>() == false && tagless, void>::type
    jacobian(OutValue&& outvalue, OutJacobian&& outjacobian, const Eigen::MatrixBase<Args>&... args)
    {
        jacobian_impl_autodiff(DefaultTag{}, std::forward<OutValue>(outvalue), std::forward<OutJacobian>(outjacobian), args...);
    }

    // define DefaultTag for jacobian
    template <typename OutValue, typename OutJacobian, typename... Args, typename Dummy = void>
    EIGEN_STRONG_INLINE typename std::enable_if<tagless, Dummy>::type
    jacobian(DefaultTag, OutValue&& outvalue, OutJacobian&& outjacobian, const Eigen::MatrixBase<Args>&... args)
    {
        jacobian(std::forward<OutValue>(outvalue), std::forward<OutJacobian>(outjacobian), args...);
    }

    template<typename T, typename... Args>
    static auto test_user_wsum(int) -> decltype(std::declval<T>().wsum_impl(std::declval<Args>()...), std::true_type{});
    template<typename T, typename... Args>
    static auto test_user_wsum(long) -> std::false_type;
    template<typename T, typename... Args>
    struct has_user_wsum : decltype(test_user_wsum<T, Args...>(0)){};

    // user specified wsum code with tag
    template <typename Tag, typename Weight, typename... Args, typename scalar_t = typename Eigen::MatrixBase<Weight>::Scalar>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_wsum<Derived, Tag, Eigen::MatrixBase<Weight>, Eigen::MatrixBase<Args>...>() == true && !tagless, scalar_t>::type
    wsum(Tag&& tag, const Eigen::MatrixBase<Weight>& w, const Eigen::MatrixBase<Args>&... args)
    {
        return static_cast<Derived*>(this)->wsum_impl(std::forward<Tag>(tag), w, args...);
    }

    // user specified wsum code without tag
    template <typename Weight, typename... Args, typename scalar_t = typename Eigen::MatrixBase<Weight>::Scalar>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_wsum<Derived, Eigen::MatrixBase<Weight>, Eigen::MatrixBase<Args>...>() == true && tagless, scalar_t>::type
    wsum(const Eigen::MatrixBase<Weight>& w, const Eigen::MatrixBase<Args>&... args)
    {
        return static_cast<Derived*>(this)->wsum_impl(w, args...);
    }

    // default internal wsum code
    template <typename Tag, typename Weight, typename... Args, typename scalar_t = typename Eigen::MatrixBase<Weight>::Scalar>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_wsum<Derived, Tag, Eigen::MatrixBase<Weight>, Eigen::MatrixBase<Args>...>() == false && !tagless, scalar_t>::type
    wsum(Tag&& tag, const Eigen::MatrixBase<Weight>& w, const Eigen::MatrixBase<Args>&... args)
    {
        return wsum_impl_autodiff(std::forward<Tag>(tag), w, args...);
    }

    // delegate wsum without tag
    template <typename Weight, typename... Args, typename scalar_t = typename Eigen::MatrixBase<Weight>::Scalar>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_wsum<Derived, Eigen::MatrixBase<Weight>, Eigen::MatrixBase<Args>...>() == false && tagless, scalar_t>::type
    wsum(const Eigen::MatrixBase<Weight>& w, const Eigen::MatrixBase<Args>&... args)
    {
        return wsum_impl_autodiff(DefaultTag{}, w, args...);
    }

    // define DefaultTag for wsum
    template <typename Weight, typename... Args, typename scalar_t = typename Eigen::MatrixBase<Weight>::Scalar>
    EIGEN_STRONG_INLINE typename std::enable_if<tagless, scalar_t>::type
    wsum(DefaultTag, const Eigen::MatrixBase<Weight>& w, const Eigen::MatrixBase<Args>&... args)
    {
        return wsum(w, args...);
    }

    template<typename T, typename... Args>
    static auto test_user_gradient(int) -> decltype(std::declval<T>().gradient_impl(std::declval<Args>()...), std::true_type{});
    template<typename T, typename... Args>
    static auto test_user_gradient(long) -> std::false_type;
    template<typename T, typename... Args>
    struct has_user_gradient : decltype(test_user_gradient<T, Args...>(0)){};

    // user specified gradient code with tag
    template <typename Tag, typename Weight, typename Gradient, typename... Args, typename scalar_t = typename Eigen::MatrixBase<Weight>::Scalar>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_gradient<Derived, Tag, Gradient, Eigen::MatrixBase<Weight>, Eigen::MatrixBase<Args>...>() == true && !tagless, scalar_t>::type
    gradient(Tag&& tag, Gradient&& gradient, const Eigen::MatrixBase<Weight>& w, const Eigen::MatrixBase<Args>&... args)
    {
        return static_cast<Derived*>(this)->gradient_impl(std::forward<Tag>(tag), gradient, w, args...);
    }

    // user specified gradient code without tag
    template <typename Weight, typename Gradient, typename... Args, typename scalar_t = typename Eigen::MatrixBase<Weight>::Scalar>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_gradient<Derived, Gradient, Eigen::MatrixBase<Weight>, Eigen::MatrixBase<Args>...>() == true && tagless, scalar_t>::type
    gradient(Gradient&& gradient, const Eigen::MatrixBase<Weight>& w, const Eigen::MatrixBase<Args>&... args)
    {
        return static_cast<Derived*>(this)->gradient_impl(std::forward<Gradient>(gradient), w, args...);
    }

    // default internal gradient code
    template <typename Tag, typename Weight, typename Gradient, typename... Args, typename scalar_t = typename Eigen::MatrixBase<Weight>::Scalar>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_gradient<Derived, Tag, Gradient, Eigen::MatrixBase<Weight>, Eigen::MatrixBase<Args>...>() == false && !tagless, scalar_t>::type
    gradient(Tag&& tag, Gradient&& gradient, const Eigen::MatrixBase<Weight>& w, const Eigen::MatrixBase<Args>&... args)
    {
        return gradient_impl_autodiff(std::forward<Tag>(tag), std::forward<Gradient>(gradient), w, args...);
    }

    // delegate gradient without tag
    template <typename Weight, typename Gradient, typename... Args, typename scalar_t = typename Eigen::MatrixBase<Weight>::Scalar>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_gradient<Derived, Gradient, Eigen::MatrixBase<Weight>, Eigen::MatrixBase<Args>...>() == false && tagless, scalar_t>::type
    gradient(Gradient&& gradient, const Eigen::MatrixBase<Weight>& w, const Eigen::MatrixBase<Args>&... args)
    {
        return gradient_impl_autodiff(DefaultTag{}, std::forward<Gradient>(gradient), w, args...);
    }

    // define DefaultTag for gradient
    template <typename Weight, typename Gradient, typename... Args, typename scalar_t = typename Eigen::MatrixBase<Weight>::Scalar>
    EIGEN_STRONG_INLINE typename std::enable_if<tagless, scalar_t>::type
    gradient(DefaultTag, Gradient&& gradient, const Eigen::MatrixBase<Weight>& w, const Eigen::MatrixBase<Args>&... args)
    {
        return gradient(std::forward<Gradient>(gradient), w, args...);
    }

    template<typename T, typename... Args>
    static auto test_user_hessian(int) -> decltype(std::declval<T>().hessian_impl(std::declval<Args>()...), std::true_type{});
    template<typename T, typename... Args>
    static auto test_user_hessian(long) -> std::false_type;
    template<typename T, typename... Args>
    struct has_user_hessian : decltype(test_user_hessian<T, Args...>(0)){};

    // user specified hessian code with tag
    template <typename Tag, typename Weight, typename Gradient, typename Hessian, typename... Args, typename scalar_t = typename Eigen::MatrixBase<Weight>::Scalar>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_hessian<Derived, Tag, Gradient, Hessian, Eigen::MatrixBase<Weight>, Eigen::MatrixBase<Args>...>() == true && !tagless, scalar_t>::type
    hessian(Tag&& tag, Gradient&& gradient, Hessian&& hessian, const Eigen::MatrixBase<Weight>& w, const Eigen::MatrixBase<Args>&... args)
    {
        return static_cast<Derived*>(this)->hessian_impl(std::forward<Tag>(tag), std::forward<Gradient>(gradient), std::forward<Hessian>(hessian), w, args...);
    }

    // user specified hessian code without tag
    template <typename Weight, typename Gradient, typename Hessian, typename... Args, typename scalar_t = typename Eigen::MatrixBase<Weight>::Scalar>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_hessian<Derived, Gradient, Hessian, Eigen::MatrixBase<Weight>, Eigen::MatrixBase<Args>...>() == true && tagless, scalar_t>::type
    hessian(Gradient&& gradient, Hessian&& hessian, const Eigen::MatrixBase<Weight>& w, const Eigen::MatrixBase<Args>&... args)
    {
        return static_cast<Derived*>(this)->hessian_impl(std::forward<Gradient>(gradient), std::forward<Hessian>(hessian), w, args...);
    }

    // default internal hessian code
    template <typename Tag, typename Weight, typename Gradient, typename Hessian, typename... Args, typename scalar_t = typename Eigen::MatrixBase<Weight>::Scalar>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_hessian<Derived, Tag, Gradient, Hessian, Eigen::MatrixBase<Weight>, Eigen::MatrixBase<Args>...>() == false && !tagless, scalar_t>::type
    hessian(Tag&& tag, Gradient&& gradient, Hessian&& hessian, const Eigen::MatrixBase<Weight>& w, const Eigen::MatrixBase<Args>&... args)
    {
        return hessian_impl_autodiff(std::forward<Tag>(tag), gradient, hessian, w, args...);
    }

    // delegate hessian without tag
    template <typename Weight, typename Gradient, typename Hessian, typename... Args, typename scalar_t = typename Eigen::MatrixBase<Weight>::Scalar>
    EIGEN_STRONG_INLINE typename std::enable_if<has_user_hessian<Derived, Gradient, Hessian, Eigen::MatrixBase<Weight>, Eigen::MatrixBase<Args>...>() == false && tagless, scalar_t>::type
    hessian(Gradient&& gradient, Hessian&& hessian, const Eigen::MatrixBase<Weight>& w, const Eigen::MatrixBase<Args>&... args)
    {
        return hessian_impl_autodiff(DefaultTag{}, std::forward<Gradient>(gradient), std::forward<Hessian>(hessian), w, args...);
    }

    // define DefaultTag for hessian
    template <typename Weight, typename Gradient, typename Hessian, typename... Args, typename scalar_t = typename Eigen::MatrixBase<Weight>::Scalar>
    EIGEN_STRONG_INLINE typename std::enable_if<tagless, scalar_t>::type
    hessian(DefaultTag, Gradient&& gradient, Hessian&& hessian, const Eigen::MatrixBase<Weight>& w, const Eigen::MatrixBase<Args>&... args)
    {
        return hessian(std::forward<Gradient>(gradient), std::forward<Hessian>(hessian), w, args...);
    }

private:

    //
    // Eigen Autodiff Implementations
    //


    /**
     * Class that provides implementations of jacobian_impl, hessian_impl, gradient_impl
     * using the Eigen AutoDiff tool
     */

    // Compute the jacobian with eigen autodiff
    template<typename Tag, typename OutValue, typename OutJacobian, typename... Args>
    EIGEN_STRONG_INLINE void
    jacobian_impl_autodiff(
            Tag&& tag, // Function to call
            const OutValue& outvalue, const OutJacobian& outjacobian, // Outputs
            const Eigen::MatrixBase<Args>&... args) noexcept // Function arguments
    {
        // Compute the scalar type
        using Scalar = lampc::meta::get_scalar_t<Eigen::MatrixBase<Args>...>;

        // Get the total number of inputs
        constexpr size_t num_inputs = lampc::meta::sum_template<Eigen::MatrixBase<Args>::RowsAtCompileTime...>();

        // First order derivative
        using AD_scalar = Eigen::AutoDiffScalar<Eigen::Vector<Scalar, num_inputs>>;
        using AD_Output = Eigen::Vector<AD_scalar,std::remove_reference_t<OutValue>::RowsAtCompileTime>;

        // Convert the arguments to AD variables, and call the function
        AD_Output out;
        seed_and_call(std::forward<Tag>(tag), out, make_ad<num_inputs>(args)...);

        // Copy out into output variables
        for(int i=0; i<out.rows(); i++)
        {
            // We cast away the constness to allow temporary expressions: https://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html
            const_cast<OutValue&>(outvalue)(i) = out[i].value();
            const_cast<OutJacobian&>(outjacobian)(i, Eigen::all) = out[i].derivatives().transpose();
        }
    }

    // Sets the input derivatives to the identity.
    // Assumes that the derivative matrix is initially zero
    template<typename Arg>
    static EIGEN_STRONG_INLINE int
    AD_Seed(Eigen::MatrixBase<Arg>& x, int offset)
    {
        for (int i=0; i<x.rows(); i++)
            x[i].derivatives().coeffRef(i + offset) = 1;
        return offset + x.rows();
    }

    // Take a vector input and return an AD version of the vector
    template<size_t num_inputs, typename X,
            typename AD_scalar = Eigen::AutoDiffScalar<Eigen::Vector<typename X::Scalar, num_inputs>>>
    static EIGEN_STRONG_INLINE auto
    make_ad(const Eigen::MatrixBase<X>& x)
    {
        constexpr size_t n = X::RowsAtCompileTime;
        Eigen::Vector<AD_scalar, n> y;
        y = x;
        for (int i=0; i<y.rows(); i++) {
            y[i].derivatives().setZero();
        }
        return y;
    }

    template<typename Tag, typename Output, typename... Args>
    EIGEN_STRONG_INLINE void
    seed_and_call(Tag&& tag, Output& out, Eigen::MatrixBase<Args>&&... args)
    {
        // Set derivative equal to identity
        int offset = 0;
        (void)std::initializer_list<int>{
            (
                offset = AD_Seed(args, offset), // Set to unit vectors
                0
            )...
        };

        function(std::forward<Tag>(tag), out, std::forward<Args>(args)...);
    }

    /**
     * Returns the value w'*f(x).
     * outgradient += gradient(w'*f(x))
     * outhessian += hessian(w'*f(x))
     */
    template<typename Tag, typename Weight, typename Gradient, typename Hessian,
            typename... Args, typename scalar_t = typename Eigen::MatrixBase<Weight>::Scalar>
    EIGEN_STRONG_INLINE scalar_t
    hessian_impl_autodiff(Tag&& tag,
                          const Gradient& gradient, const Hessian& hessian,
                          const Eigen::MatrixBase<Weight>& weight,
                          const Eigen::MatrixBase<Args>&... args) noexcept
    {
        constexpr size_t num_outputs = Eigen::MatrixBase<Weight>::RowsAtCompileTime;
        constexpr size_t num_inputs = meta::sum_template<Args::RowsAtCompileTime...>();

        // Second order derivative
        using AD_scalar = Eigen::AutoDiffScalar<Eigen::Vector<scalar_t, num_inputs>>;
        using outerDerivatives = Eigen::Vector<AD_scalar, num_inputs>;
        using outerADScalar = Eigen::AutoDiffScalar<outerDerivatives>;
        using outerAD_t = Eigen::Vector<outerADScalar, num_outputs>;

        // using value_t = typename Eigen::Vector<scalar_t, num_outputs>;
        // using jacobian_t = typename Eigen::Matrix<scalar_t, num_outputs, num_inputs>;
        // using hessian_t = typename Eigen::Matrix<scalar_t, num_inputs, num_inputs>;

        // Convert to AD variables for the inputs and call our function
        outerAD_t out;
        seed_and_call2(std::forward<Tag>(tag), out, make_ad2<outerADScalar>(args)...);

        scalar_t value = 0;

        // Copy into buffers
        for(int i=0; i<num_outputs; i++)
        {
            value += weight(i) * out[i].value().value();
            // We cast away the constness to allow temporary expressions: https://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html
            const_cast<Gradient&>(gradient) += weight(i) * out[i].value().derivatives();

            for (int j = 0; j < num_inputs; j++) {
                const_cast<Hessian&>(hessian)(j,Eigen::all) += weight(i) * out[i].derivatives()(j).derivatives().transpose();
            }
        }

        return value;
    }


    // Take a vector input and return a AD version of the vector
    template<typename outerADScalar, typename X>
    EIGEN_STRONG_INLINE auto make_ad2(const Eigen::MatrixBase<X>& x)
    {
        constexpr size_t n = X::RowsAtCompileTime;
        Eigen::Vector<outerADScalar, n> y;
        // y = x;
        for (int i=0; i<n; i++) {
            y(i).value().value() = x(i);
            y(i).value().derivatives().setZero();
            y(i).derivatives().setZero();
            for (int j = 0; j < n; j++) {
                y(i).derivatives()(j).derivatives().setZero();
            }
        }
        return y;
    }

    // Sets the input derivatives to the identity.
    // Assumes that the derivative matrix is initially zero
    template <typename Arg>
    EIGEN_STRONG_INLINE int
    AD_Seed2(Eigen::MatrixBase<Arg> &x, int offset)
    {
        for (int i=0; i<Arg::RowsAtCompileTime; i++)
        {
            x(i).value().derivatives().coeffRef(i + offset) = 1;
            x(i).derivatives().coeffRef(i + offset) = 1;
        }

        return offset + x.rows();
    }

    template<typename Tag, typename Output, typename... Args>
    EIGEN_STRONG_INLINE void seed_and_call2(Tag&& tag, Output& out, Eigen::MatrixBase<Args>&&... args)
    {
        // Set derivative equal to identity
        int offset = 0;
        (void)std::initializer_list<int>{
            (
                offset = AD_Seed2(args, offset), // Set to unit vectors
                0
            )...
        };

        // Call our functionxxx
        function(std::forward<std::decay_t<Tag>>(tag), out, std::forward<Args>(args)...);
    }

    /**
      * Returns the value w'*f(x)
      */
    template<typename Tag, typename Weight, typename... Args, typename scalar_t = typename Eigen::MatrixBase<Weight>::Scalar>
    inline scalar_t
    wsum_impl_autodiff(Tag&& tag,
                       const Eigen::MatrixBase<Weight>& weight,
                       const Eigen::MatrixBase<Args>&... args) noexcept
    {
        constexpr size_t num_outputs = Eigen::MatrixBase<Weight>::RowsAtCompileTime;
        Eigen::Vector<scalar_t, num_outputs> value;
        function(std::forward<Tag>(tag), value, std::forward<Args>(args)...);
        return weight.dot(value);
    }

    /**
     * Returns the value w'*f(x) and *adds* the gradient to gradient
     */
    template<typename Tag, typename Weight, typename Gradient,
            typename... Args, typename scalar_t = typename Eigen::MatrixBase<Weight>::Scalar>
    inline scalar_t
    gradient_impl_autodiff(Tag&& tag,
                           const Gradient& gradient,
                           const Eigen::MatrixBase<Weight>& weight,
                           const Eigen::MatrixBase<Args>&... args) noexcept
    {
        constexpr size_t num_outputs = Eigen::MatrixBase<Weight>::RowsAtCompileTime;
        constexpr size_t num_inputs = meta::sum_template<Args::RowsAtCompileTime...>();

        // Call the (possibly overloaded) jacobian
        Eigen::Vector<scalar_t, num_outputs> value;
        Eigen::Matrix<scalar_t, num_outputs, num_inputs> jacobian;
        value.array() = 0; jacobian.array() = 0;
        this->jacobian(std::forward<Tag>(tag), value,jacobian, args...);
        // We cast away the constness to allow temporary expressions: https://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html
        const_cast<Gradient&>(gradient) += weight.transpose() * jacobian;
        return weight.dot(value);
    }

};

}; // namespace lampc

#endif // __LAMPC__FUNCTION_TAG_HPP