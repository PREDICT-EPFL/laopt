#ifndef __LAMPC__FUNCTION_HPP
#define __LAMPC__FUNCTION_HPP

// Defines differentiable dense functions
#include "Eigen/Dense"
#include "unsupported/Eigen/AutoDiff"

template<typename Derived>
struct func_traits;

namespace lampc {

struct Eval {};
struct Jacobian {};
struct Hessian {};
struct Gradient {};

template<typename Arg, typename... Args>
struct get_scalar
{
    using type = typename Arg::Scalar;
};
template<typename... Args>
using get_scalar_t = typename get_scalar<Args...>::type;


template<typename Derived>
struct MakeDifferentiable
{
    template<typename OutValue, typename... Args> //, typename = typename std::enable_if<sizeof...(Args) == num_input_vectors>::type>
    EIGEN_STRONG_INLINE void
    operator()(lampc::Eval, 
               OutValue&& outvalue, 
               const Eigen::MatrixBase<Args>&... args) noexcept
    {
        outvalue = static_cast<Derived*>(this)->impl(args...);
    }

    /**
     * Writes the output to the vector-like object outvalue and the matrix-like object outjacobian
     *   vector-like = an object that can be assigned a vector of type value_t and indexed like outvalue(rows)
     *   matrix-like = an object that can be assigned and indexed as outjacobian(rows, cols)
     * Does not set the internal value or jacobian buffers
     */
    template<typename OutValue, typename OutJacobian, typename... Args>
                    // typename = typename std::enable_if<sizeof...(Args) == num_input_vectors>::type>
    EIGEN_STRONG_INLINE void
    operator()(lampc::Jacobian,
               OutValue&& outvalue, OutJacobian&& outjacobian,
               const Eigen::MatrixBase<Args>&... args) noexcept
    {
        constexpr size_t num_inputs = meta::sum_template<Args::RowsAtCompileTime...>();

        // Convert to AD variables for the inputs and call our function
        auto out = seed_and_call(make_ad<num_inputs>(args)...).eval();

        constexpr size_t num_outputs = decltype(out)::RowsAtCompileTime;
        using scalar_t = get_scalar_t<Args...>;
        using value_t = typename Eigen::Vector<scalar_t, num_outputs>;
        using jacobian_t = typename Eigen::Matrix<scalar_t, num_outputs, num_inputs>;

        // Copy MakeJacobian into output variables
        for(int i=0; i<out.rows(); i++)
        {
            outvalue(i) = out[i].value();
            outjacobian(i,Eigen::all) = out[i].derivatives().transpose();
        }
    }

private:

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

    // Take a vector input and return a AD version of the vector
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

    template<typename... Args>
    EIGEN_STRONG_INLINE auto
    seed_and_call(Eigen::MatrixBase<Args>&&... args)
    {
        // Set derivative equal to identity
        int offset = 0;
        (void)std::initializer_list<int>{ 
            (
                offset = AD_Seed(args, offset), // Set to unit vectors
                0
            )...
        };

        return static_cast<Derived*>(this)->impl(args...);
    }

public:

	/**
	 * Compute the hessian of each output
     * 
	 */
    template<typename OutValue, typename OutJacobian, typename OutHessian, size_t num_outputs, typename... Args>
    EIGEN_STRONG_INLINE void
    operator()(lampc::Hessian,
               OutValue& outvalue, OutJacobian& outjacobian, std::array<OutHessian, num_outputs>& outhessian,
               const Eigen::MatrixBase<Args>&... args) noexcept
    {
        constexpr size_t num_inputs = meta::sum_template<Args::RowsAtCompileTime...>();

        // Second order derivative
        using scalar_t = get_scalar_t<Args...>;
        using AD_scalar = Eigen::AutoDiffScalar<Eigen::Vector<scalar_t, num_inputs>>;
        using outerDerivatives = Eigen::Vector<AD_scalar, num_inputs>;
        using outerADScalar = Eigen::AutoDiffScalar<outerDerivatives>;
        using outerAD_t = Eigen::Vector<outerADScalar, num_outputs>;  

        using value_t = typename Eigen::Vector<scalar_t, num_outputs>;
        using jacobian_t = typename Eigen::Matrix<scalar_t, num_outputs, num_inputs>;
        using hessian_t = typename Eigen::Matrix<scalar_t, num_inputs, num_inputs>;

        // Convert to AD variables for the inputs and call our function
        outerAD_t out = seed_and_call2(make_ad2<outerADScalar>(args)...).eval();

        // Copy into buffers
        for(int i=0; i<num_outputs; i++)
        {
            outvalue(i) = out[i].value().value();
            outjacobian.row(i) = out[i].value().derivatives();
            for (int j = 0; j < num_inputs; j++) {
                outhessian[i].template middleRows<1>(j) = out[i].derivatives()(j).derivatives().transpose();
            }
        }
    }

private:

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

    template<typename... Args>
    EIGEN_STRONG_INLINE auto seed_and_call2(Eigen::MatrixBase<Args>&&... args)
    {
        // Set derivative equal to identity
        int offset = 0;
        (void)std::initializer_list<int>{ 
            (
                offset = AD_Seed2(args, offset), // Set to unit vectors
                0
            )...
        };

        // Call our function
        return static_cast<Derived*>(this)->impl(args...);
    }

public:
	/**
	 * Define this function as g(x) = w'*f(x), then this computes the value of g
	 */

    /**
     * Returns the value w'*f(x)
     */
    template<typename Weight, typename... Args>
    EIGEN_STRONG_INLINE auto
    weightedsum(lampc::Eval,
                const Eigen::MatrixBase<Weight>& weight,
                const Eigen::MatrixBase<Args>&... args) noexcept
    {
        using scalar_t = typename Eigen::MatrixBase<Weight>::Scalar;
        constexpr size_t num_outputs = Eigen::MatrixBase<Weight>::RowsAtCompileTime;

        // Call the (possibly overloaded) eval
        Eigen::Vector<scalar_t, num_outputs> value;
        static_cast<Derived*>(this)->operator()(lampc::Eval(), value, args...);
    	return weight.dot(value);
    }

    /**
     * Returns the value w'*f(x).
     * outgradient += gradient(w'*f(x))
     */
    template<typename OutGradient, typename Weight, typename... Args>
    EIGEN_STRONG_INLINE auto
    weightedsum(lampc::Gradient,
                OutGradient&& outgradient,
                const Eigen::MatrixBase<Weight>& weight,
                const Eigen::MatrixBase<Args>&... args) noexcept
    {
        using scalar_t = typename Eigen::MatrixBase<Weight>::Scalar;
        constexpr size_t num_outputs = Eigen::MatrixBase<Weight>::RowsAtCompileTime;
        constexpr size_t num_inputs = meta::sum_template<Args::RowsAtCompileTime...>();

        // Call the (possibly overloaded) jacobian
        Eigen::Vector<scalar_t, num_outputs> value;
        Eigen::Matrix<scalar_t, num_outputs, num_inputs> jacobian;
        value.array() = 0; jacobian.array() = 0;
        static_cast<Derived*>(this)->operator()(lampc::Jacobian(), value,jacobian, args...);

        outgradient += jacobian.transpose() * weight;
        return weight.dot(value);
    }

    /**
     * Returns the value w'*f(x).
     * outgradient += gradient(w'*f(x))
     * outhessian += hessian(w'*f(x))
     */
    template<typename OutGradient, typename OutHessian, typename Weight, typename... Args>
    EIGEN_STRONG_INLINE auto
    weightedsum(lampc::Hessian,
                OutGradient&& outgradient, OutHessian&& outhessian,
                const Eigen::MatrixBase<Weight>& weight,
                const Eigen::MatrixBase<Args>&... args) noexcept
    {
        using scalar_t = typename Eigen::MatrixBase<Weight>::Scalar;
        constexpr size_t num_outputs = Eigen::MatrixBase<Weight>::RowsAtCompileTime;
        constexpr size_t num_inputs = meta::sum_template<Args::RowsAtCompileTime...>();

     	// Note: We re-code this here, rather than call operator()(Hessian) in order
    	//       to avoid the time overhead of computing the hessian output array.


        // Second order derivative
        using AD_scalar = Eigen::AutoDiffScalar<Eigen::Vector<scalar_t, num_inputs>>;
        using outerDerivatives = Eigen::Vector<AD_scalar, num_inputs>;
        using outerADScalar = Eigen::AutoDiffScalar<outerDerivatives>;
        using outerAD_t = Eigen::Vector<outerADScalar, num_outputs>;  

        using value_t = typename Eigen::Vector<scalar_t, num_outputs>;
        using jacobian_t = typename Eigen::Matrix<scalar_t, num_outputs, num_inputs>;
        using hessian_t = typename Eigen::Matrix<scalar_t, num_inputs, num_inputs>;

        // Convert to AD variables for the inputs and call our function
        outerAD_t out = seed_and_call2(make_ad2<outerADScalar>(args)...).eval();

        scalar_t value = 0;

        // Copy into buffers
        for(int i=0; i<num_outputs; i++)
        {
            value += weight(i) * out[i].value().value();
			outgradient += weight(i) * out[i].value().derivatives();

            for (int j = 0; j < num_inputs; j++) {
                outhessian(j,Eigen::all) += weight(i) * out[i].derivatives()(j).derivatives().transpose();
            }
        }

    	return value;
    }
};

};

#endif // __LAMPC__FUNCTION_HPP