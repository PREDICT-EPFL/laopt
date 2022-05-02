#ifndef __LAMPC__FUNCTION_HPP
#define __LAMPC__FUNCTION_HPP

// Defines differentiable dense functions
#include "Eigen/Dense"
#include "unsupported/Eigen/AutoDiff"

namespace lampc {

struct Eval {};
struct Jacobian {};
struct Hessian {};
struct Gradient {};


template<typename Derived, typename scalar_t_, size_t num_outputs_, size_t... input_sizes>
struct MakeDifferentiable
{
    static constexpr size_t num_inputs = lampc::meta::sum_template<input_sizes...>();  // Total number of inputs
	static constexpr size_t num_outputs = num_outputs_;
	using scalar_t = scalar_t_;

	// Output types
    using value_t = Eigen::Vector<scalar_t, num_outputs>;
    using jacobian_t = Eigen::Matrix<scalar_t, num_outputs, num_inputs>;
    using hessian_t = Eigen::Matrix<scalar_t, num_inputs, num_inputs>;
    using hessian_array_t = std::array<hessian_t, num_outputs>;  // Hessian for each of the outputs in a vector-function

    /**
     * Writes the output to the vector-like object outvalue 
     * vector-like = an object that can be assigned a vector of type value_t
     */
    template<typename OutValue>
    EIGEN_STRONG_INLINE void
    operator()(const Eigen::Ref< const Eigen::Vector<scalar_t, input_sizes> >&... args, OutValue&& outvalue) noexcept
    {
        outvalue = static_cast<Derived*>(this)->impl(args...);
    }

    /**
     * Writes the output to the vector-like object outvalue and the matrix-like object outjacobian
     *   vector-like = an object that can be assigned a vector of type value_t and indexed like outvalue(rows)
     *   matrix-like = an object that can be assigned and indexed as outjacobian(rows, cols)
     * Does not set the internal value or jacobian buffers
     */
    template<typename OutValue, typename OutJacobian>
    EIGEN_STRONG_INLINE void
    operator()(const Eigen::Ref<const Eigen::Vector<scalar_t, input_sizes>>&... args, 
    		   OutValue&& outvalue, OutJacobian&& outjacobian) noexcept
    {
        // Convert to AD variables for the inputs and call our function
        AD_output_t out = seed_and_call(make_ad(args)...);

		value_t value;  // We use temporary buffers here in order to avoid writing directly to BSMatrix objects 1-element at a time
		jacobian_t jacobian;

        // Copy MakeJacobian into output variables
        for(int i=0; i<num_outputs; i++)
        {
            value(i) = out[i].value();
            jacobian.row(i) = out[i].derivatives();
        }

        outvalue = value;
        outjacobian = jacobian;
    }

private:

    // First order derivative
    using AD_scalar = Eigen::AutoDiffScalar<Eigen::Matrix<scalar_t, num_inputs, 1>>;
    using AD_output_t = Eigen::Matrix<AD_scalar, num_outputs, 1>;  

    // Sets the input derivatives to the identity. 
    // Assumes that the derivative matrix is initially zero
    template <typename vec>
    static constexpr int AD_Seed(vec &x, int offset)
    {
        for (int i=0; i<x.rows(); i++)
            x[i].derivatives().coeffRef(i + offset) = 1;
        return offset + x.rows();
    }

    // Take a vector input and return a AD version of the vector
    template<int n>
    static EIGEN_STRONG_INLINE 
    Eigen::Matrix<AD_scalar, n, 1> 
    make_ad(const Eigen::Ref<const Eigen::Matrix<scalar_t, n, 1>> x)
    // static EIGEN_STRONG_INLINE auto make_ad(const Arg& x)
    {
        Eigen::Matrix<AD_scalar, n, 1> y;
        y = x;
        for (int i=0; i<y.rows(); i++) {
            y[i].derivatives().setZero();
        }
        return y;
    }

    EIGEN_STRONG_INLINE Eigen::Matrix<AD_scalar, num_outputs, 1>
    seed_and_call(Eigen::Matrix<AD_scalar, input_sizes, 1>... args)
    {
        // Set derivative equal to identity
        int offset = 0;
        (void)std::initializer_list<int>{ 
            (
                offset = AD_Seed(args, offset), // Set to unit vectors
                0
            )...
        };
        return static_cast<Derived*>(this)->template impl<AD_scalar>(args...);
    }

    // Second order derivative
    using outerDerivatives = Eigen::Matrix<AD_scalar, num_inputs, 1>;
    using outerADScalar = Eigen::AutoDiffScalar<outerDerivatives>;
    using outerAD_t = Eigen::Matrix<outerADScalar, num_outputs, 1>;  

public:

    template<typename OutValue, typename OutJacobian, typename OutHessianArray>
    EIGEN_STRONG_INLINE void
    operator()(const Eigen::Ref<const Eigen::Matrix<scalar_t, input_sizes, 1>>&... args,
    		   OutValue&& outvalue, OutJacobian&& outjacobian, OutHessianArray&& outhessian) noexcept
    {
        // Convert to AD variables for the inputs and call our function
        outerAD_t out = seed_and_call2(make_ad2<input_sizes>(args)...);

        // We use buffers here so we can do block-copies at the end
        value_t value;
        jacobian_t jacobian;
        hessian_array_t hessian;

        // Copy into buffers
        for(int i=0; i<num_outputs; i++)
        {
            value(i) = out[i].value().value();
            jacobian.row(i) = out[i].value().derivatives();
            for (int j = 0; j < num_inputs; j++) {
                hessian[i].template middleRows<1>(j) = out[i].derivatives()(j).derivatives().transpose();
            }
        }

        outvalue = value;
        outjacobian = jacobian;
        for(int i=0; i<num_outputs; i++)
        	outhessian[i] = hessian[i];
    }

private:

    // Take a vector input and return a AD version of the vector
    template<int n>
    static EIGEN_STRONG_INLINE Eigen::Matrix<outerADScalar, n, 1> 
        make_ad2(const Eigen::Ref<const Eigen::Matrix<scalar_t, n, 1>> x)
    {
        Eigen::Matrix<outerADScalar, n, 1> y;
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
    template <typename vec>
    static constexpr int AD_Seed2(vec &x, int offset)
    {
        for (int i=0; i<x.rows(); i++)
        {
            x(i).value().derivatives().coeffRef(i + offset) = 1;
            x(i).derivatives().coeffRef(i + offset) = 1;
        }

        return offset + x.rows();
    }

    EIGEN_STRONG_INLINE outerAD_t seed_and_call2(Eigen::Matrix<outerADScalar, input_sizes, 1>... args)
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
        return static_cast<Derived*>(this)->template impl<outerADScalar>(args...);
    }

public:

	/**
	 * Tagged function calls
	 */

    EIGEN_STRONG_INLINE value_t
    operator()(lampc::Eval, const Eigen::Ref< const Eigen::Vector<scalar_t, input_sizes> >&... args) noexcept        
    {
    	value_t value;
    	value.array() = 0;
    	static_cast<Derived*>(this)->operator()(args..., value);
    	return value;
    }

    EIGEN_STRONG_INLINE std::pair<value_t, jacobian_t>
    operator()(lampc::Jacobian, const Eigen::Ref<const Eigen::Vector<scalar_t, input_sizes>>&... args) noexcept
    {
		std::pair<value_t, jacobian_t> ret;
		std::get<0>(ret).array() = 0;
		std::get<1>(ret).array() = 0;
		static_cast<Derived*>(this)->operator()(args..., std::get<0>(ret), std::get<1>(ret));
		return ret;
    }

    EIGEN_STRONG_INLINE std::tuple<value_t, jacobian_t, hessian_array_t>
    operator()(lampc::Hessian, const Eigen::Ref<const Eigen::Matrix<scalar_t, input_sizes, 1>>&... args) noexcept
    {
    	std::tuple<value_t, jacobian_t, hessian_array_t> ret;
		std::get<0>(ret).array() = 0;
		std::get<1>(ret).array() = 0;
		for(int i=0; i<num_outputs; i++) std::get<2>(ret)[i].array() = 0;
    	static_cast<Derived*>(this)->operator()(args..., std::get<0>(ret), std::get<1>(ret), std::get<2>(ret));
    	return ret;
    }

	/**
	 * Define this function as g(x) = w'*f(x), then this computes the value of g
	 */

    /**
     * Returns the value w'*f(x)
     */
    EIGEN_STRONG_INLINE scalar_t
    weightedsum(const Eigen::Ref<const Eigen::Vector<scalar_t, input_sizes>>&... args,
    		    const Eigen::Ref<const Eigen::Vector<scalar_t, num_outputs>>& weight) noexcept
    {
    	return weight.transpose() * operator()(lampc::Eval(), args...);
    }

    /**
     * Returns the value w'*f(x) and its gradient
     */
    template<typename OutGradient>
    EIGEN_STRONG_INLINE scalar_t
    weightedsum(const Eigen::Ref<const Eigen::Vector<scalar_t, input_sizes>>&... args, 
    			OutGradient&& outgradient,
    			const Eigen::Ref<const Eigen::Vector<scalar_t, num_outputs>>& weight) noexcept
    {
    	auto val = operator()(lampc::Jacobian(), args...);
    	outgradient += std::get<1>(val).transpose() * weight; // Jacobian
    	return weight.transpose() * std::get<0>(val); // Value
    }

    /**
     * Returns the value w'*f(x), its gradient and hessian
     */
    template<typename OutGradient, typename OutHessian>
    EIGEN_STRONG_INLINE scalar_t
    weightedsum(const Eigen::Ref<const Eigen::Vector<scalar_t, input_sizes>>&... args, 
    			const Eigen::Ref<const Eigen::Vector<scalar_t, num_outputs>>& weight,
    			OutGradient&& outgradient, OutHessian&& outhessian) noexcept
    {
    	auto val = operator()(lampc::Hessian(), args...);
    	outgradient += std::get<1>(val).transpose() * weight; // Jacobian
    	for(int i=0; i<num_outputs; i++)
    		outhessian += weight(i) * std::get<2>(val)[i]; // Hessian
    	return weight.transpose() * std::get<0>(val); // Value
    }
};

};

#endif // __LAMPC__FUNCTION_HPP