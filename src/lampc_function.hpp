#ifndef __LAMPC__FUNCTION_HPP
#define __LAMPC__FUNCTION_HPP

// Defines differentiable dense functions
#include "Eigen/Dense"
#include "unsupported/Eigen/AutoDiff"

namespace lampc {

struct Eval {};
struct Jacobian {};
struct Hessian {};

template<typename _scalar_t, size_t num_outputs, size_t... input_sizes>
struct FuncInfo
{
    using scalar_t = _scalar_t;
    static constexpr int num_inputs = lampc::meta::sum_template<input_sizes...>();  // Total number of inputs
};

template<typename Derived, typename Info>
struct MakeEval;

template<typename Derived, typename scalar_t_, size_t num_outputs_, size_t... input_sizes>
struct MakeEval<Derived, FuncInfo<scalar_t_, num_outputs_, input_sizes...>>
{
	static constexpr size_t num_outputs = num_outputs_;
	using scalar_t = scalar_t_;

    using value_t = Eigen::Vector<scalar_t, num_outputs>;

    /**
     * Returns the value of the function
     */
    EIGEN_STRONG_INLINE value_t
    operator()(lampc::Eval, const Eigen::Ref< const Eigen::Vector<scalar_t, input_sizes> >&... args) noexcept        
    {
        return static_cast<Derived*>(this)->impl(args...);
    }

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
     * Define this function as g(x) = w'*f(x), then this computes the value of g
     * 
     * Returns the value w'*f(x)
     */
    EIGEN_STRONG_INLINE scalar_t
    weightedsum(const Eigen::Ref<const Eigen::Vector<scalar_t, input_sizes>>&... args,
    			const Eigen::Ref<const Eigen::Vector<scalar_t, num_outputs>>& weight) noexcept
    {
    	return weight.transpose() * operator()(lampc::Eval(), args...);
    }

};

template<typename Derived, typename Info>
struct MakeJacobian;

template<typename Derived, typename scalar_t, size_t num_outputs, size_t... input_sizes>
struct MakeJacobian<Derived, FuncInfo<scalar_t, num_outputs, input_sizes...>>
            : public MakeEval<Derived, FuncInfo<scalar_t, num_outputs, input_sizes...>>
{
    using Info = FuncInfo<scalar_t, num_outputs, input_sizes...>;
    using MakeEval<Derived, FuncInfo<scalar_t, num_outputs, input_sizes...>>::operator();
    using typename MakeEval<Derived, FuncInfo<scalar_t, num_outputs, input_sizes...>>::value_t;

    using jacobian_t = Eigen::Matrix<scalar_t, num_outputs, Info::num_inputs>;

    // First order derivative
    using AD_scalar = Eigen::AutoDiffScalar<Eigen::Matrix<scalar_t, Info::num_inputs, 1>>;
    using AD_output_t = Eigen::Matrix<AD_scalar, num_outputs, 1>;  

    /**
     * Sets the internal value and jacobian buffers
     * Returns the value and dense jacobian of the function
     */
    EIGEN_STRONG_INLINE auto
    operator()(lampc::Jacobian, const Eigen::Ref<const Eigen::Vector<scalar_t, input_sizes>>&... args) noexcept
    {
        // Convert to AD variables for the inputs and call our function
        AD_output_t out = seed_and_call(make_ad(args)...);

		std::pair<value_t, jacobian_t> ret;

        // Copy MakeJacobian into output variables
        for(int i=0; i<num_outputs; i++)
        {
            std::get<0>(ret)(i) = out[i].value();
            std::get<1>(ret).row(i) = out[i].derivatives();
        }

        return ret;
    }


    /**
     * Writes the output to the vector-like object outvalue and the matrix-like object outjacobian
     *   vector-like = an object that can be assigned a vector of type value_t and indexed like outvalue(rows)
     *   matrix-like = an object that can be assigned and indexed as outjacobian(rows, cols)
     * Does not set the internal value or jacobian buffers
     */
    template<typename OutValue, typename OutJacobian>
    EIGEN_STRONG_INLINE void
    operator()(const Eigen::Ref<const Eigen::Vector<scalar_t, input_sizes>>&... args, OutValue&& outvalue, OutJacobian&& outjacobian) noexcept
    {
    	std::tie(outvalue, outjacobian) = operator()(lampc::Jacobian(), args...);
    }

    /**
     * Define this function as g(x) = w'*f(x), then this computes the value and gradient of g
     * 
     * Returns the value w'*f(x)
     * Adds outgradient += w'*jacobian(f(x))
     */
    template<typename OutGradient>
    EIGEN_STRONG_INLINE scalar_t
    weightedsum(const Eigen::Ref<const Eigen::Vector<scalar_t, input_sizes>>&... args, 
    			OutGradient&& outgradient,
    			const Eigen::Ref<const Eigen::Vector<scalar_t, num_outputs>>& weight) noexcept
    {
    	auto val = operator()(lampc::Jacobian(), args...);
    	outgradient = weight.transpose() * std::get<1>(val); // Jacobian
    	return weight.transpose() * std::get<0>(val); // Value
    }


private:

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
};


template<typename Derived, typename Info>
struct MakeHessian;

template<typename Derived, typename scalar_t, size_t num_outputs, size_t... input_sizes>
struct MakeHessian<Derived, FuncInfo<scalar_t, num_outputs, input_sizes...>>
            : public MakeJacobian<Derived, FuncInfo<scalar_t, num_outputs, input_sizes...>>
{
    using Info = FuncInfo<scalar_t, num_outputs, input_sizes...>;
    using Jacobian = MakeJacobian<Derived, FuncInfo<scalar_t, num_outputs, input_sizes...>>;
    using Jacobian::operator();
    using typename Jacobian::jacobian_t;
    using typename Jacobian::value_t;

    // Hessian for each of the outputs in an array
    using hessian_single_t = Eigen::Matrix<scalar_t, Info::num_inputs, Info::num_inputs>;
    using hessian_t = std::array<hessian_single_t, num_outputs>;    
    hessian_t hessian;

    // Second order derivative
    using AD_scalar = Eigen::AutoDiffScalar<Eigen::Matrix<scalar_t, Info::num_inputs, 1>>;
    using outerDerivatives = Eigen::Matrix<AD_scalar, Info::num_inputs, 1>;
    using outerADScalar = Eigen::AutoDiffScalar<outerDerivatives>;
    using outerAD_t = Eigen::Matrix<outerADScalar, num_outputs, 1>;  

    EIGEN_STRONG_INLINE std::tuple<value_t, jacobian_t, hessian_t>
    operator()(lampc::Hessian, const Eigen::Ref<const Eigen::Matrix<scalar_t, input_sizes, 1>>&... args) noexcept
    {
        // Convert to AD variables for the inputs and call our function
        outerAD_t out = seed_and_call2(make_ad2<input_sizes>(args)...);

        std::tuple<value_t, jacobian_t, hessian_t> ret;

        // Copy MakeHessian into output variables
        for(int i=0; i<num_outputs; i++)
        {
            std::get<0>(ret)(i) = out[i].value().value();
            std::get<1>(ret).row(i) = out[i].value().derivatives();
            for (int j = 0; j < Info::num_inputs; j++) {
                std::get<2>(ret)[i].template middleRows<1>(j) = out[i].derivatives()(j).derivatives().transpose();
            }
        }

        return ret;
    }

    /**
     * Define this function as g(x) = w'*f(x), then this computes the value, gradient and hessian of g
     * 
     * Returns the value w'*f(x)
     * Adds outgradient += w'*jacobian(f(x))
     * Adds outhessian += sum wi*hessian(f_i(x))
     */
    template<typename OutGradient, typename OutHessian>
    EIGEN_STRONG_INLINE scalar_t
    weightedsum(const Eigen::Ref<const Eigen::Vector<scalar_t, input_sizes>>&... args, 
    			OutGradient&& outgradient, OutHessian&& outhessian, 
    			const Eigen::Ref<const Eigen::Vector<scalar_t, num_outputs>>& weight) noexcept
    {
    	auto val = operator()(lampc::Hessian(), args...);
    	outgradient += weight.transpose() * std::get<1>(val); // Jacobian
    	for(int i=0; i<num_outputs; i++)
    		outhessian += weight(i) * std::get<2>(val)[i]; // Hessian
    	return weight.transpose() * std::get<0>(val); // Value
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
};
};

#endif // __LAMPC__FUNCTION_HPP