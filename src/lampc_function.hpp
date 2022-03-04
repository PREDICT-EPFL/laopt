#ifndef __LAMPC__FUNCTION_HPP
#define __LAMPC__FUNCTION_HPP

// Defines differentiable dense functions

#include "Eigen/Dense"
#include <Eigen/Sparse>
#include "unsupported/Eigen/AutoDiff"

#include "lampc_utility.hpp"

#include "map.hpp"

using namespace Eigen;

namespace lampc {

// User macro to define a differentiable function
#define really_unparen(...) __VA_ARGS__

#define GET_VAR_SIZE(name, len) len
#define GET_VAR_SIZE_PAIR(pair) GET_VAR_SIZE pair
#define GET_VAR_NAME(name, len) name
#define GET_VAR_NAME_PAIR(pair) GET_VAR_NAME pair
#define GET_OUT_SIZE(name, len) len
#define GET_OUT_SIZE_PAIR(pair) GET_OUT_SIZE pair
#define GET_VAR_LIST(name, len) const Eigen::Ref<const Eigen::Matrix<T, len, 1>> name
#define GET_VAR_LIST_PAIR(pair) GET_VAR_LIST pair

/** @file */

/**
 * Macro to define a differentiable function.
 * 
 * For example:
 * 
 * \code{.cpp}
 *  FUNCTION(quadratic, scalar_t, param_t, (out, 2), (x, 1), (y, 2))
 *  {
 *      out << x(0) + y(0), 
 *             p.q * x(0) + y(0) + 2 * y(1);
 *  };
 * \endcode
 *  
 * Will expand to the structure
 * \code{.cpp}
 * struct quadratic_
 * {
 *    template<typename T>
 *    static EIGEN_STRONG_INLINE void impl(const param_t& p, 
 *               Eigen::Ref<Eigen::Matrix<T, 2, 1> out,
 *               const Eigen::Ref<const Eigen::Matrix<T, 1, 1> x,
 *               const Eigen::Ref<const Eigen::Matrix<T, 2, 1> y)
 *    {
 *      quadratic_impl<T>(p, out, x, y);
 *    }
 * };
 * 
 * using quadratic = lampc::Jacobian<quadratic_, scalar_t, param_t, 2, 1, 2>;
 * 
 * template<typename T>
 * static EIGEN_STRONG_INLINE void quadratic_impl(const param_t& p, 
 *            Eigen::Ref<Eigen::Matrix<T, 2, 1> out,
 *            const Eigen::Ref<const Eigen::Matrix<T, 1, 1> x,
 *            const Eigen::Ref<const Eigen::Matrix<T, 2, 1> y)
 * {
 *      out << x(0) + y(0), 
 *             p.q * x(0) + y(0) + 2 * y(1);
 * }
 * \endcode 
 */
 #define FUNCTION(func_name, scalar_t, param_t, out_pair, ...) \
	struct func_name##_ \
	{ \
		static constexpr const char* name=#func_name;\
		template<typename T> \
		static EIGEN_STRONG_INLINE void impl(const param_t& p,\
		Eigen::Ref<Eigen::Matrix<T, GET_VAR_SIZE_PAIR(out_pair), 1>> GET_VAR_NAME_PAIR(out_pair), \
		MAP_LIST(GET_VAR_LIST_PAIR, __VA_ARGS__) \
		) noexcept \
		{ \
			func_name##_impl<T>(p, GET_VAR_NAME_PAIR(out_pair), MAP_LIST(GET_VAR_NAME_PAIR, __VA_ARGS__)); \
		} \
	}; \
	using func_name = lampc::Jacobian<func_name##_, scalar_t, param_t, \
	GET_VAR_SIZE_PAIR(out_pair), \
	MAP_LIST(GET_VAR_SIZE_PAIR, __VA_ARGS__) \
	>; \
	template<typename T> \
	static EIGEN_STRONG_INLINE void func_name##_impl(const param_t& p,\
	Eigen::Ref<Eigen::Matrix<T, GET_VAR_SIZE_PAIR(out_pair), 1>> GET_VAR_NAME_PAIR(out_pair), \
	MAP_LIST(GET_VAR_LIST_PAIR, __VA_ARGS__) \
	) noexcept


/*************************************************************
	 Jacobian computation
 *************************************************************/

namespace detail 
{
	template<typename scalar_t, int num_outputs, int num_inputs>
	struct jacobian_return_t
	{
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW

		Eigen::Matrix<scalar_t, num_outputs, 1> val;
		Eigen::Matrix<scalar_t, num_outputs, num_inputs> jacobian;
	};

	template<typename scalar_t, int num_outputs, int num_inputs>
	struct hessian_return_t
	{
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW

		Eigen::Matrix<scalar_t, num_outputs, 1> val;
		Eigen::Matrix<scalar_t, num_outputs, num_inputs> jacobian;
		using hessian_t = Eigen::Matrix<scalar_t, num_inputs, num_inputs>;
		std::array<hessian_t, num_outputs> hessian;
	};
};

template<typename Func_, typename scalar_t_, typename param_t_, int num_outputs_, int... input_sizes>
struct Jacobian // < FunctionTraits<Func, scalar_t, param_t, num_outputs, input_sizes...> >
{
	using Func = Func_;
	using param_t = param_t_;
	using scalar_t = scalar_t_;

	static constexpr const char* name = Func::name;

	static constexpr int num_inputs = meta::sum_template<input_sizes...>();  // Total number of inputs
	static constexpr int num_input_vars = sizeof...(input_sizes);  // Number of input vector variables
	static constexpr int num_outputs = num_outputs_;

	static std::vector<int> get_input_sizes()
	{
		return std::vector<int>{input_sizes...};
	}

	/**
	 * Return the number of nonzeros in the jacobian
	 * 
	 * Default implementation assumes dense jacobians.
	 * Oveload in child class for sparse.
	 */
	static constexpr int nnzJacobian()
	{
		int nnz = 0;
		auto l = {(
			nnz += input_sizes * num_outputs_,
			0
			)...};
		return nnz;
	}

	/**
	 * Returns the sparsity structure of the jacobian of this function
	 * 
	 * S = [J_var1 J_var2 ...]
	 */
	static Eigen::SparseMatrix<int> jacobianStructure()
	{
		// Default is just a dense matrix
		Eigen::MatrixX<int> S(num_outputs, num_inputs);
		S.array() = 1;
		return S.sparseView();
	}

	/**
	 * Returns the sparsity structure of the hessian of the i'th output of this function
	 */
	static Eigen::SparseMatrix<int> hessianStructure(int output_index)
	{
		// Default is just a dense matrix
		Eigen::MatrixX<int> H(num_inputs, num_inputs);
		H.array() = 1;
		return H.sparseView();
	}


	// First order derivative
	using AD_scalar = Eigen::AutoDiffScalar<Eigen::Matrix<scalar_t, num_inputs, 1>>;
	using AD_output_t = Eigen::Matrix<AD_scalar, num_outputs, 1>;  

	// Second order derivative
	using outerDerivatives = Eigen::Matrix<AD_scalar, num_inputs, 1>;
	using outerADScalar = Eigen::AutoDiffScalar<outerDerivatives>;
	using outerAD_t = Eigen::Matrix<outerADScalar, num_outputs, 1>;  

	using jacobian_return_t = detail::jacobian_return_t<scalar_t, num_outputs, num_inputs>;
	using hessian_return_t = detail::hessian_return_t<scalar_t, num_outputs, num_inputs>;

	/*
		Evaluate the function with scalar type
	 */
	template<typename T>
	static EIGEN_STRONG_INLINE void impl(
		const param_t& param,
		Eigen::Ref<Eigen::Matrix<T, num_outputs, 1>> out, 
		const Eigen::Ref<const Eigen::Matrix<T, input_sizes, 1>>&... args) 
		noexcept
	{
		Func::template impl<T>(param, out, args...);
	}


	/*
		Evaluate the function
	 */
	static EIGEN_STRONG_INLINE auto eval(
		const param_t& param,
		const Eigen::Ref<const Eigen::Matrix<scalar_t, input_sizes, 1>>&... args) 
		noexcept
	{
		Eigen::Matrix<scalar_t, num_outputs, 1> out;
		Func::template impl<scalar_t>(param, out, args...);
		return out;
	}

	/*
		Evaluate the function, and its jacobian
	 */
	static EIGEN_STRONG_INLINE auto jac(
		const param_t& param,
		const Ref<const Matrix<scalar_t, input_sizes, 1>>&... args) 
		noexcept
	{
		AD_output_t _out;

		// Convert to AD variables for the inputs and call our function
		seed_and_call(make_ad<input_sizes>(args)..., _out, param);

		// Copy Jacobian into output variables
		jacobian_return_t ret;
		for(int i=0; i<num_outputs; i++)
		{
			ret.val(i) = _out[i].value();
			ret.jacobian.row(i) = _out[i].derivatives();
		}

		return ret;
	}

	/*
		Evaluate the function, its jacobian and hessian
	 */
	static EIGEN_STRONG_INLINE auto hessian(
		const param_t& param,
		const Ref<const Matrix<scalar_t, input_sizes, 1>>&... args) 
		noexcept
	{
		outerAD_t _out;

		// Convert to AD variables for the inputs and call our function
		seed_and_call2(make_ad2<input_sizes>(args)..., _out, param);

		// Copy Hessian into output variables
		hessian_return_t ret;
		for(int i=0; i<num_outputs; i++)
		{
			ret.val(i) = _out[i].value().value();
			ret.jacobian.row(i) = _out[i].value().derivatives();
			for (int j = 0; j < num_inputs; j++) {
				ret.hessian[i].template middleRows<1>(j) = _out[i].derivatives()(j).derivatives().transpose();
			}
		}

		return ret;
	}


private:

	/*********
	 Jacobians 
	 *********/

	// Take a vector input and return a AD version of the vector
	template<int n>
	static EIGEN_STRONG_INLINE Matrix<AD_scalar, n, 1> 
		make_ad(const Ref<const Matrix<scalar_t, n, 1>> x)
	{
		Matrix<AD_scalar, n, 1> y;
		y = x;
		for (int i=0; i<y.rows(); i++) {
			y[i].derivatives().setZero();
		}
		return y;
	}

	static EIGEN_STRONG_INLINE void seed_and_call(
			Matrix<AD_scalar, input_sizes, 1>... args,
			Eigen::Ref<Eigen::Matrix<AD_scalar, num_outputs, 1>> out,
			const param_t& param)
	{
		// Set derivative equal to identity
		int offset = 0;
		(void)std::initializer_list<int>{ 
			(
				offset = AD_Seed(args, offset), // Set to unit vectors
				0
			)...
		};

		// Call our function
		Func::template impl<AD_scalar>(param, out, args...);
	}

	// Sets the input derivatives to the identity. 
	// Assumes that the derivative matrix is initially zero
	template <typename vec>
	static constexpr int AD_Seed(vec &x, int offset)
	{
		for (int i=0; i<x.rows(); i++)
			x[i].derivatives().coeffRef(i + offset) = 1;
		return offset + x.rows();
	}


	/********
	 Hessians 
	 ********/

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

	static EIGEN_STRONG_INLINE void seed_and_call2(
		Eigen::Matrix<outerADScalar, input_sizes, 1>... args,
		Eigen::Ref<outerAD_t> out,
		const param_t& param)
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
		Func::template impl<outerADScalar>(param, out, args...);
	}

};

};

#endif // __LAMPC__FUNCTION_HPP