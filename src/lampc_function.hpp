#ifndef __LAMPC__FUNCTION_HPP
#define __LAMPC__FUNCTION_HPP

// Defines differentiable dense functions
#include <functional>

#include "Eigen/Dense"
#include "unsupported/Eigen/AutoDiff"

#include "lampc_utility.hpp"

using namespace Eigen;

namespace lampc {

// template<typename Fn, typename... Args, 
//         std::enable_if_t<std::is_member_pointer<std::decay_t<Fn>>{}, int> = 0 >
// constexpr decltype(auto) my_invoke(Fn&& f, Args&&... args)
//     noexcept(noexcept(std::mem_fn(f)(std::forward<Args>(args)...)))
// {
//     return std::mem_fn(f)(std::forward<Args>(args)...);
// }

// template<typename Fn, typename... Args, 
//          std::enable_if_t<!std::is_member_pointer<std::decay_t<Fn>>{}, int> = 0>
// constexpr decltype(auto) my_invoke(Fn&& f, Args&&... args)
//     noexcept(noexcept(std::forward<Fn>(f)(std::forward<Args>(args)...)))
// {
//     return std::forward<Fn>(f)(std::forward<Args>(args)...);
// }

/**
 * Used to create a differentiable function
 * 
 * Usage:
 *   using F = lampc::Function<double, 2, 2,2>;
 *   auto f = make_function(F, myfunction);
 */
#define make_function(F, name) F(name<F::scalar_t>, name<F::scalar_t, F::AD_scalar>, name<F::scalar_t, F::outerADScalar>);

/**
 * Used to add member functions to a class to make a function differentiable
 * 
 * Usage:
 *   make_differentiable(function_name, output_size, input_sizes...)
 */
#define make_differentiable(name, out_size, ...)\
    using name##_t = lampc::Function<scalar_t, out_size, __VA_ARGS__>;\
    template<typename diff_t=scalar_t, typename... Args>\
    EIGEN_STRONG_INLINE auto name(lampc::Jacobian, const Args&... args) noexcept\
    {\
    	auto self = this;\
    	return name##_t::jacobian([self](auto... args){\
    		return self->template name<typename name##_t::AD_scalar>(args...);\
    	}, args...);\
    }\
    template<typename diff_t=scalar_t, typename... Args>\
    EIGEN_STRONG_INLINE auto name(lampc::Hessian, const Args&... args) noexcept\
    {\
    	auto self = this;\
    	return name##_t::hessian([self](auto... args){\
    		return self->template name<typename name##_t::AD_scalar>(args...);\
    	}, args...);\
    }


// Tags so the user can choose the operator overload
struct Eval{};
struct Jacobian{};
struct Hessian{};






template<typename scalar_t_, int num_outputs_, int... input_sizes>
struct DFunction
{
	using scalar_t = scalar_t_;

	static constexpr int num_inputs = meta::sum_template<input_sizes...>();  // Total number of inputs
	static constexpr int num_input_vars = sizeof...(input_sizes);  // Number of input vector variables
	static constexpr int num_outputs = num_outputs_;

	// First order derivative
	using AD_scalar = Eigen::AutoDiffScalar<Eigen::Matrix<scalar_t, num_inputs, 1>>;
	using AD_output_t = Eigen::Matrix<AD_scalar, num_outputs, 1>;  

	// Second order derivative
	using outerDerivatives = Eigen::Matrix<AD_scalar, num_inputs, 1>;
	using outerADScalar = Eigen::AutoDiffScalar<outerDerivatives>;
	using outerAD_t = Eigen::Matrix<outerADScalar, num_outputs, 1>;  

	using out_t = Eigen::Vector<scalar_t, num_outputs>;
	using jacobian_t = Eigen::Matrix<scalar_t, num_outputs, num_inputs>;

	// Hessian for each of the outputs in an array
	using hessian_single_t = Eigen::Matrix<scalar_t, num_inputs, num_inputs>;
	using hessian_t = std::array<hessian_single_t, num_outputs>;

	/**
	 * Create a lambda function to call the given member function
	 */
	template<typename T, typename C, typename F>
	static auto make_member_eval(C* obj, F f)
	{
	    return [obj,f] (const Eigen::Ref<const Eigen::Vector<T, input_sizes>>&... args) -> Eigen::Vector<T, num_outputs>
	    {
	    	return (obj->*f)(args...);
	    };
	}

	template<typename F, typename... Args>
	static EIGEN_STRONG_INLINE std::pair<out_t, jacobian_t> jacobian(F f, Args&... args) noexcept
		// const Eigen::Ref<const Eigen::Matrix<scalar_t, input_sizes, 1>>... args) noexcept
	{
		// Convert to AD variables for the inputs and call our function
		AD_output_t out = seed_and_call(f, make_ad(args)...);

		// Copy Jacobian into output variables
		out_t val;
		jacobian_t jacobian;
		for(int i=0; i<num_outputs; i++)
		{
			val(i) = out[i].value();
			jacobian.row(i) = out[i].derivatives();
		}

		return std::make_pair(val, jacobian);
	}


	// EIGEN_STRONG_INLINE 
	// std::tuple<out_t, jacobian_t, hessian_t> operator()(const Ref<const Matrix<scalar_t, input_sizes, 1>>&... args, Hessian) noexcept
	// {
	// 	// Convert to AD variables for the inputs and call our function
	// 	outerAD_t out = seed_and_call2(make_ad2<input_sizes>(args)...);

	// 	// Copy Hessian into output variables
	// 	out_t val;
	// 	jacobian_t jacobian;
	// 	hessian_t hessian;
	// 	for(int i=0; i<num_outputs; i++)
	// 	{
	// 		val(i) = out[i].value().value();
	// 		jacobian.row(i) = out[i].value().derivatives();
	// 		for (int j = 0; j < num_inputs; j++) {
	// 			hessian[i].template middleRows<1>(j) = out[i].derivatives()(j).derivatives().transpose();
	// 		}
	// 	}

	// 	return std::make_tuple(val, jacobian, hessian);
	// }


private:

	/*********
	 Jacobians 
	 *********/

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
	template<typename Arg>
	static EIGEN_STRONG_INLINE auto make_ad(const Arg& x)
	{
		Matrix<AD_scalar, Arg::RowsAtCompileTime, 1> y(x.rows());
		y = x;
		for (int i=0; i<y.rows(); i++) {
			y[i].derivatives().setZero();
		}
		return y;
	}


	template<typename F, typename... Args>
	static EIGEN_STRONG_INLINE auto	seed_and_call(F f, Args&... args)
	{
		// Set derivative equal to identity
		int offset = 0;
		(void)std::initializer_list<int>{ 
			(
				offset = AD_Seed(args, offset), // Set to unit vectors
				0
			)...
		};

		return f(args...);
	}


	// /********
	//  Hessians 
	//  ********/

	// // Take a vector input and return a AD version of the vector
	// template<int n>
	// static EIGEN_STRONG_INLINE Eigen::Matrix<outerADScalar, n, 1> 
	// 	make_ad2(const Eigen::Ref<const Eigen::Matrix<scalar_t, n, 1>> x)
	// {
	// 	Eigen::Matrix<outerADScalar, n, 1> y;
	// 	// y = x;
	// 	for (int i=0; i<n; i++) {
	// 		y(i).value().value() = x(i);
	// 		y(i).value().derivatives().setZero();
	// 		y(i).derivatives().setZero();
	// 		for (int j = 0; j < n; j++) {
	// 			y(i).derivatives()(j).derivatives().setZero();
	// 		}
	// 	}
	// 	return y;
	// }

	// // Sets the input derivatives to the identity. 
	// // Assumes that the derivative matrix is initially zero
	// template <typename vec>
	// static constexpr int AD_Seed2(vec &x, int offset)
	// {
	// 	for (int i=0; i<x.rows(); i++)
	// 	{
	// 		x(i).value().derivatives().coeffRef(i + offset) = 1;
	// 		x(i).derivatives().coeffRef(i + offset) = 1;
	// 	}

	// 	return offset + x.rows();
	// }

	// EIGEN_STRONG_INLINE outerAD_t seed_and_call2(Eigen::Matrix<outerADScalar, input_sizes, 1>... args)
	// {
	// 	// Set derivative equal to identity
	// 	int offset = 0;
	// 	(void)std::initializer_list<int>{ 
	// 		(
	// 			offset = AD_Seed2(args, offset), // Set to unit vectors
	// 			0
	// 		)...
	// 	};

	// 	// Call our function
	// 	return ddfunc(args...);
	// }
};





















template<typename scalar_t_, int num_outputs_, int... input_sizes>
struct Function
{
	using scalar_t = scalar_t_;

	static constexpr int num_inputs = meta::sum_template<input_sizes...>();  // Total number of inputs
	static constexpr int num_input_vars = sizeof...(input_sizes);  // Number of input vector variables
	static constexpr int num_outputs = num_outputs_;

	// First order derivative
	using AD_scalar = Eigen::AutoDiffScalar<Eigen::Matrix<scalar_t, num_inputs, 1>>;
	using AD_output_t = Eigen::Matrix<AD_scalar, num_outputs, 1>;  

	// Second order derivative
	using outerDerivatives = Eigen::Matrix<AD_scalar, num_inputs, 1>;
	using outerADScalar = Eigen::AutoDiffScalar<outerDerivatives>;
	using outerAD_t = Eigen::Matrix<outerADScalar, num_outputs, 1>;  

	using out_t = Eigen::Vector<scalar_t, num_outputs>;
	using jacobian_t = Eigen::Matrix<scalar_t, num_outputs, num_inputs>;

	// Hessian for each of the outputs in an array
	using hessian_single_t = Eigen::Matrix<scalar_t, num_inputs, num_inputs>;
	using hessian_t = std::array<hessian_single_t, num_outputs>;

	// Function types for scalar and autodiff types
	using Func = Eigen::Vector<scalar_t, num_outputs>(const Eigen::Ref<const Eigen::Vector<scalar_t, input_sizes>>&...);
	using DFunc = Eigen::Vector<AD_scalar, num_outputs>(const Eigen::Ref<const Eigen::Vector<AD_scalar, input_sizes>>&...);
	using DDFunc = Eigen::Vector<outerADScalar, num_outputs>(const Eigen::Ref<const Eigen::Vector<outerADScalar, input_sizes>>&...);

	// Three function references to evaluate and first and second derivative overloads
	Func &func;
	DFunc &dfunc;
	DDFunc &ddfunc;

	Function(Func& func, DFunc& dfunc, DDFunc& ddfunc)
		: func(func), dfunc(dfunc), ddfunc(ddfunc)
		{}

	// Function(Func& func, DFunc& dfunc)
	// 	: func(func), dfunc(dfunc), ddfunc(nullptr)
	// 	{}

	// /**
	//  * Create a lambda function to call the given member function
	//  */
	// template<typename T, typename C, typename F>
	// static auto make_member_eval(C* obj, F f)
	// {
	//     return [obj,f] (const Eigen::Ref<const Eigen::Vector<T, input_sizes>>&... args) -> Eigen::Vector<T, num_outputs>
	//     {
	//     	return (obj->*f)(args...);
	//     };
	// }

// 	static std::vector<int> get_input_sizes()
// 	{
// 		return std::vector<int>{input_sizes...};
// 	}

// 	/**
// 	 * Return the number of nonzeros in the jacobian
// 	 * 
// 	 * Default implementation assumes dense jacobians.
// 	 * Oveload in child class for sparse.
// 	 */
// 	static constexpr int nnzJacobian()
// 	{
// 		int nnz = 0;
// 		auto l = {(
// 			nnz += input_sizes * num_outputs_,
// 			0
// 			)...};
// 		return nnz;
// 	}

// 	/**
// 	 * Returns the sparsity structure of the jacobian of this function
// 	 * 
// 	 * S = [J_var1 J_var2 ...]
// 	 */
// 	static Eigen::SparseMatrix<int> jacobianStructure()
// 	{
// 		// Default is just a dense matrix
// 		Eigen::MatrixX<int> S(num_outputs, num_inputs);
// 		S.array() = 1;

// 		return S.sparseView();
// 	}

// 	/**
// 	 * Returns the sparsity structure of the hessian of the i'th output of this function
// 	 */
// 	static Eigen::SparseMatrix<int> hessianStructure(int output_index)
// 	{
// 		// Default is just a dense matrix
// 		Eigen::MatrixX<int> H(num_inputs, num_inputs);
// 		H.array() = 1;
// 		return H.sparseView();
// 	}

	EIGEN_STRONG_INLINE 
	out_t operator()(const Eigen::Ref<const Eigen::Matrix<scalar_t, input_sizes, 1>>... args) noexcept
	{
		return func(args...);
	}

	EIGEN_STRONG_INLINE 
	out_t operator()(const Eigen::Ref<const Eigen::Matrix<scalar_t, input_sizes, 1>>... args, Eval) noexcept
	{
		return func(args...);
	}

	EIGEN_STRONG_INLINE 
	std::pair<out_t, jacobian_t> operator()(const Eigen::Ref<const Eigen::Matrix<scalar_t, input_sizes, 1>>... args, Jacobian) noexcept
	{
		// Convert to AD variables for the inputs and call our function
		AD_output_t out = seed_and_call(make_ad<input_sizes>(args)...);

		// Copy Jacobian into output variables
		out_t val;
		jacobian_t jacobian;
		for(int i=0; i<num_outputs; i++)
		{
			val(i) = out[i].value();
			jacobian.row(i) = out[i].derivatives();
		}

		return std::make_pair(val, jacobian);
	}

	template<typename F>
	static 
	EIGEN_STRONG_INLINE 
	std::pair<out_t, jacobian_t> jacobian(F f, 
		const Eigen::Ref<const Eigen::Matrix<scalar_t, input_sizes, 1>>... args) noexcept
	{
		// Convert to AD variables for the inputs and call our function
		AD_output_t out = seed_and_call_obj(f, make_ad<input_sizes>(args)...);

		// Copy Jacobian into output variables
		out_t val;
		jacobian_t jacobian;
		for(int i=0; i<num_outputs; i++)
		{
			val(i) = out[i].value();
			jacobian.row(i) = out[i].derivatives();
		}

		return std::make_pair(val, jacobian);
	}


	EIGEN_STRONG_INLINE 
	std::tuple<out_t, jacobian_t, hessian_t> operator()(const Ref<const Matrix<scalar_t, input_sizes, 1>>&... args, Hessian) noexcept
	{
		// Convert to AD variables for the inputs and call our function
		outerAD_t out = seed_and_call2(make_ad2<input_sizes>(args)...);

		// Copy Hessian into output variables
		out_t val;
		jacobian_t jacobian;
		hessian_t hessian;
		for(int i=0; i<num_outputs; i++)
		{
			val(i) = out[i].value().value();
			jacobian.row(i) = out[i].value().derivatives();
			for (int j = 0; j < num_inputs; j++) {
				hessian[i].template middleRows<1>(j) = out[i].derivatives()(j).derivatives().transpose();
			}
		}

		return std::make_tuple(val, jacobian, hessian);
	}


private:

	/*********
	 Jacobians 
	 *********/

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
	static 
	EIGEN_STRONG_INLINE 
	Matrix<AD_scalar, n, 1> 
	make_ad(const Ref<const Matrix<scalar_t, n, 1>> x)
	{
		Matrix<AD_scalar, n, 1> y;
		y = x;
		for (int i=0; i<y.rows(); i++) {
			y[i].derivatives().setZero();
		}
		return y;
	}

	EIGEN_STRONG_INLINE 
	Eigen::Matrix<AD_scalar, num_outputs, 1>
	seed_and_call(Matrix<AD_scalar, input_sizes, 1>... args)
	{
		// Set derivative equal to identity
		int offset = 0;
		(void)std::initializer_list<int>{ 
			(
				offset = AD_Seed(args, offset), // Set to unit vectors
				0
			)...
		};

		return dfunc(args...);  // Call our derivative function
	}


	template<typename F>
	static 
	EIGEN_STRONG_INLINE 
	Eigen::Matrix<AD_scalar, num_outputs, 1>
	seed_and_call_obj(F f, Matrix<AD_scalar, input_sizes, 1>... args)
	{
		// Set derivative equal to identity
		int offset = 0;
		(void)std::initializer_list<int>{ 
			(
				offset = AD_Seed(args, offset), // Set to unit vectors
				0
			)...
		};

		return f(args...);
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
		return ddfunc(args...);
	}
};

// Definitions for static members
// template<typename Func_, typename scalar_t_, typename param_t_, int num_outputs_, int... input_sizes>
// constexpr const char* Jacobian<Func_, scalar_t_, param_t_, num_outputs_, input_sizes...>::name;
template<typename scalar_t, int num_outputs_, int... input_sizes>
constexpr int Function<scalar_t,num_outputs_, input_sizes...>::num_inputs;
template<typename scalar_t, int num_outputs_, int... input_sizes>
constexpr int Function<scalar_t,num_outputs_, input_sizes...>::num_input_vars;
template<typename scalar_t, int num_outputs_, int... input_sizes>
constexpr int Function<scalar_t,num_outputs_, input_sizes...>::num_outputs;


};

#endif // __LAMPC__FUNCTION_HPP