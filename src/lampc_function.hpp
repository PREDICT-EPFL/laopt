#ifndef __LAMPC__FUNCTION_HPP
#define __LAMPC__FUNCTION_HPP

// Defines differentiable dense functions
#include <functional>
#include <algorithm>

#include "Eigen/Dense"
#include "unsupported/Eigen/AutoDiff"

#include "lampc_utility.hpp"
#include "bsmatrix.hpp"

using namespace Eigen;

namespace lampc {



/**
 * The output from a function call.
 */
template<typename Func>
struct Call
{
	using out_t = typename Func::out_t;
	using out_r = typename Eigen::Ref<out_t>;
	using scalar_t = typename Func::scalar_t;

	Eigen::Matrix<typename Func::scalar_t, Func::num_outputs, 2> bounds;
	typename Func::out_t value;

	Call<Func> operator<=(name<F> call, typename name<F>::out_r ub)  { call.bounds(all, last) = ub; return call; }
	template<typename F, typename scalar_t> name<F> operator<=(name<F> call, scalar_t ub)                 { call.bounds(all, last).array() = ub; return call; }
	template<typename F>        name<F> operator>=(name<F> call, Eigen::Ref<Eigen::VectorX<typename F::scalar_t>> lb)  { std::cout << "operator" << std::endl; call.bounds(all, 0) = lb;            return call; }
	// template<typename F, typename scalar_t> name<F> operator>=(name<F> call, scalar_t lb)                 {
	// std::cout << "operator" << std::endl;
	// std::cout << "type(lb) = " << type_name<decltype(lb)>() << std::endl;
	// std::cout << "type(out_t) = " << type_name<typename F::out_t>() << std::endl;
	// call.bounds(all, 0).array() = lb;            return call; }
	template<typename F, typename scalar_t> name<F> operator==(name<F> call, scalar_t val)                { val <= call <= val; std::cout << "HERE" << std::endl; return call; }
	template<typename F, typename scalar_t> name<F> operator==(scalar_t val, name<F> call)                { val <= call <= val; return call; }
	template<typename F, typename scalar_t> name<F> operator<=(scalar_t lb, name<F> call)                 { return call >= lb; }
	template<typename F, typename scalar_t> name<F> operator>=(scalar_t ub, name<F> call)                 { return call <= ub; } 
};


template<typename Func>
struct JacobianCall : public Call<Func>
{
	typename Func::jacobian_t jacobian;
};

template<typename Func>
struct HessianCall : public JacobianCall<Func>
{
	typename Func::hessian_t hessian;
};


// Information about the inputs to the call
struct InputInfo
{
	std::vector<Segment> inputs;
	InputInfo(std::initializer_list<Segment> in) : inputs(in) {}
};

template<typename Func>
struct CallTape : public Call<Func>, public InputInfo
{
	CallTape(std::initializer_list<Segment> inputs) : InputInfo(inputs) {}
};

template<typename Func>
struct JacobianTapeCall : public JacobianCall<Func>, public InputInfo
{
	JacobianTapeCall(std::initializer_list<Segment> inputs) : InputInfo(inputs) {}
};

template<typename Func>
struct HessianTapeCall : public HessianCall<Func>, public InputInfo
{
	HessianTapeCall(std::initializer_list<Segment> inputs) : InputInfo(inputs) {}
};


#define make_call_comparison(name) \
template<typename F>                    name<F> operator<=(name<F> call, typename name<F>::out_r ub)  { call.bounds(all, last) = ub; return call; } \
template<typename F, typename scalar_t> name<F> operator<=(name<F> call, scalar_t ub)                 { call.bounds(all, last).array() = ub; return call; } \
template<typename F>        name<F> operator>=(name<F> call, Eigen::Ref<Eigen::VectorX<typename F::scalar_t>> lb)  { std::cout << "operator" << std::endl; call.bounds(all, 0) = lb;            return call; } \
// template<typename F, typename scalar_t> name<F> operator>=(name<F> call, scalar_t lb)                 { \
// std::cout << "operator" << std::endl; \
// std::cout << "type(lb) = " << type_name<decltype(lb)>() << std::endl; \
// std::cout << "type(out_t) = " << type_name<typename F::out_t>() << std::endl; \
// call.bounds(all, 0).array() = lb;            return call; } \
template<typename F, typename scalar_t> name<F> operator==(name<F> call, scalar_t val)                { val <= call <= val; std::cout << "HERE" << std::endl; return call; } \
template<typename F, typename scalar_t> name<F> operator==(scalar_t val, name<F> call)                { val <= call <= val; return call; } \
template<typename F, typename scalar_t> name<F> operator<=(scalar_t lb, name<F> call)                 { return call >= lb; } \
template<typename F, typename scalar_t> name<F> operator>=(scalar_t ub, name<F> call)                 { return call <= ub; } 

make_call_comparison(Call)
make_call_comparison(JacobianCall)
make_call_comparison(HessianCall)

make_call_comparison(CallTape)
make_call_comparison(JacobianTapeCall)
make_call_comparison(HessianTapeCall)



/**
 * Used to add member functions to a class to make a function differentiable
 * 
 * Usage:
 *   make_differentiable(function_name, output_size, input_sizes...)
 */



#define make_eval(name, out_size, ...) \
	using name##_t = lampc::DFunction<scalar_t, out_size, __VA_ARGS__>; \
	/* Tape version of eval */\
	template<typename... V>\
	EIGEN_STRONG_INLINE auto name(lampc::Eval, const std::pair<lampc::Segment, Eigen::Ref<V>>... args) noexcept \
	{\
		auto self = this; \
		lampc::CallTape<name##_t> ret{args.first...}; \
	  name##_t::eval(ret, \
	  	[self](auto... args){return self->template name<typename name##_t::scalar_t>(args...);}, \
	  	args.second...); \
	  return ret; \
	}\
	/* Standard version of eval */\
	template<typename... V>\
	EIGEN_STRONG_INLINE auto name(lampc::Eval, Eigen::Ref<V>... args) noexcept \
	{\
		auto self = this; \
		lampc::Call<name##_t> ret; \
	  name##_t::eval(ret, \
	  	[self](auto... args){return self->template name<typename name##_t::scalar_t>(args...);}, \
	  	args...); \
		return ret; \
	}

		// ret.value = self->template name(args.second...); \
		// ret.value = self->template name(args...); \


#define make_jacobian(name, out_size, ...)\
	make_eval(name, out_size, __VA_ARGS__); \
	/* Tape version of the jacobian */\
	template<typename... Args>\
	EIGEN_STRONG_INLINE auto name(lampc::Jacobian, const std::pair<lampc::Segment, Args>... args) noexcept\
	{\
		auto self = this; \
		lampc::JacobianTapeCall<name##_t> ret{args.first...}; \
	  name##_t::jacobian(ret, \
	  	[self](auto... args){return self->template name<typename name##_t::AD_scalar>(args...);}, \
	  	args.second...); \
	  return ret; \
	}\
	/* Standard version of the jacobian */\
	template<typename... Args>\
	EIGEN_STRONG_INLINE auto name(lampc::Jacobian, const Args... args) noexcept\
	{\
		auto self = this;\
		lampc::JacobianCall<name##_t> ret; \
		name##_t::jacobian(ret, \
			[self](auto... args){return self->template name<typename name##_t::AD_scalar>(args...);}, \
			args...);\
		return ret; \
	}


#define make_hessian(name, out_size, ...)\
	make_jacobian(name, out_size, __VA_ARGS__);\
	template<typename... Args>\
	EIGEN_STRONG_INLINE auto name(lampc::Hessian, const std::pair<lampc::Segment, Args>... args) noexcept\
	{\
		auto self = this; \
		lampc::HessianTapeCall<name##_t> ret{args.first...}; \
	  name##_t::hessian(ret, \
	  	[self](auto... args){return self->template name<typename name##_t::outerADScalar>(args...);}, \
	  	args.second...); \
	  return ret; \
	}\
	template<typename... Args>\
	EIGEN_STRONG_INLINE auto name(lampc::Hessian, const Args... args) noexcept\
	{\
		auto self = this; \
		lampc::HessianCall<name##_t> ret; \
		name##_t::hessian(ret, \
			[self](auto... args){return self->template name<typename name##_t::outerADScalar>(args...);}, \
			args...); \
		return ret; \
	}

  // template<typename diff_t=scalar_t, typename... Args>\
  // EIGEN_STRONG_INLINE auto name(lampc::Hessian, const Args&... args) noexcept\
  // {\
  // 	auto self = this;\
  // 	return name##_t::hessian([self](auto... args){\
  // 		return self->template name<typename name##_t::AD_scalar>(args...);\
  // 	}, args...);\
  // }


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

	using out_t = Eigen::Vector<scalar_t, num_outputs>;
	using jacobian_t = Eigen::Matrix<scalar_t, num_outputs, num_inputs>;

	// Hessian for each of the outputs in an array
	using hessian_single_t = Eigen::Matrix<scalar_t, num_inputs, num_inputs>;
	using hessian_t = std::array<hessian_single_t, num_outputs>;

	// First order derivative
	using AD_scalar = Eigen::AutoDiffScalar<Eigen::Matrix<scalar_t, num_inputs, 1>>;
	using AD_output_t = Eigen::Matrix<AD_scalar, num_outputs, 1>;  

	// Second order derivative
	using outerDerivatives = Eigen::Matrix<AD_scalar, num_inputs, 1>;
	using outerADScalar = Eigen::AutoDiffScalar<outerDerivatives>;
	using outerAD_t = Eigen::Matrix<outerADScalar, num_outputs, 1>;  


	template<typename ret_t, // Return type (must be derived from Call)
					 typename F>     // Function to be called
	static EIGEN_STRONG_INLINE void
	eval(ret_t& ret, F f, 
		   const Eigen::Ref<const Eigen::Matrix<scalar_t, input_sizes, 1>>&... args) noexcept
	{
		ret.value = f(make_copy(args)...);
	}

	template<typename ret_t, // Return type (must be derived from JacobianCall)
					 typename F>     // Function to be called
	static EIGEN_STRONG_INLINE void
	jacobian(ret_t& ret, F f, 
		const Eigen::Ref<const Eigen::Matrix<scalar_t, input_sizes, 1>>&... args) noexcept
	{
		// Convert to AD variables for the inputs and call our function
		AD_output_t out = seed_and_call(f, make_ad(args)...);

		// Copy Jacobian into output variables
		for(int i=0; i<num_outputs; i++)
		{
			ret.value(i) = out[i].value();
			ret.jacobian.row(i) = out[i].derivatives();
		}
	}

	template<typename ret_t, // Return type (must be derived from HessianCall)
					 typename F>     // Function to be called
	static EIGEN_STRONG_INLINE void
	hessian(ret_t& ret, F f, 
		const Eigen::Ref<const Eigen::Matrix<scalar_t, input_sizes, 1>>&... args) noexcept
	{
		// Convert to AD variables for the inputs and call our function
		outerAD_t out = seed_and_call2(f, make_ad2<input_sizes>(args)...);

		// Copy Hessian into output variables
		for(int i=0; i<num_outputs; i++)
		{
			ret.value(i) = out[i].value().value();
			ret.jacobian.row(i) = out[i].value().derivatives();
			for (int j = 0; j < num_inputs; j++) {
				ret.hessian[i].template middleRows<1>(j) = out[i].derivatives()(j).derivatives().transpose();
			}
		}
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

	/********
	 * Eval
	 ********/

	// Take a vector input and return a fixed-sized version of it (copies)
	template<int n>
	static EIGEN_STRONG_INLINE Eigen::Vector<scalar_t, n> 
	make_copy(const Ref<const Matrix<scalar_t, n, 1>> x)
	{
		Eigen::Vector<scalar_t, n> y;
		y = x;
		return y;
	}



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
	static EIGEN_STRONG_INLINE 
	Matrix<AD_scalar, n, 1> 
	make_ad(const Ref<const Matrix<scalar_t, n, 1>> x)
	// static EIGEN_STRONG_INLINE auto make_ad(const Arg& x)
	{
		Matrix<AD_scalar, n, 1> y;
		y = x;
		for (int i=0; i<y.rows(); i++) {
			y[i].derivatives().setZero();
		}
		return y;
	}


	template<typename F>
	static EIGEN_STRONG_INLINE Eigen::Matrix<AD_scalar, num_outputs, 1>
	seed_and_call(F f, Eigen::Matrix<AD_scalar, input_sizes, 1>... args)
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

	template<typename F>
	static EIGEN_STRONG_INLINE outerAD_t seed_and_call2(F f, Eigen::Matrix<outerADScalar, input_sizes, 1>... args)
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
		return f(args...);
	}
};




















/**
 * Used to create a differentiable function
 * 
 * Usage:
 *   using F = lampc::Function<double, 2, 2,2>;
 *   auto f = make_function(F, myfunction);
 */
#define make_function(F, name) F(name<F::scalar_t>, name<F::scalar_t, F::AD_scalar>, name<F::scalar_t, F::outerADScalar>);


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