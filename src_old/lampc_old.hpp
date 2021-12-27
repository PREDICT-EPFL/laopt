#ifndef __LAMPC__HPP
#define __LAMPC__HPP

#include <iostream>
#include <tuple>
#include <array>

#include "map.hpp"

#include "Eigen/Dense"
#include <Eigen/Sparse>
#include "unsupported/Eigen/AutoDiff"

#include "type_name.hpp"

#include "utils/helpers.hpp"

using namespace Eigen;

namespace lampc {

/*************************************************************
	Meta-programming helper functions
 *************************************************************/

// Sum the inputs to get total number of inputs
template<int... S>
constexpr int sum_template() {
	int result = 0;
	for(auto s : { S... }) result += s;
	return result;
}

// Build an array at compile time
template <typename T, typename... Args>
constexpr std::array<T, sizeof...(Args)> make_array(Args... args)
{
	return {args...};
}

// Helper function to reshape an eigen matrix while maintaining const'ness
template <typename map_to, typename scalar_t>
constexpr auto _map_matrix(const scalar_t* x) {
	return Eigen::Map<const map_to>(x);
}

template <typename map_to, typename scalar_t>
constexpr auto _map_matrix(scalar_t* x) {
	return Eigen::Map<map_to>(x);
}


/*************************************************************
	 Short names to pass constant vectors and writable vectors
 *************************************************************/
template<typename T, std::size_t n>
using cVec = const Eigen::Ref<const Eigen::Matrix<T, n, 1>>;

template<typename T, std::size_t n>
using Vec = Eigen::Ref<Eigen::Matrix<T, n, 1>>;


/*************************************************************
	 Jacobian computation
 *************************************************************/

// template<typename func,
//       typename scalar_t,
//       typename param_t, 
//       int num_outputs_, 
//       int... input_sizes>
// struct FunctionTraits
// {
//  static const auto num_outputs = num_outputs_;
// };

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

#define FUNCTION(func_name, out_pair, ...) \
	struct func_name##_ \
	{ \
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


template<typename Func, typename scalar_t, typename param_t, int num_outputs_, int... input_sizes>
struct Jacobian // < FunctionTraits<Func, scalar_t, param_t, num_outputs, input_sizes...> >
{
	static constexpr int num_inputs = sum_template<input_sizes...>();  // Total number of inputs
	static constexpr int num_input_vars = sizeof...(input_sizes);  // Number of input vector variables
	static constexpr int num_outputs = num_outputs_;

	// First order derivative
	using AD_scalar = Eigen::AutoDiffScalar<Eigen::Matrix<scalar_t, num_inputs, 1>>;
	using AD_output_t = Eigen::Matrix<AD_scalar, num_outputs, 1>;  

	// Second order derivative
	using outerDerivatives = Eigen::Matrix<AD_scalar, num_inputs, 1>;
	using outerADScalar = Eigen::AutoDiffScalar<outerDerivatives>;
	using outerAD_t = Eigen::Matrix<outerADScalar, num_outputs, 1>;  

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
		jacobian_return_t<scalar_t, num_outputs, num_inputs> ret;
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
		hessian_return_t<scalar_t, num_outputs, num_inputs> ret;
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


/*************************************************************
	 Variables
 *************************************************************/

// User macro to define a list of variables and their order
#define DECL_VAR(ind, name, type) \
	really_unparen type
#define DECL_VAR_PAIR(pair) \
	DECL_VAR pair
#define GET_VAR(ind, name, type) \
	using name = typename variables::template get_by_index<ind>;
#define GET_VAR_PAIR(pair) \
	GET_VAR pair
#define Make_Variables(scalar_t, param_t, ...) \
	typename make_variables<scalar_t, param_t, \
	MAP_LIST(DECL_VAR_PAIR, __VA_ARGS__) \
	>::type; \
	MAP(GET_VAR_PAIR, __VA_ARGS__) \


template<typename param_t, int num_variables, typename scalar_t, 
		 typename BndFunc, 
		 std::size_t _len, std::size_t _num_vars,
		 int offset_>  // Offset into the main variable
		 // typename Prev = ZeroVariable_> // Variable defined before this one (specifies ordering)
struct var_t_
{
	static constexpr std::size_t len      = _len;      // Length of a variable of this type
	static constexpr std::size_t num_vars = _num_vars; // Number of variables in this set

	static constexpr std::size_t offset = offset_;

	using variable_t = typename dense_matrix_type_selector<scalar_t, num_variables, 1>::type;
	using vec_t = typename dense_matrix_type_selector<scalar_t, len, 1>::type;
	using mat_t = typename dense_matrix_type_selector<scalar_t, len, num_vars>::type;

	// Location of this variable in the NLP optimizer
	// var = [x1; x2; x3; ...]
	// var[offset] = this variable
	//
	// var{0} = x1, var{1} = x2, var{2} = x3 ...
	// var{index} = this variable

	// static constexpr std::size_t offset    = Prev::next;               // Offset into the main variable
	// static constexpr std::size_t next      = offset + len * num_vars;  // Offset where next variable should go
	// static constexpr std::size_t var_index = Prev::var_index + Prev::num_vars; // Variable index

	// Get indexed offset
	static constexpr std::size_t o(int ind = 0) 
	{
		assert(ind < num_vars);
		return offset + ind * len;
	}

	// Static programatic interface !!!!!!!!

	// Return ind variable segment of var
	static EIGEN_STRONG_INLINE constexpr Eigen::Ref<vec_t> get(
		Eigen::Ref<variable_t> var, const int ind = 0)
	{
		assert(ind < num_vars);
		assert(var.rows() == num_variables && var.cols() == 1);
		return var.template segment<len>(offset + ind * len);
	}

	// static EIGEN_STRONG_INLINE constexpr const Eigen::Ref<const vec_t> get(
	// 	const Eigen::Ref<const variable_t> var, const int ind = 0)
	// {
	// 	assert(var.rows() == num_variables && var.cols() == 1);
	// 	return var.template segment<len>(offset + ind * len);
	// }

	/*
		Get the upper and lower bounds for this variable
	 */
	static EIGEN_STRONG_INLINE void get_bounds(const param_t& param, 
		Eigen::Ref<variable_t> lb, Eigen::Ref<variable_t> ub) noexcept
	{
		assert(lb.rows() == num_variables && lb.cols() == 1);
		assert(ub.rows() == num_variables && ub.cols() == 1);

		for(int i=0; i<num_vars; i++)
			BndFunc::eval(param, i, get(lb, i), get(ub, i));
	}
};


/** User convenience interface to get variables
 */
template<typename var_t>
struct user_var_t
{
	using variable_t = typename var_t::variable_t;
	using vec_t = typename var_t::vec_t;

	// Keep pointer to a base variable vector to reference against
	Eigen::Ref<variable_t> ref_var;
	user_var_t(Eigen::Ref<variable_t> _var) : ref_var(_var) {};

	// // Return ind variable segment of var
	// EIGEN_STRONG_INLINE Eigen::Ref<vec_t> 
	// 	operator()(Eigen::Ref<variable_t> var, const int ind) const
	// {
	// 	assert(ind < num_vars);
	// 	assert(var.rows() == num_variables && var.cols() == 1);
	// 	return var.template segment<len>(offset + ind * len);
	// }

	// EIGEN_STRONG_INLINE Eigen::Ref<vec_t> 
	// 	operator()(Eigen::Ref<variable_t> var) const
	// {
	// 	assert(var.rows() == num_variables && var.cols() == 1);
	// 	return _map_matrix<Eigen::Matrix<scalar_t, len, num_vars>>(var.template segment<len * num_vars>(offset).data());
	// }

	// Return ind variable segment of reference variable
	EIGEN_STRONG_INLINE Eigen::Ref<vec_t> 
		operator()(const int ind = 0) const
	{
		return var_t::get(ref_var, ind);
	}
};


// List of variables
template<typename scalar_t, typename param_t, std::size_t num_variables_, typename var_tuple, typename Index>
struct VariableList_t;

template<typename scalar_t, typename param_t, std::size_t num_variables_, typename var_tuple, std::size_t... ind>
struct VariableList_t<scalar_t, param_t, num_variables_, var_tuple, std::integer_sequence<std::size_t, ind...>>
{
	// static constexpr std::size_t num_variables = sum_template<Vars::len * Vars::num_vars...>();
	static constexpr std::size_t num_variables = num_variables_;
	using variable_t = typename dense_matrix_type_selector<scalar_t, num_variables, 1>::type;

	template<int i>
	static constexpr auto get()
	{
		return var_tuple::template get<i>();
	}

	static EIGEN_STRONG_INLINE void get_bounds(const param_t& param, 
				Eigen::Ref<variable_t> lb, Eigen::Ref<variable_t> ub) noexcept
	{
		assert(lb.rows() == num_variables && lb.cols() == 1);
		assert(ub.rows() == num_variables && ub.cols() == 1);

		(void)std::initializer_list<int>{ 
			(
				std::tuple_element<ind, var_tuple>::type::get_bounds(param, lb, ub),
				0
			)...
		};
	}

	template<int i>
	using get_by_index = typename std::remove_reference<decltype(std::get<i>(var_tuple()))>::type;

	// Represents an iterator over a variable set
	template<typename variable_set, int start=0, int step=0>
	struct itr
	{
		static constexpr std::size_t var_len = variable_set::len;
		using vec_t = typename variable_set::vec_t;

		// Return index variable segment of var
		static EIGEN_STRONG_INLINE constexpr Eigen::Ref<vec_t> get(Eigen::Ref<variable_t> var, int index = 0)
		{
			return variable_set::get(var, start + index * step);
		}

		static EIGEN_STRONG_INLINE constexpr std::size_t offset(int index = 0)
		{
			return variable_set::o(start + index * step);
		}
	};
};



// User-interface type used to construct variable list
template<typename BndFunc_, std::size_t _len, std::size_t _num_vars>
struct var_t
{
	using BndFunc = BndFunc_;
	static constexpr std::size_t len = _len;
	static constexpr std::size_t num_vars = _num_vars;
};


// Returns a VariableList_t type
template<typename scalar_t, typename param_t,
		 typename... Vars> // List of var_t
struct make_variables
{

	static constexpr std::size_t num_variables = sum_template<Vars::len * Vars::num_vars...>();
	using variable_t = Eigen::Matrix<scalar_t, num_variables, 1>;

	static constexpr std::array<std::size_t, sizeof...(Vars)+1> cumulative_sum(std::size_t seed = 0) 
	{ 
		return{ {0, seed += Vars::len * Vars::num_vars ... } };
	}

	static constexpr std::array<std::size_t, sizeof...(Vars) + 1> a = cumulative_sum();

	// Convert array into a tuple
	template<typename Array, std::size_t... I>
	static constexpr auto a2t_impl(const Array& a, std::index_sequence<I...>)
	{
		return std::make_tuple(a[I]...);
	}

	template<typename T, std::size_t N, typename Indices = std::make_index_sequence<N - 1>>
	static constexpr auto a2t(const std::array<T, N>& a)
	{
		return a2t_impl(a, Indices{});
	}

	static constexpr auto offsets = a2t(cumulative_sum());

	// Helper to convert var_t to variable_t
	template<std::size_t... ind>
	static constexpr auto make_variable_tuple(std::index_sequence<ind...>)
	{
		return std::make_tuple(
					var_t_<param_t, num_variables, scalar_t, 
							 typename Vars::BndFunc,
							 Vars::len,
							 Vars::num_vars,
							 std::get<ind>(offsets)>()
				...);
	}

	using index = std::make_integer_sequence<std::size_t, sizeof...(Vars)>;
	static constexpr auto variable_tup = make_variable_tuple(index{});

	template<std::size_t... ind>
	static void show_offsets_impl(std::index_sequence<ind...>)
	{
		std::cout << "num_variables = " << num_variables << std::endl;
		(void)std::initializer_list<int>{ 
			(
				std::cout << "offset = " << std::get<ind>(offsets),
				std::cout << " len = " << Vars::len,
				std::cout << " num_vars = " << Vars::num_vars,
				std::cout << std::endl,
				0
			)...
		};
	}

	static void show_offsets()
	{
		show_offsets_impl(index{});
	}


	using type = VariableList_t<scalar_t, param_t, num_variables, decltype(variable_tup), index>;
};


/*************************************************************
	 Constraints
 *************************************************************/

/*************************************************************
	A single constraint, or collection of constraints of the same type.
	A collection is a constraint that's calling the same function 
	with the same variable sets, but with diffent indices
 *************************************************************/
template<typename, typename, int, typename...>
struct con_t;

template<typename BndFunc,   // Function returning the upper and lower bounds
		 typename Func,      // The function to be called
		 typename scalar_t_,
		 typename param_t_, 
		 int num_outputs,    // Output size of the function
		 int... input_sizes, // Input sizes of the function
		 int num_iterations, // Number of iterations in the constraint collection
		 typename... Var_t>  // Variable Index's (i.e., range objects)
struct con_t<BndFunc, Jacobian<Func, scalar_t_, param_t_, num_outputs, input_sizes...>, num_iterations, Var_t...>
{
	using param_t = param_t_;
	using scalar_t = scalar_t_;
	using jacFunc = Jacobian<Func, scalar_t, param_t, num_outputs, input_sizes...>;

	// Total vector length required to store this constraint collection
	static constexpr std::size_t constraint_vector_length = num_iterations * num_outputs;

	// Index into the constraint vector where this constraint starts
	int constraint_offset = 0; 

	// Index into a compressed sparse jacobian where each (constraint, variable) block starts
	Eigen::Matrix<std::size_t, num_iterations, sizeof...(input_sizes)> jacobian_offsets;

	// Index into a compressed sparse hessian where each (variable, variable) block starts
	using hessian_offset_t = Eigen::Matrix<std::size_t, sizeof...(input_sizes), sizeof...(input_sizes)>;
	std::array<hessian_offset_t, num_iterations> hessian_offsets;

	// Compute the number of non-zeros in the sparse jacobian
	static constexpr std::size_t nnz_jacobian = jacFunc::num_inputs * num_iterations * num_outputs;

	con_t()
	{}


	/*
		Get the upper and lower bounds for this constraint
	 */
	template<typename constraint_t>
	EIGEN_STRONG_INLINE void get_bounds(const param_t& param, constraint_t& lb, constraint_t& ub) const noexcept
	{
		for(int i=0; i<num_iterations; i++)
		{
			BndFunc::eval(
				param, 
				i,
				lb.template segment<num_outputs>(offset(i)),
				ub.template segment<num_outputs>(offset(i)));
		}
	}


private:
	/*
		Functions to copy the constraints, jacobian and hessian into the global
		variables in dense and sparse formats
	 */
	template<typename ret_t, typename constraint_t>
	EIGEN_STRONG_INLINE void copy_to_constraint(int iteration, ret_t& ret, constraint_t& con) const noexcept
	{
		con.template segment<num_outputs>(offset(iteration)) = ret.val;
	}

	template<typename ret_t, typename jacobian_t>
	EIGEN_STRONG_INLINE void copy_to_jacobian_dense(int iteration, ret_t& ret, jacobian_t& jac) const noexcept
	{
		// Write the jacobian into the dense jacobian matrix
		int var_offset = 0;
		(void)std::initializer_list<int>{ 
			(
				jac.template block<num_outputs, input_sizes>(offset(iteration), Var_t::offset(iteration))
					= ret.jacobian.template block<num_outputs, input_sizes>(0, var_offset),
				var_offset += input_sizes,
				0
			)...
		};
	}

	template<typename ret_t>
	EIGEN_STRONG_INLINE void copy_to_jacobian_sparse(int iteration, ret_t& ret, Eigen::Ref<Eigen::SparseMatrix<scalar_t>> jac) const noexcept
	{
		// Write the jacobian into the sparse jacobian matrix
		int var_offset = 0;
		int var_num = 0;
		(void)std::initializer_list<int>{ 
			(
				get_sparse_block<num_outputs, Var_t::var_len>(jac, Var_t::offset(iteration), jacobian_offsets(iteration, var_num))
					= ret.jacobian.template block<num_outputs, input_sizes>(0, var_offset),
				var_num++,
				var_offset += input_sizes,
				0
			)...
		};
	}


public:
	/*
		Evaluate the constraint
	 */
	template<typename variable_t, typename constraint_t>
	EIGEN_STRONG_INLINE void operator()(
		const param_t& param,
		const variable_t& var,
		constraint_t& con)
		const noexcept
	{
		for(int i=0; i<num_iterations; i++)
		{
			Func::template impl<scalar_t>(
				param, 
				con.template segment<num_outputs>(offset(i)), // Output
				Var_t::get(var, i) ...);  // Inputs
		}
	}

	/*
		Evaluate the constraint, and its jacobian in dense format
	 */
	template<typename variable_t, typename constraint_t>
	EIGEN_STRONG_INLINE auto operator()(
		const param_t& param,
		const variable_t& var,
		constraint_t& con,
		Eigen::Ref<Eigen::Matrix<scalar_t, constraint_t::RowsAtCompileTime, variable_t::RowsAtCompileTime>> jac)
		const noexcept
	{
		for(int i=0; i<num_iterations; i++)
		{
			auto ret = jacFunc::jac(param, Var_t::get(var, i)...);
			copy_to_constraint(i, ret, con);
			copy_to_jacobian_dense(i, ret, jac);
		}
	}

	/*
		Evaluate the constraint, and its jacobian in sparse format
	 */
	template<typename variable_t, typename constraint_t>
	EIGEN_STRONG_INLINE auto operator()(
		const param_t& param,
		const variable_t& var,
		constraint_t& con,
		Eigen::Ref<Eigen::SparseMatrix<scalar_t>> jac)
		const noexcept
	{
		for(int i=0; i<num_iterations; i++)
		{
			auto ret = jacFunc::jac(param, Var_t::get(var, i)...);
			copy_to_constraint(i, ret, con);
			copy_to_jacobian_sparse(i, ret, jac);
		}
	}

	/**********************************************************************
		Weighted sum versions of the constraint.

		We interpret the constraint as a weighted sum of functions w'f(x)
		and compute the value, gradient and hessian
	 **********************************************************************/

private:
	template<typename ret_t, typename gradient_t, typename weights_t>
	void add_to_gradient(int iteration, ret_t& ret, gradient_t& gradient, const weights_t& weights) const
	{
		auto local_grad = weights.template segment<num_outputs>(offset(iteration)).transpose() * ret.jacobian;

		// Write into the gradient vector in the right locations
		int var_offset = 0;
		(void)std::initializer_list<int>{ 
			(
				Var_t::get(gradient, iteration) += local_grad.template segment<input_sizes>(var_offset),
				var_offset += input_sizes,
				0
			)...
		};
	}

	template<typename ret_t, typename hessian_t, typename hessian_multiplier_t>
	EIGEN_STRONG_INLINE void add_to_hessian_dense(int iteration, ret_t& ret, 
													hessian_t& hessian, 
													const hessian_multiplier_t& hessian_multiplier)
	 const noexcept
	{
		// Write the hessian into the dense hessian matrix
		for(int output_index=0; output_index<num_outputs; output_index++)
		{
			ret.hessian[output_index] *= hessian_multiplier(offset(iteration) + output_index);

			int var_offset = 0;
			(void)std::initializer_list<int>{ 
				(
					// copy_dense_hessian_blockrow<input_sizes, variable_t::RowsAtCompileTime>(
					copy_dense_hessian_blockrow(
						// Rows of the global hessian for this variable
						hessian.template block<input_sizes, hessian_t::ColsAtCompileTime>(Var_t::offset(iteration), 0), 
						// Rows of the function hessian for this variable
						ret.hessian[output_index].template block<input_sizes, jacFunc::num_inputs>(var_offset, 0),
						iteration // Iteration number
					),
					var_offset += input_sizes,
					0
				)...
			};
		}
	}

	template<typename ret_t, typename hessian_multiplier_t>
	EIGEN_STRONG_INLINE void add_to_hessian_sparse(int iteration, ret_t& ret, 
													Eigen::Ref<Eigen::SparseMatrix<scalar_t>> hessian, 
													const hessian_multiplier_t& hessian_multiplier) const noexcept
	{
		std::array<std::size_t, sizeof...(Var_t)> var_offsets = {Var_t::offset(iteration)...};
		std::array<std::size_t, sizeof...(Var_t)> var_sizes = {Var_t::var_len...};

		// Write the hessian into the sparse hessian matrix
		for(int output_index=0; output_index<num_outputs; output_index++)
		{
			ret.hessian[output_index] *= hessian_multiplier(offset(iteration) + output_index);

			std::size_t row_offset = 0;
			for(int row=0; row<sizeof...(Var_t); row++)
			{
				std::size_t col_offset = 0;
				for(int col=0; col<sizeof...(Var_t); col++)
				{
					 get_sparse_block_dynamic(hessian, var_sizes[row], var_sizes[col], var_offsets[col], hessian_offsets[iteration](row, col))
						+= ret.hessian[output_index].block(row_offset, col_offset, var_sizes[row], var_sizes[col]);
					col_offset += var_sizes[col];
				}
				row_offset += var_sizes[row];
			}
		}
	}

public:

	/*
		Returns w'*f(x)
	 */
	template<typename variable_t, typename constraint_t>
	EIGEN_STRONG_INLINE scalar_t weighted_sum(
		const param_t& param,
		const variable_t& var,
		const constraint_t& weights)
		const noexcept
	{
		Eigen::Matrix<scalar_t, num_outputs, 1> tmp;
		scalar_t out = 0;
		for(int i=0; i<num_iterations; i++)
		{
			Func::template impl<scalar_t>(
				param, 
				tmp, // Output
				Var_t::get(var, i) ...);  // Inputs
			out += weights.template segment<num_outputs>(offset(i)).dot(tmp);
		}
		return out;
	}

	/*
		Returns w'*f(x) and gradient w'*grad f(x)
	 */
	template<typename variable_t, typename constraint_t>
	EIGEN_STRONG_INLINE scalar_t weighted_sum(
		const param_t& param,
		const variable_t& var,
		const constraint_t& weights,
		Eigen::Ref<Eigen::Matrix<scalar_t, variable_t::RowsAtCompileTime, 1>> gradient)
		const noexcept
	{
		scalar_t out = 0;        
		for(int i=0; i<num_iterations; i++)
		{
			auto ret = jacFunc::jac(param, Var_t::get(var, i)...);
			out += weights.template segment<num_outputs>(offset(i)).dot(ret.val);
			add_to_gradient(i, ret, gradient, weights);
		}
		return out;
	}

	/*
		Returns w'*f(x), gradient w'*grad f(x) and hessian H = sum_i w_i nabla^2 f_i(x) for dense H
	 */
	template<typename variable_t, typename constraint_t>
	EIGEN_STRONG_INLINE auto weighted_sum(
		const param_t& param,
		const variable_t& var,
		const constraint_t& weights,
		Eigen::Ref<Eigen::Matrix<scalar_t, variable_t::RowsAtCompileTime, 1>> gradient,
		Eigen::Ref<Eigen::Matrix<scalar_t, variable_t::RowsAtCompileTime, variable_t::RowsAtCompileTime>> hessian)
		const noexcept
	{
		scalar_t out = 0;        
		for(int i=0; i<num_iterations; i++)
		{
			auto ret = jacFunc::hessian(param, Var_t::get(var, i)...);
			out += weights.template segment<num_outputs>(offset(i)).dot(ret.val);
			add_to_gradient(i, ret, gradient, weights);
			add_to_hessian_dense(i, ret, hessian, weights);
		}
		return out;
	}

	/*
		Returns w'*f(x), gradient w'*grad f(x) and hessian H = sum_i w_i nabla^2 f_i(x) for sparse H
	 */
	template<typename variable_t, typename constraint_t>
	EIGEN_STRONG_INLINE auto weighted_sum(
		const param_t& param,
		const variable_t& var,
		const constraint_t& weights,
		Eigen::Ref<Eigen::Matrix<scalar_t, variable_t::RowsAtCompileTime, 1>> gradient,
		Eigen::Ref<Eigen::SparseMatrix<scalar_t>> hessian)
		const noexcept
	{
		scalar_t out = 0;        
		for(int i=0; i<num_iterations; i++)
		{
			auto ret = jacFunc::hessian(param, Var_t::get(var, i)...);
			out += weights.template segment<num_outputs>(offset(i)).dot(ret.val);
			add_to_gradient(i, ret, gradient, weights);
			add_to_hessian_sparse(i, ret, hessian, weights);
		}
		return out;
	}



	/*
		Sets the offset, and returns the new offset
	 */
	std::size_t set_offset(std::size_t offset_)
	{
		constraint_offset = offset_;
		return constraint_offset + num_outputs * num_iterations;
	}

	/*
		Return the offset into the constraint vector for the 
		ind'th constraint
	 */
	std::size_t offset(int ind = 0) const
	{
		return constraint_offset + ind * num_outputs;
	}

	/*
		Adds the non-zero elements to the triplet vector for the given constraint
	*/
	void jac_get_sparsity_triplet(std::vector<Eigen::Triplet<scalar_t>>& trip) const
	{
		for(int i=0; i<num_iterations; i++)
		{
			(void)std::initializer_list<int>{ 
				(
					get_sparsity_triplet_block(trip, offset(i), num_outputs, Var_t::offset(i), Var_t::var_len),
					0
				)...
			};
		}
	}

	/*
		Adds the non-zero elements to the triplet vector for the given constraint
	*/
	static constexpr void hessian_get_sparsity_triplet(std::vector<Eigen::Triplet<scalar_t>>& trip)
	{
		for(int i=0; i<num_iterations; i++)
		{
			std::array<std::size_t, sizeof...(Var_t)> var_offsets = {Var_t::offset(i)...};
			std::array<std::size_t, sizeof...(Var_t)> var_sizes = {Var_t::var_len...};

			for(int row=0; row<sizeof...(Var_t); row++)
				for(int col=0; col<sizeof...(Var_t); col++)
					get_sparsity_triplet_block(trip, var_offsets[row], var_sizes[row], var_offsets[col], var_sizes[col]);
		}
	}

	/*
		Set sparsity offsets. This can only be called once the global constraint structure is known 
		from constraints_impl

		J is a compressed matrix whose structure was set by constraints_impl.initialize_sparse_jacobian

		!! This function is only called from the constraints constructor !!
	 */
	void jac_set_sparsity_offsets(Eigen::SparseMatrix<scalar_t>& J)
	{
		for(int i=0; i<num_iterations; i++)
		{
			// Get the index at the start of each constraint / variable block
			int var_ind = 0;
			(void)std::initializer_list<int>{ 
				(
					jacobian_offsets(i, var_ind) = J.coeff(offset(i), Var_t::offset(i)),
					var_ind++,
					0
				)...
			};
		}
	}

	/*
		Set sparsity offsets. This can only be called once the global constraint structure is known 
		from constraints_impl

		H is a compressed matrix whose structure was set by constraints_impl.initialize_sparse_hessian

		!! This function is only called from the constraints constructor !!
	 */
	void hessian_set_sparsity_offsets(Eigen::SparseMatrix<scalar_t>& H)
	{
		for(int i=0; i<num_iterations; i++)
		{
			std::array<std::size_t, sizeof...(Var_t)> var_offsets = {Var_t::offset(i)...};

			// Get the index at the start of each constraint / variable block
			for(int row=0; row<sizeof...(Var_t); row++)
				for(int col=0; col<sizeof...(Var_t); col++)
				{
					hessian_offsets[i](row, col) = H.coeff(var_offsets[row], var_offsets[col]);
				}
		}
	}

	/*
		Print out the elements of the jacobian that this constraint 
		will impact for debugging
	 */
	void jac_print_sparsity_structure(Eigen::SparseMatrix<scalar_t>& J)
	{
		scalar_t* values = J.valuePtr();
		for(int i=0; i<J.nonZeros(); i++)
			values[i] = -1.0;

		for(int i=0; i<num_iterations; i++)
		{
			// Write the jacobian into the sparse jacobian matrix
			int var_num = 0;
			(void)std::initializer_list<int>{ 
				(
					get_sparse_block<num_outputs, Var_t::var_len>(J, Var_t::offset(i), jacobian_offsets(i, var_num)).array()
						= (double)i + (double)(var_num+1) / 10.0,
					var_num++,
					0
				)...
			};
		}
	}


private:
	static constexpr void get_sparsity_triplet_block(std::vector<Eigen::Triplet<scalar_t>>& trip, 
													 int start_row, int row_len, int start_col, int col_len)
	{
		for(int row=start_row; row<start_row + row_len; row++)
			for(int col=start_col; col<start_col + col_len; col++)
				trip.emplace_back(row, col, 1.0);
	}

	/*
		Returns a dense map to a block of the hessian for a particular pair of variables
	 */
	template<std::size_t row_len, std::size_t col_len>
	Eigen::Map<Eigen::Matrix<scalar_t, row_len, col_len>, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
		 get_sparse_block(Eigen::Ref<Eigen::SparseMatrix<scalar_t>> sparse_matrix,
							int col, int valueStart) const
	{
		using Map_t = Eigen::Map<Eigen::Matrix<scalar_t, row_len, col_len>, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;
		using Stride_t = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;

		int nnz = sparse_matrix.outerIndexPtr()[col+1] - sparse_matrix.outerIndexPtr()[col];
		scalar_t* data = sparse_matrix.valuePtr() + valueStart;

		// Terrible fix: Eigen seems to mixup inner and outer strides for row vectors
		int outerStride, innerStride;
		if(row_len == 1)
		{
			outerStride = 1;
			innerStride = nnz;
		} else
		{
			innerStride = 1;
			outerStride = nnz;
		}

		// return Map_t(data, Stride_t(nnz, 2));
		return Map_t(data, Stride_t(outerStride, innerStride));
	}

	Eigen::Map<Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
		 get_sparse_block_dynamic(Eigen::Ref<Eigen::SparseMatrix<scalar_t>> sparse_matrix,
							std::size_t row_len, std::size_t col_len, int col, int valueStart) const
	{
		using Map_t = Eigen::Map<Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;
		using Stride_t = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;

		int nnz = sparse_matrix.outerIndexPtr()[col+1] - sparse_matrix.outerIndexPtr()[col];
		scalar_t* data = sparse_matrix.valuePtr() + valueStart;

		// Terrible fix: Eigen seems to mixup inner and outer strides for row vectors
		int outerStride, innerStride;
		// if(row_len == 1)
		// {
		//     outerStride = 1;
		//     innerStride = nnz;
		// } else
		// {
			innerStride = 1;
			outerStride = nnz;
		return Map_t(data, row_len, col_len, Stride_t(outerStride, innerStride));
	}


	/*
		Copies a block-row of a dense hessian into the right location of the full hessian
	 */
	template<typename target_t,   // Length of the variable row being copied
			 typename source_t> // Total number of variables in the target hessian
	EIGEN_STRONG_INLINE void copy_dense_hessian_blockrow(
		target_t&& target_hessian_row,
		const source_t& source_hessian_row,
		// Eigen::Ref<Eigen::Matrix<scalar_t, num_rows, num_inputs>> target_hessian_row,
		// const Eigen::Ref<const Eigen::Matrix<scalar_t, num_rows, jacFunc::num_inputs>>& source_hessian_row,
		const int iteration_number) const
	{
		constexpr auto num_rows = target_t::RowsAtCompileTime;

		// Write the hessian into the dense hessian matrix
		int var_offset = 0;
		(void)std::initializer_list<int>{ 
			(
				target_hessian_row.template block<num_rows, input_sizes>(0, Var_t::offset(iteration_number))
					+= source_hessian_row.template block<num_rows, input_sizes>(0, var_offset),
				var_offset += input_sizes,
				0
			)...
		};        
	}

};

/*************************************************************
	A set of constraints

	Virtual - not meant to be used.
 *************************************************************/
template<std::size_t num_variables, 
		 typename cons_t,  // Tuple of types con_t
		 std::size_t... ind>  // Index of length(cons_t)
struct constraintset_base
{
	cons_t cons; // Tuple of constraints

	using scalar_t = typename std::tuple_element_t<0, cons_t>::scalar_t;
	using param_t = typename std::tuple_element_t<0, cons_t>::param_t;
	static constexpr std::size_t num_constraints = sum_template<std::tuple_element_t<ind, cons_t>::constraint_vector_length...>();

	std::size_t nnz_hessian;

	// // Offset into the value vector in each column where the first elemtent of the lower-triangular part of the matrix is stored
	// Eigen::Matrix<Eigen::Index, num_variables, 1> hessian_lower_triangular_offsets; 

	constraintset_base() : cons()
	{
		// Set the offsets / ordering
		int offset = 0;
		(void)std::initializer_list<int>{ 
			(
				offset = std::get<ind>(cons).set_offset(offset),
				0
			)...
		};

		// Store the offset locations for each block of the jacobian
		SparseMatrix<scalar_t> J(num_constraints, num_variables);
		initialize_sparse_jacobian(J);

		scalar_t* values = J.valuePtr();
		for(int i=0; i<J.nonZeros(); i++)
			values[i] = i;

		(void)std::initializer_list<int>{ 
			(
				std::get<ind>(cons).jac_set_sparsity_offsets(J),
				0
			)...
		};


		// Store the offset locations for each block of the hessian
		SparseMatrix<scalar_t> H(num_variables, num_variables);
		initialize_sparse_hessian(H);

		values = H.valuePtr();
		for(int i=0; i<H.nonZeros(); i++)
			values[i] = i;

		(void)std::initializer_list<int>{ 
			(
				std::get<ind>(cons).hessian_set_sparsity_offsets(H),
				0
			)...
		};

		nnz_hessian = H.nonZeros();
	}

	/*
		Set non-zeros of J to match the sparsity structure of the constraint Jacobian
	 */
	EIGEN_STRONG_INLINE void initialize_sparse_jacobian(SparseMatrix<scalar_t>& J)
	{
		std::vector<Eigen::Triplet<scalar_t>> trip;

		(void)std::initializer_list<int>{ 
			(
				std::get<ind>(cons).jac_get_sparsity_triplet(trip),
				0
			)...
		};

		J.setFromTriplets(trip.begin(), trip.end());
		J.makeCompressed();
	}

	/*
		Set non-zeros of H to match the sparsity structure of the constraint Jacobian
	 */
	static constexpr void initialize_sparse_hessian(SparseMatrix<scalar_t>& H)
	{
		std::vector<Eigen::Triplet<scalar_t>> trip;

		(void)std::initializer_list<int>{ 
			(
				std::tuple_element_t<ind, cons_t>::hessian_get_sparsity_triplet(trip),
				// std::get<ind>(cons).hessian_get_sparsity_triplet(trip),
				0
			)...
		};

		H.setFromTriplets(trip.begin(), trip.end());
		H.makeCompressed();
	}
};


/*************************************************************
	A set of constraints of different types

	This set computes values and jacobians
 *************************************************************/
template<std::size_t num_variables, typename cons_t, typename Index>
struct constraints_impl;

template<std::size_t num_variables, typename cons_t, std::size_t... ind>
struct constraints_impl<num_variables, cons_t, std::integer_sequence<std::size_t, ind...>> 
		: constraintset_base<num_variables, cons_t, ind...>
{
	// Number of non-zeros in jacobian
	static constexpr std::size_t nnz_jacobian = sum_template<std::tuple_element_t<ind, cons_t>::nnz_jacobian...>();

	// Expose elements from the base class
	using base_t = constraintset_base<num_variables, cons_t, ind...>;
	using typename base_t::scalar_t;
	using base_t::cons;
	using base_t::num_constraints;
	using typename base_t::param_t;

	using variable_t = typename dense_matrix_type_selector<scalar_t, num_variables, 1>::type;
	using constraint_t = typename dense_matrix_type_selector<scalar_t, num_constraints, 1>::type;
	using jacobian_dense_t = typename dense_matrix_type_selector<scalar_t, num_constraints, num_variables>::type;

	constraints_impl() : base_t()
	{}

	/*
		Evaluate all constraints
	 */
	EIGEN_STRONG_INLINE void operator()(
		const param_t& param,
		const Eigen::Ref<const variable_t> var,
		Eigen::Ref<constraint_t> con)
		const noexcept
	{
		assert(var.rows() == num_variables && var.cols() == 1);
		assert(con.rows() == num_constraints && con.cols() == 1);

		std::cout << "\n\nConstraints\n";

		std::cout << "type(var) = " << type_name<decltype(var)>() << std::endl;
		std::cout << "type(con) = " << type_name<decltype(con)>() << std::endl;
		// std::get<0>(cons)(param, var, con);
		std::cout << "\n\n";

		// con.array() = 0;        
		// (void)std::initializer_list<int>{ 
		// 	(
		// 		std::get<ind>(cons)(param, var, con),
		// 		0
		// 	)...
		// };
	}

	/*
		Evaluate all constraints and dense jacobians
	 */
	EIGEN_STRONG_INLINE auto operator()(
		const param_t& param,
		const Eigen::Ref<const variable_t> var,
		Eigen::Ref<constraint_t> con,
		Eigen::Ref<jacobian_dense_t> jac)
		const noexcept
	{
		assert(var.rows() == num_variables && var.cols() == 1);
		assert(con.rows() == num_constraints && con.cols() == 1);
		assert(jac.rows() == num_constraints && jac.cols() == num_variables);

		con.array() = 0;
		jac.array() = 0;
		(void)std::initializer_list<int>{ 
			(
				std::get<ind>(cons)(param, var, con, jac),
				0
			)...
		};
	}

	/*
		Evaluate all constraints and their jacobian in sparse format
	 */
	EIGEN_STRONG_INLINE auto operator()(
		const param_t& param,
		const Eigen::Ref<const variable_t> var,
		Eigen::Ref<constraint_t> con,
		Eigen::Ref<Eigen::SparseMatrix<scalar_t>> jac)
		const noexcept
	{
		assert(var.rows() == num_variables && var.cols() == 1);
		assert(con.rows() == num_constraints && con.cols() == 1);
		assert(jac.rows() == num_constraints && jac.cols() == num_variables);

		con.array() = 0;
		auto ptr = jac.valuePtr();
		for(int i=0; i<jac.nonZeros(); i++) ptr[i] = 0;

		(void)std::initializer_list<int>{ 
			(
				std::get<ind>(cons)(param, var, con, jac),
				0
			)...
		};
	}

	/*
		Get the lower and upper bounds for this constraint
	 */
	EIGEN_STRONG_INLINE void get_bounds(const param_t& param, Eigen::Ref<constraint_t> lb, Eigen::Ref<constraint_t> ub)
	{
		assert(lb.rows() == num_constraints && lb.cols() == 1);
		assert(ub.rows() == num_constraints && ub.cols() == 1);

		(void)std::initializer_list<int>{ 
			(
				std::get<ind>(cons).get_bounds(param, lb, ub),
				0
			)...
		};
	}
};



/****************************************************************
	Objective function

	Takes the form w' * f(x)
	Represents f as the vector-valued constraint_impl
 ****************************************************************/
template<std::size_t num_variables, typename F_t, typename Index>
struct objective_impl;

template<std::size_t num_variables, typename F_t, std::size_t... ind>
struct objective_impl<num_variables, F_t, std::integer_sequence<std::size_t, ind...>>
		: constraintset_base<num_variables, F_t, ind...>
{
	// Expose elements from the base class
	using base_t = constraintset_base<num_variables, F_t, ind...>;
	using typename base_t::scalar_t;
	using base_t::cons;
	using base_t::num_constraints;
	using typename base_t::param_t;

	using variable_t = typename Eigen::Matrix<scalar_t, num_variables, 1>;

	// Weights
	using weight_t = Eigen::Matrix<scalar_t, num_constraints, 1>;
	weight_t w = weight_t::Ones();

	objective_impl() : base_t()
	{}

	/*
		Evaluate objective
	 */
	EIGEN_STRONG_INLINE scalar_t operator()(
		const param_t& param,
		const Eigen::Ref<const variable_t> var)
		const noexcept
	{
		scalar_t out = 0;
		(void)std::initializer_list<int>{ 
			(
				out += std::get<ind>(cons).weighted_sum(param, var, w),
				0
			)...
		};
		return out;
	}

	/*
		Evaluate objective and gradient
	 */
	EIGEN_STRONG_INLINE scalar_t operator()(
		const param_t& param,
		const Eigen::Ref<const variable_t> var,
		Eigen::Ref<variable_t> gradient)
		const noexcept
	{
		gradient.array() = 0;
		scalar_t out = 0;
		(void)std::initializer_list<int>{ 
			(
				out += std::get<ind>(cons).weighted_sum(param, var, w, gradient),
				0
			)...
		};
		return out;
	}

	/*
		Evaluate objective, gradient and hessian in dense form
	 */
	EIGEN_STRONG_INLINE scalar_t operator()(
		const param_t& param,
		const Eigen::Ref<const variable_t> var,
		Eigen::Ref<Eigen::Matrix<scalar_t, num_variables, 1>> gradient,
		Eigen::Ref<Eigen::Matrix<scalar_t, num_variables, num_variables>> hessian)
		const noexcept
	{
		hessian.array() = 0;
		gradient.array() = 0;
		scalar_t out = 0;
		(void)std::initializer_list<int>{ 
			(
				out += std::get<ind>(cons).weighted_sum(param, var, w, gradient, hessian),
				0
			)...
		};
		return out;
	}

	/*
		Evaluate objective, gradient and sparse hessian
	 */
	EIGEN_STRONG_INLINE auto operator()(
		const param_t& param,
		const Eigen::Ref<const variable_t> var,
		Eigen::Ref<Eigen::Matrix<scalar_t, num_variables, 1>> gradient,
		Eigen::Ref<Eigen::SparseMatrix<scalar_t>> hessian)
		const noexcept
	{
		auto ptr = hessian.valuePtr();
		for(int i=0; i<hessian.nonZeros(); i++) ptr[i] = 0;
		gradient.array() = 0;
		scalar_t out = 0;

		// TODO: add a static-assert that all sparse matrices have the right number of non-zeros

		(void)std::initializer_list<int>{ 
			(
				out += std::get<ind>(cons).weighted_sum(param, var, w, gradient, hessian),
				0
			)...
		};
		return out;
	}
};

/****************************************************************
	Lagrangian

	L = obj + lam_ineq' * ineq + lam_eq' * eq + lam_var' * var
 ****************************************************************/
template<typename variables_t, typename obj_tuple, std::size_t num_obj_, 
								 typename eq_tuple, std::size_t num_eq_,
								 typename ineq_tuple, std::size_t num_ineq_>
struct lagrangian_impl
{
	// TODO: Buffer bounds computation somewhere

	// The full list of functions in the lagrangian
	using lag_tuple = decltype(std::tuple_cat<obj_tuple, eq_tuple, ineq_tuple>(
			std::declval<obj_tuple>(),
			std::declval<eq_tuple>(),
			std::declval<ineq_tuple>()));

	static constexpr std::size_t num_obj = num_obj_;
	static constexpr std::size_t num_eq = num_eq_;
	static constexpr std::size_t num_ineq = num_ineq_;

	// Reference to the variables
	variables_t& variables;

	// Weighted sum representation of the lagrangian
	using lag_index = std::make_integer_sequence<std::size_t, std::tuple_size<lag_tuple>::value>;
	using lagrangian_t = objective_impl<variables_t::num_variables, lag_tuple, lag_index>;
	lagrangian_t lagrangian;

	lagrangian_impl(variables_t& variables_) : variables(variables_), lagrangian()
	{}

	// Input types
	using scalar_t = typename lagrangian_t::scalar_t;
	using param_t  = typename lagrangian_t::param_t;

	using variable_vec  = typename Eigen::Matrix<scalar_t, variables_t::num_variables, 1>;
	using eq_dual_vec   = typename Eigen::Matrix<scalar_t, num_eq, 1>;
	using ineq_dual_vec = typename Eigen::Matrix<scalar_t, num_ineq, 1>;
	using var_dual_vec  = typename Eigen::Matrix<scalar_t, variables_t::num_variables, 1>;

	EIGEN_STRONG_INLINE std::size_t nnz_hessian()
	{
		return lagrangian.nnz_hessian;
	}

	/*
		Evaluate lagrangian
	 */
	EIGEN_STRONG_INLINE scalar_t operator()(
		const param_t& param,
		const Eigen::Ref<const variable_vec> var,
		const scalar_t obj_factor,
		const Eigen::Ref<const eq_dual_vec> eq_dual,
		const Eigen::Ref<const ineq_dual_vec> ineq_dual,
		const Eigen::Ref<const var_dual_vec> var_dual)
		noexcept
	{
		variable_vec lb; 
		variable_vec ub;
		variables.get_bounds(param, lb, ub);

		lagrangian.w.template head<num_obj>().array() = obj_factor;
		lagrangian.w.template segment<num_eq>(num_obj) = eq_dual;
		lagrangian.w.template tail<num_ineq>() = ineq_dual;

		return lagrangian(param, var) 
				+ var.dot(var_dual) - var_dual.array().min(0).matrix().dot(lb) + var_dual.array().max(0).matrix().dot(ub);
	}

	/*
		Evaluate objective and gradient
	 */
	EIGEN_STRONG_INLINE scalar_t operator()(
		const param_t& param,
		const Eigen::Ref<const variable_vec> var,
		const scalar_t obj_factor,
		const Eigen::Ref<const eq_dual_vec> eq_dual,
		const Eigen::Ref<const ineq_dual_vec> ineq_dual,
		const Eigen::Ref<const var_dual_vec> var_dual,
		Eigen::Ref<variable_vec> gradient)
		noexcept
	{
		variable_vec lb; 
		variable_vec ub;
		variables.get_bounds(param, lb, ub);

		lagrangian.w.template head<num_obj>().array() = obj_factor;
		lagrangian.w.template segment<num_eq>(num_obj) = eq_dual;
		lagrangian.w.template tail<num_ineq>() = ineq_dual;

		scalar_t lag = lagrangian(param, var, gradient) 
						+ var.dot(var_dual) - var_dual.array().min(0).matrix().dot(lb) + var_dual.array().max(0).matrix().dot(ub);

		gradient += var_dual;
		return lag;
	}

	/*
		Evaluate objective, gradient and hessian in dense form
	 */
	EIGEN_STRONG_INLINE scalar_t operator()(
		const param_t& param,
		const Eigen::Ref<const variable_vec> var,
		const scalar_t obj_factor,
		const Eigen::Ref<const eq_dual_vec> eq_dual,
		const Eigen::Ref<const ineq_dual_vec> ineq_dual,
		const Eigen::Ref<const var_dual_vec> var_dual,
		Eigen::Ref<variable_vec> gradient,
		Eigen::Ref<Eigen::Matrix<scalar_t, variables_t::num_variables, variables_t::num_variables>> hessian)
		noexcept
	{
		variable_vec lb; 
		variable_vec ub;
		variables.get_bounds(param, lb, ub);

		lagrangian.w.template head<num_obj>().array() = obj_factor;
		lagrangian.w.template segment<num_eq>(num_obj) = eq_dual;
		lagrangian.w.template tail<num_ineq>() = ineq_dual;

		scalar_t lag = lagrangian(param, var, gradient, hessian) 
						+ var.dot(var_dual) - var_dual.array().min(0).matrix().dot(lb) + var_dual.array().max(0).matrix().dot(ub);

		gradient += var_dual;
		return lag;
	}

	/*
		Evaluate objective, gradient and sparse hessian
	 */
	EIGEN_STRONG_INLINE scalar_t operator()(
		const param_t& param,
		const Eigen::Ref<const variable_vec> var,
		const scalar_t obj_factor,
		const Eigen::Ref<const eq_dual_vec> eq_dual,
		const Eigen::Ref<const ineq_dual_vec> ineq_dual,
		const Eigen::Ref<const var_dual_vec> var_dual,
		Eigen::Ref<variable_vec> gradient,
		Eigen::Ref<Eigen::SparseMatrix<scalar_t>> hessian)
		noexcept
	{
		variable_vec lb; 
		variable_vec ub;
		variables.get_bounds(param, lb, ub);

		lagrangian.w.template head<num_obj>().array() = obj_factor;
		lagrangian.w.template segment<num_eq>(num_obj) = eq_dual;
		lagrangian.w.template tail<num_ineq>() = ineq_dual;

		scalar_t lag = lagrangian(param, var, gradient, hessian) 
						+ var.dot(var_dual) - var_dual.array().min(0).matrix().dot(lb) + var_dual.array().max(0).matrix().dot(ub);

		gradient += var_dual;
		return lag;
	}

	/*
		Set non-zeros of H to match the sparsity structure of the lagrangian Hessian
	 */
	static constexpr void initialize_sparse_hessian(SparseMatrix<scalar_t>& H)
	{
		lagrangian_t::initialize_sparse_hessian(H);
	}
};


/*************************************************************
	Selection of sparse / dense matrices
 *************************************************************/

// // Puts small fixed-sized matrices on the stack, and large ones on the heap
// template<typename scalar_t, int Rows, int Cols>
// struct dense_matrix_type_selector
// {
//     enum { allocate_dynamic = (Rows == Eigen::Dynamic) || (Cols == Eigen::Dynamic) ? 1 : 0,
//            // allocate_static =  (!allocate_dynamic) && (Rows * Cols * sizeof (scalar_t) < 10000) ? 1 : 0,
//            allocate_static =  (!allocate_dynamic) && (Rows * Cols * sizeof (scalar_t) < EIGEN_STACK_ALLOCATION_LIMIT) ? 1 : 0,
//            cols = (Cols == 1) ? 1 : Eigen::Dynamic};
//     using type = typename std::conditional<allocate_static, Eigen::Matrix<scalar_t, Rows, Cols>,
//                                                             Eigen::Matrix<scalar_t, Eigen::Dynamic, cols>>::type;
// };

// enum
// {
//     DENSE  = 0,
//     SPARSE = 1
// };

// Selects sparse or dense matrix as user-specification
template<typename scalar_t, int rows, int cols, int MatrixType = DENSE>
struct matrix_type_selector
{
    using type = typename std::conditional<MatrixType == SPARSE, 
    								Eigen::SparseMatrix<scalar_t>,
                 		typename dense_matrix_type_selector<scalar_t, rows, cols>::type>::type;
};


/** Define NLP variable and matrix types.
 * 
 * This is a trait class that decides between sparse / dense and stack / heap allocation.
 * 
 * This must be used everywhere for consistency in types, or we can get very confusing 
 * assertions from eigen for passing heap allocated matrices to functions expecting 
 * stack allocation.
 *
 */
template<typename scalar_t, int num_variables, int num_equalities, int num_inequalities, int MatrixType = DENSE>
struct make_var_types
{
	using variable_vec              = typename dense_matrix_type_selector<scalar_t, num_variables, 1>::type;
	using equalities_vec            = typename dense_matrix_type_selector<scalar_t, num_equalities, 1>::type;
	using equalities_jacobian_mat   = typename matrix_type_selector<scalar_t, num_equalities,  num_variables, MatrixType>::type;
	using inequalities_vec          = typename dense_matrix_type_selector<scalar_t, num_inequalities, 1>::type;
	using inequalities_jacobian_mat = typename matrix_type_selector<scalar_t, num_inequalities,  num_variables, MatrixType>::type;
	using constraints_vec           = typename dense_matrix_type_selector<scalar_t, num_equalities + num_inequalities, 1>::type;
	using constraints_jacobian_mat  = typename matrix_type_selector<scalar_t, num_equalities + num_inequalities,  num_variables, MatrixType>::type;
	using obj_gradient_vec          = typename dense_matrix_type_selector<scalar_t, num_variables, 1>::type;
	using obj_hessian_mat           = typename matrix_type_selector<scalar_t, num_variables, num_variables, MatrixType>::type;
	using obj_vec                   = scalar_t;

	// Allocate memory for variable types
	// If they're fixed-size, then this does nothing. 
	// If sparse or dynamic, then this sets the size correctly
	static variable_vec init_variable_vec() 			       {return variable_vec{num_variables, 1};}
	static equalities_vec init_equalities_vec() 			     {return equalities_vec{num_equalities, 1};}
	static equalities_jacobian_mat init_equalities_jacobian_mat()   {return equalities_jacobian_mat{num_equalities,  num_variables};}
	static inequalities_vec init_inequalities_vec() 			   {return inequalities_vec{num_inequalities, 1};}
	static inequalities_jacobian_mat init_inequalities_jacobian_mat() {return inequalities_jacobian_mat{num_inequalities,  num_variables};}
	static constraints_vec init_constraints_vec() 			     {return constraints_vec{num_equalities + num_inequalities, 1};}
	static constraints_jacobian_mat init_constraints_jacobian_mat()  {return constraints_jacobian_mat{num_equalities + num_inequalities,  num_variables};}
	static obj_gradient_vec init_obj_gradient_vec() 			   {return obj_gradient_vec{num_variables, 1};}
	static obj_hessian_mat init_obj_hessian_mat() 			     {return obj_hessian_mat{num_variables, num_variables};}
};


/****************************************************************
 * A problem description containing constraints and objective
 * 
 * /todo Add sparse matrices
 * 
 ****************************************************************/
template<typename variables_t, typename eq_tuple, typename ineq_tuple, typename obj_tuple, int MatrixType = DENSE>
struct make_problem
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	// TODO: deal with empty constraints

private:
	static constexpr std::size_t num_obj = std::tuple_size<obj_tuple>::value;
	static constexpr std::size_t num_eq = std::tuple_size<eq_tuple>::value;
	static constexpr std::size_t num_ineq = std::tuple_size<ineq_tuple>::value;

	// Index sequences to run through the constraints and objective functions
	using eq_index = std::make_integer_sequence<std::size_t, num_eq>;
	using ineq_index = std::make_integer_sequence<std::size_t, num_ineq>;
	using obj_index = std::make_integer_sequence<std::size_t, num_obj>;

	// Eq + ineq
	using con_tuple = decltype(std::tuple_cat<eq_tuple, ineq_tuple>(
			std::declval<eq_tuple>(),
			std::declval<ineq_tuple>()));
	using con_index = std::make_integer_sequence<std::size_t, num_eq + num_ineq>;


public:
	static constexpr std::size_t num_variables = variables_t::num_variables;

	using equalities_t = constraints_impl<num_variables, eq_tuple, eq_index>;
	using inequalities_t = constraints_impl<num_variables, ineq_tuple, ineq_index>;
	using constraints_t = constraints_impl<num_variables, con_tuple, con_index>;
	using objective_t = objective_impl<num_variables, obj_tuple, obj_index>;

	// Lagrangian
	using lag_eq_t = objective_impl<num_variables, eq_tuple, eq_index>;
	using lag_ineq_t = objective_impl<num_variables, ineq_tuple, ineq_index>;
	using lagrangian_t = lagrangian_impl<variables_t, 
										 obj_tuple, objective_t::num_constraints,
										 eq_tuple, equalities_t::num_constraints,
										 ineq_tuple, inequalities_t::num_constraints>;

	variables_t    variables;
	objective_t    objective;
	equalities_t   equalities;
	inequalities_t inequalities;
	constraints_t  constraints;
	lagrangian_t   lagrangian;

	make_problem() : variables(), objective(), equalities(), inequalities(), constraints(),
					 lagrangian(variables)
	{}

	// Expose required constants
	using scalar_t = typename objective_t::scalar_t;

	static constexpr std::size_t num_equalities = equalities_t::num_constraints;
	static constexpr std::size_t nnz_equalities_jacobian = equalities_t::nnz_jacobian;

	static constexpr std::size_t num_inequalities = inequalities_t::num_constraints;
	static constexpr std::size_t nnz_inequalities_jacobian = inequalities_t::nnz_jacobian;

	static constexpr std::size_t num_constraints = constraints_t::num_constraints;
	static constexpr std::size_t nnz_constraints_jacobian = constraints_t::nnz_jacobian;

	// NLP variable types
	using var_types = make_var_types<scalar_t, num_variables, num_equalities, num_inequalities, MatrixType>;

	using variable_vec              = typename var_types::variable_vec;
	using equalities_vec            = typename var_types::equalities_vec;
	using equalities_jacobian_mat   = typename var_types::equalities_jacobian_mat;
	using inequalities_vec          = typename var_types::inequalities_vec;
	using inequalities_jacobian_mat = typename var_types::inequalities_jacobian_mat;
	using constraints_vec           = typename var_types::constraints_vec;
	using constraints_jacobian_mat  = typename var_types::constraints_jacobian_mat;
	using obj_gradient_vec          = typename var_types::obj_gradient_vec;
	using obj_hessian_mat           = typename var_types::obj_hessian_mat;
	using obj_vec                   = typename var_types::obj_vec;

	using param_t = typename objective_t::param_t;

	EIGEN_STRONG_INLINE void setBounds(param_t param,
										 Eigen::Ref<variable_vec> lb_x, Eigen::Ref<variable_vec> ub_x,
										 Eigen::Ref<inequalities_vec> lb_g, Eigen::Ref<inequalities_vec> ub_g)
	{
		inequalities.get_bounds(param, lb_g, ub_g);
		variables.get_bounds(param, lb_x, ub_x);
	}

	friend std::ostream& operator<<( std::ostream& o, const make_problem& p ) {
		o << "======= NLP Problem =======\n";
		o << std::endl;
		o << "Problem size:\n";
		o << "\tNumber of equalities   = " << p.num_equalities << std::endl;
		o << "\tNumber of inequalities = " << p.num_inequalities << std::endl;
		o << "\tNumber of constraints (eq + ineq)  = " << p.num_constraints << std::endl;
		o << std::endl;
		o << "Sparsity information:\n";
		o << "\tNNZ equalities jacobian   = " << p.nnz_equalities_jacobian << std::endl;
		o << "\tNNZ inequalities jacobian = " << p.nnz_inequalities_jacobian << std::endl;
		o << "\tNNZ constraints jacobian  = " << p.nnz_constraints_jacobian << std::endl;

		return o;
	}

	/// Dump a linearization of the problem around the given operating point.
	/**
	 * @param o Output stream to write to
	 * @param p Parameter values
	 * @param primal Primal vector to linearize around
	 * @param eq_dual Equalities dual vector
	 * @param ineq_dual Inequalities dual vector
	 * @param var_dual Variable bounds dual vector
	 */
	void print_linearization(std::ostream& o, param_t& p, variable_vec& primal, 
		typename lagrangian_t::eq_dual_vec&   eq_dual,
		typename lagrangian_t::ineq_dual_vec& ineq_dual,
		typename lagrangian_t::var_dual_vec&  var_dual)
	{
		print_linearization_impl(o, p, primal, &eq_dual, &ineq_dual, &var_dual);
	}

	/// Dump a linearization of the problem around the given operating point.
	/**
	 * Doesn't print the Lagrangian, since we don't have the dual.
	 * 
	 * @param o Output stream to write to
	 * @param p Parameter values
	 * @param primal Primal vector to linearize around
	 */
	void print_linearization(std::ostream& o, param_t& p, variable_vec& primal)
	{
		print_linearization_impl(o, p, primal, NULL, NULL, NULL);
	}

private:
	void print_linearization_impl(std::ostream& o, param_t& p, variable_vec& primal, 
		typename lagrangian_t::eq_dual_vec*   eq_dual = NULL,
		typename lagrangian_t::ineq_dual_vec* ineq_dual = NULL,
		typename lagrangian_t::var_dual_vec*  var_dual = NULL)
	{
		o << "***********************************************\n";
		o << "*** Printing a linearization of the problem ***\n";
		o << "***********************************************\n";

		o << "==> Linearizing around the point <==\n";

		if(eq_dual)
		{
			o << "Primal               : " << primal.transpose() << std::endl;
			o << "Equalities dual      : " << eq_dual->transpose() << std::endl;
			o << "Inequalities dual    : " << ineq_dual->transpose() << std::endl;
			o << "Variable bounds dual : " << var_dual->transpose() << std::endl;
		} else
		{
			o << "Primal : " << primal.transpose() << std::endl;			
		}

		o << "\n\n";
		o << "=======================================\n";
		o << "=== Test computation of constraints ===\n";
		o << "=======================================\n";

		o << "==> Computing value of constraints <==\n";
		constraints_vec con;
		con.setZero();
		constraints(p, primal, con);
		o << "con = " << con.transpose() << std::endl;
		o << "\n\n";

		o << "==> Computing dense jacobian <==\n";
		constraints_jacobian_mat J;
		J.setZero();
		constraints(p, primal, con, J);
		o << "con = " << con.transpose() << std::endl;
		o << "J = \n" << J << std::endl;
		o << "\n\n";

		o << "==> Computing sparse jacobian <==\n";
		Eigen::SparseMatrix<scalar_t> sJ(num_constraints, num_variables);
		constraints.initialize_sparse_jacobian(sJ);
		constraints(p, primal, con, sJ);
		o << "con = " << con.transpose() << std::endl;
		o << "sJ = \n" << Eigen::MatrixX<scalar_t>(sJ) << std::endl;
		o << "\n\n";

		o << "==> Computing lower and upper bounds <==\n";
		constraints_vec lb, ub;
		lb.setZero(); ub.setZero();
		constraints.get_bounds(p, lb, ub);
		o << "lb = " << lb.transpose() << std::endl;
		o << "ub = " << ub.transpose() << std::endl;
		o << "\n\n";

		o << "==> Computing variable bounds <==\n";
		variable_vec x_lb, x_ub;
		x_lb.setZero(); x_ub.setZero();
		variables.get_bounds(p, x_lb, x_ub);
		o << "x_lb = " << x_lb.transpose() << std::endl;
		o << "x_ub = " << x_ub.transpose() << std::endl;
		o << "\n\n";

		o << "=====================================\n";
		o << "=== Test computation of objective ===\n";
		o << "=====================================\n";

		o << "==> Objective value <==\n";
		o << "obj = " << objective(p, primal) << std::endl;
		o << "\n\n";

		o << "==> Objective value and gradient <==\n";
		variable_vec grad;
		grad.setZero();
		auto val = objective(p, primal, grad);
		o << "obj = " << val << std::endl;
		o << "gradient = " << grad.transpose() << std::endl;
		o << "\n\n";

		o << "==> Objective value, gradient and dense Hessian <==\n";
		obj_hessian_mat hessian;
		val = objective(p, primal, grad, hessian);
		o << "obj = " << val << std::endl;
		o << "gradient = " << grad.transpose() << std::endl;
		o << "hessian = \n" << hessian << std::endl;
		o << "\n\n";

		o << "==> Objective value, gradient and sparse Hessian <==\n";
		Eigen::SparseMatrix<scalar_t> s_hessian(num_variables, num_variables);
		objective.initialize_sparse_hessian(s_hessian);

		val = objective(p, primal, grad, s_hessian);
		o << "obj = " << val << std::endl;
		o << "gradient = " << grad.transpose() << std::endl;
		o << "hessian = \n" << Eigen::MatrixX<scalar_t>(s_hessian) << std::endl;

		if(eq_dual) 
		{
			o << "\n\n";
			o << "======================================\n";
			o << "=== Test computation of lagrangian ===\n";
			o << "======================================\n";

			Eigen::SparseMatrix<scalar_t> l_hessian(num_variables, num_variables);
			lagrangian.initialize_sparse_hessian(l_hessian);

			val = lagrangian(p, primal, 1.0, *eq_dual, *ineq_dual, *var_dual, grad, l_hessian);
			o << "obj = " << val << std::endl;
			o << "gradient = " << grad.transpose() << std::endl;
			o << "hessian = \n" << Eigen::MatrixX<scalar_t>(l_hessian) << std::endl;
		}
	}
};

}

#endif // __LAMPC__HPP