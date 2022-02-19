#include <tuple>
#include <numeric>

#include "bsmatrix.hpp"
 
#include "Eigen/Dense"
#include "Eigen/Sparse"

namespace LA {

struct variable_info_t
{
	int offset;
	int size;
};

template<typename scalar_t, typename param_t, typename Functions, 
		 typename VariableInfo, typename FunctionInfo,
		 int... functionCalls>
class Function
{
private:
	template<typename T>
	static constexpr T total(std::initializer_list<T> elements)
	{
		T size = 0;
		for(const T &e : elements)
			size += e;
		return size;
	}

	static constexpr int compute_inputSize()
	{
		int size = 0;
		for(int i=0; i<VariableInfo::numVariables; i++)
			size += VariableInfo::variable_info[i].size;
		return size;
	}

public:
	static constexpr int inputSize = compute_inputSize(); 
	static constexpr int outputSize 
				= total({std::tuple_element<functionCalls, Functions>::type::num_outputs...});

	using input_t = Eigen::Vector<scalar_t, inputSize>;
	using output_t = Eigen::Vector<scalar_t, outputSize>;


private: // Everything to do with the jacobian

	// Number of columns in all the nonzero blocks
	static constexpr int nnzBlockColumns 
				= total({std::tuple_element<functionCalls, Functions>::type::num_inputs...});

	// Query each function for the nnz in its jacobian
	static constexpr int nnzJacobian 
				= total({std::tuple_element<functionCalls, Functions>::type::nnzJacobian()...});

	static std::array<int, sizeof...(functionCalls)> rowSizes()
	{
		return {std::tuple_element<functionCalls, Functions>::type::num_outputs...};
	}

	static std::array<int, VariableInfo::numVariables> colSizes()
	{
		std::array<int, VariableInfo::numVariables> sizes;
		for(int i=0; i<VariableInfo::numVariables; i++)
			sizes[i] = VariableInfo::variable_info[i].size;
		return sizes;
	}

	using jacobian_bs_t = BS::BSMatrix<scalar_t, 
					          FunctionInfo::numFunctionCalls, VariableInfo::numVariables,
					          outputSize, inputSize,
					          FunctionInfo::totalNumArgs, nnzBlockColumns, nnzJacobian>;
	jacobian_bs_t jacobian_bs;

	// Locations to write the jacobian blocks store in argument call-order
	Eigen::Vector<int, FunctionInfo::totalNumArgs> jacobian_block_indices;

	void addRowBlock(int row, Eigen::VectorX<int> args)
	{
		std::sort(std::begin(args), std::end(args), std::less<int>());
		std::cout << "Adding denseBlock at " << row << std::endl;
		for(auto arg : args) jacobian_bs.addDenseBlock(row, arg);
	}

	/**
	* Compute the Jacobian block indices for the function in the order of the 
	* arguments.
	* 
	* i.e., if our function is f(x,y), then this returns the block indices ix, iy
	* such that block index of jacobian_f(x) is ix, and jacobian_f(y) is iy
	*/
	void getJacobianBlockIndices(const int row, const Eigen::Ref<const Eigen::VectorX<int>>& args, 
															 Eigen::Ref<Eigen::VectorX<int>> offsets)
	{
		for(int i=0; i<args.rows(); i++)
			offsets[i] = jacobian_bs.getBlockIndex(row, args[i]);
	}

	void initialize_jacobian()
	{
		// Don't really understand why, but we need to copy the compile-time array
		// into a run-time one or the compiler complains here...
		Eigen::Vector<int, FunctionInfo::totalNumArgs> functionArgs;
		for(int i=0; i<FunctionInfo::totalNumArgs; i++)
		 	functionArgs[i] = FunctionInfo::functionArguments[i];

    	// Fill in the nonZero blocks
    	int arg_index = 0;
    	int num_args = 0;
    	int blockRow = 0;
    	auto l = { (
    		num_args = std::tuple_element<functionCalls, Functions>::type::num_input_vars,
				addRowBlock(blockRow, functionArgs.segment(arg_index, num_args)),
				arg_index += num_args,
				blockRow++,
				0)...};

    	jacobian_bs.finalize();

    	// Grab the indices to the start of each non-zero block in function call order
    	arg_index = 0;
    	blockRow = 0;
    	auto l2 = { (
    		num_args = std::tuple_element<functionCalls, Functions>::type::num_input_vars,
				getJacobianBlockIndices(blockRow, 
															  functionArgs.segment(arg_index, num_args), 
															  jacobian_block_indices.segment(arg_index, num_args)),
				arg_index += num_args,
				blockRow++,
				0)...};
	}


public:
	Eigen::SparseMatrix<scalar_t> &jacobian;


/** Everything to do with the hessian
 * 
 * For the hessian, we're interpreting this vector function as a weighted sum
 * of scalar functions
 *   f(x) = sum_i weights_i * f_i(x)
 * 
 */
private: 

	// using hessian_bs_t = BS::BSMatrix<scalar_t, 
	// 				          VariableInfo::numVariables, VariableInfo::numVariables,
	// 				          inputSize, inputSize,
	// 				          FunctionInfo::hessian_nnzBlocks,
	// 				          FunctionInfo::hessian_nnzBlockColumns,
	// 				          FunctionInfo::hessian_nnzEstimate>;
	// hessian_bs_t hessian_bs;

public:
	// Eigen::SparseMatrix<scalar_t> &hessian;

private:
	template<typename f, int... Is>
	void call_F(
		param_t& param,
		const Eigen::Ref<const input_t>& x,
		Eigen::Ref<Eigen::Vector<scalar_t, f::num_outputs>> out,
		int index,  // Index into the argInfo vector
		std::integer_sequence<int, Is...>)
	{
		// Pull the variables from the functionArguments list
		// We have to write this out in full here because we're expanding the parameter pack to generate
		// the arguments
		// This should be entirely compiled out since everything is static constexpr
		out = f::eval(param, 
						x.segment(
							VariableInfo::variable_info[FunctionInfo::functionArguments[index + Is]].offset,
							VariableInfo::variable_info[FunctionInfo::functionArguments[index + Is]].size)...);
	}

	// Evaluate the function and copy the jacobian into jacobian_BS
	template<typename f, int... Is>
	void call_jacobian_F(
		param_t& param,
		const Eigen::Ref<const input_t>& x,
		Eigen::Ref<Eigen::Vector<scalar_t, f::num_outputs>> out,
		int index,  // Index into the argInfo vector
		std::integer_sequence<int, Is...>)
	{
		typename f::jacobian_return_t jac = f::jac(param, 						
						x.segment(
							VariableInfo::variable_info[FunctionInfo::functionArguments[index + Is]].offset,
							VariableInfo::variable_info[FunctionInfo::functionArguments[index + Is]].size)...);

		out = jac.val;

		// Copy the jacobian into the right blocks
		int offset = 0;
		auto l = {(
			jacobian_bs.setBlockByIndex(jacobian_block_indices[index + Is], 
										jac.jacobian.middleCols(offset,
										VariableInfo::variable_info[FunctionInfo::functionArguments[index + Is]].size)),
			offset += VariableInfo::variable_info[FunctionInfo::functionArguments[index + Is]].size,
			0
			)...};
	}

public:
	/**
	 * Evalue the Function for the variable x
	 */
	void eval(param_t &param, const Eigen::Ref<const input_t>& x, Eigen::Ref<output_t> out)
	{
		int output_index = 0;
		int arg_index = 0;

		/*
		 * Iterate over each function in the set and evaluate.
		 */		 
		auto l = {(
			call_F<typename std::tuple_element<functionCalls, Functions>::type>(
				param, 
				x,
				out.segment(output_index, std::tuple_element<functionCalls, Functions>::type::num_outputs),
				arg_index,
				std::make_integer_sequence<int, std::tuple_element<functionCalls, Functions>::type::num_input_vars>()),

			arg_index += std::tuple_element<functionCalls, Functions>::type::num_input_vars,
			output_index += std::tuple_element<functionCalls, Functions>::type::num_outputs,
			0)...};
	}

	/**
	 * Evalue the Function and its jacobian for the variable x
	 * 
	 * The jacobian member is updated
	 */
	void eval_jacobian(
		param_t &param,
		const Eigen::Ref<const input_t>& x,
		Eigen::Ref<output_t> out)
	{
		int output_index = 0;
		int arg_index = 0;

		auto l = {(
			call_jacobian_F<typename std::tuple_element<functionCalls, Functions>::type>(
				param, 
				x,
				out.segment(output_index, std::tuple_element<functionCalls, Functions>::type::num_outputs),
				arg_index,
				std::make_integer_sequence<int, std::tuple_element<functionCalls, Functions>::type::num_input_vars>()),

			arg_index += std::tuple_element<functionCalls, Functions>::type::num_input_vars,
			output_index += std::tuple_element<functionCalls, Functions>::type::num_outputs,
			0)...};
	}

	/**
	 * Computes the hessian of the function f(x) = sum_i weights_i * f_i(x)
	 */
	void eval_hessian(
		param_t &param,
		const Eigen::Ref<const input_t>& x,
		const Eigen::Ref<const output_t>& weights,
		Eigen::Ref<output_t> out)
	{
		int output_index = 0;
		int arg_index = 0;

		// H.array() = 0;

		// auto l = {(
		// 	call_hessian_F<typename std::tuple_element<functionCalls, Functions>::type>(
		// 		param, 
		// 		x,
		// 		out.segment(output_index, std::tuple_element<functionCalls, Functions>::type::num_outputs),
		// 		arg_index,
		// 		std::make_integer_sequence<int, std::tuple_element<functionCalls, Functions>::type::num_input_vars>()),

		// 	arg_index += std::tuple_element<functionCalls, Functions>::type::num_input_vars,
		// 	output_index += std::tuple_element<functionCalls, Functions>::type::num_outputs,
		// 	0)...};
	}

	Function() 
		:	jacobian_bs(rowSizes(), colSizes()), jacobian(jacobian_bs.S)
			// hessian_bs(colSizes(), colSizes()), hessian(hessian_bs.S)
	{
		initialize_jacobian();
		// initialize_hessian();
	}
};

};