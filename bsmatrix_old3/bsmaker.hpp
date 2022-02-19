#ifndef __BSMAKER_HPP
#define __BSMAKER_HPP

#include "bsmatrix.hpp"

namespace BS {

/**
 * Generator object to do the calculations for producing a BSMatrix
 */
template<typename Scalar, typename ConstraintTuple, typename VariableTuple>
class BSMatrixFactory;

template<typename Scalar, typename... Constraints, typename... Variables>
class BSMatrixFactory<Scalar, std::tuple<Constraints...>, std::tuple<Variables...>> 
{
  static constexpr int numRows = detail::sum_int_template<Constraints::size...>();
  static constexpr int numColumns = detail::sum_int_template<Variables::size...>();
  static constexpr int numBlockRows = sizeof...(Constraints);
  static constexpr int numBlockColumns = sizeof...(Variables);

  // Total number of columns in non-zero blocks
  static constexpr int nnzBlockColumns = detail::sum_int_template<Constraints::totalVariables...>();

  // Number of non-zero blocks
  static constexpr int nnzBlocks = detail::sum_int_template<Constraints::numVariables...>();

  // Number of non-zeros assuming dense blocks
  static constexpr int nnzEstimate = detail::sum_int_template<Constraints::size * Constraints::numVariables...>(); 

  // The type that we're producing
  using BSType_ = BS::BSMatrix<Scalar, 
  														 numBlockRows, numBlockColumns, 
  														 numRows, numColumns,
  														 nnzBlocks,
  														 nnzBlockColumns,
  														 nnzEstimate>;

	template<typename... Ts>
	static constexpr Eigen::Vector<int, sizeof...(Ts)> makeSizeArray()
	{
		Eigen::Vector<int, sizeof...(Ts)> sizes;
		int i = 0;
		auto l = { (sizes[i] = Ts::size, i++, 0)... };
		return sizes;
	}

	template<typename... local_vars>
	static constexpr void makeBlocks(BSType_& M, int blockRow, std::tuple<local_vars...>)
	{
		// Get indices of arguments
		Eigen::Vector<int, sizeof...(local_vars)> columns{detail::get_index<local_vars, Variables...>()...};
		std::sort(std::begin(columns), std::end(columns), std::less<int>());

		for(auto column : columns) M.addDenseBlock(blockRow, column);
			addDenseBlock(int blockRow, int blockColumn)
	}

public:
	using BSType = BSType_;
	using constraint_t = Eigen::Vector<Scalar, numRows>;
	using variable_t = Eigen::Vector<Scalar, numColumns>;

    static constexpr BSType make()
    {
    	// Create our BSMatrix
    	BSType M(makeSizeArray<Constraints...>(), makeSizeArray<Variables...>());

    	// Fill in the nonZero blocks
    	// Iterate over constraints
    	int blockRow = 0;
    	auto l = { (
				makeBlocks(M, blockRow, typename Constraints::varTuple()),
				blockRow++,
				0)...
    	};

    	M.finalize();

	    return M;
    }
};




/**
 * Compiles the tags into indices
 * 
 * Constraints here must be derived from FunctionConstraint
 */
template<typename Scalar, typename param_t, typename ConstraintTuple, typename VariableTuple>
class FunctionFactory;

template<typename Scalar, typename param_t, typename... Constraints, typename... Variables>
class FunctionFactory<Scalar, param_t, std::tuple<Constraints...>, std::tuple<Variables...>> 
{
  static constexpr int numOutputs = detail::sum_int_template<Constraints::size...>();
  static constexpr int numInputs = detail::sum_int_template<Variables::size...>();

  using VariableTuple = std::tuple<Variables...>;
  using ConstraintTuple = std::tuple<Constraints...>;
  using jacobian_factory_t = BSMatrixFactory<Scalar, ConstraintTuple, VariableTuple>;
  using jacobian_BS_t = typename jacobian_factory_t::BSType;

  // Default constructor is private => cannot instantiate
  FunctionFactory() {}

  /**
   * Compute the Jacobian block indices for the function in the order of the 
   * arguments.
   * 
   * i.e., if our function is f(x,y), then this returns the block indices ix, iy
   * such that block index of jacobian_f(x) is ix, and jacobian_f(y) is iy
   */
  template<int numArgs>
  static constexpr void getJacobianBlockIndices(int blockRow, 
  																							const Eigen::Ref<const Eigen::Vector<int, numArgs>>& blockColumns, 
  																							jacobian_BS_t &J, 
  																							Eigen::Ref<Eigen::Vector<int, numArgs>> out)
  {
  	for(int i=0; i<numArgs; i++)
  		out[i] = J.getBlockIndex(blockRow, blockColumns[i]);
  }

public:

	// The produced FunctionSet type
	using type = FunctionSet<Scalar, param_t, numInputs, numOutputs, 
										jacobian_BS_t,
										typename Constraints::Function...>;


  /** 
   * Convert from tag-representation to indices. 
   * 
   * If editing: This function must be able to be run at compile-time, 
   * or the code will be massive (check the debug build size)
   */
  static constexpr type make()
  {
		// Get the indices of the arguments in order
		Eigen::Vector<int, type::totalNumArgs> indices;
		int ind = 0;
		auto l1 = {(
			indices.segment(ind, Constraints::numVariables) = Constraints::template get_indices<VariableTuple>(),
			ind += Constraints::numVariables,
			0
			)...};

		// Build a vector of offsets into the dense variable vector
		Eigen::Vector<int, sizeof...(Variables)> var_offsets;
		ind = 0;
		int offset = 0;
		auto l2 = {(
			var_offsets[ind] = offset,
			offset += Variables::size, 
			ind++,
			0)...};

		// Build a vector of variable lengths
		Eigen::Vector<int, sizeof...(Variables)> var_lengths;
		ind = 0;
		auto l3 = {(
			var_lengths[ind] = Variables::size, 
			ind++,
			0)...};

		// Construct the jacobian here so that we can retrieve the block indices
		jacobian_BS_t J = jacobian_factory_t::make();

		// Get the block indices for each jacobian block
		Eigen::Vector<int, type::totalNumArgs> jacobian_indices;
		ind = 0;
		auto l4 = {(
			getJacobianBlockIndices<Constraints::numVariables>
														(detail::get_index<Constraints>(ConstraintTuple()), // Row
														 Constraints::template get_indices<VariableTuple>(), // Columns
														 J,
														 jacobian_indices.segment(ind, Constraints::numVariables)),
			ind += Constraints::numVariables,
			0
			)...};

		// Use the tag-info to fill in the argInfo data
		Eigen::Vector<typename type::argInfo_t, type::totalNumArgs> argInfo;

		for(int i=0; i<type::totalNumArgs; i++)
		{
			argInfo[i].index = indices[i];
			argInfo[i].offset = var_offsets[indices[i]];
			argInfo[i].len = var_lengths[indices[i]];

			argInfo[i].jacobianBlockIndex = jacobian_indices[i];
		}

  	type f(argInfo, J);
  	return f;
  }
};


/**
 * The compiled problem class.
 * 
 * eq_t, ineq_t, obj_t are all FunctionSets
 * 
 */
template<typename equalities_t_, typename inequalities_t_, typename objective_t_>
class Problem
{
public:
	using equalities_t = equalities_t_;
	using inequalities_t = inequalities_t_;
	using objective_t = objective_t_;

	equalities_t equalities;
	inequalities_t inequalities;
	objective_t objective;

	Problem(equalities_t equalities_, inequalities_t inequalities_, objective_t objective_)
	: equalities(equalities_), inequalities(inequalities_), objective(objective_)
	{}
};


template<typename scalar_t, typename param_t, 
				 typename variables_tuple, typename eq_tuple, typename ineq_tuple, typename obj_tuple>
class ProblemCompiler
{
    using eq_factory = FunctionFactory<scalar_t, param_t, eq_tuple, variables_tuple>;
    using ineq_factory = FunctionFactory<scalar_t, param_t, ineq_tuple, variables_tuple>;
    using obj_factory = FunctionFactory<scalar_t, param_t, obj_tuple, variables_tuple>;

public:
    using problem_t = Problem<typename eq_factory::type, typename ineq_factory::type, typename obj_factory::type>;

    static problem_t make()
    {
    	return problem_t(eq_factory::make(), ineq_factory::make(), obj_factory::make());
    }
};


};

#endif