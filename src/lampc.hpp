#ifndef __LAMPC__HPP
#define __LAMPC__HPP

// #include "map.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>

// #include "utils/helpers.hpp"

#include "lampc_utility.hpp"
#include "lampc_function.hpp"
// #include "lampc_impl.hpp"
#include "la_functionset.hpp"

#define SEG(len,offset) template segment<len>(offset)

/**
 * Information about location of a variable in a vector
 */
struct variable_info_t
{
	int offset;
	int size;
	std::string name;
};

/**
 * Copy information for a sparse block matrix
 */
template<typename T>
struct sparseblock_info
{
	// T source_index;
	T target_index;
	T block_length;
};


/**
 * Operations to add
 * - zero a sub-matrix
 * - add a source matrix to the target
 * - overwrite a sub-matrix, while zeroing elements that aren't in source
 */




template<typename scalar_t>
struct abstract_function_t
{
	/**
	 * Copy a sub-matrix into another, larger matrix
	 * 
	 * The blocks array must have been built with the build_copy_sequence function 
	 * with the same sparsity structures.
	 * 
	 * Note: Only copies the non-zero elements of source. 
	 * The zero elements in source are untouched in target.
	 */
	static void copy_submatrix(Eigen::Ref<Eigen::SparseMatrix<scalar_t>> target, 
						const Eigen::SparseMatrix<scalar_t> &source, 
						const sparseblock_info<int> *blocks, const int num_blocks)
	{
		scalar_t* targetPtr = target.valuePtr();
		scalar_t* sourcePtr = source.valuePtr();
		for(int i=0; i<num_blocks; i++)
		{
			memcpy(targetPtr + blocks[i].target_index, sourcePtr, sizeof(scalar_t) * blocks[i].block_length);
			sourcePtr += blocks[i].block_length;
		}
	};

	static void copy_submatrix(Eigen::Ref<Eigen::SparseMatrix<scalar_t>> target, 
						const Eigen::MatrixX<scalar_t> &source, 
						const sparseblock_info<int> *blocks, const int num_blocks)
	{
		scalar_t* targetPtr = target.valuePtr();
		const scalar_t* sourcePtr = source.data();
		for(int i=0; i<num_blocks; i++)
		{
			memcpy(targetPtr + blocks[i].target_index, sourcePtr, sizeof(scalar_t) * blocks[i].block_length);
			sourcePtr += blocks[i].block_length;
		}
	};
};

template<typename scalar_t, typename out_t, typename jacobian_t>
struct function_util_t : public abstract_function_t<scalar_t>
{
	/** 
	 * Copy the LAMPC_Function output into the right place in the jacobian
	 */
	template<typename jacobian_output_t>
	static inline void setJ(Eigen::Ref<out_t> out, Eigen::Ref<jacobian_t> jacobian, // Values to write into
				 	 const int offset, // Offset into out for the evaluation
				 	 const sparseblock_info<int> *jac_seq, // Writing sequence for the jacobian
				 	 const int num_blocks, 
				 	 const jacobian_output_t &J) // Input
	{
	      out.template segment<jacobian_output_t::num_outputs>(offset) = J.val;
	      abstract_function_t<scalar_t>::copy_submatrix(jacobian, J.jacobian, jac_seq, num_blocks);
	}
};

template<typename scalar_t, typename gradient_t, typename weight_t>
struct weightedsum_util_t : public abstract_function_t<scalar_t>
{
	/** 
	 * Copy the LAMPC_Function output into the gradient
	 */
	template<typename jacobian_output_t>
	static inline void setGrad(scalar_t &val, Eigen::Ref<gradient_t> grad, 
				  const sparseblock_info<int> *var_info, int num_vars, // Offsets of the vars into grad
				  const Eigen::Ref<const Eigen::Vector<scalar_t, jacobian_output_t::num_outputs>> w, 
				  const jacobian_output_t &J)
	{
		val += w.dot(J.val);
		auto g = w.transpose() * J.jacobian;
		int offset = 0;
		int varlen = 0;
		for(int i=0; i<num_vars; i++)
		{
			varlen = var_info[i].block_length;
			grad.segment(var_info[i].target_index, varlen) += g.segment(offset, varlen);
			offset += varlen;
		}
	}
};

#endif // __LAMPC__HPP