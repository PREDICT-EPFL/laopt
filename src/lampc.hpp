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
struct seqinfo
{
	int index;
	int length;
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
	 * 
	 * source is a contiguous vector (either source.valuePtr or source.data if sparse or dense)
	 * blocks[num_blocks] stores the sequence of copies required to copy source into a pre-specified
	 * location in target (location specified during construction in build_copy_sequence))
	 * 
	 * block[i] gives the index into target.valuePtr where source.valuePtr[j] should be copied
	 */
	static void copy_submatrix(Eigen::Ref<Eigen::SparseMatrix<scalar_t>> target, 
						const Eigen::SparseMatrix<scalar_t> &source, 
						const seqinfo *blocks, const int num_blocks)
	{
		scalar_t* targetPtr = target.valuePtr();
		const scalar_t* sourcePtr = source.valuePtr();
		for(int i=0; i<num_blocks; i++)
		{
			// for(int j=0; j<blocks[i].length; j++) sourcePtr[j] = i+2; // For debugging
			memcpy(targetPtr + blocks[i].index, sourcePtr, sizeof(scalar_t) * blocks[i].length);
			sourcePtr += blocks[i].length;
		}
	};

	static void copy_submatrix(Eigen::Ref<Eigen::SparseMatrix<scalar_t>> target, 
						const Eigen::MatrixX<scalar_t> &source, 
						const seqinfo *blocks, const int num_blocks)
	{
		scalar_t* targetPtr = target.valuePtr();
		const scalar_t* sourcePtr = source.data();
		for(int i=0; i<num_blocks; i++)
		{
			memcpy(targetPtr + blocks[i].index, sourcePtr, sizeof(scalar_t) * blocks[i].length);
			sourcePtr += blocks[i].length;
		}
	};
};

template<typename scalar_t, typename out_t, typename jacobian_t>
struct function_util_t : public abstract_function_t<scalar_t>
{
	/** 
	 * Copy the block jacobian output into the right place in the jacobian
	 * 
	 * Add the jacobian and value of function fi to jacobian and out
	 * 
	 * jacobian_output_t must containt two members
	 * - val = vector of length jacobian_output_t::num_outputs
	 * - jacobian = matrix of size jacobian_output_t::num_outputs x sum_i var_info[i].block_length
	 * 
	 * jac_seq specifies the locations and sizes of the num_vars sets of columns of the jacobian.
	 * for the i'th variable, we add grad += <wi, jacobian(:,var_info[i].offset:var_info[i].block_length)
	 */
	template<typename jacobian_output_t>
	static inline void setJ(Eigen::Ref<out_t> out, Eigen::Ref<jacobian_t> jacobian, // Values to write into
				 	 const int offset, // Offset into out for the evaluation
				 	 const seqinfo *jac_seq, // Writing sequence for the jacobian
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
	 * Accumulate the block gradient into the gradient
	 * 
	 * Add the gradient and value of function <wi, fi> to grad and val
	 * 
	 * jacobian_output_t must containt two members
	 * - val = vector of length jacobian_output_t::num_outputs
	 * - jacobian = matrix of size jacobian_output_t::num_outputs x sum_i var_info[i].block_length
	 * 
	 * var_info specifies the locations and sizes of the num_vars sets of columns of the jacobian.
	 * for the i'th variable, we add grad += <wi, jacobian(:,var_info[i].offset:var_info[i].block_length)
	 */
	template<typename jacobian_output_t>
	static inline void accGrad(scalar_t &val, Eigen::Ref<gradient_t> grad, 
				  const seqinfo *var_info, int num_vars, // Offsets of the vars into grad
				  const Eigen::Ref<const Eigen::Vector<scalar_t, jacobian_output_t::num_outputs>> w, 
				  const jacobian_output_t &J)
	{
		val += w.dot(J.val);
		auto g = w.transpose() * J.jacobian;
		int offset = 0;
		int varlen = 0;
		for(int i=0; i<num_vars; i++)
		{
			varlen = var_info[i].length;
			grad.segment(var_info[i].index, varlen) += g.segment(offset, varlen);
			offset += varlen;
		}
	}
};

#endif // __LAMPC__HPP