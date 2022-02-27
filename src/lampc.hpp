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

/**
 * Copy a sub-matrix into another, larger matrix
 * 
 * The blocks array must have been built with the build_copy_sequence function 
 * with the same sparsity structures.
 * 
 * Note: Only copies the non-zero elements of source. 
 * The zero elements in source are untouched in target.
 */
template<typename T>
void copy_submatrix(Eigen::SparseMatrix<T> &target, const Eigen::SparseMatrix<T> &source, 
					const sparseblock_info<int> *blocks, const int num_blocks)
{
	T* targetPtr = target.valuePtr();
	T* sourcePtr = source.valuePtr();
	for(int i=0; i<num_blocks; i++)
	{
		memcpy(targetPtr + blocks[i].target_index, sourcePtr, sizeof(T) * blocks[i].block_length);
		sourcePtr += blocks[i].block_length;
	}
};

template<typename T>
void copy_submatrix(Eigen::SparseMatrix<T> &target, const Eigen::MatrixX<T> &source, 
					const sparseblock_info<int> *blocks, const int num_blocks)
{
	T* targetPtr = target.valuePtr();
	const T* sourcePtr = source.data();
	for(int i=0; i<num_blocks; i++)
	{
		memcpy(targetPtr + blocks[i].target_index, sourcePtr, sizeof(T) * blocks[i].block_length);
		sourcePtr += blocks[i].block_length;
	}
};


/** 
 * Copy the LAMPC_Function output into the right place in the jacobian
 */
template<int len, typename scalar_t,
		 typename out_t, typename jacobian_t,
		 typename jacobian_output_t>
inline void setJ(out_t &out, jacobian_t &jacobian, // Values to write into
			 	 const int offset, // Offset into out for the evaluation
			 	 const sparseblock_info<int> *jac_seq, // Writing sequence for the jacobian
			 	 const int num_blocks, 
			 	 const jacobian_output_t &J) // Input
{
      out.template segment<len>(offset) = J.val;
      copy_submatrix<scalar_t>(jacobian, J.jacobian, jac_seq, num_blocks);
}



#endif // __LAMPC__HPP