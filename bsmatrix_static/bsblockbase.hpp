#ifndef __BSBLOCKBASE_HPP
#define __BSBLOCKBASE_HPP

#include "Eigen/Dense"
#include "lampc_utility.hpp"

namespace BS
{

// Tag class for columns
template<typename Scalar, int len_>
struct Column
{
	static const int len = len_;
	static const std::string name;
};

template <typename Derived> 
struct BlockTraits;

/** Base for all blocks
 */
template<typename Derived>
struct Block
{
    using col = typename BlockTraits<Derived>::col;
    using dense_matrix_t = typename BlockTraits<Derived>::dense_matrix_t;
    using Scalar = typename BlockTraits<Derived>::Scalar;
	static const int rows = BlockTraits<Derived>::rows;
	static const int cols = BlockTraits<Derived>::cols;

	EIGEN_STRONG_INLINE void toDense(Eigen::Ref<dense_matrix_t> out)
	{
        static_cast<Derived*>(this)->toDense_impl(out);
	}

	/* Copy column and innerIndex into given vector
	 * return number of elements copied (nnz of this column)
	 */
	template<typename StorageIndex>
	EIGEN_STRONG_INLINE int toSparseColumn(
		std::size_t col,         // Which column of this matrix we want
		std::size_t row_offset,  // First index of the column in the block sparse matrix
		Scalar* x,       // Target to copy the data column to
		StorageIndex* innerIndex) // Target to copy the inner index to
	{
		return static_cast<Derived*>(this)->toSparseColumn_impl(col, row_offset, x, innerIndex);
	}

	// inline operator=(Eigen::Ref<other> )

	// inline void setBlock(const Eigen::Ref<const dense_matrix_t>& in)
	// {
	// 	static_cast<Derived*>(this)->setBlock_impl(in);
	// }

	static void info()
	{
		Derived::info();
	}

	inline std::size_t nonZeros()
	{
		return static_cast<Derived*>(this)->nonZeros_impl();
	}
};

};

#endif // __BSBLOCKBASE_HPP
