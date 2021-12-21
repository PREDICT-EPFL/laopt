#ifndef __BSDENSEBLOCK_HPP
#define __BSDENSEBLOCK_HPP

#include "bsblockbase.hpp"
#include "Eigen/Dense"

namespace BS
{

template<typename Scalar, int rows_, typename col_>
struct DenseBlock : Block<DenseBlock<Scalar, rows_, col_>>
{
	using Derived = DenseBlock<Scalar, rows_, col_>;
    using col = typename BlockTraits<Derived>::col;
    using dense_matrix_t = typename BlockTraits<Derived>::dense_matrix_t;
	static const int rows = BlockTraits<Derived>::rows;
	static const int cols = BlockTraits<Derived>::cols;

	// Dense storage
	dense_matrix_t m_matrix;

	EIGEN_STRONG_INLINE void toDense_impl(Eigen::Ref<dense_matrix_t> out)
	{
		out = m_matrix;
	}

	static void info()
	{
		// std::cout << "    Denseblock : " << rows << " x " << cols << " | column " << type_name<col>() << std::endl;
	}	

	constexpr inline std::size_t nonZeros_impl()
	{
		return rows * cols;
	}


	/* Copy column and innerIndex into given vector
	 * return number of elements copied (nnz of this column)
	 */
	template<typename StorageIndex>
	EIGEN_STRONG_INLINE int toSparseColumn_impl(
		std::size_t col,         // Which column of this matrix we want
		std::size_t row_offset,  // First index of the column in the block sparse matrix
		Scalar* x,       // Target to copy the data column to
		StorageIndex* innerIndex) // Target to copy the inner index to
	{
		memcpy(x, m_matrix.col(col).data(), sizeof(Scalar) * rows);
		for(int i=0; i<rows; i++)
			innerIndex[i] = row_offset + i;
		return rows;
	}

};

// BlockTraits specialization for DenseBlock
template<typename Scalar_, int rows_, typename col_>
struct BlockTraits<DenseBlock<Scalar_, rows_, col_> > {
	using Scalar = Scalar_;
	using col = col_;
	static const int rows = rows_;
	static const int cols = col::len;
	using dense_matrix_t = Eigen::Matrix<Scalar, rows, cols>;
};

};

#endif // __BSDENSEBLOCK_HPP