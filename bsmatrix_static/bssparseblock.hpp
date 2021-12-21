#ifndef __BSSPARSEBLOCK_HPP
#define __BSSPARSEBLOCK_HPP

#include "bsblockbase.hpp"
#include "Eigen/Sparse"

namespace BS
{

template<typename Scalar_, int rows_, typename col_>
struct SparseBlock : Block<SparseBlock<Scalar_, rows_, col_>>
{
	using Derived = SparseBlock<Scalar_, rows_, col_>;
    using col = typename BlockTraits<Derived>::col;
    using dense_matrix_t = typename BlockTraits<Derived>::dense_matrix_t;
    using Scalar = typename BlockTraits<Derived>::Scalar;
	static const int rows = BlockTraits<Derived>::rows;
	static const int cols = BlockTraits<Derived>::cols;

	// Sparse storage
	Eigen::SparseMatrix<Scalar> m_matrix;

	SparseBlock()
	{
		m_matrix.resize(rows, cols);
	}

	inline void toDense_impl(Eigen::Ref<dense_matrix_t> out)
	{
		out = dense_matrix_t(m_matrix);
	}

	static void info()
	{
		std::cout << "    Sparseblock : " << rows << " x " << cols << " | column " << type_name<col>() << std::endl;
	}	

	inline std::size_t nonZeros_impl()
	{
		return m_matrix.nonZeros();
	}
};

// BlockTraits specialization for DenseBlock
template<typename Scalar_, int rows_, typename col_>
struct BlockTraits<SparseBlock<Scalar_, rows_, col_> > {
	using col = col_;
	using Scalar = Scalar_;
	static const int rows = rows_;
	static const int cols = col::len;
	using dense_matrix_t = Eigen::Matrix<Scalar, rows, cols>;
};

};

#endif // __BSSPARSEBLOCK_HPP