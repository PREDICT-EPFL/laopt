#ifndef __BSZEROBLOCK_HPP
#define __BSZEROBLOCK_HPP

#include "bsblockbase.hpp"

namespace BS
{

template<typename Scalar_, int rows_, typename col_>
struct ZeroBlock : Block<ZeroBlock<Scalar_, rows_, col_>>
{
	using Derived = ZeroBlock<Scalar_, rows_, col_>;
    using col = typename BlockTraits<Derived>::col;
    using dense_matrix_t = typename BlockTraits<Derived>::dense_matrix_t;
	static const int rows = BlockTraits<Derived>::rows;
	static const int cols = BlockTraits<Derived>::cols;

	inline void toDense_impl(Eigen::Ref<dense_matrix_t> out)
	{
		out = dense_matrix_t::Zero();
	}

	static void info()
	{
		std::cout << "    ZeroBlock : " << rows << " x " << cols << " | column " << type_name<col>() << std::endl;
	}	

	constexpr inline std::size_t nonZeros_impl()
	{
		return 0;
	}
};

// BlockTraits specialization for ZeroBlock
template<typename Scalar_, int rows_, typename col_>
struct BlockTraits<ZeroBlock<Scalar_, rows_, col_> > {
	using Scalar = Scalar_;
	using col = col_;
	static const int rows = rows_;
	static const int cols = col::len;
	using dense_matrix_t = Eigen::Matrix<Scalar, rows, cols>;
};

};

#endif // #__BSZEROBLOCK_HPP
