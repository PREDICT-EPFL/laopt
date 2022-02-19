#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "Eigen/Core"

#include "type_name.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>


typedef std::chrono::time_point<std::chrono::system_clock> time_point;
static time_point get_time()
{
    /** OS dependent */
#ifdef __APPLE__
    return std::chrono::system_clock::now();
#elif defined _WIN32 || defined _WIN64 || defined _MSC_VER
    return std::chrono::system_clock::now();
#else
    return std::chrono::high_resolution_clock::now();
#endif
}


// TODO: CRTF to make block interfaces consistent

// namespace internal
// {

// template<typename T> 
// struct BlockTraits
// {};

// template<typename Scalar_, int Rows_, int Cols_>
// struct BlockTraits<BlockZero<Scalar_, Rows_, Cols_> >
// {
// };

// }

// template<typename Derived>
// class BlockBase
// {
// 	using Scalar = Derived::Scalar;

// public:
// 	EIGEN_STRONG_INLINE int getColumn(
// 		int col,         // Which column of this matrix we want
// 		int row_offset,  // First index of the column in the block sparse matrix
// 		Scalar* x,       // Target to copy the data column to
// 		int* innerIndex) // Target to copy the inner index to
// 	{
// 		return 0;
// 	}
// };


template<typename Scalar, int Rows_ = Eigen::Dynamic, int Cols_ = Eigen::Dynamic>
class BlockZero
{
	int m_rows;
	int m_cols;

public:
	BlockZero() : m_rows(Rows_), m_cols(Cols_)
	{}

	EIGEN_STRONG_INLINE void resize(int rows, int cols)
	{
		assert((Rows_ == Eigen::Dynamic || Rows_ == rows) && "Cannot resize a static matrix");
		assert((Cols_ == Eigen::Dynamic || Cols_ == cols) && "Cannot resize a static matrix");
		m_rows = rows;
		m_cols = cols;
	}

	EIGEN_STRONG_INLINE int getColumn(
		int col,         // Which column of this matrix we want
		int row_offset,  // First index of the column in the block sparse matrix
		Scalar* x,       // Target to copy the data column to
		int* innerIndex) // Target to copy the inner index to
	{
		return 0;
	}

	constexpr EIGEN_STRONG_INLINE int nnz() {return 0;}

	EIGEN_STRONG_INLINE int rows() {return m_rows;}
	EIGEN_STRONG_INLINE int cols() {return m_cols;}
};

template<typename matrix_t>
class BlockDense
{
	using Scalar = typename matrix_t::Scalar;

	///// TODO: Prevent access to matrix so that it can't be resized

public:

	matrix_t matrix;

	/* Copy column and innerIndex into given vector
	 * return number of elements copied (nnz of this column)
	 */
	EIGEN_STRONG_INLINE int getColumn(
		int col,         // Which column of this matrix we want
		int row_offset,  // First index of the column in the block sparse matrix
		Scalar* x,       // Target to copy the data column to
		int* innerIndex) // Target to copy the inner index to
	{
		int num_elements = matrix.rows();
		memcpy(x, matrix.col(col).data(), sizeof(Scalar) * num_elements);
		for(int i=0; i<num_elements; i++)
			innerIndex[i] = row_offset + i;
		return num_elements;
	}

	EIGEN_STRONG_INLINE void resize(int rows, int cols)
	{
		matrix.resize(rows, cols);
	}

	EIGEN_STRONG_INLINE int nnz()
	{
		return matrix.rows() * matrix.cols();
	}

	EIGEN_STRONG_INLINE int rows() {return matrix.rows();}
	EIGEN_STRONG_INLINE int cols() {return matrix.cols();}
};

template<typename Scalar>
class BlockSparse
{
	///// TODO: Prevent access to matrix so that it can't be resized

public:
	Eigen::SparseMatrix<Scalar> matrix;

	EIGEN_STRONG_INLINE int getColumn(
		int col,         // Which column of this matrix we want
		int row_offset,  // First index of the column in the block sparse matrix
		Scalar* x,       // Target to copy the data column to
		int* innerIndex) // Target to copy the inner index to
	{
		auto outer_ptr = matrix.outerIndexPtr();
		auto inner_ptr = matrix.innerIndexPtr();

		int num_elements = outer_ptr[col+1] - outer_ptr[col];
		memcpy(x, matrix.col(col).data(), sizeof(Scalar) * num_elements);
		memcpy(innerIndex, inner_ptr + outer_ptr[col], sizeof(int) * num_elements);
		return num_elements;
	}

	EIGEN_STRONG_INLINE void resize(int rows, int cols)
	{
		matrix.resize(rows, cols);
	}

	EIGEN_STRONG_INLINE int nnz()
	{
		/////// TODO
		// Check if the matrix hasn't been initialized, and if not return NaN	
		return matrix.nnz();
	}

	EIGEN_STRONG_INLINE int rows() {return matrix.rows();}
	EIGEN_STRONG_INLINE int cols() {return matrix.cols();}
};

/////// TODO: Change this from a column to an "Inner" block column or row
template<typename Scalar, typename inner_t, typename Index>
class BSInner;

template<typename Scalar, typename inner_t, std::size_t... ind>
class BSInner<Scalar, inner_t, std::integer_sequence<std::size_t, ind...>> 
{
	inner_t m_inner;

public:

	// Copy one column in compressed sparse column format into x
	EIGEN_STRONG_INLINE int getColumn(
		int col,         // Which column of this matrix we want
		Scalar* x,       // Target to copy the data column to
		int* innerIndex) // Target to copy the inner index to
	{
		int total = 0;  // Total number of elements written
		int num;        // Number of elements written in this block
		int row = 0;    // Current row of insertion
        (void)std::initializer_list<int>{
            (
            	num = std::get<ind>(m_inner).getColumn(col, row, x, innerIndex),
            	row += std::get<ind>(m_inner).rows(),
                x += num,
                innerIndex += num,
                total += num,
                0
            )...
        };

        return total;
	}

	// Copy block column into compressed sparse format
	EIGEN_STRONG_INLINE int toSparse(
		Scalar* x,           // Target to copy the data column to
		int* innerIndex,     // Target to copy the inner index to
		int* outerIndex,     // Target to copy the inner index to
		int offset = 0)      // Index offset for the first value
	{
		int num = 0;
		int col = 0;
		for(; col<cols(); col++)
		{
			outerIndex[col] = offset;
			num += getColumn(col, x, innerIndex);
			x += num;
			innerIndex += num;
			offset += num;
		}
		outerIndex[col] = offset;
	}


	/** Compile time access to block data
	 */
	template<int blockInd>
	EIGEN_STRONG_INLINE std::tuple_element_t<blockInd, inner_t>* getBlock()
	{
		return &(std::get<blockInd>(m_inner));
	}

	EIGEN_STRONG_INLINE int nnz()
	{
		int num = 0;
        (void)std::initializer_list<int>{ 
            (
                num += std::get<ind>(m_inner).nnz(),
                0
            )...
        };

		return num;
	}


	EIGEN_STRONG_INLINE int cols() 
	{
        (void)std::initializer_list<int>{ 
            (
                assert(std::get<ind>(m_inner).cols() == std::get<0>(m_inner).cols() && 
                		"All blocks in a column must have the same number of columns"),
                0
            )...
        };

		return std::get<0>(m_inner).cols();
	}

	EIGEN_STRONG_INLINE int rows() 
	{ 
		int rows = 0;
        (void)std::initializer_list<int>{ 
            (
                rows += std::get<ind>(m_inner).rows(),
                0
            )...
        };
		return rows;
	}

	EIGEN_STRONG_INLINE int num_blocks()
	{
		return std::tuple_size<inner_t>::value;
	}

};


template<typename Scalar, typename inner_t, typename Index>
class BSMatrix;

/**
 * A collection of BSInner types
 */
template<typename Scalar, typename inner_t, std::size_t... ind>
class BSMatrix<Scalar, inner_t, std::integer_sequence<std::size_t, ind...>> 
{
	inner_t m_columns; // Columns of the block matrix

public:

	EIGEN_STRONG_INLINE int rows()
	{
        (void)std::initializer_list<int>{ 
            (
                assert(std::get<ind>(m_columns).rows() == std::get<0>(m_columns).rows() && 
                		"All columns must have the same number of rows"),
                0
            )...
        };

		return std::get<0>(m_columns).rows();
	}

	EIGEN_STRONG_INLINE int cols()
	{
		int cols = 0;
        (void)std::initializer_list<int>{ 
            (
                cols += std::get<ind>(m_columns).cols(),
                0
            )...
        };
		return cols;
	}

	EIGEN_STRONG_INLINE int block_rows()
	{
        (void)std::initializer_list<int>{ 
            (
                assert(std::get<ind>(m_columns).num_blocks() == std::get<0>(m_columns).num_blocks() && 
                		"All columns must have the same number of blocks"),
                0
            )...
        };

		return std::get<0>(m_columns).num_blocks();
	}

	EIGEN_STRONG_INLINE int block_cols()
	{
		return std::tuple_size<inner_t>::value;
	}

	EIGEN_STRONG_INLINE int nnz()
	{
		int nnz = 0;
        (void)std::initializer_list<int>{ 
            (
                nnz += std::get<ind>(m_columns).nnz(),
                0
            )...
        };
		return nnz;
	}

	template<int row, int col>
	EIGEN_STRONG_INLINE auto getBlock()
	{
		return std::get<col>(m_columns).template getBlock<row>();
	}

	EIGEN_STRONG_INLINE void toSparse(Eigen::SparseMatrix<Scalar>& S)
	{
		S.resize(rows(), cols());
		S.reserve(nnz());
		S.makeCompressed();

		auto p = S.valuePtr();
		auto i = S.innerIndexPtr();
		auto o = S.outerIndexPtr();

		int offset = 0;  // Offset into the data
		int num;

        (void)std::initializer_list<int>{ 
            (
            	o[0] = offset,
				num = std::get<ind>(m_columns).toSparse(p, i, o, offset),
				p += num,
				i += num,
				o += std::get<ind>(m_columns).cols(),
				offset += num,
                0
            )...
        };
        o[1] = offset;
    }

	// EIGEN_STRONG_INLINE int toSparse(
	// 	Scalar* x,           // Target to copy the data column to
	// 	int* innerIndex,     // Target to copy the inner index to
	// 	int* outerIndex,     // Target to copy the inner index to
	// 	int offset = 0)      // Index offset for the first value


	// 	// for(int col=0; col<colA.cols(); col++, blkCol++)
	// 	// {
	// 	// 	o[blkCol] = offset;
	// 	// 	offset += colA.getColumn(col, p + offset, i + offset);
	// 	// }

	// 	// o[blkCol] = offset;

	// }
};


int main()
{
	/**
	 * [A Z;
	 *  B C]
	 */

	// BlockDense<Eigen::Matrix<double, 2, 3>> A;
	// A.matrix.array() = 1;

	// BlockZero<double> Z;
	// Z.resize(2, 4);

	// BlockDense<Eigen::Matrix<double, 3, 3>> B;
	// B.matrix.array() = 2;

	// BlockDense<Eigen::Matrix<double, 3, 4>> C;
	// C.matrix.array() = 3;

	// std::cout << "A = \n" << A.matrix << std::endl;
	// std::cout << "B = \n" << B.matrix << std::endl;
	// std::cout << "C = \n" << C.matrix << std::endl;

	// Eigen::SparseMatrix<double> T(10, 20);
	// std::vector<Eigen::Triplet<double>> coefficients;
	// for(int row=0; row<blkA.matrix.rows(); row++)
	// 	for(int col=0; col<blkA.matrix.cols(); col++)
	// 	{
	// 		coefficients.push_back(Eigen::Triplet<double>{row, col, blkA.matrix(row, col)});
	// 	}
	// T.setFromTriplets(coefficients.begin(), coefficients.end());
	// T.makeCompressed();
	// std::cout << "T = \n" << T << std::endl;

// 	std::cout << "========================\n";

// {
// 	Eigen::SparseMatrix<double> S(10, 20);
// 	S.reserve(A.nnz() + Z.nnz() + B.nnz() + C.nnz());
// 	std::cout << "reserve = " << A.nnz() + Z.nnz() + B.nnz() + C.nnz() << std::endl;
// 	// S.resize();
// 	S.makeCompressed();
// 	auto p = S.valuePtr();
// 	auto i = S.innerIndexPtr();
// 	auto o = S.outerIndexPtr();

// 	// std::cout << "type(i) = " << type_name<decltype(i)>() << std::endl;
// 	// std::cout << "type(o) = " << type_name<decltype(o)>() << std::endl;

// 	int offset = 0;
// 	int blkCol = 0;
// 	for(int col=0; col<A.cols(); col++, blkCol++)
// 	{
// 		o[blkCol] = offset;
// 		offset += A.getColumn(col, 0, p + offset, i + offset);
// 		offset += B.getColumn(col, A.rows(), p + offset, i + offset);
// 	}
// 	for(int col=0; col<Z.cols(); col++, blkCol++)
// 	{
// 		o[blkCol] = offset;
// 		offset += Z.getColumn(col, 0, p + offset, i + offset);
// 		offset += C.getColumn(col, Z.rows(), p + offset, i + offset);
// 	}

// 	o[blkCol] = offset;

// 	std::cout << "S = \n" << S << std::endl;
// }


{
	std::cout << "===============  Static matrices  =================\n\n";

	using column_t = std::tuple<
			BlockDense<Eigen::Matrix<double, 2, 3>>,
			BlockDense<Eigen::Matrix<double, 3, 3>>>;
	using column_index = std::make_integer_sequence<std::size_t, 
									std::tuple_size<column_t>::value>;

	BSInner<double, column_t, column_index> colA;

	colA.getBlock<0>()->matrix.array() = 1;
	colA.getBlock<1>()->matrix.array() = 2;

	std::cout << "!!! nnz = " << colA.nnz() << std::endl;


	Eigen::SparseMatrix<double> S(10, 20);
	S.reserve(colA.nnz());
	std::cout << "reserve = " << colA.nnz() << std::endl;
	S.makeCompressed();
	auto p = S.valuePtr();
	auto i = S.innerIndexPtr();
	auto o = S.outerIndexPtr();

	int offset = 0;
	int blkCol = 0;
	for(int col=0; col<colA.cols(); col++, blkCol++)
	{
		o[blkCol] = offset;
		offset += colA.getColumn(col, p + offset, i + offset);
	}
	// for(int col=0; col<Z.cols(); col++, blkCol++)
	// {
	// 	o[blkCol] = offset;
	// 	offset += Z.getColumn(col, 0, p + offset, i + offset);
	// 	offset += C.getColumn(col, Z.rows(), p + offset, i + offset);
	// }

	o[blkCol] = offset;

	std::cout << "S = \n" << S << std::endl;

	std::cout << "cols = " << colA.cols() << std::endl;
	std::cout << "rows = " << colA.rows() << std::endl;
}


{
	std::cout << "===========  Dynamic matrices  =============\n\n";
	using column_t = std::tuple<
			BlockZero<double>,
			BlockDense<Eigen::MatrixX<double>>,
			BlockDense<Eigen::Matrix<double, 3, 3>>>;
	using index = std::make_integer_sequence<std::size_t, 
							std::tuple_size<column_t>::value>;

	BSInner<double, column_t, index> colA;

	colA.getBlock<0>()->resize(4, 3);
	colA.getBlock<1>()->resize(2, 3);

	// colA.getBlock<0>()->matrix.array() = 3;
	colA.getBlock<1>()->matrix.array() = 1;
	colA.getBlock<2>()->matrix.array() = 2;

	std::cout << "!!! nnz = " << colA.nnz() << std::endl;
	std::cout << "cols = " << colA.cols() << std::endl;
	std::cout << "rows = " << colA.rows() << std::endl;


	Eigen::SparseMatrix<double> S(10, 20);
	S.reserve(colA.nnz());
	std::cout << "reserve = " << colA.nnz() << std::endl;
	S.makeCompressed();
	auto p = S.valuePtr();
	auto i = S.innerIndexPtr();
	auto o = S.outerIndexPtr();

	int offset = 0;
	int blkCol = 0;
	for(int col=0; col<colA.cols(); col++, blkCol++)
	{
		o[blkCol] = offset;
		offset += colA.getColumn(col, p + offset, i + offset);
	}
	// // // for(int col=0; col<Z.cols(); col++, blkCol++)
	// // // {
	// // // 	o[blkCol] = offset;
	// // // 	offset += Z.getColumn(col, 0, p + offset, i + offset);
	// // // 	offset += C.getColumn(col, Z.rows(), p + offset, i + offset);
	// // // }

	o[blkCol] = offset;

	std::cout << "S = \n" << S << std::endl;
}


{
	std::cout << "===========  Matrices  =============\n\n";

	/**
	 * [A 0 0  | 3
	 *  0 0 B  | 2
	 *  0 C D  | 1
	 *  - - -
	 *  3 2 2
	 */

	using A_t = Eigen::MatrixX<double>;
	using B_t = Eigen::Matrix<double, 2, 2>;
	using C_t = Eigen::MatrixX<double>;
	using D_t = Eigen::Matrix<double, 1, 2>;

	using column_0_t = std::tuple<BlockDense<A_t>, BlockZero<double, 2, 3>, BlockZero<double, 1, 3>>;
	using column_0_index = std::make_integer_sequence<std::size_t, std::tuple_size<column_0_t>::value>;
	using column_0 = BSInner<double, column_0_t, column_0_index>;

	using column_1_t = std::tuple<BlockZero<double, 3, 2>, BlockZero<double, 2, 3>, BlockDense<C_t>>;
	using column_1_index = std::make_integer_sequence<std::size_t, std::tuple_size<column_1_t>::value>;
	using column_1 = BSInner<double, column_1_t, column_1_index>;

	using column_2_t = std::tuple<BlockZero<double, 3, 2>, BlockDense<B_t>, BlockDense<D_t>>;
	using column_2_index = std::make_integer_sequence<std::size_t, std::tuple_size<column_2_t>::value>;
	using column_2 = BSInner<double, column_2_t, column_2_index>;

	using columns = std::tuple<column_0, column_1, column_2>;
	BSMatrix<double, columns, std::make_integer_sequence<std::size_t, std::tuple_size<columns>::value>> matrix;

	// Resize the dynamic matrices A and C
	auto A = matrix.getBlock<0,0>();
	auto B = matrix.getBlock<1,2>();
	auto C = matrix.getBlock<2,1>();
	auto D = matrix.getBlock<2,2>();

	A->resize(3,3);
	C->resize(1,2);

	A->matrix.array() = 1.0;
	B->matrix.array() = 2.0;
	C->matrix.array() = 3.0;
	D->matrix.array() = 4.0;

	std::cout << "nnz = " << matrix.nnz() << std::endl;
	std::cout << "Block size = " << matrix.block_rows() << " x " << matrix.block_cols() << std::endl;

}


	return 0;
}


template<int order, int size>
struct variable {};

enum VariableOrder
{
	ix,iy,iz;
};

using x = variable<ix,3>;
using y = variable<iy,2>;
using z = variable<iz,1>;

?? define order here?
using variables = std::tuple<x, y, z>;

constraint / row

- row contraints only non-zero blocks



... we need to fill in 
constraint vector   <- row offsets
constraint jacobian <- row and variable offsets
constraint hessian  <- variable offsets



Data / processing flow:

Optimizer:
- owns Jacobian data in standard format
-- requests sparsity structure from BSMatrix
- passes to BSMatrix to fill in (in dense or sparse form)

BSMatrix:
- owns a copy of the data in block-format
- has an update list?? array of function calls to update data?

Blocks:
- static?
- update their data when requested... how to do row-wise?


Jacobian...
Bottom of stack... we have functions that spit out updates of Jx, Jy, Jz simultaneously
Each constraint / block row knows how to update itself


We also normally compute several things with one call... 
	constraint and jacobian, or value, gradient and hessian

So do we have the constraints update jacobians, or the jacobians request an update?
Got to be jacobian's requesting update... since we might compute it for multiple values



=> BSMatrix is row-major
=> BSRow takes multiple forms
--> one is a "jacobian" row, which stores the data in dense form with variable offsets
--> update occurs on every read... no - need to get the current iterate in there...

BSRow needs to know about all variables, and about our non-zero blocks


How do we read this out in column-sparse format?

iterate over variables, then over rows... should still be zero-overhead




// https://stackoverflow.com/questions/15411022/how-do-i-replace-a-tuple-element-at-compile-time