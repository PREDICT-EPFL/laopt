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


template<typename scalar_t>
class BlockBase
{
public:
	virtual void get_column(Eigen::Ref<Eigen::VectorX<scalar_t>> x, int index) = 0;

	virtual int rows() = 0;
	virtual int cols() = 0;
};

template<typename matrix_t>
class BlockDense : public BlockBase<typename matrix_t::Scalar>
{
	Eigen::Ref<matrix_t> matrix;
	using scalar_t = typename matrix_t::Scalar;

public:
	BlockDense(Eigen::Ref<matrix_t> _matrix) : matrix(_matrix) {}

	virtual void get_column(Eigen::Ref<Eigen::VectorX<scalar_t>> x, int index)
	{
		// std::cout << "Getting column of matrix of type " << type_name<matrix_t>() << " \n";
		x = matrix.col(index);
	}

	virtual int rows() {return matrix.rows();}
	virtual int cols() {return matrix.cols();}
};



// /** /class BlockInfo
//  */
// template<typename scalar_t>
// class BlockInfo
// {
// 	// get_sparsity_pattern

// 	/**
// 	 * Copy the index'th column into the given vector x.
// 	 * 
// 	 * If this column is sparse, then the data is copied into x 
// 	 * in compressed column format.
// 	 */ 
// 	typedef void (*get_column_t)(Eigen::Ref<Eigen::VectorX<scalar_t>>, int);
// 	get_column_t get_column;

// public:
// 	// template<typename matrix_t>
// 	// BlockInfo(Eigen::Ref<matrix_t> matrix)
// 	// {
// 	// 	assert(false && "Unknown matrix type ")
// 	// }

// 	// Dense matrices
// 	BlockInfo(Eigen::Ref<Eigen::MatrixX<scalar_t>> matrix) :
// 		get_column([&matrix](Eigen::Ref<Eigen::VectorX<scalar_t>> x, int index)		
// 		{
// 			std::cout << "Getting column of dense matrix\n";
// 			x = matrix.col(index);
// 		})
// 		{}

// 	// Sparse matrices
// 	BlockInfo(Eigen::Ref<Eigen::SparseMatrix<scalar_t>> matrix) :
// 		get_column([&matrix](Eigen::Ref<Eigen::VectorX<scalar_t>> x, int index)		
// 		{
// 			std::cout << "Getting column of sparse matrix\n";
// 			x = matrix.col(index);
// 		})
// 		{}

// };


// template<typename scalar_t, 
// 				 int _rowblocks_at_compiletime = Eigen::Dynamic, 
// 				 int _colblocks_at_compiletime = Eigen::Dynamic>
// class BSMatrix //: public Eigen::SparseMatrix<BSMatrix<scalar_t>>
// {
// 	// using Base = Eigen::SparseMatrix<BSMatrix<scalar_t>>;

// 	const int rowblocks_at_compiletime = _rowblocks_at_compiletime;
// 	const int colblocks_at_compiletime = _colblocks_at_compiletime;

// 	int m_row_blocks, m_col_blocks;


// public:
// 	// using Base::Base;

// 	BSMatrix(int _rowblocks = Eigen::Dynamic, 
// 					 int _colblocks = Eigen::Dynamic) 
// 		: m_row_blocks(std::max(_rowblocks, rowblocks_at_compiletime)), 
// 		  m_col_blocks(std::max(_colblocks, colblocks_at_compiletime))
// 	{
// 		assert(m_row_blocks > 0 && m_col_blocks > 0 &&
// 				"Number of blocks must be specified either in the template, or in the constructor.");
// 	}

// 	int row_blocks() {return m_row_blocks;}
// 	int col_blocks() {return m_col_blocks;}

// 	template<typename matrix_t>
// 	set_block(matrix_t& mat, int row, int col)
// 	{
// 		// Create lambdas with 
// 	}
// };



// typedef double (*binOp)(double, double);
// struct MyStruct {
//     // char c;
//     binOp fn;
// };

int main()
{
	Eigen::Matrix<double, 20, 30> A;
	// A << 1,2,3,4,5,6;
	BlockDense<decltype(A)> blkA(A);

	Eigen::MatrixX<double> B(30,20);
	// B << 7,8,9,10,11,12;
	BlockDense<decltype(B)> blkB(B);


	// Eigen::VectorX<double> x(3);

	// {
	// 	x = matrix.col(index);
	// }

	// Eigen::Matrix<BlockBase<double>*, 20, 30> blks;
	// blks(1,1) = &blkA;

	BlockBase<double>* arr[] = {&blkA, &blkB};

{
	Eigen::VectorX<double> x(50);
	Eigen::VectorX<double> y(50);
  time_point start = get_time();
	for(int i=0; i<1000000; i++)
	{
		for(int i=0; i<2; i++)
		{
			for(int col=0; col<arr[i]->cols(); col++)
			{
				arr[i]->get_column(x.head(arr[i]->rows()), col);
				y.head(arr[i]->rows()) += x.head(arr[i]->rows());
			}
		}
	}
  time_point stop = get_time();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	std::cout << "y = " << y.transpose() << std::endl;
  std::cout << "Virtual time: " << std::setprecision(9) << static_cast<double>(duration.count()) << "[us] \n";
}

{
	Eigen::VectorX<double> x(50);
	Eigen::VectorX<double> y(50);
  time_point start = get_time();
	for(int i=0; i<1000000; i++)
	{
		for(int col=0; col<A.cols(); col++)
		{
			x.head(A.rows()) = A.col(col);
			y.head(A.rows()) += x.head(A.rows());
		}
		for(int col=0; col<B.cols(); col++)
		{
			x.head(B.rows()) = B.col(col);
			y.head(B.rows()) += x.head(B.rows());
		}
	}
  time_point stop = get_time();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	std::cout << "y = " << y.transpose() << std::endl;
  std::cout << "Direct time: " << std::setprecision(9) << static_cast<double>(duration.count()) << "[us] \n";
}
	// std::cout << "Got column " << col << " = " << x.transpose() << " of arr[" << i << "]" << std::endl;

	// Eigen::VectorX<double> x(2);
	// blk.get_column(x, 0);
	// std::cout << "Got column " << x.transpose() << std::endl;



	// BSMatrix<double, 2, 3> bsm;

	// std::cout << "bsm.row_blocks = " << bsm.row_blocks() << std::endl;


	//  const binOp ops[] = {
	//     {[] (double a, double b) { return a+b; } },
	//     {[] (double a, double b) { return a*2.3*b; } },
	// };

// {	
// 	double x = 0.0;
//   time_point start = get_time();
//   for(int j=0; j<1000000; j++)
//   {
//   	for(int i=0; i<2; i++)
//   		x += ops[i](1,2);
//   }
//   time_point stop = get_time();
//   auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

//   std::cout << "x = " << x << std::endl;
//   std::cout << "Lambda time: " << std::setprecision(9) << static_cast<double>(duration.count()) << "[us] \n";
// }

// {	
// 	double x = 0.0;
//   time_point start = get_time();
//   for(int j=0; j<1000000; j++)
//   {
//   		x += (1+2);
//   		x += (1*2.3*2);
//   }
//   time_point stop = get_time();
//   auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

//   std::cout << "x = " << x << std::endl;
//   std::cout << "Direct time: " << std::setprecision(9) << static_cast<double>(duration.count()) << "[us] \n";
// }

	// for(int i=0; i<2; i++)
	// {
	// 	std::cout << "ops[" << i << "](1,2) = " << ops[i](1,2) << std::endl;
	// }

	// Eigen::MatrixXd A(2,3);
	// A << 1,2,3,4,5,6;
	// Eigen::Matrix<double, 1, 4> B;
	// B << 7,8,9,10;

	// auto getA = [&A](int i){return A.col(i);};
	// auto getB = [&B](int i){return B.col(i);};

	// std::cout << "getA(0) = " << getA(0).transpose() << std::endl;

	// std::array<MyStruct, 2> functors = {{{getA}}, {{getB}}};

	return 0;
}







template<typename matrix_t>
class Block
{
	Eigen::Ref<matrix_t> matrix;
	using scalar_t = typename matrix_t::Scalar;

public:
	Block(Eigen::Ref<matrix_t> _matrix) : matrix(_matrix) {}

	inline void get_column(Eigen::Ref<Eigen::VectorX<scalar_t>> x, int index)
	{
		x = matrix.col(index);
	}

	inline int rows() {return matrix.rows();};
	inline int cols() {return matrix.cols();};
};

// template<typename matrix_t>
// class BlockDense : public BlockBase<typename matrix_t::Scalar>
// {
// 	Eigen::Ref<matrix_t> matrix;
// 	using scalar_t = typename matrix_t::Scalar;

// public:
// 	BlockDense(Eigen::Ref<matrix_t> _matrix) : matrix(_matrix) {}

// 	virtual void get_column(Eigen::Ref<Eigen::VectorX<scalar_t>> x, int index)
// 	{
// 		// std::cout << "Getting column of matrix of type " << type_name<matrix_t>() << " \n";
// 		x = matrix.col(index);
// 	}

// 	virtual int rows() {return matrix.rows();}
// 	virtual int cols() {return matrix.cols();}
// };



// /** /class BlockInfo
//  */
// template<typename scalar_t>
// class BlockInfo
// {
// 	// get_sparsity_pattern

// 	/**
// 	 * Copy the index'th column into the given vector x.
// 	 * 
// 	 * If this column is sparse, then the data is copied into x 
// 	 * in compressed column format.
// 	 */ 
// 	typedef void (*get_column_t)(Eigen::Ref<Eigen::VectorX<scalar_t>>, int);
// 	get_column_t get_column;

// public:
// 	// template<typename matrix_t>
// 	// BlockInfo(Eigen::Ref<matrix_t> matrix)
// 	// {
// 	// 	assert(false && "Unknown matrix type ")
// 	// }

// 	// Dense matrices
// 	BlockInfo(Eigen::Ref<Eigen::MatrixX<scalar_t>> matrix) :
// 		get_column([&matrix](Eigen::Ref<Eigen::VectorX<scalar_t>> x, int index)		
// 		{
// 			std::cout << "Getting column of dense matrix\n";
// 			x = matrix.col(index);
// 		})
// 		{}

// 	// Sparse matrices
// 	BlockInfo(Eigen::Ref<Eigen::SparseMatrix<scalar_t>> matrix) :
// 		get_column([&matrix](Eigen::Ref<Eigen::VectorX<scalar_t>> x, int index)		
// 		{
// 			std::cout << "Getting column of sparse matrix\n";
// 			x = matrix.col(index);
// 		})
// 		{}

// };


// template<typename scalar_t, 
// 				 int _rowblocks_at_compiletime = Eigen::Dynamic, 
// 				 int _colblocks_at_compiletime = Eigen::Dynamic>
// class BSMatrix //: public Eigen::SparseMatrix<BSMatrix<scalar_t>>
// {
// 	// using Base = Eigen::SparseMatrix<BSMatrix<scalar_t>>;

// 	const int rowblocks_at_compiletime = _rowblocks_at_compiletime;
// 	const int colblocks_at_compiletime = _colblocks_at_compiletime;

// 	int m_row_blocks, m_col_blocks;


// public:
// 	// using Base::Base;

// 	BSMatrix(int _rowblocks = Eigen::Dynamic, 
// 					 int _colblocks = Eigen::Dynamic) 
// 		: m_row_blocks(std::max(_rowblocks, rowblocks_at_compiletime)), 
// 		  m_col_blocks(std::max(_colblocks, colblocks_at_compiletime))
// 	{
// 		assert(m_row_blocks > 0 && m_col_blocks > 0 &&
// 				"Number of blocks must be specified either in the template, or in the constructor.");
// 	}

// 	int row_blocks() {return m_row_blocks;}
// 	int col_blocks() {return m_col_blocks;}

// 	template<typename matrix_t>
// 	set_block(matrix_t& mat, int row, int col)
// 	{
// 		// Create lambdas with 
// 	}
// };


/**
 * We store data in a modified block sparse column format.
 * 
 * Assume our matrix has the format
 * 
 * [A 0
 *  0 B
 *  C D]
 * 
 * Where each block A, B, C, D can be one of 
 * 
 */
template<typename Scalar> // Something here to specify block sizes and nnz
class BSMatrix
{
    enum { Options = internal::traits<Derived>::Options };

    DenseStorage<Scalar, MaxSizeAtCompileTime, RowsAtCompileTime, ColsAtCompileTime, Options> m_storage;

	Index m_outerSize;
	Index m_innerSize;
	StorageIndex* m_outerIndex;
	StorageIndex* m_innerNonZeros;     // optional, if null then the data is compressed
	Storage m_data;

	struct block_data_t
	{
		enum block_type
		{
			zero,
			dense,
			sparse,
			colloc
		};

		// Offset to the start of the data for this block
		int offset_data;
		int offset_InnerIndices;
		int offset_OuterStarts;

		// Matrix size
		int rows, cols, nnz;

		block_type type;

		// Additional data specific to this type (sparsity pattern)
		void* data = NULL;

		block_data_t(Eigen::Ref<Eigen::MatrixX<scalar_t>> mat)
		{
			
		}

		block_data_t(Eigen::Ref<Eigen::SparseMatrix<scalar_t>> mat)
		{
			
		}
	};

	block_data_t blocks[block_rows][block_cols];

	constexpr nnz()
	{

	}

public:
	BSMatrix(matrix_t... matrices) 
	{
		for(int i=0; i<)


		data.resize(nnz(), 1);
	}


};





// typedef double (*binOp)(double, double);
// struct MyStruct {
//     // char c;
//     binOp fn;
// };

#define n 20
#define m 30

template<typename scalar_t, typename... block_t>
class BSMatrix
{
	std::tuple<block_t& ...> blocks;

public:
	BSMatrix(block_t&... _blocks) :
		blocks{_blocks...}
		{}

	auto get_block(int block)
	{
		switch(block)
		{
			case 0: return std::get<0>(blocks);
			case 1: return std::get<1>(blocks);
			// case 2: std::get<2>(blocks).get_column(x.head(std::get<2>(blocks).rows()), index);
			// case 3: std::get<3>(blocks).get_column(x.head(std::get<3>(blocks).rows()), index);
			// case 4: std::get<4>(blocks).get_column(x.head(std::get<4>(blocks).rows()), index);
			// case 5: std::get<5>(blocks).get_column(x.head(std::get<5>(blocks).rows()), index);
			// case 6: std::get<6>(blocks).get_column(x.head(std::get<6>(blocks).rows()), index);
			// case 7: std::get<7>(blocks).get_column(x.head(std::get<7>(blocks).rows()), index);
			// case 8: std::get<8>(blocks).get_column(x.head(std::get<8>(blocks).rows()), index);
			// case 9: std::get<9>(blocks).get_column(x.head(std::get<9>(blocks).rows()), index);
		}
	}

	inline void get_column(Eigen::Ref<Eigen::VectorX<scalar_t>> x, int block, int index)
	{
		auto B = get_block(block);

		B.get_column(x.head(B.rows()), index);
	}

	// inline rows(int block)
	// {

	// }
};

int main()
{

	Eigen::Matrix<double, n, m> A;
	A.array() = 1;
	// A.head(6) << 1,2,3,4,5,6;
	Block<decltype(A)> blkA(A);

	Eigen::MatrixX<double> B(m,n);
	B.array() = 2;
	// B.head(6) << 7,8,9,10,11,12;
	Block<decltype(B)> blkB(B);


	// BSMatrix<decltype(A), decltype(B)> bsmat(A, B);
	BSMatrix<double, decltype(blkA), decltype(blkB)> bsmat(blkA, blkB);

	Eigen::VectorX<double> x(50);
	bsmat.get_column(x, 0, 0);

	// std::cout << "x = " << x.transpose() << std::endl;

	// Eigen::Matrix<BlockBase<double>*, n, m> blks;
	// blks(1,1) = &blkA;

	// BlockBase<double>* arr[] = {&blkA, &blkB};

// {
// 	Eigen::VectorX<double> x(50);
// 	Eigen::VectorX<double> y(50);
//   time_point start = get_time();
// 	for(int i=0; i<1000000; i++)
// 	{
// 		for(int i=0; i<2; i++)
// 		{
// 			for(int col=0; col<bsmat.->cols(); col++)
// 			{
// 				arr[i]->get_column(x.head(arr[i]->rows()), col);
// 				y.head(arr[i]->rows()) += x.head(arr[i]->rows());
// 			}
// 		}
// 	}
//   time_point stop = get_time();
//   auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
// 	std::cout << "y = " << y.transpose() << std::endl;
//   std::cout << "Virtual time: " << std::setprecision(9) << static_cast<double>(duration.count()) << "[us] \n";
// }

// {
// 	Eigen::VectorX<double> x(50);
// 	Eigen::VectorX<double> y(50);
//   time_point start = get_time();
// 	for(int i=0; i<1000000; i++)
// 	{
// 		for(int col=0; col<A.cols(); col++)
// 		{
// 			x.head(A.rows()) = A.col(col);
// 			y.head(A.rows()) += x.head(A.rows());
// 		}
// 		for(int col=0; col<B.cols(); col++)
// 		{
// 			x.head(B.rows()) = B.col(col);
// 			y.head(B.rows()) += x.head(B.rows());
// 		}
// 	}
//   time_point stop = get_time();
//   auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
// 	std::cout << "y = " << y.transpose() << std::endl;
//   std::cout << "Direct time: " << std::setprecision(9) << static_cast<double>(duration.count()) << "[us] \n";
// }
	// std::cout << "Got column " << col << " = " << x.transpose() << " of arr[" << i << "]" << std::endl;

	// Eigen::VectorX<double> x(2);
	// blk.get_column(x, 0);
	// std::cout << "Got column " << x.transpose() << std::endl;



	// BSMatrix<double, 2, 3> bsm;

	// std::cout << "bsm.row_blocks = " << bsm.row_blocks() << std::endl;


	//  const binOp ops[] = {
	//     {[] (double a, double b) { return a+b; } },
	//     {[] (double a, double b) { return a*2.3*b; } },
	// };

// {	
// 	double x = 0.0;
//   time_point start = get_time();
//   for(int j=0; j<1000000; j++)
//   {
//   	for(int i=0; i<2; i++)
//   		x += ops[i](1,2);
//   }
//   time_point stop = get_time();
//   auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

//   std::cout << "x = " << x << std::endl;
//   std::cout << "Lambda time: " << std::setprecision(9) << static_cast<double>(duration.count()) << "[us] \n";
// }

// {	
// 	double x = 0.0;
//   time_point start = get_time();
//   for(int j=0; j<1000000; j++)
//   {
//   		x += (1+2);
//   		x += (1*2.3*2);
//   }
//   time_point stop = get_time();
//   auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

//   std::cout << "x = " << x << std::endl;
//   std::cout << "Direct time: " << std::setprecision(9) << static_cast<double>(duration.count()) << "[us] \n";
// }

	// for(int i=0; i<2; i++)
	// {
	// 	std::cout << "ops[" << i << "](1,2) = " << ops[i](1,2) << std::endl;
	// }

	// Eigen::MatrixXd A(2,3);
	// A << 1,2,3,4,5,6;
	// Eigen::Matrix<double, 1, 4> B;
	// B << 7,8,9,10;

	// auto getA = [&A](int i){return A.col(i);};
	// auto getB = [&B](int i){return B.col(i);};

	// std::cout << "getA(0) = " << getA(0).transpose() << std::endl;

	// std::array<MyStruct, 2> functors = {{{getA}}, {{getB}}};

	return 0;
}




// What would a "manual" version of this look like?
/**
 * [A 0
 *  0 B
 *  C D]
 */
template<typename... matrix_types>
class BSMatrix
{

	Eigen::MatrixX<Scalar> A;
	Eigen::Matrix<Scalar, 2, 3> B;
	Eigen::SparseMatrix<Scalar> C;
	Eigen::Colloc<Scalar> D;



	// The collective representation
	Eigen::SparseMatrix S;

	int blockRows, blockCols;

	void reserve()
	{

	}

public:
	BSMatrix() : A(3, 2), C(4,2), D(4, 3)
	{
		rows = A.rows() + B.rows() + C.rows();
		cols = A.cols() + B.cols();
		S.resize(rows, cols);

		int nnz = A.rows() * A.cols() + ...;

		blockRows = 3;
		blockCols = 2;
	}

	void to_sparsematrix(Eigen::SparseMatrix S)
	{
		int ptr = 0; // Offset into S data
		for(int blkRow=0; blkRow<blockRows; blkRow++)
		{
			for(int blkCol=0; blkCol<blockCols; blkCol++)
			{
				ptr += A.getcolumn(&(S.data[ptr]), &(S.innerStride));
			}
		}
	}
}



BSMatrix
{
	/**
	 * Holds "pointers" to objects that can
	 */
}






// template<typename Scalar, int rows = Eigen::Dynamic, int cols = Eigen::Dynamic>
// void test_input(Eigen::Matrix<Scalar, rows, cols>& A)
// {
// 	std::cout << "A = \n" << A << std::endl << "type(A) = " << type_name<decltype(A)>() << std::endl;
// 	std::cout << "mem(A) = " << &A << std::endl;


// }

// /** A type that can represent either a dense or a sparse matrix
//  */
// template<typename Scalar>
// class Block
// {
// 	enum block_type
// 	{
// 		zero,
// 		dense,
// 		sparse
// 	};

// 	// If we don't use this, then it doesn't take any memory
// 	SparseMatrix m_sparse;

// 	Scalar* data

//  public:
//  	Block(int rows, int cols)
//  	{

//  	}
// };


// template<typename Scalar>
// Block<Scalar> make_block(int rows, int cols, block_type type)
// {
// 	Block<Scalar>
// }




{
	time_point start = get_time();
	for(int i=0; i<1000000; i++)
	{
		A = A.array() + 4;
	}
	time_point stop = get_time();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	std::cout << "Static time: " << std::setprecision(9) << static_cast<double>(duration.count()) << "[us] \n";
	std::cout << "A = \n" << A << std::endl;
}

{
	time_point start = get_time();
	for(int i=0; i<1000000; i++)
	{
		B = B.array() + 4;
	}
	time_point stop = get_time();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	std::cout << "Dynamic time: " << std::setprecision(9) << static_cast<double>(duration.count()) << "[us] \n";
	std::cout << "A = \n" << A << std::endl;
}

{
	double data[6];
	data[0] = 1;
	data[1] = 2;
	data[2] = 3;
	data[3] = 4;
	data[4] = 5;
	data[5] = 6;

	Eigen::Map<Eigen::MatrixX<double>> map_dynamic(data, 2, 3);
	time_point start = get_time();
	for(int i=0; i<1000000; i++)
	{
		map_dynamic = map_dynamic.array() + 4;
	}
	time_point stop = get_time();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	std::cout << "Dynamic time: " << std::setprecision(9) << static_cast<double>(duration.count()) << "[us] \n";
	std::cout << "map_dynamic = \n" << map_dynamic << std::endl;
}

{
	double data[6];
	data[0] = 1;
	data[1] = 2;
	data[2] = 3;
	data[3] = 4;
	data[4] = 5;
	data[5] = 6;

	Eigen::Map<Eigen::Matrix<double, 2, 3>, Eigen::Aligned> map_static(A.data());
	time_point start = get_time();
	for(int i=0; i<1000000; i++)
	{
		map_static = map_static.array() + 4;
	}
	time_point stop = get_time();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	std::cout << "Static time: " << std::setprecision(9) << static_cast<double>(duration.count()) << "[us] \n";
	std::cout << "map_static = \n" << map_static << std::endl;
}

#ifdef NDEBUG
std::cout << "NDEBUG!!\n";
#else
std::cout << "no NDEBUG!!\n";
#endif

