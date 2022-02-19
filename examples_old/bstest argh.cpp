#include <iostream>
#include <tuple>
#include <utility>
#include <numeric>
#include <functional>
#include <iomanip>

#include "type_name.hpp"

#include "Eigen/Dense"
#include "Eigen/Sparse"

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

// /*
// Generic function to find an element in vector and also its position.
// It returns a pair of bool & int i.e.
// bool : Represents if element is present in vector or not.
// int : Represents the index of element in vector if its found else -1
// */
// template < typename T>
// int findInVector(const std::vector<T>  & vecOfElements, const T  & element)
// {
//     auto it = std::find(vecOfElements.begin(), vecOfElements.end(), element);
//     if (it != vecOfElements.end())
//         return std::distance(vecOfElements.begin(), it);
//     else
//     	return -1;
// }


// // Sum the inputs to get total number of inputs, or return 
// // Eigen::Dynamic if any of the values are Eigen::Dynamic
// // template<int... S>
// // constexpr int compute_LenAtCompileTime() {
// //     int result = 0;
// //     for(auto s : { S... }) 
// //     {
// //     	if(result == Eigen::Dynamic || s == Eigen::Dynamic)
// //     		result = Eigen::Dynamic;
// //     	else
// // 			result += s;
// //     }
// //     return result;
// // }


// /**
//  * BSMatrix<output_type>, where output_type is either dense or sparse, 
//  * 
//  * Block<output_type, input_type>
//  * 
//  * 
//  * 
//  * During construction:
//  * - BSMatrix 
//  * 		- builds an appropriate matrix (dense or sparse)
//  * 		- requests N = nonZeros from each block (or N = rows * cols for dense)
//  * 		- allocates an array valuePtr = Scalar[N] (or valuePtr = matrix.data())
//  * 		- writes 0:N to valuePtr
//  * - Iterates through column-by-column and builds the 
//  */


// // /** An object storing offset data required to copy block-information into the 
// //  * Matrix or SparseMatrix
// //  */
// // struct BaseBlock
// // {
// // 	int rows, cols;

// // 	* Returns a vector of length cols, where each element gives the 
// // 	 * number of nonzeros in that column
	 
// // 	virtual std::vector<std::size_t> columnwiseNonZeros() = 0;

// // 	BaseBlock(int rows_, int cols_)
// // 		: rows(rows_), cols(cols_)
// // 	{}
// // };






// // template<int RowsAtCompileTime = Eigen::Dynamic, int ColsAtCompileTime = Eigen::Dynamic>
// // struct Block : BaseBlock
// // {
// // 	using Index = std::size_t;
// // 	static constexpr bool is_static = (RowsAtCompileTime != Eigen::Dynamic && ColsAtCompileTime != Eigen::Dynamic);

// //     using offset_type = std::conditional_t<ColsAtCompileTime == Eigen::Dynamic,
// //     						std::vector<Index>, 
// //     						std::array<Index, is_static ? ColsAtCompileTime : 0>>;

// //     // Columwise offset into the global data vector
// //     offset_type column_offsets;

// //     // Length of each column
// //     offset_type column_lengths;

// // 	Block(int rows_ = RowsAtCompileTime, int cols_ = ColsAtCompileTime)
// // 		: BaseBlock(rows_, cols_)
// // 	{
// // 		assert((RowsAtCompileTime == Eigen::Dynamic || RowsAtCompileTime == rows) &&
// // 			"Cannot define both rows and RowsAtCompileTime unless they match");
// // 		assert((ColsAtCompileTime == Eigen::Dynamic || ColsAtCompileTime == cols) &&
// // 			"Cannot define both cols and ColsAtCompileTime unless they match");

// // 		reserve();
// // 	}

// // 	// /** Copy block B into the correct location of mat
// // 	//  */
// // 	// void setBlock(Eigen::Ref<Eigen::SparseMatrix<Scalar>> mat, 
// // 	// 			  const Eigen::Ref<const Eigen::MatrixX<Scalar>> B)
// // 	// {
// // 	// }

// // 	/** Copy block B into the correct location of mat
// // 	 */
// // 	// void setBlock(Eigen::Ref<Eigen::MatrixX<Scalar>> mat, 
// // 	// 			  const Eigen::Ref<const Eigen::Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>> B)
// // 	// {
// // 	// 	Scalar* data = mat.data();
// // 	// }

// // 	virtual std::vector<std::size_t> columnwiseNonZeros()
// // 	{
// // 		return std::vector<Index>(column_lengths.begin(), column_lengths.end());
// // 	}


// // private:
// // 	/** Allocate memory if it's not statically known.
// // 	 * 
// // 	 * Called from the constructor.
// // 	 */
// // 	template<bool is_static_ = is_static, std::enable_if_t<!is_static_, bool> = true>
// // 	inline void reserve()
// // 	{
// // 		column_offsets.reserve(cols);
// // 	}

// // 	template<bool is_static_ = is_static, std::enable_if_t<is_static_, bool> = true>
// // 	inline void reserve() {}
// // };

// // /** Dense dynamic block
// //  */
// // struct BlockX : public Block<>
// // {
// // 	BlockX(int rows_, int cols_) : Block(rows_, cols_)
// // 	{}
// // };


// struct Variable
// {
// 	int len;
// 	std::string name;

// 	Variable(int len_, std::string name_) 
// 		: name(name_), len(len_)
// 	{}

// 	// Properties that are set by finalize	
// 	int block_index = -1;
// 	int first_col = -1; // Columns in the Matrix
// 	int last_col = -1;
// };

// /** Holds a list of variables in order
//  */
// struct Variables
// {
// 	std::vector<Variable> vars;

// 	Variables(std::initializer_list<Variable> vars_) : vars(vars_)
// 	{}

// 	int len()
// 	{
// 		int acc = 0;
// 		for(auto v : vars) acc += v.len;
// 		return acc;
// 	}
// };

// enum BlockType
// {
// 	Zero,
// 	Dense,
// 	Sparse
// };

// struct Block
// {
// 	using Index = std::size_t;
// 	BlockType type;

//     std::size_t rows, cols;

//     // Columwise offset into the global data vector
//     std::vector<Index> column_offsets;

//     // Length of each column
//     std::vector<Index> column_lengths;

// 	Block(std::size_t rows_, std::size_t cols_, BlockType type_ = Dense) 
// 		: rows(rows_), cols(cols_), type(type_)
// 	{
// 		reserve();
// 	}

// 	inline void FillMe(Eigen::Ref<Eigen::SparseMatrix<Scalar>> Target, Eigen::Ref<Eigen::MatrixX<Scalar>> M)
// 	{
// 		Scalar* valuePtr = Target.valuePtr();
// 		for(int col=0; col < cols; col++)
// 			memcpy(valuePtr + column_offsets[col], 
// 				   M.column(col).data(), 
// 				   sizeof(Scalar) * rows);
// 	}

// 	inline void FillMe(Eigen::Ref<Eigen::MatrixX<Scalar>> Target, Eigen::Ref<Eigen::MatrixX<Scalar>> M)
// 	{
// 		Scalar* valuePtr = Target.data();
// 		for(int col=0; col < cols; col++)
// 			memcpy(valuePtr + column_offsets[col],  <<< Needs more information
// 				   M.column(col).data(), 
// 				   sizeof(Scalar) * rows);
// 	}

// 	inline void FillMe(Scalar* valuePtr, Eigen::Ref<Eigen::SparseMatrix<Scalar>> M)
// 	{
// 		Scalar* Mdata = M.valuePtr();
// 		for(int col=0; col < cols; col++)
// 			memcpy(valuePtr + column_offsets[col], 
// 				   Mdata + M.outerIndexPtr[col], 
// 				   sizeof(Scalar) * column_lengths[col]);
// 	}


// 	// virtual std::vector<std::size_t> columnwiseNonZeros()
// 	// {
// 	// 	return std::vector<Index>(column_lengths.begin(), column_lengths.end());
// 	// }


// private:
// 	inline void reserve()
// 	{
// 		if(type != Zero)
// 			column_offsets.reserve(cols);
// 	}
// };

// struct Constraint
// {
// 	int len;
// 	std::string name;

// 	// Of length |Variables|
// 	std::vector<Block> blocks;

// 	Constraint(int len_, std::string name_, 
// 			   Variables& variables,	// Collection of all variables in order
// 			   std::initializer_list<Variable> vars)  
// 		: name(name_), len(len_)
// 	{
// 		blocks.reserve(variables.vars.size());
// 		// for(int i=0; i<blocks.size(); i++) blocks[i] = NULL;

// 		int ind;
// 		for(auto v : vars)
// 		{
// 			// ind = findInVector(variables.vars, v);
// 			assert(ind != -1 && "Variable not found in list of variables");
// 			blocks[ind] = Block(len, v.len);
// 		}
// 	}

// 	// Properties that are set by finalize	
// 	int block_index = -1;
// 	int first_col = -1; // Columns in the Matrix
// 	int last_col = -1;
// };


// /** Holds a list of variables in order
//  */
// struct Constraints
// {
// 	std::initializer_list<Constraint> cons;

// 	Constraints(std::initializer_list<Constraint> cons_) : cons(cons_)
// 	{
// 	}

// 	int len()
// 	{
// 		int acc = 0;
// 		for(auto v : cons) acc += v.len;
// 		return acc;
// 	}
// };





// // template<typename Scalar, OutputType Type>
// // struct BSMatrix
// // {
// // 	Constraints& constraints;
// // 	Variables& variables;

// // 	BSMatrix(Constraints& constraints_, Variables& variables_) 
// // 		: constraints(constraints_), variables(variables_)
// // 	{}
// // };

// // template<typename Scalar>
// // struct BSMatrix<Scalar, DENSE>
// // {
// // 	Constraints& constraints;
// // 	Variables& variables;

// // 	Eigen::MatrixX<Scalar> m_matrix;

// // 	BSMatrix(Constraints& constraints_, Variables& variables_) 
// // 		: constraints(constraints_), variables(variables_)
// // 	{
// // 		std::cout << "BSMatrix dense\n";
// // 		m_matrix.resize(constraints.len(), variables.len());

// // 		std::cout << "m_matrix size = " << m_matrix.rows() << " x " << m_matrix.cols() << std::endl;
// // 	}
// // };

// template<typename Scalar>
// struct BSMatrix : Eigen::SparseMatrix<Scalar>
// {
// 	Constraints& constraints;
// 	Variables& variables;

// 	// std::vector<std::shared_ptr<Block>> blocks;

// 	BSMatrix(Constraints& constraints_, Variables& variables_) 
// 		: constraints(constraints_), 
// 		  variables(variables_), 
// 		  Eigen::SparseMatrix<Scalar>(constraints_.len(), variables_.len())
// 	{
// 		std::cout << "BSMatrix sparse\n";
// 		this->resize(constraints.len(), variables.len());
// 		std::cout << "BSMatrix size = " << this->rows() << " x " << this->cols() << std::endl;
// 	}
// };
 


// // /** Dense matrix
// //  */
// // template<typename MatrixType>
// // struct BSDenseMatrix : BSMatrix<MatrixType>
// // {
// // 	using Base = BSMatrix<MatrixType>;
// // 	MatrixType m_matrix;

// // 	BSDenseMatrix(Constraints& constraints_, Variables& variables_) 
// // 		: Base(constraints_, variables_)
// // 	{
// // 	}
// // };

// // /** Helper to build a ConstraintX given a set of variables
// //  */
// // template<typename... Vars>
// // auto ConstraintX(std::size_t rows, std::string name, Vars... vars)
// // {
// // 	return Constraint(rows, name, {BlockX{rows, vars.len()}...});
// // }

// struct BSTest
// {
// 	using Scalar = double;

// 	Variable x{2, "x"};
// 	Variable y{3, "y"};
// 	Variable z{4, "z"};

// 	Variables vars{x, y, z};

// 	// Constraint con1{4, "con1", vars, {x, y}};
// 	// Constraint con2{2, "con2", vars, {y}};
// 	// Constraint con3{5, "con3", vars, {z, x}};

// 	// Constraints constraints{con1, con2, con3};

// 	// // Dynamic matrix
// 	// BSMatrix<Scalar> mat{constraints, vars};

// 	BSTest()
// 	{
// 		std::vector<Block> blocks;
// 		blocks.emplace_back(3, x.len);
// 		blocks.emplace_back(4, y.len);
// 		blocks.emplace_back(5, z.len);
// 		blocks.emplace_back(6, x.len);

// 		for(auto b : blocks)
// 		{
// 			std::cout << "b.column_offsets.size() = " << b.column_offsets.size() << std::endl;
// 		}

// 		std::cout << "Total variable len = " << vars.len() << std::endl;
// 		// std::cout << "Total constraint len = " << constraints.len() << std::endl;

// 		// Eigen::Matrix<Scalar, 30, 40> mat;
// 		// // Eigen::Matrix<Scalar, 3, 4> B;
// 		// Eigen::MatrixX<Scalar> B(3, 4);

// 		// B.array() = 1;
// 		// blkX.setBlock(mat, B);
// 		// blkX.setBlock(mat, B);
// 	}
// };

// int main()
// {
// 	BSTest test;

// 	return 0;
// }


// template<int size_,  // Variable size
// 		 int N_ = 1> // Number of copies of this variable
// struct Variable
// {
// public:
// 	static constexpr int size = size_;
// 	static constexpr int N = N_;

// 	template<int ind_>
// 	struct ind
// 	{
// 		static_assert(ind_ < N, "Requesting sub-variable larger than N");
// 		static constexpr int size = size_;
// 	};
// };

// typename<typename Var, int start=0, int step=1>
// struct itr
// {

// };

// template<int size_, int num_iterations, typename... vars> // Takes itr types
// struct Constraint
// {
// 	static constexpr int size = size_;

// 	using column_t = std::array<std::size_t, size>;
// 	std::array<column_t, num_iterations> offsets;
// 	std::array<column_t, num_iterations> column_sizes; // Used for sparse-only

// 	static void printVariables()
// 	{
// 	    (void)std::initializer_list<int>{
// 	        (
// 	        	std::cout << "var size = " << vars::size << std::endl,
// 	        	0
// 	        )...
// 	    };
// 	}
// };

// template<typename Scalar, typename... constraint_t>
// void BSMatrix(Eigen::SparseMatrix<Scalar>, constraint_t... constraint)
// {
// 	// Compute the offset data for each constraint and 
// 	// return matrix with correct sparsity structure
// }

// template<std::size_t N>
// struct BSTest
// {
// 	// struct x : Variable<2, 3> {};
// 	// struct u : Variable<3, 2> {};

// 	struct y : Variable<0, 5> {};
// 	struct x : Variable<1, 2, 3> {};
// 	struct u : Variable<2, 3, 2> {};

// 	struct con1 : Constraint<3, N, itr<y>, itr<x, 0>, itr<x, 1>, itr<u, 0>> {};
// 	struct con2 : Constraint<3, 3, itr<y>, itr<x, 0>, itr<x, 1>, itr<u, 0>> {};

// 	std::tuple<con1, con2> constraints;
// 	std::tuple<u, y, x> variables;
// 	Eigen::SparseMatrix<double> J;

// 	Eigen::SparseMatrix<double> H;
// 	struct con_x : Constraint<x::size, x> {};
// 	struct con_y : Constraint<y::size, y, x> {};
// 	struct con_u : Constraint<u::size, u> {};

// 	void test()
// 	{
// 		// Does offset computations
// 		BSMatrix(J, constraints, variables);

// 		for(int i=0; i<N; i++)
// 			// constraints.set<con1, 0>(i, B);
// 			constraints.set<con1, ALL>(i, jacobian(F<i>, x));

// 		for(int i=0; i<N; i++)
// 		{
// 			constraints.setBlock<con1, y>(i, );
// 		}
// 	}
// };



struct Base
{
	virtual int testVirtual(int x) noexcept = 0;
};

struct Child final : Base
{
	const int i;

	Child(int i_) : i(i_) {}

	virtual int testVirtual(int x) noexcept
	{
		return x*x;
	}
};

struct Direct
{
	const int i;

	Direct(int i_) : i(i_) {}

	int testVirtual(int x) noexcept
	{
		return x*x;
	}
};


int main()
{

	std::array<Base*, 10> arr;
	for(int i=0; i<10; i++)
		arr[i] = new Child(i);

	Direct f0(0);
	Direct f1(1);
	Direct f2(2);
	Direct f3(3);
	Direct f4(4);
	Direct f5(5);
	Direct f6(6);
	Direct f7(7);
	Direct f8(8);
	Direct f9(9);

{	
	int acc = 0;
    time_point start = get_time();
	for(int i=0; i<100000000; i++)
	{
		for(auto f : arr)
			acc += f->testVirtual(i);
	}
    time_point stop = get_time();

    std::cout << "acc = " << acc << std::endl;
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "Vector time " << std::setprecision(9)
              << static_cast<double>(duration.count()) / 100000000 << " [microseconds]" << "\n";
}

{	
	int acc = 0;
    time_point start = get_time();
	for(int i=0; i<100000000; i++)
	{
		acc += f0.testVirtual(i);
		acc += f1.testVirtual(i);
		acc += f2.testVirtual(i);
		acc += f3.testVirtual(i);
		acc += f4.testVirtual(i);
		acc += f5.testVirtual(i);
		acc += f6.testVirtual(i);
		acc += f7.testVirtual(i);
		acc += f8.testVirtual(i);
		acc += f9.testVirtual(i);
	}
    time_point stop = get_time();

    std::cout << "acc = " << acc << std::endl;
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "Vector time " << std::setprecision(9)
              << static_cast<double>(duration.count()) / 100000000 << " [microseconds]" << "\n";
}

	return 0;
}
