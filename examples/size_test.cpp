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

// Sum the inputs to get total number of inputs
template<int... S>
constexpr int sum_template() {
    int result = 0;
    for(auto s : { S... }) result += s;
    return result;
}


/////////////////////////////////////////////////////////////
// Implementation of conjunction and disjunction in C++11 
namespace meta {
template<bool...> struct bool_pack{};

template<bool... Bs>
using conjunction = std::is_same<bool_pack<true,Bs...>, bool_pack<Bs..., true>>;

template <bool B>
using bool_constant = std::integral_constant<bool, B>; // redefining C++17 bool_constant helper

template<bool... Bs>
struct disjunction : bool_constant<!conjunction<!Bs...>::value>{};

template<typename T, typename... Ts>
struct contains : disjunction<std::is_same<T, Ts>::value...>
{};

/** Return the index into the tuple where type matches x
 * Returns -1 if x is not found.
 */
template<typename x, typename... elements>
constexpr int get_index(int value_on_fail = -1)
{
	int ind = 0;
	int out = value_on_fail;
    (void)std::initializer_list<int>{
        (
			out = std::is_same<x, elements>::value ? ind : out,
			ind++,
            0
        )...
    };
    return out;
}
/////////////////////////////////////////////////////////////

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
    using matrix_t = typename BlockTraits<Derived>::matrix_t;
	static const int rows = BlockTraits<Derived>::rows;
	static const int cols = BlockTraits<Derived>::cols;

	inline void toDense(Eigen::Ref<matrix_t> out)
	{
        static_cast<Derived*>(this)->toDense_impl(out);
	}

	// inline operator=(Eigen::Ref<other> )

	// inline void setBlock(const Eigen::Ref<const matrix_t>& in)
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

template<typename Scalar, int rows_, typename col_>
struct ZeroBlock : Block<ZeroBlock<Scalar, rows_, col_>>
{
	using Derived = ZeroBlock<Scalar, rows_, col_>;
    using col = typename BlockTraits<Derived>::col;
    using matrix_t = typename BlockTraits<Derived>::matrix_t;
	static const int rows = BlockTraits<Derived>::rows;
	static const int cols = BlockTraits<Derived>::cols;

	inline void toDense_impl(Eigen::Ref<matrix_t> out)
	{
		out = matrix_t::Zero();
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

template<typename Scalar, int rows_, typename col_>
struct DenseBlock : Block<DenseBlock<Scalar, rows_, col_>>
{
	using Derived = DenseBlock<Scalar, rows_, col_>;
    using col = typename BlockTraits<Derived>::col;
    using matrix_t = typename BlockTraits<Derived>::matrix_t;
	static const int rows = BlockTraits<Derived>::rows;
	static const int cols = BlockTraits<Derived>::cols;

	// Dense storage
	matrix_t m_matrix;

	inline void toDense_impl(Eigen::Ref<matrix_t> out)
	{
		out = m_matrix;
	}

	static void info()
	{
		std::cout << "    Denseblock : " << rows << " x " << cols << " | column " << type_name<col>() << std::endl;
	}	

	constexpr inline std::size_t nonZeros_impl()
	{
		return rows * cols;
	}
};

template<typename Scalar, int rows_, typename col_>
struct SparseBlock : Block<SparseBlock<Scalar, rows_, col_>>
{
	using Derived = SparseBlock<Scalar, rows_, col_>;
    using col = typename BlockTraits<Derived>::col;
    using matrix_t = typename BlockTraits<Derived>::matrix_t;
	static const int rows = BlockTraits<Derived>::rows;
	static const int cols = BlockTraits<Derived>::cols;

	// Sparse storage
	Eigen::SparseMatrix<Scalar> m_matrix;

	inline void toDense_impl(Eigen::Ref<matrix_t> out)
	{
		out = matrix_t(m_matrix);
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
template<typename Scalar, int rows_, typename col_>
struct BlockTraits<DenseBlock<Scalar, rows_, col_> > {
	using col = col_;
	static const int rows = rows_;
	static const int cols = col::len;
	using matrix_t = Eigen::Matrix<Scalar, rows, cols>;
};

// BlockTraits specialization for ZeroBlock
template<typename Scalar, int rows_, typename col_>
struct BlockTraits<ZeroBlock<Scalar, rows_, col_> > {
	using col = col_;
	static const int rows = rows_;
	static const int cols = col::len;
	using matrix_t = Eigen::Matrix<Scalar, rows, cols>;
};

template<typename Scalar, int len_, typename... blocks>
struct Row
{
	static const int len = len_;
	// using block_types = std::tuple<blocks...>;

	using block_tuple = std::tuple<blocks...>;
	block_tuple m_blocks;


	/** Returns true if col is in row, false otherwise
	 */
	template<typename col>
	static constexpr bool contains_column()
	{
		return meta::contains<col, typename BlockTraits<blocks>::col...>::value;
	}

	// Meta-function returning true is the block is non-zero and false otherwise
	template<typename col>
	static constexpr bool is_nonzero = contains_column<col>();


	// Meta-function to get the type of the block for column col
	// Returns ZeroBlock if the block is not present
	template<typename col>
	struct block_type
	{
		// We extend the tuple so that both branches of the conditional below are valid, 
		// even when the search didn't find it
		using fake_column = int;
		using extended_tuple = std::tuple<fake_column, typename BlockTraits<blocks>::col...>;

		template<int i>
		struct get_type_by_index
		{
			using type = std::tuple_element_t<i, std::tuple<fake_column, blocks...>>;
		};

		static constexpr int index = meta::get_index<col, fake_column, typename BlockTraits<blocks>::col...>(0);
		using type = std::conditional_t<index == 0, // Didn't find it
						ZeroBlock<Scalar, len, col>,
						typename get_type_by_index<index>::type>;
	};
	template<typename col>
	using block_type_t = typename block_type<col>::type;

	/** Print basic info about the matrix
	 */
	static void info()
	{
		std::cout << "size " << len << " with " << sizeof...(blocks) << " blocks\n";
        (void)std::initializer_list<int>{
            (
            	blocks::info(),
                0
            )...
        };
	}

	// // This block does not exist - so return zero block
	// template<typename col, 
	// 		 std::enable_if_t<!contains_column<col>(), bool> = true>
	// inline const block_type_t<col> get()
	// {
	// 	return ZeroBlock<Scalar, len, col>();
	// }

	// This block does exist - so return reference to dense block
	// template<typename col, 
	// 		 std::enable_if_t<contains_column<col>(), bool> = true>
	template<typename col>
	inline block_type_t<col>& get()
	{
		const int ind = meta::get_index<col, typename BlockTraits<blocks>::col...>();
		return std::get<ind>(m_blocks);
	}

};


// Helper to produce a row of all dense blocks
template<typename Scalar, int len, typename... vars>
struct DenseRow : Row<Scalar, len, DenseBlock<Scalar, len, vars>...>
{};

template<typename Scalar, typename columnTuple, typename rowTuple>
struct BSMatrix;

template<typename Scalar, typename... columns, typename... rows>
struct BSMatrix<Scalar, std::tuple<columns...>, std::tuple<rows...>>
{
	static constexpr std::size_t num_rows = sum_template<rows::len...>();
	static constexpr std::size_t num_cols = sum_template<columns::len...>();

	using rowTuple = std::tuple<rows...>;
	rowTuple m_rows;

	// Meta-function to get the type of the block at (row, column)
	template<typename row, typename col>
	struct block_type
	{
		static constexpr int index = meta::get_index<row, rows...>();
		using row_type = std::tuple_element_t<index, rowTuple>;
		using type = typename row_type::template block_type_t<col>;
	};
	template<typename row, typename col>
	using block_type_t = typename block_type<row, col>::type;

	// Meta-function returning true is the block is non-zero and false otherwise
	template<typename row, typename col>
	static constexpr bool is_nonzero = row::template is_nonzero<col>;

public:

	/** Print basic info about the matrix
	 */
	static void info()
	{
		std::cout << "Block sparse matrix with " << sizeof...(rows) << " x " 
				  << sizeof...(columns) << " blocks\n";

        (void)std::initializer_list<int>{
            (
            	std::cout << "  " << type_name<rows>() << ": ",
            	rows::info(),
                0
            )...
        };
	}

	template<typename row, typename column>
	void block_info()
	{
		std::get<row>(m_rows).template get<column>().info();
	}

	template<typename row>
	static constexpr int row_len()
	{
		// Make a test here to confirm that row is in the index?
		return row::len;
	}

	template<typename column>
	static constexpr int column_len()
	{
		return column::len;
	}

	/** Returns a reference to the requested block
	 */
	template<typename row, typename column>
	inline block_type_t<row, column>& get()
	{ // xxxxxx
		// row& x = std::get<row>(m_rows);
		return std::get<row>(m_rows).template get<column>();
	}

private:

	template<typename row, typename col, 
			 std::enable_if_t<is_nonzero<row, col>, bool> = true>
	inline void toDense_block(Eigen::Ref<typename block_type_t<row, col>::matrix_t> B)
	{
		get<row, col>().toDense(B);
	}

	template<typename row, typename col, 
			 std::enable_if_t<!is_nonzero<row, col>, bool> = true>
	inline void toDense_block(Eigen::Ref<typename block_type_t<row, col>::matrix_t> B)
	{
	}


	/** Copy the row into the given dense matrix
	 */
	template<typename row>
	inline void toDense_row(Eigen::Ref<Eigen::Matrix<Scalar, row::len, num_cols>> D)
	{
		int col = 0;
	    (void)std::initializer_list<int>{
	        (
	        	toDense_block<row, columns>(D.template block<row::len, columns::len>(0, col)),
				col += columns::len,
	            0
	        )...
	    };
	}


	// Compute the number of non-zeros in this block
	template<typename row, typename col, 
			 std::enable_if_t<is_nonzero<row, col>, bool> = true>
	inline std::size_t nonZeros_block()
	{
		return get<row, col>().nonZeros();
	}

	template<typename row, typename col, 
			 std::enable_if_t<!is_nonzero<row, col>, bool> = true>
	inline std::size_t nonZeros_block()
	{
		return 0;
	}

	// Compute the number of non-zeros in this row
	template<typename row>
	constexpr std::size_t nonZeros_row()
	{
		std::size_t nnz = 0;
	    (void)std::initializer_list<int>{
	        (
	        	nnz += nonZeros_block<row, columns>(),
	            0
	        )...
	    };
	    return nnz;
	}

public:
	/** Copy into a dense matrix
	 */
	inline void toDense(Eigen::Ref<Eigen::Matrix<Scalar, num_rows, num_cols>> D)
	{
		D.array() = 0;
		int row = 0;
	    (void)std::initializer_list<int>{
	        (
				toDense_row<rows>(D.template block<rows::len, num_cols>(row, 0)),
				row += rows::len,
	            0
	        )...
	    };

	}

	/** Copy into a sparse matrix
	 */
	inline void toSparse(Eigen::SparseMatrix<Scalar>& S)
	{
		S.reserve(nonZeros());
		int row = 0;
	    (void)std::initializer_list<int>{
	        (
				toDense_row<rows>(D.template block<rows::len, num_cols>(row, 0)),
				row += rows::len,
	            0
	        )...
	    };

	}



	inline std::size_t nonZeros()
	{
		std::size_t nnz = 0;
	    (void)std::initializer_list<int>{
	        (
				nnz += nonZeros_row<rows>(),
	            0
	        )...
	    };
	    return nnz;
	}

	/** Reserve memory in the sparse matrix
	 */
	inline void reserve(Eigen::SparseMatrix<Scalar>& S)
	{
		S.reserve(nonZeros());
	}
};



using Scalar = double;

// Define column tags
struct x : Column<Scalar, 2> {};
struct y : Column<Scalar, 3> {};
struct z : Column<Scalar, 4> {};

int main()
{
	struct con1 : DenseRow<Scalar, 3, x, y> {};
	struct con2 : DenseRow<Scalar, 2, x> {};
	struct con3 : DenseRow<Scalar, 4, z, y> {};

	using mat_t = BSMatrix<Scalar, std::tuple<x, y, z>, std::tuple<con1, con2, con3>>;
	mat_t mat;

	mat.info();

	std::cout << "\n\n======================\n\n" << std::endl;

	std::cout << "type(con1, x) = " << type_name<mat_t::block_type_t<con1, x>>() << std::endl;
	std::cout << "type(con1, y) = " << type_name<mat_t::block_type_t<con1, y>>() << std::endl;
	std::cout << "type(con1, z) = " << type_name<mat_t::block_type_t<con1, z>>() << std::endl;

	std::cout << "type(con2, x) = " << type_name<mat_t::block_type_t<con2, x>>() << std::endl;
	std::cout << "type(con2, y) = " << type_name<mat_t::block_type_t<con2, y>>() << std::endl;
	std::cout << "type(con2, z) = " << type_name<mat_t::block_type_t<con2, z>>() << std::endl;

	std::cout << "type(con3, x) = " << type_name<mat_t::block_type_t<con3, x>>() << std::endl;
	std::cout << "type(con3, y) = " << type_name<mat_t::block_type_t<con3, y>>() << std::endl;
	std::cout << "type(con3, z) = " << type_name<mat_t::block_type_t<con3, z>>() << std::endl;

	std::cout << "\n\n======================\n\n" << std::endl;

	std::cout << "type(mat.get<con1, x>()) = " << type_name<decltype(mat.get<con1, x>())>() << std::endl;
	std::cout << "type(mat.get<con1, y>()) = " << type_name<decltype(mat.get<con1, y>())>() << std::endl;
	std::cout << "type(mat.get<con1, z>()) = " << type_name<decltype(mat.get<con1, z>())>() << std::endl;

	std::cout << "type(mat.get<con2, x>()) = " << type_name<decltype(mat.get<con2, x>())>() << std::endl;
	std::cout << "type(mat.get<con2, y>()) = " << type_name<decltype(mat.get<con2, y>())>() << std::endl;
	std::cout << "type(mat.get<con2, z>()) = " << type_name<decltype(mat.get<con2, z>())>() << std::endl;

	std::cout << "type(mat.get<con3, x>()) = " << type_name<decltype(mat.get<con3, x>())>() << std::endl;
	std::cout << "type(mat.get<con3, y>()) = " << type_name<decltype(mat.get<con3, y>())>() << std::endl;
	std::cout << "type(mat.get<con3, z>()) = " << type_name<decltype(mat.get<con3, z>())>() << std::endl;

	std::cout << "\n\n======================\n\n" << std::endl;

	std::cout << "mat_t::is_nonzero<con1, x> = " << mat_t::is_nonzero<con1, x> << std::endl;
	std::cout << "mat_t::is_nonzero<con1, y> = " << mat_t::is_nonzero<con1, y> << std::endl;
	std::cout << "mat_t::is_nonzero<con1, z> = " << mat_t::is_nonzero<con1, z> << std::endl;

	std::cout << "mat_t::is_nonzero<con2, x> = " << mat_t::is_nonzero<con2, x> << std::endl;
	std::cout << "mat_t::is_nonzero<con2, y> = " << mat_t::is_nonzero<con2, y> << std::endl;
	std::cout << "mat_t::is_nonzero<con2, z> = " << mat_t::is_nonzero<con2, z> << std::endl;

	std::cout << "mat_t::is_nonzero<con3, x> = " << mat_t::is_nonzero<con3, x> << std::endl;
	std::cout << "mat_t::is_nonzero<con3, y> = " << mat_t::is_nonzero<con3, y> << std::endl;
	std::cout << "mat_t::is_nonzero<con3, z> = " << mat_t::is_nonzero<con3, z> << std::endl;

	std::cout << "\n\n======================\n\n" << std::endl;

	auto& q = mat.get<con1, x>().m_matrix;
	q.array() = 1;
	mat.get<con1, y>().m_matrix.array() = 2;
	mat.get<con2, x>().m_matrix.array() = 3;
	mat.get<con3, z>().m_matrix.array() = 4;
	mat.get<con3, y>().m_matrix.array() = 5;

	Eigen::Matrix<Scalar, mat_t::num_rows, mat_t::num_cols> D;
	mat.toDense(D);

	std::cout << "D = \n" << D << std::endl;


	Eigen::SparseMatrix<Scalar> S(mat_t::num_rows, mat_t::num_cols);
	mat.reserve(S);
	S.makeCompressed();

	std::cout << "nnz S = " << S.nonZeros() << std::endl;

	mat.toSparse(S);

	return 0;
}