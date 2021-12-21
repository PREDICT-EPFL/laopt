#ifndef __BSMATRIXCORE_HPP
#define __BSMATRIXCORE_HPP

#include "Eigen/Dense"
#include "Eigen/Sparse"

#include "bsblockbase.hpp"
#include "bsutil.hpp"
#include "bszeroblock.hpp"

namespace BS
{

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
		// return row.template get<column>();
		return std::get<row>(m_rows).template get<column>();
	}

private:

	template<typename row, typename col, 
			 std::enable_if_t<is_nonzero<row, col>, bool> = true>
	inline void toDense_block(Eigen::Ref<typename block_type_t<row, col>::dense_matrix_t> B)
	{
		get<row, col>().toDense(B);
	}

	template<typename row, typename col, 
			 std::enable_if_t<!is_nonzero<row, col>, bool> = true>
	inline void toDense_block(Eigen::Ref<typename block_type_t<row, col>::dense_matrix_t> B)
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


private:

	template<typename row, typename column, typename StorageIndex,
			 std::enable_if_t<is_nonzero<row, column>, bool> = true>
	EIGEN_STRONG_INLINE std::size_t toSparse_block(
		std::size_t   col,
		std::size_t   row_offset,
		Scalar*       valuePtr,
		StorageIndex* innerIndexPtr)
	{
		return get<row, column>().toSparseColumn(col, row_offset, valuePtr, innerIndexPtr);
	}

	template<typename row, typename column, typename StorageIndex,
			 std::enable_if_t<!is_nonzero<row, column>, bool> = true>
	EIGEN_STRONG_INLINE std::size_t toSparse_block(
		std::size_t   col,
		std::size_t   row_offset,
		Scalar*       valuePtr,
		StorageIndex* innerIndexPtr)
	{
		return 0;
	}


	/** Copy columns associated to column to the sparse matrix
	 * 
	 * Returns the number of elements copied
	 */
	template<typename column, typename StorageIndex>
	EIGEN_STRONG_INLINE std::size_t toSparse_column(
		Scalar*       valuePtr,
		StorageIndex* innerIndexPtr,
		StorageIndex* outerIndexPtr)
	{
    	std::size_t num; // Number of non-zeros in given column of given block
    	std::size_t total = 0; // Total number of non-zeros in this block-column
    	std::size_t row_offset; // Row start index for each block

    	// Copy each column 
    	for(int col=0; col<column::len; col++)
    	{
    		row_offset = 0;

	    	// Copy into the sparse matrix in block-row order
		    (void)std::initializer_list<int>{
		        (
		        	num = toSparse_block<rows, column>(col, row_offset, valuePtr, innerIndexPtr),
		        	row_offset += rows::len,
					innerIndexPtr += num,
					valuePtr += num,
					total += num,
		            0
		        )...
		    };
			// Compute the start-location of the next column
			outerIndexPtr[col+1] = outerIndexPtr[0] + total;
		}
		return total;
	}


public:
	/** Copy into a sparse matrix in compressed column order
	 */
	inline void toSparse(Eigen::SparseMatrix<Scalar>& S)
	{
		S.resize(num_rows, num_cols);
		S.reserve(nonZeros()); // Should do nothing if space is already reserved

		using StorageIndex = typename Eigen::SparseMatrix<Scalar>::StorageIndex;

    	Scalar* valuePtr = S.valuePtr();
    	StorageIndex* innerIndexPtr = S.innerIndexPtr();
    	StorageIndex* outerIndexPtr = S.outerIndexPtr();

    	// Copy into the sparse matrix one block-column at a time
    	outerIndexPtr[0] = 0; // Location of the first data point
		std::size_t num;
	    (void)std::initializer_list<int>{
	        (
				num = toSparse_column<columns, StorageIndex>(valuePtr, 
															 innerIndexPtr, 
															 outerIndexPtr),
				valuePtr += num,
				innerIndexPtr += num,
				outerIndexPtr += columns::len,
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

};

#endif // __BSMATRIXCORE_HPP