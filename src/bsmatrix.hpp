#ifndef __BSMATRIX_HPP
#define __BSMATRIX_HPP

#include <array>
#include <iostream>
#include <iomanip>

#include "Eigen/Dense"
#include "Eigen/Sparse"

#include "lampc_utility.hpp"

namespace BS {

	namespace detail {
		/** Return the sum of the integers
		 */
		template<int... i>
		constexpr int sum_int_template()
		{
			int acc = 0;
			auto l = { (acc += i, 0)... };
			return acc;
		}
	};

template<typename Scalar, 
				 int _numBlockRows, int _numBlockColumns, // Number of blocks
				 int rowsAtCompileTime, int colsAtCompileTime, // Total size of the matrix
				 int nnzBlocks, // Number of non-zero blocks
				 int nnzBlockColumns, // Number of columns in the nonzero blocks
				 int nnzEstimate // Estimate of the number of non-zeros
				 >
class BSMatrix
{
public:
	static const int numBlockRows = _numBlockRows;
	static const int numBlockColumns = _numBlockColumns;

	// Size of each block
	Eigen::Vector<int, numBlockColumns> columnSizes;
	Eigen::Vector<int, numBlockRows> rowSizes;

	// Offset of each block in scalars
	// Note valid until after finalize is called
	Eigen::Vector<int, numBlockColumns> columnIndices;
	Eigen::Vector<int, numBlockRows> rowIndices;

private:

	/**
	 * Blocks are stored in a compressed row format
	 * 
	 * [B1 B2 0 B3]
	 * [0  B4 0 0 ]
	 * [B5 0  0 B6]
	 * 
	 * Is stored as 
	 * blocks = [B1, B2, B3, B4, B5, B6]
	 * 
	 * Each block stores its block-location and location in the dense matrix.
	 * 
	 * We store a row-vector that provides an offset to the first 
	 * non-zero block of each row.
	 */

	/** 
	 * Offset into the blocks vector of the first non-zero block of this row
	 * 
	 * [B1 B2 0 B3]
	 * [0  B4 0 0 ]
	 * [0  0  0 0 ]
	 * [B5 0  0 B6]
	 * 
	 * This matrix would have a blockRows = [0 3 4 4 6]
	 * The first block in row r is blockRows[r]
	 * The number of non-zero columns in row r is blockRows[r+1] - blockRows[r]
	 */
	Eigen::Vector<int, numBlockRows + 1> blockRows;

	struct Block
	{
		int blkColumnOffset; // Offset for the first column into blkColumnInfo vector
		int row, column; // Location of the top left element
		int blockRow, blockColumn; // Block location
	};
	Eigen::Vector<Block, nnzBlocks> blocks;
	int next_block = 0; // Pointer to the first unused element of blocks

	struct BlockColumnInfo
	{
		int valuePtr_offset; // Offset into the sparse matrix valuePtr
		int nnz; // Number of non-zeros in this block-column
	};

	// Offsets of each block column into output sparse matrix
	Eigen::Vector<BlockColumnInfo, nnzBlockColumns> blkColumnInfo;
	int next_blkColumnInfo = 0; // Pointer to the first unused element of column_offsets


public:
	Eigen::SparseMatrix<Scalar> S{rowsAtCompileTime, colsAtCompileTime};

	// BSMatrix(Eigen::Vector<int, numBlockRows> rowSizes_, Eigen::Vector<int, numBlockColumns> columnSizes_)
	BSMatrix(std::array<int, numBlockRows> rowSizes_, std::array<int, numBlockColumns> columnSizes_)
		: rowSizes(rowSizes_.data()), columnSizes(columnSizes_.data())
	{
		// rowSizes = rowSizes_;
		// columnSizes = columnSizes_;

		S.reserve(nnzEstimate);
	}

	/**
	 * Set a block to non-zero. Must be specified in compressed row-major order.
	 */
	void addDenseBlock(int blockRow, int blockColumn)
	{
		assert(next_blkColumnInfo <= blkColumnInfo.size());
		assert(next_block <= blocks.size());

		// Check that we're getting the blocks in compressed row-major order
		if(next_block > 0)
		{
			assert(blocks[next_block-1].blockRow <= blockRow);
			if(blocks[next_block-1].blockRow == blockRow)
				assert(blocks[next_block-1].blockColumn <= blockColumn);
		}

		Block &block = blocks[next_block++];

		block.blkColumnOffset = next_blkColumnInfo; 
		next_blkColumnInfo += columnSizes[blockColumn];

		// Assuming this is a dense block
		int nnz = rowSizes(blockRow) * columnSizes(blockColumn);

		// Location of the block
		block.row = rowSizes.head(blockRow).sum();
		block.column = columnSizes.head(blockColumn).sum();

		block.blockRow = blockRow;
		block.blockColumn = blockColumn;

		// Reserve memory for the block
		for(int r=block.row; r<block.row + rowSizes[blockRow]; r++)
			for(int c=block.column; c<block.column + columnSizes[blockColumn]; c++)
				S.insert(r,c) = 1;
	}

	/**
	 * Must be called after all nonzero blocks have been declared
	 */
	void finalize()
	{
		// Check that all blocks have been filled
		assert(next_blkColumnInfo == blkColumnInfo.size());
		assert(next_block == blocks.size());

		// Get pointers into the blocks to accelerate rowwise access
		blockRows(0) = 0;
		for(int r=1; r<blockRows.size()-1; r++)
		{
			int i;
			for(i=0; i<blocks.size(); i++)
				if(blocks(i).blockRow >= r) break;
			blockRows(r) = i;
		}
		blockRows(blockRows.size()-1) = nnzBlocks;

		S.makeCompressed();

		// Get pointers to the start of each block-column 
		// and fill in the blkColumnInfo vector

		// First we write the offsets into the valuePtr so we can find
		// the offset locations
		Scalar* data = S.valuePtr();
		for(int i=0; i<S.nonZeros(); i++)
			data[i] = i;

		for(const auto &block : blocks)
		{
			for(int ind=0; ind<columnSizes[block.blockColumn]; ind++)
			{
				BlockColumnInfo& info = blkColumnInfo[block.blkColumnOffset + ind];

				// Get the offset into the valueptr for the first non-zero element in this column
				// Iterate over all non-zeros in this column to find the first one larger than our block row
				info.valuePtr_offset = 0; // Will stay zero of nnz = 0 
				for (typename Eigen::SparseMatrix<Scalar>::InnerIterator it(S, block.column + ind); it; ++it)
				{
					if(it.row() >= block.row)
					{
						info.valuePtr_offset = it.value();
						break;
					}
				}

				// Compute the number of non-zeros in the column
				info.nnz = 0;
				for (typename Eigen::SparseMatrix<Scalar>::InnerIterator it(S, block.column + ind); it; ++it)
					if(it.row() >= block.row && it.row() < block.row + rowSizes[block.blockRow])
						info.nnz++;
			}
		}

		// Set the row indices
		rowIndices[0] = 0;
		for(int i=1; i<numBlockRows; i++)
			rowIndices[i] = rowIndices[i-1] + rowSizes[i-1];

		// Set the column indices
		columnIndices[0] = 0;
		for(int i=1; i<numBlockColumns; i++)
			columnIndices[i] = columnIndices[i-1] + columnSizes[i-1];
	}

	/** 
	 * Returns true if block is nonzero.
	 */
	bool isNonzero(int blockRow, int blockColumn)
	{
		return getBlockIndex(blockRow, blockColumn) >= 0;
	}

	/** 
	 * Get a dense copy of a block.
	 */
	Eigen::MatrixX<Scalar> getBlock(int blockRow, int blockColumn)
	{
		Eigen::MatrixX<Scalar> mat(rowSizes[blockRow], columnSizes[blockColumn]);
		mat.array() = 0;

		int ind = getBlockIndex(blockRow, blockColumn);
		if(ind >= 0)
		{
			Block &blk = blocks[ind];
			for(int i=0; i<mat.cols(); i++)
			{
				auto &info = blkColumnInfo[blk.blkColumnOffset + i];				
				assert(info.nnz == rowSizes[blockRow] && "Only dense blocks implemented so far.");

				mat.col(i) = Eigen::Map<Eigen::VectorX<Scalar>>(S.valuePtr() + info.valuePtr_offset, rowSizes[blockRow]);
			}
		}

		return mat;
	}

	/** 
	 * Copy the dense block into the BSMatrix.
	 */
	void setBlockByIndex(int ind, const Eigen::Ref<const Eigen::MatrixX<Scalar>>& mat)
	{
		Block &blk = blocks[ind];
		assert(ind >= 0 && "Attempt to set zero block");

		int blockRow = blk.blockRow;
		int blockColumn = blk.blockColumn;
		assert(mat.rows() == rowSizes[blockRow] && mat.cols() == columnSizes[blockColumn] && "setBlock: Matrix is the wrong size");

		for(int i=0; i<mat.cols(); i++)
		{
			auto &info = blkColumnInfo[blk.blkColumnOffset + i];				
			assert(info.nnz == rowSizes[blockRow] && "Only dense blocks implemented so far.");

			Eigen::Map<Eigen::VectorX<Scalar>>(S.valuePtr() + info.valuePtr_offset, rowSizes[blockRow]) = mat.col(i);
		}
	}

	/**
	 * Get index into blocks. 
	 * If block doesn't exist, returns -1.
	 */
	int getBlockIndex(int blockRow, int blockColumn)
	{
		// For now, we just do a simple linear search assuming that the number of blocks per row is small
		for(int ind = blockRows[blockRow]; ind < blockRows[blockRow+1]; ind++)
			if(blocks[ind].blockColumn == blockColumn) return ind;
		return -1;
	}

	/** 
	 * Copy the dense block into the BSMatrix.
	 */
	void setBlock(int blockRow, int blockColumn, const Eigen::Ref<const Eigen::MatrixX<Scalar>>& mat)
	{
		int ind = getBlockIndex(blockRow, blockColumn);
		assert(ind >= 0 && "Attempt to set zero block");

		setBlockByIndex(ind, mat);
	}


	// /** 
	//  * Copy the given matrix into block (r, c) of the SparseMatrix S
	//  */
	// void set(Eigen::SparseMatrix<Scalar> S, int r, int c, Eigen::MatrixX<Scalar> B)
	// {
	// 	assert(blocks[r][c].nonZero);

	// 	int offset = blocks[r][c].offset;
	// 	// for(int i=0; i<blocks[r][c].
	// }
};


// /**
//  * FunctionSet
//  * 
//  * Take a list of functions [F1;...;FN] and their arguments such that 
//  * we can evaluate them as a single function F = [F1;...;FN]
//  */
// template<typename Scalar, typename param_t, 
// 				int numInputs_, int numOutputs_,
// 				typename jacobian_BSMatrix_t,
// 				typename... F> // The functions to call in order
// class FunctionSetx
// {
// public:
// 	static const int totalNumArgs = detail::sum_int_template<F::num_input_vars...>();

// 	static const int numInputs = numInputs_;
// 	static const int numOutputs = numOutputs_;

// 	using variable_t = Eigen::Vector<Scalar, numInputs>;
// 	using output_t = Eigen::Vector<Scalar, numOutputs>;

// 	using jacobian_t = Eigen::SparseMatrix<Scalar>;


// private:
// 	jacobian_BSMatrix_t jacobian_BS;


// public: // Public for the Factory class, not the user
// 	/**
// 	 * We build an array of info about the arguments in the calling sequence.
// 	 * 
// 	 * [F1(arg1, arg2); F2(arg3, arg4, arg5); ... ]
// 	 */
// 	struct argInfo_t
// 	{
// 		int index;  // Index of the block-variable
// 		int offset;  // Offset into the dense variable
// 		int len;  // Length

// 		int jacobianBlockIndex; // Index into the block for this argument's jacobian
// 	};

// 	Eigen::Vector<argInfo_t, totalNumArgs> argInfo;

// private:
// 	template<typename f, int... Is>
// 	void call_F(
// 		param_t& param,
// 		const Eigen::Ref<const Eigen::VectorX<Scalar>>& x,
// 		Eigen::Ref<Eigen::VectorX<Scalar>> out,
// 		int index,  // Index into the argInfo vector
// 		std::integer_sequence<int, Is...>)
// 	{
// 		out = f::eval(param, 
// 									x.segment(argInfo[index + Is].offset,
// 														argInfo[index + Is].len)...);
// 	}

// 	// Evaluate the function and copy the jacobian into jacobian_BS
// 	template<typename f, int... Is>
// 	void call_jacobian_F(
// 		param_t& param,
// 		const Eigen::Ref<const Eigen::VectorX<Scalar>>& x,
// 		Eigen::Ref<Eigen::VectorX<Scalar>> out,
// 		int index,  // Index into the argInfo vector
// 		std::integer_sequence<int, Is...>)
// 	{
// 		// f::jac(param, 
// 		// 	std::make_pair(x.segment(argInfo[index + Is].offset, argInfo[index + Is].len), 
// 		// 	columnOffsets.segment(argInfo[index+Is].offset, argInfo[index + Is].len))...)

// 		typename f::jacobian_return_t jac = f::jac(param, x.segment(argInfo[index + Is].offset,
// 													argInfo[index + Is].len)...);
// 		out = jac.val;

// 		// Copy the jacobian into the right blocks
// 		int offset = 0;
// 		auto l = {(
// 			jacobian_BS.setBlockByIndex(argInfo[index + Is].jacobianBlockIndex, 
// 																	jac.jacobian.middleCols(offset,argInfo[index + Is].len)),
// 			offset += argInfo[index + Is].len,
// 			0
// 			)...};
// 	}

// public:

// 	/**
// 	 * Evalue the FunctionSet for the variable x
// 	 */
// 	void eval(
// 		param_t &param,
// 		const Eigen::Ref<const Eigen::Vector<Scalar, numInputs>>& x,
// 		Eigen::Ref<Eigen::Vector<Scalar, numOutputs>> out)
// 	{
// 		int output_index = 0;
// 		int arg_index = 0;

// 		/**
// 		 * Iterate over each function in the set and evaluate.
// 		 * 
// 		 * We're unrolling the loop here, so there is one line of 
// 		 * generated code per function in the set.
// 		 * However, the templated function callF is overloaded only by
// 		 * the underlying function, and not by the constraint. i.e., we've
// 		 * decoupled the variable tag from the function tag here. This means
// 		 * that callF is generated only once per unique function called.
// 		 */
// 		auto l = {(
// 			call_F<F>(
// 				param, 
// 				x,
// 				out.segment(output_index, F::num_outputs),
// 				arg_index,
// 				std::make_integer_sequence<int, F::num_input_vars>()),

// 			arg_index += F::num_input_vars,
// 			output_index += F::num_outputs,
// 			0)...};
// 	}

// 	/**
// 	 * Evalue the jacobian of the FunctionSet for the variable x
// 	 */
// 	void jacobian(
// 		param_t &param,
// 		const Eigen::Ref<const Eigen::Vector<Scalar, numInputs>>& x,
// 		Eigen::Ref<Eigen::Vector<Scalar, numOutputs>> out)
// 	{
// 		int output_index = 0;
// 		int arg_index = 0;

// 		auto l = {(
// 			call_jacobian_F<F>(
// 				param, 
// 				x,
// 				out.segment(output_index, F::num_outputs),
// 				arg_index,
// 				std::make_integer_sequence<int, F::num_input_vars>()),

// 			arg_index += F::num_input_vars,
// 			output_index += F::num_outputs,
// 			0)...};
// 	}

// 	/**
// 	 * Returns a reference to the jacobian matrix
// 	 * 
// 	 * This is overwritten when jacobian is called.
// 	 */
// 	Eigen::SparseMatrix<Scalar>& get_jacobian()
// 	{
// 		return jacobian_BS.S;
// 	}


// 	FunctionSetx(Eigen::Vector<argInfo_t, totalNumArgs> argInfo_, jacobian_BSMatrix_t jacobian_BS_)
// 		: argInfo(argInfo_), jacobian_BS(jacobian_BS_)
// 		{}
// };

};

#endif // __BSMATRIX_HPP
