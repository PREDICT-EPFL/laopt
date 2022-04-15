#ifndef __BSMATRIX_HPP
#define __BSMATRIX_HPP

#include <functional>

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace lampc
{

/**
 * Copy information for a sparse block matrix
 */
struct Segment
{
	int index;  // Index into the target.valuePtr()
	int length; // Number of element to copy
};
struct CopyInfo
{
	int segment_index;  // Index into segments
	int num_segments_to_copy;  // Number of segments to copy to execute this task
};

std::ostream &operator<<(std::ostream &os, std::vector<Segment> const &sequence) 
{
	for(auto& seg: sequence)
		os << "(" << seg.index << "," << seg.length << ")";
	return os;
}
std::ostream &operator<<(std::ostream &os, std::vector<CopyInfo> const &sequence) 
{
	for(auto& seg: sequence)
		os << "(" << seg.segment_index << "," << seg.num_segments_to_copy << ")";
	return os;
}


/**
 * Class to record the copy sequence for a series of sparse matrices.
 */
template<typename scalar_t>
class BSMatrixTape
{
public:
	// Sparse matrix structure
	// S.valuePtr = [0,1,2,...]
	SparseMatrix<int> S;
	
	// List of non-zero coefficients
	std::vector< Eigen::Triplet<int> > trip;

	// Partition of the next block to copy
	std::vector<Segment> row_partition;
	std::vector<Segment> col_partition;

	// The copy sequence data (output of the tape)
	std::vector<Segment> copy_sequence;
	std::vector<int> copy_lengths; // copy_lengths[i] == number of segments to execute for the i'th copy

	enum State { 
			build_structure, // Initial state. Recording the structure of S
			create_copy_sequence};  // Phase II. Recoding the copy sequence

	State state = build_structure;

	/**
	 * Return the location in the target where ind in the source should be copied
	 * 
	 * partition = {{index_i,len_i}, ...}
	 * Partitions a vector of length sum len_i into segements starting at the index_i's
	 */
	int get_target_location(int ind, std::vector<Segment> &partition)
	{
		for(auto &segment: partition)
		{
			if(ind >= segment.length) ind -= segment.length;
			else return segment.index + ind;
		}
		throw std::runtime_error("Invalid index passed to get_target_location");
		assert(0 && "Invalid index passed to get_target_location");
		return -1;
	}

	/**
	 * Copy source into target
	 * 
	 * The source is partitioned into blocks and copied into target according to
	 * - rows = {{target_row, len}, ...}
	 * - cols = {{target_col, len}, ...}
	 * 
	 * Returns a vector of Segment specifying the copies to be done on the source data in 
	 * data-contiguous order to achieve the requested sparse block-copy.
	 */
	std::vector<Segment> build_copy_sequence(const Eigen::SparseMatrix<int>& source)
	{
		if(!source.isCompressed()) std::runtime_error( "Source matrix must be in compressed format" );

		// We iterate over the source in data-continuous order, defining the copy sequence to the target
		std::vector<Segment> seq; // The copying sequence

		for (int c=0; c<source.outerSize(); ++c)
		{
			int tcol = get_target_location(c, col_partition);
			for (SparseMatrix<int>::InnerIterator it(source,c); it; ++it)
			{
				int trow = get_target_location(it.row(), row_partition);
				int tindex = S.coeffRef(trow,tcol); // Index into the data at the target location

				// std::cout << fmt::format("source ({:3d},{:3d}) -> target ({:3d},{:3d}) [{:3d}]", it.row(), c, trow, tcol, tindex);

				// The next target index if the data is contiguous
				if(seq.size() == 0 || tindex != seq.back().index + seq.back().length)
				{
					// std::cout << "  NON-CONTIGUOUS\n";
					seq.push_back({.index = tindex, .length=1});
				}
				else
				{
					// std::cout << "  CONTIGUOUS\n";
					seq.back().length++;
				}
			}		
		}

		return seq;
	}

	void finalize_partition(int rows, int cols)
	{
		if(row_partition.back().length == -1) row_partition.back().length = rows;
		if(col_partition.back().length == -1) col_partition.back().length = cols;
	}

	/**
	 * Computes the current shape of the matrix, before the sparsity pattern is complete
	 */
	std::pair<int,int> nonfinalized_shape()
	{
		int rows = 0;
		int cols = 0;
		for(auto &t : trip) 
		{
			rows = std::max(rows, t.row() + 1);
			cols = std::max(cols, t.col() + 1);
		}
		return std::make_pair(rows, cols);
	}

	/**
	 * Execute the copy operation on this block
	 */
	void record_op(const Eigen::MatrixX<scalar_t>& block)
	{
		finalize_partition(block.rows(), block.cols());

		// Iterate over partition
		for(auto& row_seg: row_partition)
			for(auto& col_seg: col_partition)
				// Fill in the non-zeros
		    for(int r=0; r<row_seg.length; r++)
		    	for(int c=0; c<col_seg.length; c++)
		    		trip.push_back(Eigen::Triplet<int>(r + row_seg.index, c + col_seg.index, 1));

		if(state == create_copy_sequence) // Copy block sparsity structure
		{
			Eigen::MatrixX<scalar_t> B(block);
			B.array() = 1;
			std::vector<Segment> v = build_copy_sequence(B.sparseView().template cast<int>());

			copy_sequence.insert(copy_sequence.end(), v.begin(), v.end());
			copy_lengths.push_back(v.size());
		}

		row_partition.clear();
	}

public:
	/**
	 * Record the data to copy block to (target_row, target_column)
	 */
	void operator=(const Eigen::SparseMatrix<scalar_t>& block)
	{
		finalize_partition(block.rows(), block.cols());

		std::cout << "NOT IMPLEMENTED YET!!!\n";

		if(state == create_copy_sequence) // Copy block sparsity structure
		{
			// Record copy data to copy block to (target_row, target_column)
			std::vector<Segment> v = build_copy_sequence(block);
		}

		row_partition.clear();
	}	

	/**
	 * Copy the given block into the BSMatrix according to the partition set in operator()
	 * 
	 * Assumption: The input matrix is contiguous. Don't change this to a Ref.
	 */
	void operator=(const Eigen::MatrixX<scalar_t>& block)
	{
		record_op(block);
	}
	void operator+=(const Eigen::MatrixX<scalar_t>& block)
	{
		record_op(block);
	}
	void operator-=(const Eigen::MatrixX<scalar_t>& block)
	{
		record_op(block);
	}

	/**
	 * Set the target rows and columns according to the given partition.
	 * 
	 * The source is partitioned into blocks and copied into target according to
	 * - rows = {{target_row, len}, ...}
	 * - cols = {{target_col, len}, ...}
	 * 
	 * The blocks are taken contiguously from the matrix to be copied in.
	 */
	BSMatrixTape& operator()(std::vector<Segment> rows,
								 					 std::vector<Segment> cols)
	{
		row_partition = rows;
		col_partition = cols;
		return *this;
	}

	BSMatrixTape& operator()(std::vector<Segment> rows, int col)
	{
		if(col == -1)	std::tie(std::ignore, col) = nonfinalized_shape();
		return operator()(rows, {{col,-1}});
	}
	BSMatrixTape& operator()(int row, std::vector<Segment> cols)
	{
		if(row == -1) std::tie(row,std::ignore) = nonfinalized_shape();
		return operator()({{row,-1}}, cols);
	}

	BSMatrixTape& operator()(int row, int col)
	{
		// We want to append this matrix after the last row that we've seen so far
		if(row == -1) std::tie(row,std::ignore) = nonfinalized_shape();
		if(col == -1)	std::tie(std::ignore, col) = nonfinalized_shape();
		return operator()({{row,-1}}, {{col,-1}});
	}

	/**
	 * Called when all copy operations have been completed once, which fixes the sparsity structure.
	 * 
	 * If rows and cols aren't specified, then they will be taken as large enough to contain all the
	 * blocks copied in.
	 */
	void finalize_structure(int rows=-1, int cols=-1)
	{
		if(rows == -1 || cols == -1)
		{
			// Determine the size of the matrix as the largest element
			std::tie(rows, cols) = nonfinalized_shape();
		}
		S.resize(rows, cols);
		S.setFromTriplets(trip.begin(), trip.end());
		S.makeCompressed();

		// Write the index of each element of S to its data matrix
		for(int i=0; i<S.nonZeros(); i++)
			S.valuePtr()[i] = i;

		// Clear the triplet representation, so that the -1's in the operator() are
		// correctly created in the second call sequence.
		trip.clear(); 

		state = State::create_copy_sequence;
	}

	/**
	 * Produce the information required to execute the recorded copies
	 */
	void generate_copy_data(std::vector<Segment>& segments, 
													std::vector<CopyInfo>& copies, 
													Eigen::SparseMatrix<scalar_t>& sparsity_structure)
	{
		int offset = 0;
		for(int i=0; i<copy_lengths.size(); i++)
		{
			copies.push_back({.segment_index=offset, .num_segments_to_copy=copy_lengths[i]});
			offset += copy_lengths[i];
		}
		segments = copy_sequence;
		sparsity_structure = S.template cast<scalar_t>();
	}

	int rows() {return S.rows();}
	int cols() {return S.cols();}

	Eigen::SparseMatrix<int> get_sparsity_structure()
	{
		Eigen::SparseMatrix<int> ret = S;
		for(int i=0; i<ret.nonZeros(); i++) ret.valuePtr()[i] = 1;
		return ret;
	}
};



template<typename scalar_t>
class BSMatrix
{
private:

	scalar_t* target; // Where we're going to write the data

	Eigen::SparseMatrix<scalar_t> sparsity_structure;
	std::vector<Segment>  segments;
	std::vector<CopyInfo> copies;
	int copy_index;

	// Execute the next copy in the sequence
	template<typename Op>
	inline void execute_operation(Op op, const scalar_t *source)
	{
		int segment_index = copies[copy_index].segment_index;
		for(int i=0; i<copies[copy_index].num_segments_to_copy; i++)
		{
			Segment seg = segments[segment_index + i];
			op(Eigen::Map<Eigen::VectorX<scalar_t>>(target+seg.index, seg.length),
				Eigen::Map<const Eigen::VectorX<scalar_t>>(source, seg.length));
			source += seg.length;
		}
		copy_index++;

		if(copy_index == copies.size()) copy_index = 0;
	}


public:

	BSMatrix() {}; // Must call initialize_from_tape before using
	void initialize_from_tape(BSMatrixTape<scalar_t> tape)
	{
		tape.generate_copy_data(segments, copies, sparsity_structure);
		copy_index = 0;
		for(int i=0; i<sparsity_structure.nonZeros(); i++)
			sparsity_structure.valuePtr()[i] = 1;
	}

	// Initialize from a tape instance
	BSMatrix(BSMatrixTape<scalar_t> tape) 
		: target(NULL)
	{
		initialize_from_tape(tape);
	}

	/**
	 * Initialize S to the right sparsity structure and set it
	 * as the target
	 */
	void initialize_matrix(SparseMatrix<scalar_t>& S)
	{
		S = sparsity_structure;
		set_target(S);
	}

	/**
	 * Use the given matrix as the target.
	 * Must already have been initialized to the correct sparsity structure!
	 */
	void set_target(Eigen::SparseMatrix<scalar_t>& S)
	{
		target = S.valuePtr();
	}
	void set_target(Eigen::Ref<Eigen::MatrixX<scalar_t>> S)
	{
		target = S.data();
	}

	/**
	 * Copy the given block into the target matrix
	 */
	void operator=(const Eigen::SparseMatrix<scalar_t>& block)
	{
		execute_operation([](auto&& a, auto&& b){a=b;}, block.valuePtr());
	}	

	// Assumption: The input matrix is contiguous. Don't change this to a Ref.
	void operator=(const Eigen::MatrixX<scalar_t>& block)
	{
		execute_operation([](auto&& a, auto&& b){a=b;}, block.data());
	}

	/**
	 * Copy the given block into the target matrix
	 */
	void operator+=(const Eigen::SparseMatrix<scalar_t>& block)
	{
		execute_operation([](auto&& a, auto&& b){a+=b;}, block.valuePtr());
	}	

	// Assumption: The input matrix is contiguous. Don't change this to a Ref.
	void operator+=(const Eigen::MatrixX<scalar_t>& block)
	{
		execute_operation([](auto&& a, auto&& b){a+=b;}, block.data());
	}

	/**
	 * Copy the given block into the target matrix
	 */
	void operator-=(const Eigen::SparseMatrix<scalar_t>& block)
	{
		execute_operation([](auto&& a, auto&& b){a-=b;}, block.valuePtr());
	}	

	// Assumption: The input matrix is contiguous. Don't change this to a Ref.
	void operator-=(const Eigen::MatrixX<scalar_t>& block)
	{
		execute_operation([](auto&& a, auto&& b){a-=b;}, block.data());
	}

	// These all compile out. Not used in deployment.
	inline BSMatrix<scalar_t>& operator()(std::vector<Segment> rows, std::vector<Segment> cols)	{return *this;}
	inline BSMatrix<scalar_t>& operator()(std::vector<Segment> rows, int col)	{return *this;}
	inline BSMatrix<scalar_t>& operator()(int row, std::vector<Segment> cols)	{return *this;}
	inline BSMatrix<scalar_t>& operator()(int row, int col)	{return *this;}
	void finalize_structure(int rows=-1, int cols=-1)	{}	
};



};


#endif // __BSMATRIX_HPP
