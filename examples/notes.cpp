#include <iostream>
#include <numeric>
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include <type_traits>
// #include "unsupported/Eigen/AutoDiff"

#include "lampc_function.hpp"

using namespace Eigen;

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
	 */
	void operator=(const Eigen::Ref<const Eigen::MatrixX<scalar_t>>& block)
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

	/**
	 * Set the target rows and columns according to the given partition.
	 * 
	 * The source is partitioned into blocks and copied into target according to
	 * - rows = {{target_row, len}, ...}
	 * - cols = {{target_col, len}, ...}
	 * 
	 * The blocks are taken contiguously from the matrix to be copied in.
	 */
	BSMatrixTape& operator()(std::initializer_list<Segment> rows,
								 					 std::initializer_list<Segment> cols)
	{
		row_partition = rows;
		col_partition = cols;
		return *this;
	}

	BSMatrixTape& operator()(std::initializer_list<Segment> rows, int col)
	{
		if(col == -1)	std::tie(std::ignore, col) = nonfinalized_shape();
		return operator()(rows, {{col,-1}});
	}
	BSMatrixTape& operator()(int row, std::initializer_list<Segment> cols)
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
	void do_copy(const scalar_t *source)
	{
		int segment_index = copies[copy_index].segment_index;
		for(int i=0; i<copies[copy_index].num_segments_to_copy; i++)
		{
			Segment seg = segments[segment_index + i];
			Eigen::Map<Eigen::VectorX<scalar_t>>(target+seg.index, seg.length) = 
				Eigen::Map<const Eigen::VectorX<scalar_t>>(source, seg.length);
			source += seg.length;
		}
		copy_index++;

		if(copy_index == copies.size()) copy_index = 0;
	}

public:

	// Initialize from a tape instance
	BSMatrix(BSMatrixTape<scalar_t> tape) 
		: target(NULL)
	{
		tape.generate_copy_data(segments, copies, sparsity_structure);
		copy_index = 0;
	}

	/**
	 * Initialize S to the right sparsity structure and set it
	 * as the target
	 */
	void initialize_matrix(SparseMatrix<scalar_t>& S)
	{
		S = sparsity_structure;
		target = S.valuePtr();
	}

	/**
	 * Copy the given block into the target matrix
	 */
	void operator=(const Eigen::SparseMatrix<scalar_t>& block)
	{
		do_copy(block.valuePtr());
	}	

	void operator=(const Eigen::Ref<const Eigen::MatrixX<scalar_t>>& block)
	{
		do_copy(block.data());
	}

	// These all compile out. Not used in deployment.
	inline BSMatrix& operator()(std::initializer_list<Segment> rows, std::initializer_list<Segment> cols)	{return *this;}
	inline BSMatrix& operator()(std::initializer_list<Segment> rows, int col)	{return *this;}
	inline BSMatrix& operator()(int row, std::initializer_list<Segment> cols)	{return *this;}
	inline BSMatrix& operator()(int row, int col)	{return *this;}
	void finalize_structure(int rows=-1, int cols=-1)	{}	
};



template<typename scalar_t>
struct Variable;

template<typename scalar_t>
struct Problem
{
	Variable<scalar_t> variable(int n, int m=1)
	{
		Variable<scalar_t> v(n,m,size);
		variables.push_back(v);
		size = compute_size(); // Update the total size of the problem variable

		// Return with move symantics is required, 
		// since we need the reference stored in variables to remain valid
		return v; 
	}

	int size = 0;

	/**
	 * Called once the problem is defined. Defines the optimization variable.
	 */
	void register_variables(std::initializer_list<std::reference_wrapper<Variable<scalar_t>>> vars)
	{
		variables = vars;

		int offset = 0;
		for(auto v: variables)
		{
			v.get().initialize(offset);
			offset += v.get().num_elements();
		}

		size = compute_size(); // Once all the variables are registered, the size doesn't change
	}

	void set_variable(Eigen::Ref<Eigen::VectorX<scalar_t>> var)
	{
		for(auto v: variables)
			v.get().set_var(var);
	}

private:
	std::vector<std::reference_wrapper<Variable<scalar_t>>> variables;

	int compute_size() // Total number of optimization variables
	{
		return std::accumulate(variables.begin(), variables.end(), 0, 
							[](int total, std::reference_wrapper<Variable<scalar_t>> var) {return total + var.get().num_elements();});
	}
};


/**
 * Represents a variable, which is an offset into
 * a global decision variable.
 */
template<typename scalar_t>
struct Variable
{
	Variable(int n, int m, int offset) 
		: n(n), m(m), offset(offset), 
			src(NULL,n,m) // Variable is invalid until set_var is called
		{}

	inline int num_elements() { return n*m; } // Total number of elements in the variable (n*m)
	inline int size() { return n; } // Size of a single variable
	inline std::pair<int,int> shape() { return make_pair(n,m); } // Shape of the variable matrix

	/** 
	 * Return the matrix
	 */
	inline Eigen::Map<Eigen::MatrixX<scalar_t>> operator()()
	{
		return src;
	}

	/** 
	 * Return a reference to the index'th variable in the matrix
	 */
	inline Eigen::Ref<Eigen::VectorX<scalar_t>> operator()(int index)
	{
		return src(Eigen::all,index);
	}

	/**
	 * Return a pair of {offset, length}
	 */
	inline Segment operator[](int index)
	{
		return Segment{.index = offset + index * n, .length = n}	;
	}


friend Problem<scalar_t>;
protected:
	inline void set_var(Eigen::Ref<Eigen::VectorX<scalar_t>> var)
	{
	  new (&src) Eigen::Map<Eigen::MatrixX<scalar_t>>(var.data() + offset, n, m);
	}

private:

	// Offset into the global variable
	int offset;
	int n,m;

	// Reference to start of this variable in the global variable
	Eigen::Map<Eigen::MatrixX<scalar_t>> src;
};




/**
 * General RK4 integrator
 * 
 * Takes a pointer to the ODE and a step size, and returns the RK4 integration.
 */
template<typename diff_t, int n, int m>
using SysFunc_ptr = std::add_pointer_t<Eigen::Vector<diff_t, n>(
		const Eigen::Ref<const Eigen::Vector<diff_t, n>>&, 
		const Eigen::Ref<const Eigen::Vector<diff_t, m>>&)>;

template<typename scalar_t=double, typename diff_t=scalar_t, int n, int m>
Eigen::Vector<diff_t, n> rk4(SysFunc_ptr<diff_t,n,m> ode, 
															 const Eigen::Ref<const Eigen::Vector<diff_t, n>>& x, 
															 const Eigen::Ref<const Eigen::Vector<diff_t, m>>& u,
															 const scalar_t _h)
{
	diff_t h = static_cast<diff_t>(_h);
  Eigen::Vector<diff_t, n> k1 = ode(x,       u);
  Eigen::Vector<diff_t, n> k2 = ode(x+h/static_cast<diff_t>(2.0)*k1,u);
  Eigen::Vector<diff_t, n> k3 = ode(x+h/static_cast<diff_t>(2.0)*k2,u);
  Eigen::Vector<diff_t, n> k4 = ode(x+h*k3,  u);
  return x + h/static_cast<diff_t>(6.0) * (k1 + static_cast<diff_t>(2.0)*k2 + static_cast<diff_t>(2.0)*k3 + k4);
}

/**
 * A multiple-shooting constraint.
 */
template<typename scalar_t=double, typename diff_t=scalar_t>
Eigen::Vector<diff_t, 2> linear_system(
		const Eigen::Ref<const Eigen::Vector<diff_t, 2>>& x, 
		const Eigen::Ref<const Eigen::Vector<diff_t, 1>>& u)
{
	Eigen::Matrix<scalar_t, 2, 2> A{{1,2},{3,4}};
	Eigen::Matrix<scalar_t, 2, 1> B{{5},{6}};
	return A.template cast<diff_t>() * x + B.template cast<diff_t>() * u;
}


template<typename scalar_t=double, typename diff_t=scalar_t>
Vector<diff_t, 2> sys(const Ref<const Matrix<diff_t, 2, 1>>& x, const Ref<const Vector<diff_t, 1>>& u)
{
  Matrix<scalar_t, 2, 2> A{{2,2},{3,4}};
  Vector<scalar_t, 2> B = A(all, 0);
  return ((((x(0)) * (A.template cast<diff_t>()).array()).matrix()) * (x)) + ((B.template cast<diff_t>()) * (u));
}

template<typename scalar_t=double, typename diff_t=scalar_t>
Vector<diff_t, 2> _dsys(
		const Ref<const Vector<diff_t, 2>>& xp, 
		const Ref<const Vector<diff_t, 2>>& x, 
		const Ref<const Vector<diff_t, 1>>& u)
{
	return xp - rk4<scalar_t, diff_t, 2, 1>(sys<scalar_t, diff_t>, x, u, 0.1);
}


int main()
{
	using scalar_t = float;

	Problem<scalar_t> prob;

	constexpr int N = 5;
	auto x = prob.variable(2, N);
	auto u = prob.variable(1, N-1);

	std::cout << "type(x) = " << type_name<decltype(x)>() << std::endl;
	std::cout << "&x = " << &x << std::endl;

	Eigen::VectorX<scalar_t> var(prob.size);
	prob.set_variable(var);

	for(int i=0; i<x().cols(); i++)	x(i).array() = i;
	for(int i=0; i<u().cols(); i++)	u(i).array() = i;

	std::cout << "x = \n" << x() << std::endl;
	std::cout << "u = \n" << u() << std::endl;

	// constexpr int n = 2;
	// constexpr int m = 1;
	// constexpr scalar_t h = 0.1;

	// SysFunc_ptr<scalar_t,n,m> psystem = system<scalar_t>;

	// std::cout << "system(x, u) = " << system<scalar_t>(x, u).transpose() << std::endl;
	// std::cout << "rk4(system(x, u)) = " << rk4<scalar_t, n, m>(psystem, x, u, h).transpose() << std::endl;

	using DSYS = lampc::Function<scalar_t, 2, 2,2,1>;
	auto dsys = make_function(DSYS, _dsys);

	std::cout << "dsys(x, u) = " << dsys(x(1),x(0), u(0)).transpose() << std::endl;
	std::cout << "jac dsys(x, u) = \n" << std::get<1>(dsys(x(1),x(0), u(0), lampc::Jacobian())) << std::endl;
	std::cout << "hess dsys(x, u) = \n" << std::get<2>(dsys(x(1),x(0), u(0), lampc::Hessian()))[0] << std::endl;

	auto shoot = [&](auto& tape)
	{
		for(int i=0; i<x().cols()-1; i++)
			std::tie(std::ignore, tape(-1,{x[i],u[i],x[i+1]})) = dsys(x(i+1),u(i),x(i),lampc::Jacobian());
	};

	// Create the tape to record the copy sequence
	BSMatrixTape<scalar_t> tape;
	shoot(tape);
	tape.finalize_structure();
	shoot(tape);

	std::cout << "Sparsity structure = \n" << Eigen::MatrixX<int>(tape.get_sparsity_structure()) << std::endl;
	std::cout << "Copies = " << tape.copy_sequence << std::endl;
	std::cout << "\n\n";

	// We now replace the tape with a block-sparse matrix
	// which loads the copy-sequence from the tape (or from file)
	BSMatrix<scalar_t> mat(tape);
	SparseMatrix<scalar_t> S;
	mat.initialize_matrix(S);

	// Now we call the function and it just plays back the copy sequence
	shoot(mat);
	std::cout << "S = \n" << Eigen::MatrixX<scalar_t>(S).format(Eigen::IOFormat(2)) << std::endl;

	x(2).array() = 10;
	shoot(mat);
	std::cout << "S = \n" << Eigen::MatrixX<scalar_t>(S).format(Eigen::IOFormat(2)) << std::endl;

	constexpr int NUM_EXP = 1000;
  auto start = std::chrono::steady_clock::now();
  for(int i = 0; i < NUM_EXP; ++i)
  {
		x(2)(0)++;
		shoot(mat);
  }
  auto end = std::chrono::steady_clock::now();

  std::cout << "Time to compute the block-sparse jacobian: "
      << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / (double)NUM_EXP
      << " us" << std::endl;

	return 0;
}
