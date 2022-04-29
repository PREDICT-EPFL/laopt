#ifndef __BSMATRIX_HPP
#define __BSMATRIX_HPP

#include <functional>

#include <Eigen/Dense>
#include <Eigen/Sparse>

/**
 * Bring all the Eigen seq commands into the global namespace for convenience
 */
using Eigen::seq;
using Eigen::seqN;
using Eigen::all;
using Eigen::last;
using Eigen::lastp1;
using Eigen::fix;

/**
 * Hold a sequence of Eigen seq statements
 */
template<typename... Seq>
std::tuple<Seq...> multiSeq(Seq... seq)
{
  return std::make_tuple(seq...);
}

namespace lampc
{

/**
 * Copy information for a sparse block matrix
 */
struct Segment
{
  size_t index;  // Index into the target.valuePtr()
  size_t length; // Number of element to copy

  bool operator==(const Segment other) const
  {
  	return other.index == index && other.length == length;
  }

  /**
   * Return an Eigen ArithmeticSequence representing this Segment
   */
	auto seq()
	{
		return seqN(index, length);
	};
};
struct CopyInfo
{
  size_t segment_index;  // Index into segments
  size_t num_segments_to_copy;  // Number of segments to copy to execute this task
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
 * Information required to construct a BSMatrix
 */
struct BSMatrixInfo
{
  Eigen::SparseMatrix<bool> sparsity_structure;
  std::vector<Segment> copy_segments;
  std::vector<CopyInfo> copy_info;
};

template<typename scalar_t>
struct BSMatrix
{
  Eigen::SparseMatrix<bool> sparsity_structure;
  std::vector<Segment>  segments;
  std::vector<CopyInfo> copies;
  int copy_index; // Current index into copies

  scalar_t* target; // Where we're going to write the data

  // Execute the next copy in the sequence
  template<typename Op>
  inline void execute_operation(Op op, const scalar_t *source)
  {
    int segment_index = copies[copy_index].segment_index;
    for(int i=0; i<copies[copy_index].num_segments_to_copy; i++)
    {
      Segment seg = segments[segment_index + i];
      auto tgt = Eigen::Map<Eigen::VectorX<scalar_t>>(target+seg.index, seg.length);
      const auto src = Eigen::Map<const Eigen::VectorX<scalar_t>>(source, seg.length);
      op(tgt, src);
      source += seg.length;
    }
    copy_index++;
    if(copy_index == copies.size()) copy_index = 0;
  }

public:
  /**
   * Note: The BSMatrix owns no memory, and so set_target must be called 
   * before any operations are done!
   */
  BSMatrix(Eigen::SparseMatrix<bool> sparsity_structure, 
           std::vector<Segment> copy_segments, 
           std::vector<CopyInfo> copy_info)
    : sparsity_structure(sparsity_structure),
      segments(copy_segments),
      copies(copy_info),
      copy_index(0)
  {}

  BSMatrix(BSMatrixInfo info)
  	: sparsity_structure(info.sparsity_structure), segments(info.copy_segments), copies(info.copy_info), copy_index(0)
  	{}

  /**
   * Initialize S to the right sparsity structure and set it
   * as the target
   */
  void allocate_memory(Eigen::SparseMatrix<scalar_t>& S)
  {
    S = sparsity_structure.template cast<scalar_t>();
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
   * Clear the matrix to all-zeros
   */
  void set_zero()
  {
    Eigen::Map<Eigen::VectorX<scalar_t>>(target, sparsity_structure.nonZeros()).array() = 0;
  }

  Eigen::SparseMatrix<bool> get_sparsity_structure()
  {
  	return sparsity_structure;
  }


  /**
   * Copy the given block into the target matrix
   */
  void operator=(const Eigen::SparseMatrix<scalar_t>& block)
  {
    execute_operation([](auto& a, auto& b){a=b;}, block.valuePtr());
  } 

  // Assumption: The input matrix is contiguous. Don't change this to a Ref.
  void operator=(const Eigen::MatrixX<scalar_t>& block)
  {
    execute_operation([](auto& a, auto& b){a=b;}, block.data());
  }

  /**
   * Copy the given block into the target matrix
   */
  void operator+=(const Eigen::SparseMatrix<scalar_t>& block)
  {
    execute_operation([](auto& a, auto& b){a+=b;}, block.valuePtr());
  } 

  // Assumption: The input matrix is contiguous. Don't change this to a Ref.
  void operator+=(const Eigen::MatrixX<scalar_t>& block)
  {
    execute_operation([](auto& a, auto& b){a+=b;}, block.data());
  }

  /**
   * Copy the given block into the target matrix
   */
  void operator-=(const Eigen::SparseMatrix<scalar_t>& block)
  {
    execute_operation([](auto& a, auto& b){a-=b;}, block.valuePtr());
  } 

  // Assumption: The input matrix is contiguous. Don't change this to a Ref.
  void operator-=(const Eigen::MatrixX<scalar_t>& block)
  {
    execute_operation([](auto& a, auto& b){a-=b;}, block.data());
  }

  // These all compile out. Not used in deployment.
  template<typename... RowSlice, typename... ColSlice> 
  BSMatrix<scalar_t>& operator()(std::tuple<RowSlice...> rows, std::tuple<ColSlice...> cols) {return *this;}

  template<typename... RowSlice, typename ColSlice>
  BSMatrix<scalar_t>& operator()(std::tuple<RowSlice...> rows, ColSlice cols) {return *this;}

  template<typename RowSlice, typename... ColSlice>
  BSMatrix<scalar_t>& operator()(RowSlice rows, std::tuple<ColSlice...> cols) {return *this;}

  template<typename RowSlice, typename ColSlice>
  BSMatrix<scalar_t>& operator()(RowSlice rows, ColSlice cols) {return *this;}

  void resize(int rows, int cols) {}

  /**
   * Resize the matrix by adding rows rows and cols columns
   */
  void extend(int rows, int cols) {}

  inline auto rows() {return sparsity_structure.rows();}
  inline auto cols() {return sparsity_structure.cols();}
};

template<typename T, typename Base>
struct BSSlice
{
  Base &base; // Pointer to the top-level matrix
  T M; // Matrix slice

  BSSlice(Base& base, T M) : base(base), M(M) {}

  template<typename... RowSlice, typename... ColSlice>
  auto operator()(std::tuple<RowSlice...> rows, std::tuple<ColSlice...> cols)
  {
    return base.makeSlice(M(to_index(rows, M.rows()), to_index(cols, M.cols())));
  }

  template<typename... RowSlice, typename ColSlice>
  auto operator()(std::tuple<RowSlice...> rows, ColSlice cols)
  {
    return base.makeSlice(M(to_index(rows, M.rows()), cols));
  }

  template<typename RowSlice, typename... ColSlice>
  auto operator()(RowSlice rows, std::tuple<ColSlice...> cols)
  {
    return base.makeSlice(M(rows, to_index(cols, M.cols())));
  }

  template<typename RowSlice, typename ColSlice>
  auto operator()(RowSlice rows, ColSlice cols)
  {
    return base.makeSlice(M(rows, cols));
  }

  size_t rows() {return M.rows();}
  size_t cols() {return M.cols();}

protected:
  /**
   * Convert a multi-sequence to a list-of-integer sequence
   */
  template<typename... Seq>
  std::vector<int> to_index(std::tuple<Seq...> mseq, int size)
  {
    return to_index_impl(mseq, size, std::make_index_sequence<sizeof... (Seq)>());
  }

  template<typename... Seq, size_t... I>
  std::vector<int> to_index_impl(std::tuple<Seq...> mseq, int size, std::index_sequence<I...>)
  {
    Eigen::VectorXi index(size);
    index.setLinSpaced(size,0,size-1);
    std::vector<int> ret;

    auto extend_ret = [&ret](auto sub)
    {
      ret.insert(ret.end(), sub.begin(), sub.end());
    };

    auto l = { (extend_ret(index(std::get<I>(mseq))), 0)...};

    return ret;
  }

  /**
   * Extract the sparsity pattern of a given matrix
   * 
   * Dense matrices are assumed to be dense, sparse have patterns.
   * 
   * TODO: Implement sparse base
   */
  template<typename Derived>
  Eigen::MatrixX<int> get_pattern(const Eigen::DenseBase<Derived>& mat)
  {
    return Eigen::MatrixX<int>::Constant(mat.rows(), mat.cols(), 1);    
  }
};

/**
 * A slice class where the action of each operator is captured
 */
template<typename T, typename Base>
class BSSliceTape : public BSSlice<T, Base>
{
  using BSSlice<T,Base>::get_pattern;

  /**
   * Record the sequence of memory copies to copy mat to this slice
   */
  void record_op(const Eigen::MatrixX<int>& mat)
  {
    assert(mat.rows() == this->M.rows() && mat.cols() == this->M.cols() && "You assigned a matrix of the wrong size!");

    // M is the set of indices into the sparse matrix that we'll be copying this block into
    // Compress the index sequence into contiguous blocks
    this->base.record_copy_sequence(this->M.reshaped());
  }

public:
  BSSliceTape(Base& base, T M) : BSSlice<T,Base>(base, M) {}

  /**
   * All operators just record the operation
   */
  template<typename Derived> void operator=(const Eigen::MatrixBase<Derived>& mat)  {record_op(get_pattern(mat));}
  template<typename Derived> void operator+=(const Eigen::MatrixBase<Derived>& mat) {record_op(get_pattern(mat));}
  template<typename Derived> void operator-=(const Eigen::MatrixBase<Derived>& mat) {record_op(get_pattern(mat));}
};

/**
 * A tape class to capture the copy pattern.
 */
struct BSMatrixTape : public BSSliceTape<Eigen::MatrixX<int>, BSMatrixTape>
{
  Eigen::MatrixX<int> sparsity_structure; // Must have been created a-priori

  BSMatrixTape() : BSSliceTape<Eigen::MatrixX<int>, BSMatrixTape>(*this, Eigen::MatrixX<int>()) {}

  BSMatrixTape(Eigen::MatrixX<bool> structure, size_t rows=0, size_t cols=0) :
    BSSliceTape<Eigen::MatrixX<int>, BSMatrixTape>(*this, Eigen::MatrixX<int>())
  {
  	initialize(structure, rows, cols);
  };

  void initialize(Eigen::MatrixX<bool> structure, size_t rows=0, size_t cols=0)
  {
    sparsity_structure.resizeLike(structure);

    // We set the zero elements to -1
    // We set the non-zero elements to what their index into the data of a csc-sparse matrix
    // would be
    sparsity_structure.array() = -1; // Zero elements == -1, non-zeros == 0
    int index = 0;
    for(int c=0; c<sparsity_structure.cols(); c++)
      for(int r=0; r<sparsity_structure.rows(); r++)
        if(structure(r,c) == 1) sparsity_structure(r,c) = index++;

    // Copy in the sparsity structure for the initial size matrix
    resize(rows,cols); 
  };


  template<typename Derived>
  auto makeSlice(Derived sub_matrix)
  {
    return BSSliceTape<Derived, BSMatrixTape>(*this, sub_matrix);
  }

  /**
   * Resize the matrix M.
   * 
   * Note: Invalidates all slices!
   * 
   * Note: Sparsity structure must be set before this is called.
   */
  void resize(int rows, int cols)
  {
    int curr_rows = M.rows();
    int curr_cols = M.cols();
    M.conservativeResize(rows, cols);

    // Set new elements to that from the sparsity structure
    M(all, seq(curr_cols,cols-1)) = sparsity_structure(seq(0,rows-1), seq(curr_cols,cols-1));
    M(seq(curr_rows,rows-1), all) = sparsity_structure(seq(curr_rows,rows-1), seq(0,cols-1));
  }

  /**
   * Resize the matrix by adding rows rows and cols columns
   */
  void extend(int rows, int cols)
  {
  	resize(M.rows() + rows, M.cols() + cols);
  }

public:
  // Sequence of copy operations 
  std::vector<Segment> copy_segments;
  std::vector<CopyInfo> copy_info;

public:
  /**
   * Takes a sequence of integers and converts them into a sequence of segments.
   * 
   * e.g., [1,2,3,5,6,7,2,4] will compress into 
   *  {Segment(1,3), Segment(5,3), Segment(2,1), Segment(4,1)}
   * 
   * Store this as a single "copy" operation in the tape.
   */
  void record_copy_sequence(Eigen::VectorX<int> sequence)
  {
    std::vector<Segment> segments;

    // std::cout << "recording sequence " << sequence.transpose()	 << std::endl;

    int next_contiguous = -2; // The next value if we're in a contiguous segment
    for(auto i : sequence)
    {
      if(i == next_contiguous)
      {
        next_contiguous++;
        segments.back().length++;
      } else
      {
        segments.push_back(Segment{.index=(size_t)i, .length=1});
        next_contiguous = i+1;
      }
    }

    copy_info.push_back(CopyInfo{.segment_index=copy_segments.size(), .num_segments_to_copy=segments.size()});
    copy_segments.insert(copy_segments.end(), segments.begin(), segments.end());
  }

  /**
   * Create a BSMatrix from this tape
   */
  template<typename scalar_t>
  BSMatrix<scalar_t> makeBSMatrix()
  {
    Eigen::MatrixX<bool> bob = (sparsity_structure.array() >= 0).matrix();
    return BSMatrix<scalar_t>(bob.sparseView(), copy_segments, copy_info);
  }

  /**
   * Generate the data required to produce a BSMatrix
   */
	BSMatrixInfo generate()
	{
		BSMatrixInfo info;

    Eigen::MatrixX<bool> bob = (sparsity_structure.array() >= 0).matrix();
    info.sparsity_structure = bob.sparseView();
    info.copy_segments = copy_segments;
    info.copy_info = copy_info;

    return info;
	}
};
/**
 * A slice class where all operators just set the sparsity structure
 */
template<typename T, typename Base>
class BSSliceSparsity : public BSSlice<T, Base>
{
  using BSSlice<T,Base>::get_pattern;

  template<typename Derived>
  void create_sparsity_pattern(const Eigen::MatrixBase<Derived>& mat)
  {
    assert(mat.rows() == this->M.rows() && mat.cols() == this->M.cols() && "You assigned a matrix of the wrong size!");
    this->M = get_pattern(mat); 
  }

public:
  BSSliceSparsity(Base& base, T M) : BSSlice<T,Base>(base, M) {}

  /** 
   * Returns a DENSE matrix where 1's are the non-zeros
   */
  Eigen::MatrixX<bool> get_sparsity()
  {
  	return ((this->M).array() == 1).matrix();
  }

  template<typename Derived> void operator=(const Eigen::MatrixBase<Derived>& mat)  {create_sparsity_pattern(mat);}
  template<typename Derived> void operator+=(const Eigen::MatrixBase<Derived>& mat) {create_sparsity_pattern(mat);}
  template<typename Derived> void operator-=(const Eigen::MatrixBase<Derived>& mat) {create_sparsity_pattern(mat);}
};

/**
 * A tape class to capture the sparsity pattern
 */
struct BSMatrixSparsity : public BSSliceSparsity<Eigen::MatrixX<int>, BSMatrixSparsity>
{
  BSMatrixSparsity(size_t rows=0, size_t cols=0) 
    : BSSliceSparsity<Eigen::MatrixX<int>, BSMatrixSparsity>(*this, Eigen::MatrixX<int>(0,0))
  {resize(rows,cols);};

  template<typename Derived>
  auto makeSlice(Derived sub_matrix)
  {
    return BSSliceSparsity<Derived, BSMatrixSparsity>(*this, sub_matrix);
  }

  /**
   * Resize the matrix M.
   * 
   * Note: Invalidates all slices!
   */
  void resize(int rows, int cols)
  {
    int curr_rows = M.rows();
    int curr_cols = M.cols();
    M.conservativeResize(rows, cols);

    // Set new elements to zero == sparse
    M(all, seq(curr_cols,last)).array() = 0; 
    M(seq(curr_rows,last), all).array() = 0; 
  }

  /**
   * Resize the matrix by adding rows rows and cols columns
   */
  void extend(int rows, int cols)
  {
  	resize(M.rows() + rows, M.cols() + cols);
  }

  /**
   * Create a BSMatrixTape from this sparsity structure
   */
  BSMatrixTape makeBSTape(size_t rows, size_t cols)
  {
		return BSMatrixTape(get_sparsity(), rows, cols);
  }

};

/**
 * Helper function to create a BSMatrix from a function
 * 
 * F is a callable that takes a matrix-like object
 * 
 * rows, cols = initial size of the matrix. (F can resize it)
 */
template<typename scalar_t, typename F>
BSMatrix<scalar_t> makeBSMatrix(F f, size_t rows=0, size_t cols=0)
{
	BSMatrixSparsity sparsity(rows, cols);
	f(sparsity); // Extract sparsity pattern

	auto tape = sparsity.makeBSTape(rows, cols);
	f(tape); // Extract operation sequence

	return tape.template makeBSMatrix<scalar_t>();
};

};

#endif // __BSMATRIX_HPP
