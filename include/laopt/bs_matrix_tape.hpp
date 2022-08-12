#ifndef LAOPT_BS_MATRIX_TAPE_HPP
#define LAOPT_BS_MATRIX_TAPE_HPP

#include <vector>
#include <Eigen/Dense>

#include "bs_matrix.hpp"
#include "bs_slice_base.hpp"

namespace laopt {

/**
 * A slice class where the action of each operator is captured
 */
template<typename T, typename Base>
class BSSliceTape : public BSSliceBase<T, Base>
{
    /**
     * Record the sequence of memory copies to copy mat to this slice
     */
    inline void record_op(const Eigen::MatrixX<int>& mat)
    {
        assert(mat.rows() == this->M.rows() && mat.cols() == this->M.cols() && "You assigned a matrix of the wrong size!");

        // M is the set of indices into the sparse matrix that we'll be copying this block into
        // Compress the index sequence into contiguous blocks
        this->base.record_copy_sequence(this->M.reshaped());
    }

public:
    BSSliceTape(Base& base, T M) : BSSliceBase<T, Base>(base, M) {}

    /**
     * All operators just record the operation
     */
    template<typename Derived>
    BSSliceTape& operator=(const Eigen::MatrixBase<Derived>& mat)
    {
        record_op(this->get_pattern(mat));
        return *this;
    }

    template<typename Derived>
    void operator+=(const Eigen::MatrixBase<Derived>& mat) { record_op(this->get_pattern(mat)); }

    template<typename Derived>
    void operator-=(const Eigen::MatrixBase<Derived>& mat) { record_op(this->get_pattern(mat)); }
};

/**
* A tape class to capture the copy pattern.
*/
struct BSMatrixTape : public BSSliceTape<Eigen::MatrixX<int>, BSMatrixTape>
{
    Eigen::MatrixX<int> sparsity_structure; // Must have been created a-priori

    BSMatrixTape() : BSSliceTape<Eigen::MatrixX<int>, BSMatrixTape>(*this, Eigen::MatrixX<int>()) {}

    explicit BSMatrixTape(const Eigen::MatrixX<bool>& structure, Eigen::Index rows = 0, Eigen::Index cols = 0) :
            BSSliceTape<Eigen::MatrixX<int>, BSMatrixTape>(*this, Eigen::MatrixX<int>())
    {
        initialize(structure, rows, cols);
    };

    void initialize(const Eigen::MatrixX<bool>& structure, Eigen::Index rows = 0, Eigen::Index cols = 0)
    {
        sparsity_structure.resizeLike(structure);

        // We set the zero elements to -1
        // We set the non-zero elements to what their index into the data of a csc-sparse matrix
        // would be
        sparsity_structure.array() = -1; // Zero elements == -1, non-zeros == 0
        int index = 0;
        for (int c = 0; c < sparsity_structure.cols(); c++)
            for (int r = 0; r < sparsity_structure.rows(); r++)
                if (structure(r, c) == 1) sparsity_structure(r, c) = index++;

        // Copy in the sparsity structure for the initial size matrix
        resize(rows, cols);
    };

    void set_zero() {}

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
    void resize(Eigen::Index rows, Eigen::Index cols)
    {
        Eigen::Index curr_rows = M.rows();
        Eigen::Index curr_cols = M.cols();
        M.conservativeResize(rows, cols);

        // Set new elements to that from the sparsity structure
        M(Eigen::all, Eigen::seq(curr_cols, cols - 1)) = sparsity_structure(Eigen::seq(0, rows - 1), Eigen::seq(curr_cols, cols - 1));
        M(Eigen::seq(curr_rows, rows - 1), Eigen::all) = sparsity_structure(Eigen::seq(curr_rows, rows - 1), Eigen::seq(0, cols - 1));
    }

    /**
     * Resize the matrix by adding rows rows and cols columns
     */
    void extend(Eigen::Index rows, Eigen::Index cols)
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
    void record_copy_sequence(const Eigen::VectorX<int>& sequence)
    {
        std::vector<Segment> segments;

        int next_contiguous = -2; // The next value if we're in a contiguous segment
        for (const auto& i: sequence) {
            if (i == next_contiguous) {
                next_contiguous++;
                segments.back().length++;
            } else {
                segments.push_back(Segment{.index=static_cast<size_t>(i), .length=1});
                next_contiguous = i + 1;
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
    using Info = BSMatrixInfo;

    Info generate()
    {
        Info info;

        info.rows = rows();
        info.cols = cols();

        Eigen::MatrixX<bool> bob = (sparsity_structure.array() >= 0).matrix();
        info.sparsity_structure = bob.sparseView();
        info.copy_segments = copy_segments;
        info.copy_info = copy_info;

        return info;
    }
};

} // namespace laopt

#endif // LAOPT_BS_MATRIX_TAPE_HPP
