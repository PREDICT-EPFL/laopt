#ifndef LAOPT_BS_MATRIX_TAPE_HPP
#define LAOPT_BS_MATRIX_TAPE_HPP

#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "bs_matrix.hpp"
#include "bs_slice_base.hpp"

namespace laopt {

/**
 * A slice class where the action of each operator is captured
 */
template<typename NullMat, typename Child>
class BSSliceTape : public BSSliceBase<NullMat, Child>
{
    /**
     * Record the sequence of memory copies to copy mat to this slice
     */
    template<typename Derived>
    inline void record_op(const Eigen::DenseBase<Derived>& mat)
    {
        assert(mat.rows() == this->null_mat.rows() && mat.cols() == this->null_mat.cols() && "You assigned a matrix of the wrong size!");

        auto m_row_indices = this->row_indices(this->null_mat);
        auto m_col_indices = this->col_indices(this->null_mat);

        Eigen::VectorX<int> sequence(mat.rows() * mat.cols());
        if (Eigen::DenseBase<Derived>::IsRowMajor)
        {
            Eigen::Index s_i = 0;
            for (Eigen::Index i = 0; i < (Eigen::Index) m_row_indices.size(); i++)
            {
                for (Eigen::Index j = 0; j < (Eigen::Index) m_col_indices.size(); j++)
                {
                    int csc_index = this->child.sparsity_structure.coeff(m_row_indices[i], m_col_indices[j]);
                    assert(csc_index != 0 && "Sparsity structure does not match!");
                    sequence(s_i++) = csc_index - 1;
                }
            }
        }
        else
        {
            Eigen::Index s_i = 0;
            for (Eigen::Index j = 0; j < (Eigen::Index) m_col_indices.size(); j++)
            {
                for (Eigen::Index i = 0; i < (Eigen::Index) m_row_indices.size(); i++)
                {
                    int csc_index = this->child.sparsity_structure.coeff(m_row_indices[i], m_col_indices[j]);
                    assert(csc_index != 0 && "Sparsity structure does not match!");
                    sequence(s_i++) = csc_index - 1;
                }
            }
        }

        // sequence is the set of indices into the sparse matrix that we'll be copying this block into
        // Compress the index sequence into contiguous blocks
        this->child.record_copy_sequence(sequence);
    }

public:
    BSSliceTape(Child& child, NullMat nullMat) : BSSliceBase<NullMat, Child>(child, nullMat) {}

    /**
     * All operators just record the operation
     */
    template<typename Derived>
    BSSliceTape& operator=(const Eigen::MatrixBase<Derived>& mat) { record_op(mat); return *this; }

    template<typename Derived>
    typename std::enable_if<!std::is_base_of<Eigen::MatrixBase<Derived>, Derived>::value, BSSliceTape&>::type
    operator=(const Derived& scalar)
    {
        assert(this->null_mat.rows() == 1 && this->null_mat.cols() == 1 && "You tried to assign a scalar to a matrix");
        record_op(Eigen::Matrix<Derived, 1, 1>(scalar));
        return *this;
    }

    template<typename Derived>
    BSSliceTape& operator+=(const Eigen::MatrixBase<Derived>& mat) { record_op(mat); return *this; }

    template<typename Derived>
    typename std::enable_if<!std::is_base_of<Eigen::MatrixBase<Derived>, Derived>::value, BSSliceTape&>::type
    operator+=(const Derived& scalar)
    {
        assert(this->null_mat.rows() == 1 && this->null_mat.cols() == 1 && "You tried to assign a scalar to a matrix");
        record_op(Eigen::Matrix<Derived, 1, 1>(scalar));
        return *this;
    }

    template<typename Derived>
    BSSliceTape& operator-=(const Eigen::MatrixBase<Derived>& mat) { record_op(mat); return *this; }

    template<typename Derived>
    typename std::enable_if<!std::is_base_of<Eigen::MatrixBase<Derived>, Derived>::value, BSSliceTape&>::type
    operator-=(const Derived& scalar)
    {
        assert(this->null_mat.rows() == 1 && this->null_mat.cols() == 1 && "You tried to assign a scalar to a matrix");
        record_op(Eigen::Matrix<Derived, 1, 1>(scalar));
        return *this;
    }
};

/**
* A tape class to capture the copy pattern.
*/
class BSMatrixTape : public BSSliceTape<Eigen::Map<Eigen::MatrixX<int>>, BSMatrixTape>
{
private:
    template<typename, typename>
    friend class laopt::BSSliceBase;
    template<typename, typename>
    friend class laopt::BSSliceTape;

    Eigen::SparseMatrix<int> sparsity_structure; // Must have been created a-priori

public:
    explicit BSMatrixTape(const Eigen::SparseMatrix<bool>& structure, Eigen::Index rows = 0, Eigen::Index cols = 0)
    : BSSliceTape<Eigen::Map<Eigen::MatrixX<int>>, BSMatrixTape>(*this, Eigen::Map<Eigen::MatrixX<int>>(nullptr, rows, cols))
    {
        initialize(structure, rows, cols);
    };

    void initialize(const Eigen::SparseMatrix<bool>& structure, Eigen::Index rows, Eigen::Index cols)
    {
        sparsity_structure = structure.cast<int>();
        sparsity_structure.makeCompressed();

        // We set the non-zero elements to what their index into the data of a csc-sparse matrix.
        // The index is stored in 1-indexing format to distinguish 0 elements in the sparse matrix
        // and the original 0-index.
        for (Eigen::Index i = 0; i < sparsity_structure.nonZeros(); i++) {
            sparsity_structure.valuePtr()[i] = static_cast<int>(i + 1);
        }
    };

    void set_zero() {}

    /**
     * Resize the matrix null_mat.
     *
     * Note: Invalidates all slices!
     *
     * Note: Sparsity structure must be set before this is called.
     */
    void resize(Eigen::Index rows, Eigen::Index cols)
    {
        new(&null_mat) Eigen::Map<Eigen::MatrixX<int>>(nullptr, rows, cols);
    }

    /**
     * Resize the matrix by adding rows rows and cols columns
     */
    void extend(Eigen::Index rows, Eigen::Index cols)
    {
        resize(null_mat.rows() + rows, null_mat.cols() + cols);
    }

    /**
     * Create a BSMatrix from this tape
     */
    template<typename scalar_t>
    BSMatrix<scalar_t> makeBSMatrix()
    {
        return BSMatrix<scalar_t>(sparsity_structure.cast<bool>(), copy_segments, copy_info);
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

        info.sparsity_structure = sparsity_structure.cast<bool>();
        info.copy_segments = copy_segments;
        info.copy_info = copy_info;

        return info;
    }

private:
    // Sequence of copy operations
    std::vector<Segment> copy_segments;
    std::vector<CopyInfo> copy_info;

    template<typename Derived>
    auto makeSlice(Derived sub_matrix)
    {
        return BSSliceTape<Derived, BSMatrixTape>(*this, sub_matrix);
    }

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
        for (const int& i: sequence) {
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
};

} // namespace laopt

#endif // LAOPT_BS_MATRIX_TAPE_HPP
