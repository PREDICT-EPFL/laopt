#ifndef LAOPT_BS_MATRIX_SPARSITY_HPP
#define LAOPT_BS_MATRIX_SPARSITY_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "laopt/bs_matrix/coord_matrix.hpp"
#include "laopt/bs_matrix/bs_slice_base.hpp"

namespace laopt {

/**
 * A slice class where all operators just set the sparsity pattern
 */
template<typename Child, typename NullMat>
class BSSliceSparsity : public BSSliceBase<BSSliceSparsity<Child, NullMat>, NullMat>
{
private:
    Child& child; // Pointer to the child class BSSparsity object

    template<typename, typename>
    friend class laopt::BSSliceBase;

    template<typename Derived>
    auto make_slice(Derived sub_matrix)
    {
        return BSSliceSparsity<Child, Derived>(child, sub_matrix);
    }

    /**
     * Record the position of nonzeros to assemble the sparsity pattern
     */
    template<typename Derived>
    inline void capture_sparsity(const Eigen::DenseBase<Derived>& rhs_mat)
    {
        assert(rhs_mat.rows() == this->null_mat.rows() && rhs_mat.cols() == this->null_mat.cols() && "You assigned a matrix of the wrong size!");

        for (Eigen::Index i = 0; i < this->null_mat.rows(); i++)
        {
            for (Eigen::Index j = 0; j < this->null_mat.cols(); j++)
            {
                Coord coord = this->null_mat(i, j);
                this->child.sparsity_pattern.coeffRef(coord.row, coord.col) = 1;
            }
        }
    }

    template<typename Derived>
    inline void capture_sparsity(const Eigen::DiagonalBase<Derived>& rhs_mat)
    {
        assert(rhs_mat.rows() == this->null_mat.rows() && rhs_mat.cols() == this->null_mat.cols() && "You assigned a matrix of the wrong size!");

        for (Eigen::Index i = 0; i < this->null_mat.rows(); i++)
        {
            Coord coord = this->null_mat(i, i);
            this->child.sparsity_pattern.coeffRef(coord.row, coord.col) = 1;
        }
    }

public:
    explicit BSSliceSparsity(Child& child, NullMat nullMat) : BSSliceBase<BSSliceSparsity<Child, NullMat>, NullMat>(nullMat), child(child) {}

    using BSSliceBase<BSSliceSparsity<Child, NullMat>, NullMat>::operator=;
};

struct BSMatrixSparsityInfo
{
    Eigen::VectorXi block_rows;
    Eigen::VectorXi block_cols;
    Eigen::SparseMatrix<bool> sparsity_pattern;
};

/**
 * A tape class to capture the sparsity pattern
 */
class BSMatrixSparsity : public BSSliceSparsity<BSMatrixSparsity, CoordMatrix>
{
private:
    template<typename, typename>
    friend class laopt::BSSliceSparsity;

    Eigen::VectorXi block_rows;
    Eigen::VectorXi block_cols;
    Eigen::SparseMatrix<bool> sparsity_pattern;

public:
    explicit BSMatrixSparsity(Eigen::Index rows = 0, Eigen::Index cols = 0)
            : BSSliceSparsity<BSMatrixSparsity, CoordMatrix>(*this, CoordMatrix(0, 0))
    {
        resize(rows, cols);
    };

    using BSSliceSparsity<BSMatrixSparsity, CoordMatrix>::operator=;

    void set_zero() {}

    /**
     * Resize the matrix null_mat.
     */
    void resize(Eigen::Index rows, Eigen::Index cols)
    {
        if (rows > 0) {
            block_rows.resize(1);
            block_rows(0) = int(rows);
        } else {
            block_rows.resize(0);
        }
        if (cols > 0) {
            block_cols.resize(1);
            block_cols(0) = int(rows);
        } else {
            block_cols.resize(0);
        }
        null_mat.resize(rows, cols);
        sparsity_pattern.resize(rows, cols);
    }

    /**
     * Resize the matrix by adding rows rows and cols columns
     */
    void extend(Eigen::Index rows, Eigen::Index cols)
    {
        if (rows > 0) {
            Eigen::Index n_block_rows = block_rows.rows();
            block_rows.conservativeResize(n_block_rows + 1);
            block_rows(n_block_rows) = int(rows);
        }
        if (cols > 0) {
            Eigen::Index n_block_cols = block_cols.rows();
            block_cols.conservativeResize(n_block_cols + 1);
            block_cols(n_block_cols) = int(cols);
        }
        Eigen::Index extended_rows = null_mat.rows() + rows;
        Eigen::Index extended_cols = null_mat.cols() + cols;
        null_mat.resize(extended_rows, extended_cols);
        sparsity_pattern.conservativeResize(extended_rows, extended_cols);
    }

    /**
     * Returns sparsity pattern.
     */
    Eigen::SparseMatrix<bool> get_sparsity_pattern()
    {
        sparsity_pattern.makeCompressed();
        return sparsity_pattern;
    }

    /**
     * Return a pattern that can be passed to a BSMatrixTape to initialize it
     */
    using Info = BSMatrixSparsityInfo;

    Info generate()
    {
        return {block_rows, block_cols, sparsity_pattern};
    }
};

} // namespace laopt

#endif // LAOPT_BS_MATRIX_SPARSITY_HPP
