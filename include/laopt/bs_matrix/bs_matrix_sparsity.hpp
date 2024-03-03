#ifndef LAOPT_BS_MATRIX_SPARSITY_HPP
#define LAOPT_BS_MATRIX_SPARSITY_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "laopt/bs_matrix/coord_matrix.hpp"
#include "laopt/bs_matrix/bs_slice_base.hpp"
#include "laopt/bs_matrix/bs_matrix_tape.hpp"

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

/**
 * A tape class to capture the sparsity pattern
 */
class BSMatrixSparsity : public BSSliceSparsity<BSMatrixSparsity, CoordMatrix>
{
private:
    template<typename, typename>
    friend class laopt::BSSliceSparsity;

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
     *
     * Note: Invalidates all slices!
     */
    void resize(Eigen::Index rows, Eigen::Index cols)
    {
        null_mat.resize(rows, cols);
        sparsity_pattern.conservativeResize(rows, cols);
    }

    /**
     * Resize the matrix by adding rows rows and cols columns
     */
    void extend(Eigen::Index rows, Eigen::Index cols)
    {
        resize(null_mat.rows() + rows, null_mat.cols() + cols);
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
     * Create a BSMatrixTape from this sparsity pattern
     */
    BSMatrixTape makeBSTape()
    {
        return BSMatrixTape(get_sparsity_pattern());
    }

    /**
     * Return a pattern that can be passed to a BSMatrixTape to initialize it
     */
    using Info = Eigen::SparseMatrix<bool>;

    Info generate()
    {
        return get_sparsity_pattern();
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
BSMatrix<scalar_t> makeBSMatrix(F f, Eigen::Index rows = 0, Eigen::Index cols = 0)
{
    BSMatrixSparsity sparsity(rows, cols);
    f(sparsity); // Extract sparsity pattern

    auto tape = sparsity.makeBSTape();
    f(tape); // Extract operation sequence

    return tape.template makeBSMatrix<scalar_t>();
}

} // namespace laopt

#endif // LAOPT_BS_MATRIX_SPARSITY_HPP
