#ifndef LAOPT_BS_MATRIX_SPARSITY_HPP
#define LAOPT_BS_MATRIX_SPARSITY_HPP

#include <Eigen/Dense>

#include "bs_slice_base.hpp"
#include "bs_matrix_tape.hpp"

namespace laopt {

/**
 * A slice class where all operators just set the sparsity structure
 */
template<typename T, typename Base>
class BSSliceSparsity : public BSSliceBase<T, Base>
{
    template<typename Derived>
    void create_sparsity_pattern(const Eigen::MatrixBase<Derived>& mat)
    {
        assert(mat.rows() == this->M.rows() && mat.cols() == this->M.cols() && "You assigned a matrix of the wrong size!");
        this->M = this->get_pattern(mat);
    }

public:
    BSSliceSparsity(Base& base, T M) : BSSliceBase<T, Base>(base, M) {}

    /**
     * Returns a DENSE matrix where 1's are the non-zeros
     */
    Eigen::MatrixX<bool> get_sparsity()
    {
        return ((this->M).array() == 1).matrix();
    }

    template<typename Derived>
    BSSliceSparsity& operator=(const Eigen::MatrixBase<Derived>& mat)
    {
        create_sparsity_pattern(mat);
        return *this;
    }

    template<typename Derived>
    void operator+=(const Eigen::MatrixBase<Derived>& mat) { create_sparsity_pattern(mat); }

    template<typename Derived>
    void operator-=(const Eigen::MatrixBase<Derived>& mat) { create_sparsity_pattern(mat); }
};

/**
 * A tape class to capture the sparsity pattern
 */
class BSMatrixSparsity : public BSSliceSparsity<Eigen::MatrixX<int>, BSMatrixSparsity>
{
public:
    explicit BSMatrixSparsity(Eigen::Index rows = 0, Eigen::Index cols = 0)
            : BSSliceSparsity<Eigen::MatrixX<int>, BSMatrixSparsity>(*this, Eigen::MatrixX<int>(0, 0))
    {
        resize(rows, cols);
    };

    template<typename Derived>
    auto makeSlice(Derived sub_matrix)
    {
        return BSSliceSparsity<Derived, BSMatrixSparsity>(*this, sub_matrix);
    }

    void set_zero() {}

    /**
     * Resize the matrix M.
     *
     * Note: Invalidates all slices!
     */
    void resize(Eigen::Index rows, Eigen::Index cols)
    {
        Eigen::Index curr_rows = M.rows();
        Eigen::Index curr_cols = M.cols();
        if (curr_rows == 0 || curr_cols == 0)
            M.resize(rows, cols);
        else
            M.conservativeResize(rows, cols);

        // Set new elements to zero == sparse
        if (cols > curr_cols) M(Eigen::all, Eigen::seq(curr_cols, Eigen::last)).setZero();
        if (rows > curr_rows) M(Eigen::seq(curr_rows, Eigen::last), Eigen::all).setZero();
    }

    /**
     * Resize the matrix by adding rows rows and cols columns
     */
    void extend(Eigen::Index rows, Eigen::Index cols)
    {
        resize(M.rows() + rows, M.cols() + cols);
    }

    /**
     * Create a BSMatrixTape from this sparsity structure
     */
    BSMatrixTape makeBSTape(Eigen::Index rows, Eigen::Index cols)
    {
        return BSMatrixTape(get_sparsity(), rows, cols);
    }

    /**
     * Return a structure that can be passed to a BSMatrixTape to initialize it
     */
    using Info = Eigen::MatrixX<bool>;

    Info generate()
    {
        return get_sparsity();
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

    auto tape = sparsity.makeBSTape(rows, cols);
    f(tape); // Extract operation sequence

    return tape.template makeBSMatrix<scalar_t>();
};

} // namespace laopt

#endif // LAOPT_BS_MATRIX_SPARSITY_HPP
