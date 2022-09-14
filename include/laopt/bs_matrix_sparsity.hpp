#ifndef LAOPT_BS_MATRIX_SPARSITY_HPP
#define LAOPT_BS_MATRIX_SPARSITY_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "bs_slice_base.hpp"
#include "bs_matrix_tape.hpp"

namespace laopt {

/**
 * A slice class where all operators just set the sparsity structure
 */
template<typename T, typename Base>
class BSSliceSparsity : public BSSliceBase<T, Base>
{
private:
    template<typename Derived>
    void create_sparsity_pattern(const Eigen::DenseBase<Derived>& mat)
    {
        assert(mat.rows() == this->M.rows() && mat.cols() == this->M.cols() && "You assigned a matrix of the wrong size!");

        auto m_row_indices = this->row_indices(this->M);
        auto m_col_indices = this->col_indices(this->M);

        for (Eigen::Index i = 0; i < (Eigen::Index) m_row_indices.size(); i++)
        {
            for (Eigen::Index j = 0; j < (Eigen::Index) m_col_indices.size(); j++)
            {
                this->base.sparsity_pattern.coeffRef(m_row_indices[i], m_col_indices[j]) = 1;
            }
        }
    }

public:
    BSSliceSparsity(Base& base, T M) : BSSliceBase<T, Base>(base, M) {}

    template<typename Derived>
    BSSliceSparsity& operator=(const Eigen::MatrixBase<Derived>& mat) { create_sparsity_pattern(mat); return *this; }

    template<typename Derived>
    typename std::enable_if<!std::is_base_of<Eigen::MatrixBase<Derived>, Derived>::value, BSSliceSparsity&>::type
    operator=(const Derived& scalar)
    {
        assert(this->M.rows() == 1 && this->M.cols() == 1 && "You tried to assign a scalar to a matrix");
        create_sparsity_pattern(Eigen::Matrix<Derived, 1, 1>(scalar));
        return *this;
    }

    template<typename Derived>
    BSSliceSparsity& operator+=(const Eigen::MatrixBase<Derived>& mat) { create_sparsity_pattern(mat); return *this; }

    template<typename Derived>
    typename std::enable_if<!std::is_base_of<Eigen::MatrixBase<Derived>, Derived>::value, BSSliceSparsity&>::type
    operator+=(const Derived& scalar)
    {
        assert(this->M.rows() == 1 && this->M.cols() == 1 && "You tried to assign a scalar to a matrix");
        create_sparsity_pattern(Eigen::Matrix<Derived, 1, 1>(scalar));
        return *this;
    }

    template<typename Derived>
    BSSliceSparsity& operator-=(const Eigen::MatrixBase<Derived>& mat) { create_sparsity_pattern(mat); return *this; }

    template<typename Derived>
    typename std::enable_if<!std::is_base_of<Eigen::MatrixBase<Derived>, Derived>::value, BSSliceSparsity&>::type
    operator-=(const Derived& scalar)
    {
        assert(this->M.rows() == 1 && this->M.cols() == 1 && "You tried to assign a scalar to a matrix");
        create_sparsity_pattern(Eigen::Matrix<Derived, 1, 1>(scalar));
        return *this;
    }
};

/**
 * A tape class to capture the sparsity pattern
 */
class BSMatrixSparsity : public BSSliceSparsity<Eigen::Map<Eigen::MatrixX<int>>, BSMatrixSparsity>
{
private:
    template<typename, typename>
    friend class laopt::BSSliceBase;
    template<typename, typename>
    friend class laopt::BSSliceSparsity;

    Eigen::SparseMatrix<bool> sparsity_pattern;

public:
    explicit BSMatrixSparsity(Eigen::Index rows = 0, Eigen::Index cols = 0)
            : BSSliceSparsity<Eigen::Map<Eigen::MatrixX<int>>, BSMatrixSparsity>(*this, Eigen::Map<Eigen::MatrixX<int>>(nullptr, 0, 0))
    {
        resize(rows, cols);
    };

    void set_zero() {}

    /**
     * Resize the matrix M.
     *
     * Note: Invalidates all slices!
     */
    void resize(Eigen::Index rows, Eigen::Index cols)
    {
        new(&M) Eigen::Map<Eigen::MatrixX<int>>(nullptr, rows, cols);
        sparsity_pattern.conservativeResize(rows, cols);
    }

    /**
     * Resize the matrix by adding rows rows and cols columns
     */
    void extend(Eigen::Index rows, Eigen::Index cols)
    {
        resize(M.rows() + rows, M.cols() + cols);
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
     * Create a BSMatrixTape from this sparsity structure
     */
    BSMatrixTape makeBSTape(Eigen::Index rows, Eigen::Index cols)
    {
        return BSMatrixTape(get_sparsity_pattern(), rows, cols);
    }

    /**
     * Return a structure that can be passed to a BSMatrixTape to initialize it
     */
    using Info = Eigen::SparseMatrix<bool>;

    Info generate()
    {
        return get_sparsity_pattern();
    }

private:
    template<typename Derived>
    auto makeSlice(Derived sub_matrix)
    {
        return BSSliceSparsity<Derived, BSMatrixSparsity>(*this, sub_matrix);
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
