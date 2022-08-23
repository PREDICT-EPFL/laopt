#ifndef LAOPT_BS_SLICE_BASE_HPP
#define LAOPT_BS_SLICE_BASE_HPP

#include <Eigen/Dense>

namespace laopt {

template<typename T, typename Base>
class BSSliceBase
{
protected:
    Base& base; // Pointer to the top-level matrix
    T M; // Matrix slice

    template<int BlockRows, int BlockCols, bool InnerPanel>
    Eigen::Vector<int, BlockRows> row_indices(const Eigen::Block<Eigen::Map<Eigen::MatrixX<int>>, BlockRows, BlockCols, InnerPanel>& mat)
    {
        Eigen::Vector<int, BlockRows> indices(mat.rows());
        for (Eigen::Index i = 0; i < mat.rows(); i++)
        {
            indices(i) = mat.startRow() + i;
        }
        return indices;
    }

    template<int BlockRows, int BlockCols, bool InnerPanel>
    Eigen::Vector<int, BlockCols> col_indices(const Eigen::Block<Eigen::Map<Eigen::MatrixX<int>>, BlockRows, BlockCols, InnerPanel>& mat)
    {
        Eigen::Vector<int, BlockCols> indices(mat.cols());
        for (Eigen::Index i = 0; i < mat.cols(); i++)
        {
            indices(i) = mat.startCol() + i;
        }
        return indices;
    }

    template<typename Derived, int BlockRows, int BlockCols, bool InnerPanel>
    Eigen::Vector<int, BlockRows> row_indices(const Eigen::Block<Derived, BlockRows, BlockCols, InnerPanel>& mat)
    {
        auto derived_row_indices = row_indices(mat.nestedExpression());

        Eigen::Vector<int, BlockRows> indices(mat.rows());
        for (Eigen::Index i = 0; i < mat.rows(); i++)
        {
            indices(i) = derived_row_indices[mat.startRow() + i];
        }
        return indices;
    }

    template<typename Derived, int BlockRows, int BlockCols, bool InnerPanel>
    Eigen::Vector<int, BlockCols> col_indices(const Eigen::Block<Derived, BlockRows, BlockCols, InnerPanel>& mat)
    {
        auto derived_col_indices = col_indices(mat.nestedExpression());

        Eigen::Vector<int, BlockCols> indices(mat.cols());
        for (Eigen::Index i = 0; i < mat.cols(); i++)
        {
            indices(i) = derived_col_indices[mat.startCol() + i];
        }
        return indices;
    }

    template<typename RowIndices, typename ColIndices>
    RowIndices row_indices(const Eigen::IndexedView<Eigen::Map<Eigen::MatrixX<int>>, RowIndices, ColIndices>& mat)
    {
        return mat.rowIndices();
    }

    template<typename RowIndices, typename ColIndices>
    ColIndices col_indices(const Eigen::IndexedView<Eigen::Map<Eigen::MatrixX<int>>, RowIndices, ColIndices>& mat)
    {
        return mat.colIndices();
    }

    template<typename Derived, typename RowIndices, typename ColIndices>
    Eigen::Vector<int, Eigen::IndexedView<Derived, RowIndices, ColIndices>::RowsAtCompileTime>
    row_indices(const Eigen::IndexedView<Derived, RowIndices, ColIndices>& mat)
    {
        auto derived_row_indices = row_indices(mat.nestedExpression());

        Eigen::Vector<int, Eigen::IndexedView<Derived, RowIndices, ColIndices>::RowsAtCompileTime> indices(mat.rows());
        for (Eigen::Index i = 0; i < mat.rows(); i++)
        {
            indices(i) = derived_row_indices[mat.rowIndices()[i]];
        }
        return indices;
    }

    template<typename Derived, typename RowIndices, typename ColIndices>
    Eigen::Vector<int, Eigen::IndexedView<Derived, RowIndices, ColIndices>::ColsAtCompileTime>
    col_indices(const Eigen::IndexedView<Derived, RowIndices, ColIndices>& mat)
    {
        auto derived_col_indices = col_indices(mat.nestedExpression());

        Eigen::Vector<int, Eigen::IndexedView<Derived, RowIndices, ColIndices>::RowsAtCompileTime> indices(mat.cols());
        for (Eigen::Index i = 0; i < mat.cols(); i++)
        {
            indices(i) = derived_col_indices[mat.colIndices()[i]];
        }
        return indices;
    }

public:
    BSSliceBase(Base& base, T M) : base(base), M(std::move(M)) {}

    template<typename RowSlice, typename ColSlice>
    auto operator()(RowSlice rows, ColSlice cols)
    {
        return base.makeSlice(M(rows, cols));
    }

    // Vector format
    template<typename RowSlice>
    auto operator()(RowSlice rows)
    {
        assert(M.cols() == 1 && "YOU APPLIED A VECTOR METHOD TO A MATRIX");
        return base.makeSlice(M(rows, 0));
    }

    auto row(size_t i)
    {
        return base.makeSlice(M.row(i));
    }

    auto col(size_t i)
    {
        return base.makeSlice(M.col(i));
    }

    Eigen::Index rows() { return M.rows(); }

    Eigen::Index cols() { return M.cols(); }

    // Only used in BSMatrix
    inline void reset_copy_index() {}
};

} // namespace laopt

#endif // LAOPT_BS_SLICE_BASE_HPP
