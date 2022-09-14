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

    template<typename RowIndicesT, std::size_t RowIndicesN, typename ColIndices>
    Eigen::Map<const Eigen::Vector<RowIndicesT, RowIndicesN>>
    row_indices(const Eigen::IndexedView<Eigen::Map<Eigen::MatrixX<int>>, const RowIndicesT(&)[RowIndicesN], ColIndices>& mat)
    {
        return Eigen::Map<const Eigen::Vector<RowIndicesT, RowIndicesN>>(mat.rowIndices());
    }

    template<typename RowIndices, typename ColIndices>
    ColIndices col_indices(const Eigen::IndexedView<Eigen::Map<Eigen::MatrixX<int>>, RowIndices, ColIndices>& mat)
    {
        return mat.colIndices();
    }

    template<typename RowIndices, typename ColIndicesT, std::size_t ColIndicesN>
    Eigen::Map<const Eigen::Vector<ColIndicesT, ColIndicesN>>
    col_indices(const Eigen::IndexedView<Eigen::Map<Eigen::MatrixX<int>>, RowIndices, const ColIndicesT(&)[ColIndicesN]>& mat)
    {
        return Eigen::Map<const Eigen::Vector<ColIndicesT, ColIndicesN>>(mat.colIndices());
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

    // Matrix slicing (2D)
    template<typename RowSlice, typename ColSlice, typename std::enable_if<Eigen::internal::valid_indexed_view_overload<RowSlice, ColSlice>::value>::type* dummy = nullptr>
    auto operator()(const RowSlice& row_slice, const ColSlice& col_slice)
    {
        return base.makeSlice(M(row_slice, col_slice));
    }
    // Scalar
    auto operator()(Eigen::Index i_row, Eigen::Index j_col)
    {
        return base.makeSlice(Eigen::Block<decltype(M), 1, 1>(M, i_row, j_col));
    }
    // The following three overloads are needed to handle raw Index[N] arrays.
    template<typename RowIndicesT, std::size_t RowIndicesN, typename ColIndices>
    auto operator()(const RowIndicesT (&row_indices)[RowIndicesN], const ColIndices& col_indices)
    {
        return base.makeSlice(M(row_indices, col_indices));
    }
    template<typename RowIndices, typename ColIndicesT, std::size_t ColIndicesN>
    auto operator()(const RowIndices& row_indices, const ColIndicesT (&col_indices)[ColIndicesN])
    {
        return base.makeSlice(M(row_indices, col_indices));
    }
    template<typename RowIndicesT, std::size_t RowIndicesN, typename ColIndicesT, std::size_t ColIndicesN>
    auto operator()(const RowIndicesT (&row_indices)[RowIndicesN], const ColIndicesT (&colIndices)[ColIndicesN])
    {
        return base.makeSlice(M(row_indices, colIndices));
    }

    // Vector slicing (1D)
    template<typename RowSlice, typename std::enable_if<!Eigen::internal::is_valid_index_type<RowSlice>::value>::type* dummy = nullptr>
    auto operator()(const RowSlice& row_slice)
    {
        assert(M.cols() == 1 && "YOU APPLIED A VECTOR METHOD TO A MATRIX");
        return base.makeSlice(M(row_slice, 0));
    }
    // Scalar
    auto operator()(Eigen::Index i_row)
    {
        assert(M.cols() == 1 && "YOU APPLIED A VECTOR METHOD TO A MATRIX");
        return base.makeSlice(Eigen::Block<decltype(M), 1, 1>(M, i_row, 0));
    }
    // Raw Index[N] arrays
    template<typename RowIndicesT, std::size_t RowIndicesN>
    auto operator()(const RowIndicesT (&row_indices)[RowIndicesN])
    {
        assert(M.cols() == 1 && "YOU APPLIED A VECTOR METHOD TO A MATRIX");
        return base.makeSlice(M(row_indices, 0));
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
