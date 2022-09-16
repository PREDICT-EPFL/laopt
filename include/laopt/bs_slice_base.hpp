#ifndef LAOPT_BS_SLICE_BASE_HPP
#define LAOPT_BS_SLICE_BASE_HPP

#include <Eigen/Dense>

namespace laopt {

template<typename Child, typename NullMat>
class BSSliceBase
{
protected:
    NullMat null_mat; // Memory-less Eigen matrix that the BS... object imitates. Will use its dimensions and methods

    /**
     * These static functions take a sliced sub matrix and recover from it
     * the complete set of indices (i.e., each one) that it used to occupy
     * in the original matrix it was sliced from.
     * */

    // For a block
    template<int BlockRows, int BlockCols, bool InnerPanel>
    static Eigen::Vector<int, BlockRows>
    row_indices(const Eigen::Block<Eigen::Map<Eigen::MatrixX<int>>, BlockRows, BlockCols, InnerPanel>& block)
    {
        Eigen::Vector<int, BlockRows> indices(block.rows());
        for (Eigen::Index i = 0; i < block.rows(); i++)
        {
            indices(i) = block.startRow() + i;
        }
        return indices;
    }

    template<int BlockRows, int BlockCols, bool InnerPanel>
    static Eigen::Vector<int, BlockCols>
    col_indices(const Eigen::Block<Eigen::Map<Eigen::MatrixX<int>>, BlockRows, BlockCols, InnerPanel>& block)
    {
        Eigen::Vector<int, BlockCols> indices(block.cols());
        for (Eigen::Index i = 0; i < block.cols(); i++)
        {
            indices(i) = block.startCol() + i;
        }
        return indices;
    }

    // For a recursive block
    template<typename Derived, int BlockRows, int BlockCols, bool InnerPanel>
    static Eigen::Vector<int, BlockRows>
    row_indices(const Eigen::Block<Derived, BlockRows, BlockCols, InnerPanel>& block)
    {
        auto derived_row_indices = row_indices(block.nestedExpression());

        Eigen::Vector<int, BlockRows> indices(block.rows());
        for (Eigen::Index i = 0; i < block.rows(); i++)
        {
            indices(i) = derived_row_indices[block.startRow() + i];
        }
        return indices;
    }

    template<typename Derived, int BlockRows, int BlockCols, bool InnerPanel>
    static Eigen::Vector<int, BlockCols>
    col_indices(const Eigen::Block<Derived, BlockRows, BlockCols, InnerPanel>& block)
    {
        auto derived_col_indices = col_indices(block.nestedExpression());

        Eigen::Vector<int, BlockCols> indices(block.cols());
        for (Eigen::Index i = 0; i < block.cols(); i++)
        {
            indices(i) = derived_col_indices[block.startCol() + i];
        }
        return indices;
    }

    // For an IndexedView
    template<typename RowIndices, typename ColIndices>
    static RowIndices
    row_indices(const Eigen::IndexedView<Eigen::Map<Eigen::MatrixX<int>>, RowIndices, ColIndices>& view)
    {
        return view.rowIndices();
    }

    template<typename RowIndicesT, std::size_t RowIndicesN, typename ColIndices>
    static Eigen::Map<const Eigen::Vector<RowIndicesT, RowIndicesN>>
    row_indices(const Eigen::IndexedView<Eigen::Map<Eigen::MatrixX<int>>, const RowIndicesT(&)[RowIndicesN], ColIndices>& view)
    {
        return Eigen::Map<const Eigen::Vector<RowIndicesT, RowIndicesN>>(view.rowIndices());
    }

    template<typename RowIndices, typename ColIndices>
    static ColIndices
    col_indices(const Eigen::IndexedView<Eigen::Map<Eigen::MatrixX<int>>, RowIndices, ColIndices>& view)
    {
        return view.colIndices();
    }

    template<typename RowIndices, typename ColIndicesT, std::size_t ColIndicesN>
    static Eigen::Map<const Eigen::Vector<ColIndicesT, ColIndicesN>>
    col_indices(const Eigen::IndexedView<Eigen::Map<Eigen::MatrixX<int>>, RowIndices, const ColIndicesT(&)[ColIndicesN]>& view)
    {
        return Eigen::Map<const Eigen::Vector<ColIndicesT, ColIndicesN>>(view.colIndices());
    }

    // For a recursive IndexedView
    template<typename Derived, typename RowIndices, typename ColIndices>
    static Eigen::Vector<int, Eigen::IndexedView<Derived, RowIndices, ColIndices>::RowsAtCompileTime>
    row_indices(const Eigen::IndexedView<Derived, RowIndices, ColIndices>& view)
    {
        auto derived_row_indices = row_indices(view.nestedExpression());

        Eigen::Vector<int, Eigen::IndexedView<Derived, RowIndices, ColIndices>::RowsAtCompileTime> indices(view.rows());
        for (Eigen::Index i = 0; i < view.rows(); i++)
        {
            indices(i) = derived_row_indices[view.rowIndices()[i]];
        }
        return indices;
    }

    template<typename Derived, typename RowIndices, typename ColIndices>
    static Eigen::Vector<int, Eigen::IndexedView<Derived, RowIndices, ColIndices>::ColsAtCompileTime>
    col_indices(const Eigen::IndexedView<Derived, RowIndices, ColIndices>& view)
    {
        auto derived_col_indices = col_indices(view.nestedExpression());

        Eigen::Vector<int, Eigen::IndexedView<Derived, RowIndices, ColIndices>::RowsAtCompileTime> indices(view.cols());
        for (Eigen::Index i = 0; i < view.cols(); i++)
        {
            indices(i) = derived_col_indices[view.colIndices()[i]];
        }
        return indices;
    }

public:
    explicit BSSliceBase(NullMat nullMat) : null_mat(std::move(nullMat)) {}

    // Getters for dimensions
    Eigen::Index rows() const { return null_mat.rows(); }
    Eigen::Index cols() const { return null_mat.cols(); }

    /**
     * These overloads allow to use any BS... object to be indexed and sliced by common Eigen methods.
     * They forward the resulting BS... sub matrix to the make_slice method implemented in the child class,
     * which will once again create a BS... object for the sub matrix and handle the arithmetic operation it
     * was called with.
     */

    // Using the row() or col() operator
    auto row(size_t i)
    {
        return static_cast<Child*>(this)->make_slice(null_mat.row(i));
    }
    auto col(size_t i)
    {
        return static_cast<Child*>(this)->make_slice(null_mat.col(i));
    }

    // Block out of a matrix
    template<typename RowSlice, typename ColSlice, typename std::enable_if<Eigen::internal::valid_indexed_view_overload<RowSlice, ColSlice>::value>::type* dummy = nullptr>
    auto operator()(const RowSlice& row_slice, const ColSlice& col_slice)
    {
        return static_cast<Child*>(this)->make_slice(null_mat(row_slice, col_slice));
    }
    // Scalar out of a matrix
    auto operator()(Eigen::Index i_row, Eigen::Index j_col)
    {
        return static_cast<Child*>(this)->make_slice(Eigen::Block<decltype(null_mat), 1, 1>(null_mat, i_row, j_col));
    }
    // The following three overloads are needed to handle raw Index[N] arrays.
    template<typename RowIndicesT, std::size_t RowIndicesN, typename ColIndices>
    auto operator()(const RowIndicesT (&row_indices)[RowIndicesN], const ColIndices& col_indices)
    {
        return static_cast<Child*>(this)->make_slice(null_mat(row_indices, col_indices));
    }
    template<typename RowIndices, typename ColIndicesT, std::size_t ColIndicesN>
    auto operator()(const RowIndices& row_indices, const ColIndicesT (&col_indices)[ColIndicesN])
    {
        return static_cast<Child*>(this)->make_slice(null_mat(row_indices, col_indices));
    }
    template<typename RowIndicesT, std::size_t RowIndicesN, typename ColIndicesT, std::size_t ColIndicesN>
    auto operator()(const RowIndicesT (&row_indices)[RowIndicesN], const ColIndicesT (&colIndices)[ColIndicesN])
    {
        return static_cast<Child*>(this)->make_slice(null_mat(row_indices, colIndices));
    }

    // Segment out of a vector
    template<typename RowSlice, typename std::enable_if<!Eigen::internal::is_valid_index_type<RowSlice>::value>::type* dummy = nullptr>
    auto operator()(const RowSlice& row_slice)
    {
        assert(null_mat.cols() == 1 && "YOU APPLIED A VECTOR METHOD TO A MATRIX");
        return static_cast<Child*>(this)->make_slice(null_mat(row_slice, 0));
    }
    // Scalar out of a vector
    auto operator()(Eigen::Index i_row)
    {
        assert(null_mat.cols() == 1 && "YOU APPLIED A VECTOR METHOD TO A MATRIX");
        return static_cast<Child*>(this)->make_slice(Eigen::Block<decltype(null_mat), 1, 1>(null_mat, i_row, 0));
    }
    // Raw Index[N] arrays
    template<typename RowIndicesT, std::size_t RowIndicesN>
    auto operator()(const RowIndicesT (&row_indices)[RowIndicesN])
    {
        assert(null_mat.cols() == 1 && "YOU APPLIED A VECTOR METHOD TO A MATRIX");
        return static_cast<Child*>(this)->make_slice(null_mat(row_indices, 0));
    }

    /**
     * These overloads handle the operation / right hand side.
     * They call the class-specific sparsity capture function capture_sparsity().
     */
    template<typename Derived>
    Child& operator=(const Eigen::MatrixBase<Derived>& mat)
    {
        static_cast<Child*>(this)->capture_sparsity(mat);
        return *static_cast<Child*>(this);
    }
    template<typename Derived>
    typename std::enable_if<!std::is_base_of<Eigen::MatrixBase<Derived>, Derived>::value, Child&>::type
    operator=(const Derived& scalar)
    {
        assert(this->null_mat.rows() == 1 && this->null_mat.cols() == 1 && "You tried to assign a scalar to a matrix");
        static_cast<Child*>(this)->capture_sparsity(Eigen::Matrix<Derived, 1, 1>(scalar));
        return *static_cast<Child*>(this);;
    }

    template<typename Derived>
    Child&operator+=(const Eigen::MatrixBase<Derived>& mat)
    {
        static_cast<Child*>(this)->capture_sparsity(mat);
        return *static_cast<Child*>(this);;
    }

    template<typename Derived>
    typename std::enable_if<!std::is_base_of<Eigen::MatrixBase<Derived>, Derived>::value, Child&>::type
    operator+=(const Derived& scalar)
    {
        assert(this->null_mat.rows() == 1 && this->null_mat.cols() == 1 && "You tried to assign a scalar to a matrix");
        static_cast<Child*>(this)->capture_sparsity(Eigen::Matrix<Derived, 1, 1>(scalar));
        return *static_cast<Child*>(this);;
    }

    template<typename Derived>
    Child& operator-=(const Eigen::MatrixBase<Derived>& mat)
    {
        static_cast<Child*>(this)->capture_sparsity(mat);
        return *static_cast<Child*>(this);;
    }
    template<typename Derived>
    typename std::enable_if<!std::is_base_of<Eigen::MatrixBase<Derived>, Derived>::value, Child&>::type
    operator-=(const Derived& scalar)
    {
        assert(this->null_mat.rows() == 1 && this->null_mat.cols() == 1 && "You tried to assign a scalar to a matrix");
        static_cast<Child*>(this)->capture_sparsity(Eigen::Matrix<Derived, 1, 1>(scalar));
        return *static_cast<Child*>(this);;
    }
};

} // namespace laopt

#endif // LAOPT_BS_SLICE_BASE_HPP
