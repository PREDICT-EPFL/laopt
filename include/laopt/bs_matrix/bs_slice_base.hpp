#ifndef LAOPT_BS_SLICE_BASE_HPP
#define LAOPT_BS_SLICE_BASE_HPP

#include <Eigen/Dense>

namespace laopt {

template<typename Child, typename NullMat>
class BSSliceBase
{
protected:
    NullMat null_mat; // Memory-less Eigen matrix that the BS... object imitates. Will use its dimensions and methods

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

    auto diagonal()
    {
        return static_cast<Child*>(this)->make_slice(null_mat.diagonal());
    }

    /**
     * These overloads handle the operation / right hand side.
     * They call the capture_sparsity() function which is specific to the child class.
     */
    template<typename Derived>
    Child& operator=(const Eigen::EigenBase<Derived>& mat)
    {
        static_cast<Child*>(this)->capture_sparsity(mat.derived());
        return *static_cast<Child*>(this);
    }
    template<typename Derived>
    typename std::enable_if<!std::is_base_of<Eigen::EigenBase<Derived>, Derived>::value, Child&>::type
    operator=(const Derived& scalar)
    {
        assert(this->null_mat.rows() == 1 && this->null_mat.cols() == 1 && "You tried to assign a scalar to a matrix");
        static_cast<Child*>(this)->capture_sparsity(Eigen::Matrix<Derived, 1, 1>(scalar));
        return *static_cast<Child*>(this);;
    }

    template<typename Derived>
    Child& operator+=(const Eigen::EigenBase<Derived>& mat)
    {
        static_cast<Child*>(this)->capture_sparsity(mat.derived());
        return *static_cast<Child*>(this);;
    }

    template<typename Derived>
    typename std::enable_if<!std::is_base_of<Eigen::EigenBase<Derived>, Derived>::value, Child&>::type
    operator+=(const Derived& scalar)
    {
        assert(this->null_mat.rows() == 1 && this->null_mat.cols() == 1 && "You tried to assign a scalar to a matrix");
        static_cast<Child*>(this)->capture_sparsity(Eigen::Matrix<Derived, 1, 1>(scalar));
        return *static_cast<Child*>(this);;
    }

    template<typename Derived>
    Child& operator-=(const Eigen::EigenBase<Derived>& mat)
    {
        static_cast<Child*>(this)->capture_sparsity(mat.derived());
        return *static_cast<Child*>(this);;
    }

    template<typename Derived>
    typename std::enable_if<!std::is_base_of<Eigen::EigenBase<Derived>, Derived>::value, Child&>::type
    operator-=(const Derived& scalar)
    {
        assert(this->null_mat.rows() == 1 && this->null_mat.cols() == 1 && "You tried to assign a scalar to a matrix");
        static_cast<Child*>(this)->capture_sparsity(Eigen::Matrix<Derived, 1, 1>(scalar));
        return *static_cast<Child*>(this);;
    }
};

} // namespace laopt

#endif // LAOPT_BS_SLICE_BASE_HPP
