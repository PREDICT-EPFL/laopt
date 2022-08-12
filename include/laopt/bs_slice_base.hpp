#ifndef LAOPT_BS_SLICE_BASE_HPP
#define LAOPT_BS_SLICE_BASE_HPP

#include <Eigen/Dense>

namespace laopt {

template<typename T, typename Base>
struct BSSliceBase
{
    Base& base; // Pointer to the top-level matrix
    T M; // Matrix slice

    using Scalar = double;

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
    inline void reset_copy_index() { }

protected:

    /**
     * Extract the sparsity pattern of a given matrix
     *
     * Dense matrices are assumed to be dense, sparse have patterns.
     *
     * TODO: Implement sparse base
     */
    template<typename Derived>
    Eigen::MatrixX<int> get_pattern(const Eigen::DenseBase<Derived>& mat)
    {
        return Eigen::MatrixX<int>::Constant(mat.rows(), mat.cols(), 1);
    }
};

} // namespace laopt

#endif // LAOPT_BS_SLICE_BASE_HPP
