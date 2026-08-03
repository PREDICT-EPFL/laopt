#ifndef LAOPT_BS_MATRIX_DENSE_HPP
#define LAOPT_BS_MATRIX_DENSE_HPP

#include <Eigen/Dense>

namespace laopt {

/**
 * A dense matrix whose size is determined dynamically during construction, and
 * then fixed during deployment.
 */
template<typename Derived>
class BSMatrixDenseBase
{
protected:
    Eigen::Index m_rows;
    Eigen::Index m_cols;

public:
    BSMatrixDenseBase(Eigen::Index rows, Eigen::Index cols) : m_rows(rows), m_cols(cols) {}

    // Current size of the m_matrix
    inline Eigen::Index rows() { return m_rows; }

    inline Eigen::Index cols() { return m_cols; }

    inline Eigen::Index size() { return m_rows * m_cols; }

    // Pre-allocated memory size
    inline Eigen::Index buffer_rows() { return static_cast<Derived*>(this)->m_mat.rows(); }

    inline Eigen::Index buffer_cols() { return static_cast<Derived*>(this)->m_mat.cols(); }

    // Set buffer to zero
    inline void set_zero()
    {
        static_cast<Derived*>(this)->m_mat.setZero();
    }

    // Get the current sub-m_matrix
    EIGEN_STRONG_INLINE auto value()
    {
        assert(rows() <= buffer_rows() && cols() <= buffer_cols() && "Buffer is too small");
        return static_cast<Derived*>(this)->m_mat.topLeftCorner(m_rows, m_cols);
    }

    auto data()
    {
        return static_cast<Derived*>(this)->m_mat.data();
    }

    // Grow by this number of elements
    void extend(Eigen::Index rows, Eigen::Index cols)
    {
        static_cast<Derived*>(this)->resize(this->rows() + rows, this->cols() + cols);
    }

    template<typename MDerived>
    BSMatrixDenseBase& operator=(const Eigen::MatrixBase<MDerived>& mat)
    {
        value() = mat;
        return *this;
    }

    template<typename MDerived>
    BSMatrixDenseBase& operator+=(const Eigen::MatrixBase<MDerived>& mat)
    {
        value() += mat;
        return *this;
    }

    template<typename MDerived>
    BSMatrixDenseBase& operator-=(const Eigen::MatrixBase<MDerived>& mat)
    {
        value() -= mat;
        return *this;
    }

    template<typename RowSlice, typename ColSlice>
    inline decltype(auto) operator()(const RowSlice& row_slice, const ColSlice& col_slice)
    {
        return value()(row_slice, col_slice);
    }

    // The following three overloads are needed to handle raw Index[N] arrays.
    template<typename RowIndicesT, std::size_t RowIndicesN, typename ColIndices>
    decltype(auto) operator()(const RowIndicesT (&row_indices)[RowIndicesN], const ColIndices& col_indices)
    {
        return value()(row_indices, col_indices);
    }
    template<typename RowIndices, typename ColIndicesT, std::size_t ColIndicesN>
    decltype(auto) operator()(const RowIndices& row_indices, const ColIndicesT (&col_indices)[ColIndicesN])
    {
        return value()(row_indices, col_indices);
    }
    template<typename RowIndicesT, std::size_t RowIndicesN, typename ColIndicesT, std::size_t ColIndicesN>
    decltype(auto) operator()(const RowIndicesT (&row_indices)[RowIndicesN], const ColIndicesT (&col_indices)[ColIndicesN])
    {
        return value()(row_indices, col_indices);
    }

    template<typename RowSlice>
    inline decltype(auto) operator()(const RowSlice& row_indices)
    {
        return value()(row_indices, 0);
    }

    template<typename RowIndicesT, std::size_t RowIndicesN>
    inline decltype(auto) operator()(const RowIndicesT (&row_indices)[RowIndicesN])
    {
        return value()(row_indices, 0);
    }

    /**
     * Return information required to initialize a BSDenseMatrix
     */
    struct Info
    {
        int rows, cols;
    };

    Info generate()
    {
        Info info;
        info.rows = rows();
        info.cols = cols();
        return info;
    }
};


template<typename scalar_t_>
class BSMatrixDenseConstruction : public BSMatrixDenseBase<BSMatrixDenseConstruction<scalar_t_>>
{
public:
    using scalar_t = scalar_t_;

private:
    friend BSMatrixDenseBase<BSMatrixDenseConstruction<scalar_t>>;

    // Buffer for the matrix during construction
    Eigen::MatrixX<scalar_t> m_mat;

public:
    BSMatrixDenseConstruction() : BSMatrixDenseBase<BSMatrixDenseConstruction<scalar_t_>>(0, 0), m_mat(0, 0) {}

    explicit BSMatrixDenseConstruction(const typename BSMatrixDenseBase<BSMatrixDenseConstruction<scalar_t_>>::Info& info) :
        BSMatrixDenseBase<BSMatrixDenseConstruction<scalar_t_>>(info.rows, info.cols), m_mat(info.rows, info.cols) {}

    using BSMatrixDenseBase<BSMatrixDenseConstruction<scalar_t_>>::operator=;

    void resize(Eigen::Index rows, Eigen::Index cols)
    {
        eigen_assert(rows >= 0 && cols >= 0);

        if (rows > this->m_rows || cols > this->m_cols) {
            // This should never happen during deployment
            m_mat.conservativeResize(rows, cols);
        }

        this->m_rows = rows;
        this->m_cols = cols;
    }
};

template<typename scalar_t_>
class BSMatrixDenseDeployment : public BSMatrixDenseBase<BSMatrixDenseDeployment<scalar_t_>>
{
public:
    using scalar_t = scalar_t_;

private:
    friend BSMatrixDenseBase<BSMatrixDenseDeployment<scalar_t>>;

    // Buffer for the matrix
    Eigen::Map<Eigen::MatrixX<scalar_t>> m_mat;

public:
    BSMatrixDenseDeployment() : BSMatrixDenseBase<BSMatrixDenseDeployment<scalar_t_>>(0, 0), m_mat(nullptr, 0, 0) {}

    explicit BSMatrixDenseDeployment(const typename BSMatrixDenseBase<BSMatrixDenseConstruction<scalar_t_>>::Info& info) :
        BSMatrixDenseBase<BSMatrixDenseDeployment<scalar_t_>>(info.rows, info.cols), m_mat(nullptr, info.rows, info.cols) {}
    explicit BSMatrixDenseDeployment(Eigen::Ref<Eigen::MatrixX<scalar_t>> mat) :
        BSMatrixDenseBase<BSMatrixDenseDeployment<scalar_t_>>(mat.rows(), mat.cols()), m_mat(mat.data(), mat.rows(), mat.cols()) {}

    using BSMatrixDenseBase<BSMatrixDenseDeployment<scalar_t_>>::operator=;

    void set_buffer(Eigen::Ref<Eigen::MatrixX<scalar_t>> mat)
    {
        new (&m_mat) Eigen::Map<Eigen::MatrixX<scalar_t>>(mat.data(), mat.rows(), mat.cols());
    }

    void resize(Eigen::Index rows, Eigen::Index cols)
    {
        this->m_rows = rows;
        this->m_cols = cols;
    }
};

} // namespace laopt

#endif //LAOPT_BS_MATRIX_DENSE_HPP
