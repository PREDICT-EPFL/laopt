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
    BSMatrixDenseBase() : m_rows(0), m_cols(0) {}

    // Current size of the m_matrix
    inline Eigen::Index rows() { return m_rows; }

    inline Eigen::Index cols() { return m_cols; }

    inline Eigen::Index size() { return m_rows * m_cols; }

    // Pre-allocated memory size
    inline Eigen::Index buffer_rows() { return static_cast<Derived*>(this)->m_mat.rows(); }

    inline Eigen::Index buffer_cols() { return static_cast<Derived*>(this)->m_mat.cols(); }

    // Set buffer to zero
    inline void zero_buffer()
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
        static_cast<Derived*>(this)->resize(BSMatrixDenseBase::rows() + rows, BSMatrixDenseBase::cols() + cols);
    }

    template<typename RowSlice, typename ColSlice>
    inline auto operator()(const RowSlice rows, const ColSlice cols)
    {
        return value()(rows, cols);
    }

    template<typename RowSlice>
    inline auto operator()(const RowSlice rows)
    {
        return value()(rows, 0);
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
    BSMatrixDenseConstruction() : m_mat(0, 0) {}

    explicit BSMatrixDenseConstruction(const typename BSMatrixDenseBase<BSMatrixDenseConstruction<scalar_t_>>::Info& info) : m_mat(0, 0) {}

    void resize(Eigen::Index rows, Eigen::Index cols)
    {
        if (rows > 0 && cols > 0 && (rows > this->m_rows || cols > this->m_cols)) {
            // This should never happen during deployment
            m_mat.conservativeResize(rows, cols);
        }

        this->m_rows = rows;
        this->m_cols = cols;
    }

    /**
     * Clear the matrix to all zeros
     */
    void set_zero()
    {
        m_mat.setZero();
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
    BSMatrixDenseDeployment() : m_mat(nullptr, 0, 0) {}

    explicit BSMatrixDenseDeployment(const typename BSMatrixDenseBase<BSMatrixDenseConstruction<scalar_t_>>::Info& info) : m_mat(NULL, 0, 0) {}

    explicit BSMatrixDenseDeployment(Eigen::Ref<Eigen::MatrixX<scalar_t>> mat) : m_mat(mat.data(), mat.rows(), mat.cols())
    {
        set_buffer(mat);
    }

    void set_buffer(Eigen::Ref<Eigen::MatrixX<scalar_t>> mat)
    {
        // assert(this->rows() == mat.rows() && this->cols() == mat.cols() && "Buffer is the wrong size in set_buffer");
        new(&m_mat) Eigen::Map<Eigen::MatrixX<scalar_t>>(mat.data(), mat.rows(), mat.cols());
    }

    void resize(Eigen::Index rows, Eigen::Index cols)
    {
        // assert((rows * cols == 0 || rows <= this->buffer_rows() && cols <= this->buffer_cols()) && "You're resizing during deployment to a size larger than the buffer!");
        this->m_rows = rows;
        this->m_cols = cols;
    }

    /**
     * Clear the matrix to all zeros
     */
    void set_zero()
    {
        m_mat.setZero();
    }

};

} // namespace laopt

#endif //LAOPT_BS_MATRIX_DENSE_HPP
