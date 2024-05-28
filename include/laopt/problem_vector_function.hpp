#ifndef LAOPT_PROBLEM_VECTOR_FUNCTION_HPP
#define LAOPT_PROBLEM_VECTOR_FUNCTION_HPP

namespace laopt
{

/**
 * Information about the function.
 *
 * Either sparsity or tape information, depending on Matrix and Vector type
 */
template<typename MatrixType, typename VectorType>
struct VectorFunctionInfo
{
    int rows, variables;
    typename MatrixType::Info jacobian;
    typename VectorType::Info value;
    typename VectorType::Info lb;
    typename VectorType::Info ub;
};

template<typename scalar_t>
struct VectorFunctionMemory;

/**
 * A vector-valued function with a sparse jacobian.
 */
template<typename MatrixType, typename VectorType>
class VectorFunction
{
    int m_rows;
    int m_variables;

public:
    using scalar_t = typename VectorType::scalar_t;

    VectorType value;
    MatrixType jacobian;
    VectorType lb, ub;

    VectorFunction() : m_rows(0), m_variables(0)
    {
        value.resize(m_rows, 1);
        jacobian.resize(m_rows, m_variables);
        lb.resize(m_rows, 1);
        ub.resize(m_rows, 1);
    }

    /**
     * Constructor for tape recording or deployment
     */
    template<typename OtherMatrixType, typename OtherVectorType>
    explicit VectorFunction(const VectorFunctionInfo<OtherMatrixType, OtherVectorType>& info) :
        m_rows(info.rows), m_variables(info.variables),
        value(info.value), jacobian(info.jacobian), lb(info.lb), ub(info.ub) {}

    inline int rows() { return m_rows; }

    inline int variables() { return m_variables; }

    /**
     * Increase the number of rows
     */
    void extend_rows(int rows)
    {
        m_rows += rows;
        value.extend(rows, 0);
        jacobian.extend(rows, 0);
        lb.extend(rows, 0);
        ub.extend(rows, 0);
    }

    /**
     * Increase the number of variables
     */
    void extend_variables(int variables)
    {
        m_variables += variables;
        jacobian.extend(0, variables);
    }

    template<typename Indices, typename Derived>
    void assign_lower_bound(const Indices& indices, const Eigen::MatrixBase<Derived>& lb_)
    {
        lb(indices) = lb_;
    }

    template<typename Indices, typename Derived>
    void assign_lower_bound(const Indices& indices, const Derived& lb_)
    {
        lb(indices).array() = lb_;
    }

    template<typename Indices, typename Derived>
    void assign_upper_bound(const Indices& indices, const Eigen::MatrixBase<Derived>& ub_)
    {
        ub(indices) = ub_;
    }

    template<typename Indices, typename Derived>
    void assign_upper_bound(const Indices& indices, const Derived& ub_)
    {
        ub(indices).array() = ub_;
    }

    VectorFunctionInfo<MatrixType, VectorType> generate()
    {
        VectorFunctionInfo<MatrixType, VectorType> info;

        info.rows = jacobian.rows();
        info.variables = jacobian.cols();

        info.jacobian = jacobian.generate();
        info.value = value.generate();
        info.lb = lb.generate();
        info.ub = ub.generate();

        return info;
    }
};

/**
 * A helper class that can be used to create all the required memory for
 * a VectorFunction if the memory isn't held externally.
 */
template<typename scalar_t>
struct VectorFunctionMemory {
    Eigen::SparseMatrix<scalar_t> jacobian;
    Eigen::VectorX<scalar_t> value;
    Eigen::VectorX<scalar_t> lb;
    Eigen::VectorX<scalar_t> ub;

    // Pointer to the valuePtr of the jacobian
    Eigen::Map<Eigen::VectorX<scalar_t>> jacobian_buffer;

    template<typename MatrixType, typename VectorType>
    explicit VectorFunctionMemory(VectorFunction<MatrixType, VectorType>& f) :
            value(f.rows(), 1),
            lb(f.rows(), 1),
            ub(f.rows(), 1),
            jacobian_buffer(nullptr, 0)
    {
        f.jacobian.allocate_memory(jacobian);
        new (&jacobian_buffer) Eigen::Map<Eigen::VectorX<scalar_t>>(jacobian.valuePtr(), jacobian.nonZeros());

        // Set this memory as the buffer for the function
        f.jacobian.set_target(jacobian);
        f.value.set_buffer(value);
        f.lb.set_buffer(lb);
        f.ub.set_buffer(ub);
    }
};

} // namespace laopt

#endif // LAOPT_PROBLEM_VECTOR_FUNCTION_HPP
