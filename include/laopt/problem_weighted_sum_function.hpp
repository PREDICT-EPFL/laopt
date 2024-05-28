#ifndef LAOPT_PROBLEM_WEIGHTED_SUM_FUNCTION_HPP
#define LAOPT_PROBLEM_WEIGHTED_SUM_FUNCTION_HPP

namespace laopt
{

/**
 * Information about a weighted sum function.
 *
 * Either sparsity or tape information, depending on Matrix and Vector
 */
template<typename MatrixType, typename VectorType>
struct WeightedSumFunctionInfo
{
    int rows, variables;
    typename MatrixType::Info hessian;
    typename VectorType::Info gradient;
    typename VectorType::Info weights;
};

template<typename scalar_t>
struct WeightedSumFunctionMemory;

/**
 * A scalar function of the form f(x) = sum wi fi(x), with a sparse hessian.
 */
template<typename MatrixType, typename VectorType>
class WeightedSumFunction
{
    int m_rows;
    int m_variables;

public:
    using scalar_t = typename VectorType::scalar_t;

    VectorType weights;
    scalar_t value;
    VectorType gradient;
    MatrixType hessian;

    /**
     * Sparsity discovery constructor
     */
    WeightedSumFunction() : m_rows(0), m_variables(0)
    {
        hessian.resize(m_variables, m_variables);
        gradient.resize(m_variables, 1);
        weights.resize(m_rows, 1);
    }

    /**
     * Constructor for tape recording or deployment
     */
    template<typename OtherMatrixType, typename OtherVectorType>
    explicit WeightedSumFunction(const WeightedSumFunctionInfo<OtherMatrixType, OtherVectorType>& info) :
        m_rows(info.rows), m_variables(info.variables),
        weights(info.weights), gradient(info.gradient), hessian(info.hessian) {}

    inline int rows() { return m_rows; }

    inline int variables() { return m_variables; }

    /**
     * Increase the number of rows in the function
     */
    void extend_rows(int rows)
    {
        m_rows += rows;
        weights.extend(rows, 0);
    }

    /**
     * Increase the number of rows in the function
     */
    void extend_variables(int variables)
    {
        m_variables += variables;
        hessian.extend(variables, variables);
        gradient.extend(variables, 0);
    }

    WeightedSumFunctionInfo<MatrixType, VectorType> generate()
    {
        WeightedSumFunctionInfo<MatrixType, VectorType> info;

        info.rows = weights.rows();
        info.variables = hessian.cols();

        info.hessian = hessian.generate();
        info.gradient = gradient.generate();
        info.weights = weights.generate();

        return info;
    }
};

/**
 * A helper class that can be used to create all the required memory for
 * a WeightedSumFunction if the memory isn't held externally.
 */
template<typename scalar_t>
struct WeightedSumFunctionMemory
{
    Eigen::SparseMatrix<scalar_t> hessian;
    Eigen::VectorX<scalar_t> gradient;
    Eigen::VectorX<scalar_t> weights;

    // Pointer to the valuePtr of the hessian
    Eigen::Map<Eigen::VectorX<scalar_t>> hessian_buffer;

    template<typename MatrixType, typename VectorType>
    explicit WeightedSumFunctionMemory(WeightedSumFunction<MatrixType, VectorType>& w) :
            gradient(w.variables()),
            weights(w.rows()),
            hessian_buffer(nullptr, 0)
    {
        weights.array() = 1;

        w.hessian.allocate_memory(hessian);
        new (&hessian_buffer) Eigen::Map<Eigen::VectorX<scalar_t>>(hessian.valuePtr(), hessian.nonZeros());

        // Set this memory as the buffer for the function
        w.hessian.set_target(hessian);
        w.gradient.set_buffer(gradient);
        w.weights.set_buffer(weights);
    }
};

} // namespace laopt

#endif // LAOPT_PROBLEM_WEIGHTED_SUM_FUNCTION_HPP
