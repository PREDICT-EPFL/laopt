#ifndef LAOPT_PROBLEM_WEIGHTED_SUM_FUNCTION_HPP
#define LAOPT_PROBLEM_WEIGHTED_SUM_FUNCTION_HPP

namespace laopt
{

/**
 * Information about a weighted sum function.
 *
 * Either sparsity or tape information, depending on Matrix and Vector
 */
template<typename Matrix, typename Vector>
struct WeightedSumFunctionInfo
{
    int rows, variables;
    typename Matrix::Info hessian;
    typename Vector::Info gradient;
    typename Vector::Info weights;
};

template<typename scalar_t>
struct WeightedSumFunctionMemory;

/**
 * A scalar function of the form f(x) = sum wi fi(x), with a sparse hessian.
 */
template<typename Matrix, typename Vector>
class WeightedSumFunction
{
    int m_rows;
    int m_variables;

public:
    using scalar_t = typename Vector::scalar_t;

    Vector weights;
    scalar_t value;
    Vector gradient;
    Matrix hessian;

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
    template<typename TMatrix, typename TVector>
    explicit WeightedSumFunction(const WeightedSumFunctionInfo<TMatrix, TVector>& info) :
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

    WeightedSumFunctionInfo<Matrix,Vector> generate()
    {
        WeightedSumFunctionInfo<Matrix,Vector> info;

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

    template<typename Matrix, typename Vector>
    explicit WeightedSumFunctionMemory(WeightedSumFunction<Matrix, Vector>& w) :
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
