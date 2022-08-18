#ifndef LAOPT_BS_MATRIX_HPP
#define LAOPT_BS_MATRIX_HPP

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace laopt
{

/**
 * Copy information for a sparse block matrix
 */
struct Segment
{
    size_t index;  // Index into the target.valuePtr()
    size_t length; // Number of element to copy

    bool operator==(const Segment other) const {
        return other.index == index && other.length == length;
    }

    /**
     * Return an Eigen ArithmeticSequence representing this Segment
     */
    inline auto seq() const {
        return Eigen::seqN(index, length);
    };
};

inline std::ostream &operator<<(std::ostream &os, std::vector<Segment> const &sequence)
{
    for (auto &seg: sequence)
    {
        os << "(" << seg.index << "," << seg.length << ")";
    }
    return os;
}

struct CopyInfo
{
    size_t segment_index;  // Index into segments
    size_t num_segments_to_copy;  // Number of segments to copy to execute this task
};

inline std::ostream &operator<<(std::ostream &os, std::vector<CopyInfo> const &sequence) 
{
    for (auto&seg: sequence)
    {
        os << "(" << seg.segment_index << "," << seg.num_segments_to_copy << ")";
    }
    return os;
}

/**
 * Information required to construct a BSMatrix
 */
struct BSMatrixInfo
{
    Eigen::Index rows = 0;
    Eigen::Index cols = 0;
    Eigen::SparseMatrix<bool> sparsity_structure;
    std::vector<Segment> copy_segments;
    std::vector<CopyInfo> copy_info;
};

template<typename scalar_t>
class BSMatrix
{
    Eigen::SparseMatrix<bool> sparsity_structure;
    const std::vector<Segment> segments;
    const std::vector<CopyInfo> copies;
    size_t copy_index; // Current index into copies

    scalar_t *target = nullptr; // Where we're going to write the data

    inline void reset_copy_index() { copy_index = 0; }

    // Execute the next copy in the sequence
    template<typename Op>
    inline void execute_operation(Op op, const scalar_t *source)
    {
        int segment_index = copies[copy_index].segment_index;
        for (size_t i = 0; i < copies[copy_index].num_segments_to_copy; i++)
        {
            size_t length = segments[segment_index + i].length;
            size_t index = segments[segment_index + i].index;
            auto tgt = Eigen::Map<Eigen::VectorX<scalar_t>>(target + index, length);
            const auto src = Eigen::Map<const Eigen::VectorX<scalar_t>>(source, length);
            op(tgt, src);
            source += length;
        }
        copy_index++;
        if (copy_index == copies.size()) reset_copy_index();
    }

public:
    /**
     * Note: The BSMatrix owns no memory, and so set_target must be called
     * before any operations are done!
     */
    BSMatrix(const Eigen::SparseMatrix<bool>& sparsity_structure,
             std::vector<Segment> copy_segments,
             std::vector<CopyInfo> copy_info)
            : sparsity_structure(sparsity_structure),
              segments(std::move(copy_segments)),
              copies(std::move(copy_info)),
              copy_index(0) {}

    explicit BSMatrix(const BSMatrixInfo& info)
            : sparsity_structure(info.sparsity_structure),
              segments(info.copy_segments),
              copies(info.copy_info),
              copy_index(0) {}

    /**
     * Initialize S to the right sparsity structure and set it
     * as the target
     */
    void allocate_memory(Eigen::SparseMatrix<scalar_t>& S)
    {
        S = sparsity_structure.eval().template cast<scalar_t>();
        set_target(S);
    }

    /**
     * Use the given matrix as the target.
     * Must already have been initialized to the correct sparsity structure!
     */
    void set_target(Eigen::SparseMatrix<scalar_t>& S)
    {
        target = S.valuePtr();
    }

    void set_target(Eigen::Ref<Eigen::MatrixX<scalar_t>> S)
    {
        assert(S.rows() * S.cols() == sparsity_structure.nonZeros() && "Buffer size too small");
        target = S.data();
    }

    /**
     * Clear the matrix to all-zeros
     */
    void set_zero()
    {
        if (target != nullptr)
        {
            Eigen::Map<Eigen::VectorX<scalar_t>>(target, sparsity_structure.nonZeros()).array() = 0;
        }
    }

    Eigen::SparseMatrix<bool> get_sparsity_structure()
    {
        return sparsity_structure;
    }

    // Assumption: The input matrix is contiguous.
    template<typename Derived>
    inline BSMatrix& operator=(const Eigen::MatrixBase<Derived>& block)
    {
        // MatrixBase may or may not be an expression. As a result, we call
        // eval, which evaluates into a contiguous temporary if required,
        // or is just a noop if not.
        // Note: Avoid temporaries here - they require malloc and are slow.
        execute_operation([](auto& a, auto& b) { a = b; }, block.eval().data());
        return *this;
    }

    // Assumption: The input matrix is contiguous. Don't change this to a Ref.
    template<typename Derived>
    void inline operator+=(const Derived& block)
    {
        execute_operation([](auto& a, auto& b) { a += b; }, block.eval().data());
    }

    // Assumption: The input matrix is contiguous. Don't change this to a Ref.
    template<typename Derived>
    void operator-=(const Derived& block)
    {
        execute_operation([](auto& a, auto& b) { a -= b; }, block.eval().data());
    }

    // These all compile out. Not used in deployment.
    template<typename RowSlice, typename ColSlice>
    BSMatrix<scalar_t>& operator()(RowSlice rows, ColSlice cols) { return *this; }

    // Vector format
    template<typename RowSlice>
    BSMatrix<scalar_t>& operator()(RowSlice rows) { return *this; }

    auto row(size_t i) { return *this; }

    auto col(size_t i) { return *this; }

    void resize(int rows, int cols) {}

    /**
     * Resize the matrix by adding rows rows and cols columns
     */
    void extend(int rows, int cols) {}

    inline auto rows() { return sparsity_structure.rows(); }

    inline auto cols() { return sparsity_structure.cols(); }
};

} // namespace laopt

#endif // LAOPT_BS_MATRIX_HPP
