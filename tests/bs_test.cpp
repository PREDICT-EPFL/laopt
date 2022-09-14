/**
 * Unit test for the construction and computation of LAMPC Block Sparse Matrices.
 */

#include <iostream>
#include <chrono>

#include "laopt/laopt.hpp"

#include "test_utils.hpp"
#include "gtest/gtest.h"

namespace {

template<typename Problem>
void test_BSMatrix_problem(Problem& problem)
{
    using scalar_t = typename Problem::scalar_t;

    // Get the sparsity structure
    laopt::BSMatrixSparsity sparsity(problem.rows, problem.cols);
    problem.eval(sparsity);

//    std::cout << "sparsity = \n" << Eigen::MatrixX<bool>(sparsity.get_sparsity_pattern()) << std::endl;

    // Confirm the correct sparsity structure
    EXPECT_TRUE(problem.expected_sparsity().sparseView().isApprox(sparsity.get_sparsity_pattern(), 0));

    // Get the copy sequence
    laopt::BSMatrixTape tape(sparsity.get_sparsity_pattern(), sparsity.rows(), sparsity.cols());
    problem.eval(tape);

//    std::cout << "copy sequence = " << tape.generate().copy_segments << std::endl;

    // Confirm the correct copy sequence
    std::vector<laopt::Segment> expected_copy_sequence = problem.expected_copy_sequence();
    EXPECT_TRUE(std::equal(expected_copy_sequence.begin(), expected_copy_sequence.end(), tape.generate().copy_segments.begin()));

    // Build the BSMatrix
    auto BS = tape.template makeBSMatrix<scalar_t>();
    Eigen::SparseMatrix<scalar_t> S;
    BS.allocate_memory(S);

    // Run the copy
    BS.set_zero();
    problem.eval(BS);

//    std::cout << "result = \n" << Eigen::MatrixX<scalar_t>(S) << std::endl;

    EXPECT_TRUE(problem.expected_result().sparseView().isApprox(S, 0));
}

template<typename Problem>
void test_BSMatrixDense_problem(Problem& problem)
{
    using scalar_t = typename Problem::scalar_t;

    laopt::BSMatrixDenseConstruction<scalar_t> D;
    D.resize(problem.rows, problem.cols);
    D.set_zero();
    problem.eval(D);
    EXPECT_EQ(D.value(), problem.expected_result());

    laopt::BSMatrixDenseDeployment<scalar_t> DD;
    DD.resize(problem.rows, problem.cols);
    // We expect an error without a buffer set
    ASSERT_DEATH(problem.eval(DD), "");

    // Test with exact buffer size
    Eigen::MatrixX<scalar_t> mat_exact(problem.rows, problem.cols);
    DD.set_buffer(mat_exact);
    DD.set_zero();
    problem.eval(DD);
    EXPECT_EQ(DD.value(), problem.expected_result());

    // Test with larger buffer size
    Eigen::MatrixX<scalar_t> mat_big(problem.rows + 10, problem.cols + 8);
    DD.set_buffer(mat_big);
    DD.set_zero();
    problem.eval(DD);
    EXPECT_EQ(DD.value(), problem.expected_result());
}

/**
 * Form a block-diagonal matrix using Slicing
 */
struct SlicingProblem
{
    using scalar_t = double;
    Eigen::MatrixX<scalar_t> A;
    Eigen::MatrixX<scalar_t> B;

    Eigen::Index rows = 6;
    Eigen::Index cols = 8;

    SlicingProblem() : A(2, 3), B(4, 5)
    {
        A << 1, 2, 3,
             4, 5, 6;
        B <<  7,  8,  9, 10, 11,
             12, 13, 14, 15, 16,
             17, 18, 19, 20, 21,
             22, 23, 24, 25, 26;
    }

    template<typename Tape>
    void eval(Tape& tape)
    {
        tape(Eigen::seqN(0, A.rows()), Eigen::seqN(0, A.cols())) = A;
        tape(Eigen::seqN(Eigen::lastp1 - B.rows(), B.rows()), Eigen::seqN(Eigen::lastp1 - B.cols(), B.cols())) = B;
    }

    static Eigen::MatrixX<bool> expected_sparsity()
    {
        Eigen::MatrixX<bool> sparsity(6, 8);
        sparsity << 1, 1, 1, 0, 0, 0, 0, 0,
                    1, 1, 1, 0, 0, 0, 0, 0,
                    0, 0, 0, 1, 1, 1, 1, 1,
                    0, 0, 0, 1, 1, 1, 1, 1,
                    0, 0, 0, 1, 1, 1, 1, 1,
                    0, 0, 0, 1, 1, 1, 1, 1;
        return sparsity;
    }

    static std::vector<laopt::Segment> expected_copy_sequence()
    {
        return {{0,6}, {6,20}};
    }

    static Eigen::MatrixX<scalar_t> expected_result()
    {
        Eigen::MatrixX<scalar_t> result(6, 8);
        result << 1, 2, 3, 0,   0,  0,  0,  0,
                  4, 5, 6, 0,   0,  0,  0,  0,
                  0, 0, 0, 7,   8,  9, 10, 11,
                  0, 0, 0, 12, 13, 14, 15, 16,
                  0, 0, 0, 17, 18, 19, 20, 21,
                  0, 0, 0, 22, 23, 24, 25, 26;
        return result;
    }
};

TEST(BSMatrix, Construction_Slicing) {
    SlicingProblem problem;
    test_BSMatrix_problem(problem);
}

TEST(BSMatrixDense, Construction_Slicing) {
    SlicingProblem problem;
    test_BSMatrixDense_problem(problem);
}


/**
 * Assign single elements.
 */
struct IndexingProblem
{
    using scalar_t = double;

    Eigen::Index rows = 3;
    Eigen::Index cols = 3;

    template<typename Tape>
    void eval(Tape& tape)
    {
        tape(0, 0) = 1.0;
        tape(0, 2) = 2;
        tape(1, 1) = 3;
        tape(2, 1) = 4;
        tape(2, 2) = 5;
    }

    static Eigen::MatrixX<bool> expected_sparsity()
    {
        Eigen::MatrixX<bool> sparsity(3, 3);
        sparsity << 1, 0, 1,
                    0, 1, 0,
                    0, 1, 1;
        return sparsity;
    }

    static std::vector<laopt::Segment> expected_copy_sequence()
    {
        return {{0,1},{3,1},{1,1},{2,1},{4,1}};
    }

    static Eigen::MatrixX<scalar_t> expected_result()
    {
        Eigen::MatrixX<scalar_t> result(3, 3);
        result << 1, 0, 2,
                  0, 3, 0,
                  0, 4, 5;
        return result;
    }
};

TEST(BSMatrix, Construction_Indexing) {
    IndexingProblem problem;
    test_BSMatrix_problem(problem);
}

TEST(BSMatrixDense, Construction_Indexing) {
    IndexingProblem problem;
    test_BSMatrixDense_problem(problem);
}


/**
 * Use arrays and initializer lists.
 */
struct ArraysProblem
{
    using scalar_t = double;
    Eigen::MatrixX<scalar_t> A;
    Eigen::MatrixX<scalar_t> B;

    Eigen::Index rows = 5;
    Eigen::Index cols = 5;

    ArraysProblem() : A(2, 3), B(4, 5)
    {
        A << 1, 2, 3,
             4, 5, 6;
        B <<  7,  8,  9, 10, 11,
             12, 13, 14, 15, 16,
             17, 18, 19, 20, 21,
             22, 23, 24, 25, 26;
    }

    template<typename Tape>
    void eval(Tape& tape)
    {
        std::array<int, 3> indices = {3, 0, 1};
        tape({3, 1}, indices) = A;
        tape(indices, 2) = B(Eigen::seqN(0, Eigen::fix<3>), 1);
        tape({4}, {3, 4}) = Eigen::Matrix<double, 1, 2>::Constant(9);
    }

    static Eigen::MatrixX<bool> expected_sparsity()
    {
        Eigen::MatrixX<bool> sparsity(5, 5);
        sparsity << 0, 0, 1, 0, 0,
                    1, 1, 1, 1, 0,
                    0, 0, 0, 0, 0,
                    1, 1, 1, 1, 0,
                    0, 0, 0, 1, 1;
        return sparsity;
    }

    static std::vector<laopt::Segment> expected_copy_sequence()
    {
        return {{8,1},{7,1},{1,1},{0,1},{3,1},{2,1},{6,1},{4,2},{9,2}};
    }

    static Eigen::MatrixX<scalar_t> expected_result()
    {
        Eigen::MatrixX<scalar_t> result(5, 5);
        result << 0, 0, 13, 0, 0,
                  5, 6, 18, 4, 0,
                  0, 0,  0, 0, 0,
                  2, 3,  8, 1, 0,
                  0, 0,  0, 9, 9;
        return result;
    }
};

TEST(BSMatrix, Construction_Arrays) {
    ArraysProblem problem;
    test_BSMatrix_problem(problem);
}

TEST(BSMatrixDense, Construction_Arrays) {
    ArraysProblem problem;
    test_BSMatrixDense_problem(problem);
}


/**
 * Partition two matrices and re-assemble into a block-sparse matrix
 */
struct ConcatenateProblem
{
    using scalar_t = double;
    Eigen::MatrixX<scalar_t> A;
    Eigen::MatrixX<scalar_t> B;

    Eigen::Index rows = 20;
    Eigen::Index cols = 20;

    ConcatenateProblem() : A(2, 3), B(5, 5)
    {
        A << 1, 2, 3,
             4, 5, 6;
        B <<  7,  8,  9, 10, 11,
             12, 13, 14, 15, 16,
             17, 18, 19, 20, 21,
             22, 23, 24, 25, 26,
             27, 28, 29, 30, 31;
    }

    template<typename Tape>
    void eval(Tape& tape)
    {
        tape(Eigen::seqN(10, A.rows()),
             laopt::concatenate_indices(Eigen::Vector<int, 2>{10, 11}, Eigen::Vector<int, 1>{5})) = A;
        tape(laopt::concatenate_indices(Eigen::Vector<int, 2>{6, 7}, Eigen::Vector<int, 3>{0, 1, 2}),
             laopt::concatenate_indices(Eigen::Vector<int, 1>{7}, Eigen::Vector<int, 1>{5}, Eigen::Vector<int, 1>{3},
                                        Eigen::Vector<int, 1>{1}, Eigen::Vector<int, 1>{0})) = B;
    }

    static Eigen::MatrixX<bool> expected_sparsity()
    {
        Eigen::MatrixX<bool> sparsity(20, 20);
        sparsity << 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
        return sparsity;
    }

    static std::vector<laopt::Segment> expected_copy_sequence()
    {
        return {{27,4},{20,2},{25,2},{22,3},{18,2},{15,3},{13,2},{10,3},{8,2},{5,3},{3,2},{0,3}};
    }

    static Eigen::MatrixX<scalar_t> expected_result()
    {
        Eigen::MatrixX<scalar_t> result(20, 20);
        result << 21, 20,  0, 19,  0, 18,  0, 17,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                  26, 25,  0, 24,  0, 23,  0, 22,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                  31, 30,  0, 29,  0, 28,  0, 27,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                  11, 10,  0,  9,  0,  8,  0,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                  16, 15,  0, 14,  0, 13,  0, 12,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                   0,  0,  0,  0,  0,  3,  0,  0,  0,  0,  1,  2,  0,  0,  0,  0,  0,  0,  0,  0,
                   0,  0,  0,  0,  0,  6,  0,  0,  0,  0,  4,  5,  0,  0,  0,  0,  0,  0,  0,  0,
                   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0;
        return result;
    }
};

TEST(BSMatrix, Construction_Concatenate) {
    ConcatenateProblem problem;
    test_BSMatrix_problem(problem);
}

TEST(BSMatrixDense, Construction_Concatenate) {
    ConcatenateProblem problem;
    test_BSMatrixDense_problem(problem);
}


/**
 * Form a block-diagonal matrix and then add to it
 */
struct SumProblem
{
    using scalar_t = double;
    Eigen::MatrixX<scalar_t> A;
    Eigen::MatrixX<scalar_t> B;

    Eigen::Index rows = 10;
    Eigen::Index cols = 9;

    SumProblem() : A(2, 3), B(4, 5)
    {
        A.setConstant(1);
        B.setConstant(2);
    }

    template<typename Tape>
    void eval(Tape& tape)
    {
        tape(Eigen::seqN(0, A.rows()), Eigen::seqN(0, A.cols())) = A;
        tape(Eigen::seqN(Eigen::lastp1 - B.rows(), B.rows()), Eigen::seqN(Eigen::lastp1 - B.cols(), B.cols())) = B;
        tape(Eigen::seqN(1, B.rows()), Eigen::seqN(2, B.cols())) += B;
        tape(Eigen::seqN(Eigen::lastp1 - B.rows(), B.rows()), Eigen::seqN(2, B.cols())) += B;
        tape(Eigen::seqN(5, A.rows()), Eigen::seqN(5, A.cols())) += A;
    }

    static Eigen::MatrixX<bool> expected_sparsity()
    {
        Eigen::MatrixX<bool> sparsity(10, 9);
        sparsity << 1, 1, 1, 0, 0, 0, 0, 0, 0,
                    1, 1, 1, 1, 1, 1, 1, 0, 0,
                    0, 0, 1, 1, 1, 1, 1, 0, 0,
                    0, 0, 1, 1, 1, 1, 1, 0, 0,
                    0, 0, 1, 1, 1, 1, 1, 0, 0,
                    0, 0, 0, 0, 0, 1, 1, 1, 0,
                    0, 0, 1, 1, 1, 1, 1, 1, 1,
                    0, 0, 1, 1, 1, 1, 1, 1, 1,
                    0, 0, 1, 1, 1, 1, 1, 1, 1,
                    0, 0, 1, 1, 1, 1, 1, 1, 1;
        return sparsity;
    }

    static std::vector<laopt::Segment> expected_copy_sequence()
    {
        return {{0,6},{25,4},{34,4},{43,4},{48,8},{5,4},{13,4},{21,4},{29,4},{38,4},{9,4},{17,4},{25,4},{34,4},{43,4},{33,2},{42,2},{47,2}};
    }

    static Eigen::MatrixX<scalar_t> expected_result()
    {
        Eigen::MatrixX<scalar_t> result(10, 9);
        result << 1, 1, 1, 0, 0, 0, 0, 0, 0,
                  1, 1, 3, 2, 2, 2, 2, 0, 0,
                  0, 0, 2, 2, 2, 2, 2, 0, 0,
                  0, 0, 2, 2, 2, 2, 2, 0, 0,
                  0, 0, 2, 2, 2, 2, 2, 0, 0,
                  0, 0, 0, 0, 0, 1, 1, 1, 0,
                  0, 0, 2, 2, 4, 5, 5, 3, 2,
                  0, 0, 2, 2, 4, 4, 4, 2, 2,
                  0, 0, 2, 2, 4, 4, 4, 2, 2,
                  0, 0, 2, 2, 4, 4, 4, 2, 2;
        return result;
    }
};

TEST(BSMatrix, Construction_Simple_Sum) {
    SumProblem problem;
    test_BSMatrix_problem(problem);
}

TEST(BSMatrixDense, Construction_Simple_Sum) {
    SumProblem problem;
    test_BSMatrixDense_problem(problem);
}


template<typename scalar_t, template<int, int> class Matrix>
void test_assignment_speeds()
{
    const std::array<int, 5> rows{6, 7, 0, 1, 2};
    const std::array<int, 5> cols{7, 5, 3, 1, 0};

    laopt::BSMatrixDenseDeployment<scalar_t> M;
    Matrix<10, 10> buffer(10, 10);
    M.set_buffer(buffer);
    M.resize(10, 10);

    Matrix<5, 5> target(5, 5);
    // Eigen::Matrix<scalar_t, 20, 20> target_buffer;
    // auto target = target_buffer(seqN(3,fix<5>), seqN(4,fix<5>));

    std::cout << "type(target) = " << laopt::type_name<decltype(target)>() << std::endl;

    Matrix<10, 10> source(10, 10);
    Eigen::Map<Eigen::Matrix<scalar_t, 5, 5>> map(buffer.data());
    Eigen::Map<Eigen::Matrix<scalar_t, 10, 10>> map_buffer(buffer.data());

    auto start = std::chrono::steady_clock::now();
    auto end = std::chrono::steady_clock::now();
    size_t NUM_EXP = 1000000;

    double acc;
    acc = 0;
    buffer.array() = 0;
    target.array() = 0;
    source.array() = 0;
    start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < NUM_EXP; ++i) {
        buffer(6, 7) = i + 1;
        target = source(rows, cols);
        acc += target(0, 0);
    }
    end = std::chrono::steady_clock::now();
    // std::cout << " acc = " << acc << std::endl;
    std::cout << std::setw(40) << "source(rows, cols): "
              << std::setw(10) << std::fixed << std::setprecision(2) << std::right
              << std::chrono::duration_cast<std::chrono::nanoseconds>((end - start)).count() / (double) NUM_EXP
              << " ns" << " (" << acc << ")" << std::endl;

    acc = 0;
    buffer.array() = 0;
    target.array() = 0;
    source.array() = 0;
    start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < NUM_EXP; ++i) {
        buffer(6, 7) = i + 1;
        target = buffer(rows, cols);
        acc += target(0, 0);
    }
    end = std::chrono::steady_clock::now();
    // std::cout << " acc = " << acc << std::endl;
    std::cout << std::setw(40) << "buffer(rows, cols): "
              << std::setw(10) << std::fixed << std::setprecision(2) << std::right
              << std::chrono::duration_cast<std::chrono::nanoseconds>((end - start)).count() / (double) NUM_EXP
              << " ns" << " (" << acc << ")" << std::endl;

    acc = 0;
    buffer.array() = 0;
    target.array() = 0;
    source.array() = 0;
    start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < NUM_EXP; ++i) {
        source(0, 0) = i + 1;
        target = map;
        acc += target(0, 0);
    }
    end = std::chrono::steady_clock::now();
    // std::cout << " acc = " << acc << std::endl;
    std::cout << std::setw(40) << "fixed map: "
              << std::setw(10) << std::fixed << std::setprecision(2) << std::right
              << std::chrono::duration_cast<std::chrono::nanoseconds>((end - start)).count() / (double) NUM_EXP
              << " ns" << " (" << acc << ")" << std::endl;

    acc = 0;
    buffer.array() = 0;
    target.array() = 0;
    source.array() = 0;
    start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < NUM_EXP; ++i) {
        map_buffer(6, 7) = i + 1;
        target = map_buffer(rows, cols);
        acc += target(0, 0);
    }
    end = std::chrono::steady_clock::now();
    // std::cout << " acc = " << acc << std::endl;
    std::cout << std::setw(40) << "fixed map(rows, cols): "
              << std::setw(10) << std::fixed << std::setprecision(2) << std::right
              << std::chrono::duration_cast<std::chrono::nanoseconds>((end - start)).count() / (double) NUM_EXP
              << " ns" << " (" << acc << ")" << std::endl;
}


template<int Rows, int Cols>
using MatrixDynamic = Eigen::MatrixX<double>;
template<int Rows, int Cols>
using MatrixStatic = Eigen::Matrix<double, Rows, Cols>;

template<int Rows, int Cols>
using Matrix = MatrixStatic<Rows, Cols>;

TEST(BSMatrixDense, AccessTime)
{
    using scalar_t = double;

    std::cout << "\n\n";
    std::cout << "Assignment to dynamic matrices" << std::endl;
    test_assignment_speeds<scalar_t, MatrixDynamic>();
    std::cout << "\n\n";

    std::cout << "Assignment to static matrices" << std::endl;
    test_assignment_speeds<scalar_t, MatrixStatic>();
    std::cout << "\n\n";
}

}