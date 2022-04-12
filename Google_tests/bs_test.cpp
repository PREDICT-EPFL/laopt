/**
 * Unit test for the construction and computation of LAMPC Block Sparse Matrices.
 */

#include <iostream>

#include "lampc.hpp"
#include <chrono>

#include "test_utils.hpp"
#include "gtest/gtest.h"


namespace {


/**
 * Form a block-diagonal matrix
 */
TEST(BSMatrix, Construction) {
    using scalar_t = double;
    lampc::BSMatrixTape<scalar_t> tape;

    Eigen::MatrixX<scalar_t> A(2,3);
    Eigen::MatrixX<scalar_t> B(4,5);

    // Create the matrix blkdiag(A,B)
    auto F = [&](auto& tape){
        tape(0,0) = A;
        tape(-1,-1) = B;
    };

    F(tape);
    tape.finalize_structure();
    F(tape);

    // Confirm the correct sparsity structure
    {
        auto S = triplet_to_sparse<int>(6,8,{{0,0,1},{1,0,1},{0,1,1},{1,1,1},{0,2,1},{1,2,1},{2,3,1},{3,3,1},{4,3,1},{5,3,1},{2,4,1},{3,4,1},{4,4,1},{5,4,1},{2,5,1},{3,5,1},{4,5,1},{5,5,1},{2,6,1},{3,6,1},{4,6,1},{5,6,1},{2,7,1},{3,7,1},{4,7,1},{5,7,1}});    
        EXPECT_TRUE(S.isApprox(tape.get_sparsity_structure(), 0));
    }

    // Confirm the correct copy process
    lampc::BSMatrix<scalar_t> mat(tape);
    SparseMatrix<scalar_t> S;
    mat.initialize_matrix(S);

    A << 1,2,3,4,5,6;
    B << 7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26;
    F(mat);

    {
        auto ground = triplet_to_sparse<double>(6,8,{{0,0,1},{1,0,4},{0,1,2},{1,1,5},{0,2,3},{1,2,6},{2,3,7},{3,3,12},{4,3,17},{5,3,22},{2,4,8},{3,4,13},{4,4,18},{5,4,23},{2,5,9},{3,5,14},{4,5,19},{5,5,24},{2,6,10},{3,6,15},{4,6,20},{5,6,25},{2,7,11},{3,7,16},{4,7,21},{5,7,0},{5,7,26}});
        EXPECT_TRUE(ground.isApprox(S, 0));
    }
}

/**
 * Partition two matrices and re-assemble into a block-sparse matrix
 */
TEST(BSMatrix, Construction_Complex) {
    using scalar_t = double;
    lampc::BSMatrixTape<scalar_t> tape;

    Eigen::MatrixX<scalar_t> A(2,3);
    Eigen::MatrixX<scalar_t> B(5,5);

    auto F = [&](auto& tape){
        tape(10,{{10,2},{5,1}}) = A;
        tape({{6,2},{0,3}},{{7,1},{5,1},{3,1},{1,1},{0,1}}) = B;
    };

    F(tape);
    tape.finalize_structure(20,20);
    F(tape);

    // Check sparsity structure
    {
        auto ground = triplet_to_sparse<int>(20,20,{{0,0,1},{1,0,1},{2,0,1},{6,0,1},{7,0,1},{0,1,1},{1,1,1},{2,1,1},{6,1,1},{7,1,1},{0,3,1},{1,3,1},{2,3,1},{6,3,1},{7,3,1},{0,5,1},{1,5,1},{2,5,1},{6,5,1},{7,5,1},{10,5,1},{11,5,1},{0,7,1},{1,7,1},{2,7,1},{6,7,1},{7,7,1},{10,10,1},{11,10,1},{10,11,1},{11,11,1}});
        EXPECT_TRUE(ground.isApprox(tape.get_sparsity_structure(), 0));
    }

    // Check copy
    lampc::BSMatrix<scalar_t> mat(tape);
    SparseMatrix<scalar_t> S;
    mat.initialize_matrix(S);

    A << 1,2,3,4,5,6;
    B << 7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31;
    F(mat);

    {
        auto ground = triplet_to_sparse<double>(20,20,{{0,0,21},{1,0,26},{2,0,31},{6,0,11},{7,0,16},{0,1,20},{1,1,25},{2,1,30},{6,1,10},{7,1,15},{0,3,19},{1,3,24},{2,3,29},{6,3,9},{7,3,14},{0,5,18},{1,5,23},{2,5,28},{6,5,8},{7,5,13},{10,5,3},{11,5,6},{0,7,17},{1,7,22},{2,7,27},{6,7,7},{7,7,12},{10,10,1},{11,10,4},{10,11,2},{11,11,5}});
        EXPECT_TRUE(ground.isApprox(S, 0));
    }    
}

TEST(BSJacobian, Construction_Simple) {
    using scalar_t = double;
    lampc::BSJacobianTape<scalar_t> tape;

    Eigen::VectorX<scalar_t> A(2);
    Eigen::VectorX<scalar_t> B(4);

    Eigen::MatrixX<scalar_t> JA(2,3);
    Eigen::MatrixX<scalar_t> JB(4,5);

    // Create the matrix blkdiag(A,B)
    auto F = [&](auto& tape){
        tape(0,0) = std::make_tuple(A, JA);
        tape(-1,-1) = std::make_tuple(B, JB);
    };

    F(tape);
    tape.finalize_structure();
    F(tape);

    // Check sparsity structure
    {
        auto ground = triplet_to_sparse<int>(6,8,{{0,0,1},{1,0,1},{0,1,1},{1,1,1},{0,2,1},{1,2,1},{2,3,1},{3,3,1},{4,3,1},{5,3,1},{2,4,1},{3,4,1},{4,4,1},{5,4,1},{2,5,1},{3,5,1},{4,5,1},{5,5,1},{2,6,1},{3,6,1},{4,6,1},{5,6,1},{2,7,1},{3,7,1},{4,7,1},{5,7,1}});    
        EXPECT_TRUE(ground.isApprox(tape.jacobian.get_sparsity_structure(), 0));
        EXPECT_TRUE(tape.value.get_sparsity_structure().isApprox(Eigen::Vector<int, 6>::Constant(1), 0));
    }

    // Check copy
    lampc::BSJacobian<scalar_t> mat(tape);
    Eigen::SparseMatrix<scalar_t> S;
    Eigen::VectorX<scalar_t> value(6);
    mat.initialize_jacobian(S);
    mat.set_target_value(value);

    A << 1,2;
    B << 3,4,5,6;

    JA << 1,2,3,4,5,6;
    JB << 7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26;
    F(mat);

    {
        auto ground = triplet_to_sparse<double>(6,8,{{0,0,1},{1,0,4},{0,1,2},{1,1,5},{0,2,3},{1,2,6},{2,3,7},{3,3,12},{4,3,17},{5,3,22},{2,4,8},{3,4,13},{4,4,18},{5,4,23},{2,5,9},{3,5,14},{4,5,19},{5,5,24},{2,6,10},{3,6,15},{4,6,20},{5,6,25},{2,7,11},{3,7,16},{4,7,21},{5,7,0},{5,7,26}});
        EXPECT_TRUE(ground.isApprox(S, 0));
    }

    // Check speed of copy for small problems
    const std::size_t NUM_EXP = 10000;
    auto start = std::chrono::steady_clock::now();
    for(int i = 0; i < NUM_EXP; ++i)
    {
        JB(0,0) = i;
        F(mat);
    }
    auto end = std::chrono::steady_clock::now();
    {
        JB(0,0) = 7;
        F(mat);
        auto ground = triplet_to_sparse<double>(6,8,{{0,0,1},{1,0,4},{0,1,2},{1,1,5},{0,2,3},{1,2,6},{2,3,7},{3,3,12},{4,3,17},{5,3,22},{2,4,8},{3,4,13},{4,4,18},{5,4,23},{2,5,9},{3,5,14},{4,5,19},{5,5,24},{2,6,10},{3,6,15},{4,6,20},{5,6,25},{2,7,11},{3,7,16},{4,7,21},{5,7,0},{5,7,26}});
        EXPECT_TRUE(ground.isApprox(S, 0));
    }
    // If this takes more than 2us, then we've done something to slow down the copy
    EXPECT_LT(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / (double)(NUM_EXP), 2);

    // std::cout << "Computation time : "
    //     << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / (double)(NUM_EXP)
    //     << " us" << std::endl;
}

}