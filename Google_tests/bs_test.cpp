/**
 * Unit test for the construction and computation of LAMPC Block Sparse Matrices.
 */

#include <iostream>
#include <chrono>

#include "bsmatrix.hpp"
#include "lampc_utility.hpp"

#include "test_utils.hpp"
#include "gtest/gtest.h"


namespace {

/**
 * Form a block-diagonal matrix
 */
TEST(BSMatrix, Construction) {
    using scalar_t = double;
    Eigen::MatrixX<scalar_t> A(2,3);
    Eigen::MatrixX<scalar_t> B(4,5);

    A << 1,2,3,4,5,6;
    B << 7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26;

    // Create the matrix blkdiag(A,B)
    auto F = [&](auto& tape){
        tape(seqN(0,A.rows()),seqN(0,A.cols())) = A;
        tape(seqN(lastp1-B.rows(),B.rows()),seqN(lastp1-B.cols(),B.cols())) = B;
    };

    // Get the sparsity structure
    lampc::BSMatrixSparsity sparsity(6,8);
    F(sparsity);

    // Confirm the correct sparsity structure
    {
        Eigen::MatrixX<bool> ground(6,8);
        ground << 1,1,1,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,1,1,1,1,1,0,0,0,1,1,1,1,1,0,0,0,1,1,1,1,1;
        EXPECT_EQ(ground, sparsity.get_sparsity());
    }

    // Get the copy sequence
    lampc::BSMatrixTape tape(sparsity.get_sparsity(), sparsity.rows(), sparsity.cols());
    F(tape);

    // Confirm the correct copy sequence
    {
        std::vector<lampc::Segment> ground = {lampc::Segment{0,6}, lampc::Segment{6,20}};
        EXPECT_TRUE(std::equal(ground.begin(), ground.end(), tape.copy_segments.begin()));
    }

    // Build the BSMatrix
    auto BS = tape.template makeBSMatrix<scalar_t>();
    Eigen::SparseMatrix<scalar_t> S;
    BS.initialize_matrix(S);

    // Run the copy
    BS.set_zero();
    F(BS);

    {
        auto ground = triplet_to_sparse<double>(6,8,{{0,0,1},{1,0,4},{0,1,2},{1,1,5},{0,2,3},{1,2,6},{2,3,7},{3,3,12},{4,3,17},{5,3,22},{2,4,8},{3,4,13},{4,4,18},{5,4,23},{2,5,9},{3,5,14},{4,5,19},{5,5,24},{2,6,10},{3,6,15},{4,6,20},{5,6,25},{2,7,11},{3,7,16},{4,7,21},{5,7,26}});
        EXPECT_TRUE(ground.isApprox(S, 0));
    }
}

/**
 * Partition two matrices and re-assemble into a block-sparse matrix
 */
TEST(BSMatrix, Construction_Complex) {
    using scalar_t = double;

    Eigen::MatrixX<scalar_t> A(2,3);
    Eigen::MatrixX<scalar_t> B(5,5);
    A << 1,2,3,4,5,6;
    B << 7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31;

    // Partitioning and assembly function
    auto F = [&](auto& tape){
        tape(seqN(10,A.rows()),multiSeq(seqN(10,2),seqN(5,1))) = A;
        tape(multiSeq(seqN(6,2),seqN(0,3)), multiSeq(seqN(7,1),seqN(5,1),seqN(3,1),seqN(1,1),seqN(0,1))) = B;
    };

    auto BS = lampc::template makeBSMatrix<scalar_t>(F, 20,20);
    Eigen::SparseMatrix<scalar_t> S;
    BS.initialize_matrix(S);

    // Check sparsity structure
    {
        auto ground = triplet_to_sparse<double>(20,20,{{0,0,1},{1,0,1},{2,0,1},{6,0,1},{7,0,1},{0,1,1},{1,1,1},{2,1,1},{6,1,1},{7,1,1},{0,3,1},{1,3,1},{2,3,1},{6,3,1},{7,3,1},{0,5,1},{1,5,1},{2,5,1},{6,5,1},{7,5,1},{10,5,1},{11,5,1},{0,7,1},{1,7,1},{2,7,1},{6,7,1},{7,7,1},{10,10,1},{11,10,1},{10,11,1},{11,11,1}});
        EXPECT_TRUE(ground.isApprox(S, 0));
    }

    // Check copy
    BS.set_zero();
    F(BS);

    {
        auto ground = triplet_to_sparse<double>(20,20,{{0,0,21},{1,0,26},{2,0,31},{6,0,11},{7,0,16},{0,1,20},{1,1,25},{2,1,30},{6,1,10},{7,1,15},{0,3,19},{1,3,24},{2,3,29},{6,3,9},{7,3,14},{0,5,18},{1,5,23},{2,5,28},{6,5,8},{7,5,13},{10,5,3},{11,5,6},{0,7,17},{1,7,22},{2,7,27},{6,7,7},{7,7,12},{10,10,1},{11,10,4},{10,11,2},{11,11,5}});
        EXPECT_TRUE(ground.isApprox(S, 0));
    }    
}


/**
 * Form a block-diagonal matrix and then add to it
 */
TEST(BSMatrix, SimpleSum) {
    using scalar_t = double;

    Eigen::MatrixX<scalar_t> A(2,3);
    Eigen::MatrixX<scalar_t> B(4,5);
    A.setConstant(1);
    B.setConstant(2);

    // Create the matrix blkdiag(A,B)
    auto F = [&](auto& mat){
        mat(seqN(0,A.rows()),seqN(0,A.cols())) = A;
        mat(seqN(lastp1-B.rows(),B.rows()),seqN(lastp1-B.cols(),B.cols())) = B;
        mat(seqN(1,B.rows()), seqN(2,B.cols())) += B;
        mat(seqN(lastp1-B.rows(),B.rows()), seqN(2,B.cols())) += B;
        mat(seqN(5,A.rows()), seqN(5,A.cols())) += A;
    };

    auto BS = lampc::template makeBSMatrix<scalar_t>(F, 10,9);
    Eigen::SparseMatrix<scalar_t> S;
    BS.initialize_matrix(S);

    // Confirm the correct sparsity structure
    {
        auto ground = triplet_to_sparse<bool>(10,9,{{0,0,1},{1,0,1},{0,1,1},{1,1,1},{0,2,1},{1,2,1},{2,2,1},{3,2,1},{4,2,1},{6,2,1},{7,2,1},{8,2,1},{9,2,1},{1,3,1},{2,3,1},{3,3,1},{4,3,1},{6,3,1},{7,3,1},{8,3,1},{9,3,1},{1,4,1},{2,4,1},{3,4,1},{4,4,1},{6,4,1},{7,4,1},{8,4,1},{9,4,1},{1,5,1},{2,5,1},{3,5,1},{4,5,1},{5,5,1},{6,5,1},{7,5,1},{8,5,1},{9,5,1},{1,6,1},{2,6,1},{3,6,1},{4,6,1},{5,6,1},{6,6,1},{7,6,1},{8,6,1},{9,6,1},{5,7,1},{6,7,1},{7,7,1},{8,7,1},{9,7,1},{6,8,1},{7,8,1},{8,8,1},{9,8,1}});
        EXPECT_TRUE(ground.isApprox(BS.get_sparsity_structure(), 0));
    }

    // Confirm the correct copy process
    BS.set_zero();
    F(BS);

    {
        auto ground = triplet_to_sparse<double>(10,9,{{0,0,1},{1,0,1},{0,1,1},{1,1,1},{0,2,1},{1,2,3},{2,2,2},{3,2,2},{4,2,2},{6,2,2},{7,2,2},{8,2,2},{9,2,2},{1,3,2},{2,3,2},{3,3,2},{4,3,2},{6,3,2},{7,3,2},{8,3,2},{9,3,2},{1,4,2},{2,4,2},{3,4,2},{4,4,2},{6,4,4},{7,4,4},{8,4,4},{9,4,4},{1,5,2},{2,5,2},{3,5,2},{4,5,2},{5,5,1},{6,5,5},{7,5,4},{8,5,4},{9,5,4},{1,6,2},{2,6,2},{3,6,2},{4,6,2},{5,6,1},{6,6,5},{7,6,4},{8,6,4},{9,6,4},{5,7,1},{6,7,3},{7,7,2},{8,7,2},{9,7,2},{6,8,2},{7,8,2},{8,8,2},{9,8,2}});
        EXPECT_TRUE(ground.isApprox(S, 0));
    }
}

// // /**
// //  * Test the weighted sum
// //  */
// // TEST(WeightedSum, Simple) {
// //     WeightedSumTape<double> wsum_tape;

// //     Eigen::VectorX<scalar_t> A(2);
// //     Eigen::VectorX<scalar_t> B(4);

// //     Eigen::MatrixX<scalar_t> JA(2,3);
// //     Eigen::MatrixX<scalar_t> JB(4,5);

// //     Eigen::MatrixX<scalar_t> HA(3,3);
// //     Eigen::MatrixX<scalar_t> HB(5,5);

// //     Index obj_index; 

// //      // Create the matrix blkdiag(A,B)
// //     auto F = [&](auto& wsum){
// //         wsum += wsum.weight(rows) * std::make_tuple(A, JA, HA);
// //         wsum += wsum.weight(-1) * std::make_tuple(B, JB, HB);
// //     };


// //     auto F = [&](auto& wsum){
// //         wsum(0)  += wsum.weight(-1) * std::make_tuple(A, JA, HA);
// //         wsum(-1) += wsum.weight(-1) * std::make_tuple(B, JB, HB);
// //     };


// //     F(wsum_tape);
// //     wsum_tape.finalize_structure();
// //     F(wsum_tape);
 
// //   template<typename D, typename T>
// //   void objective(T& lag)
// //   {
// //     // for(int i=0; i<x().cols()-1; i++)
// //     //   obj({x[i],u[i]}) += obj.w(-1) * stage_cost(D(),x(i),u(i));
// //     // obj({x[i],u[i],xs,us}) += obj.w(-1) * terminal_cost(D(),x(i),u(i),xs,us);

// //     // for(int i=0; i<x().cols()-1; i++)
// //     //   lagrangian({x[i],u[i]}) += lagrangian.w(-1) * stage_cost(D(),x(i),u(i));
// //     // lagrangian({x[i],u[i],xs,us}) += lagrangian.w(-1) * terminal_cost(D(),x(i),u(i),xs,us);

// //     // lag = obj + lag.w(-1) * equalities + lag.w(-1) * inequalities;
// //     lag = obj + dual_eq * equalities + dual_ineq * inequalities;
// //   };

// //   template<typename D, typename T>
// //   void constraints(T& g)
// //   {
// //     for(int i=0; i<x().cols()-1; i++)
// //       g(-1,{x[i+1],x[i],u[i]}).con(i,"name") = this->dsys_equality(D(),x(i+1),x(i),u(i)); // Sets both the value and the jacobian
// //   };


// //     auto dual_eq = lag.w(-1);
// //     auto dual_ineq = lag.w(-1);



// // }


}