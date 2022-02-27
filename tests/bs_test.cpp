/**
 * Unit test for the construction and computation of LAMPC functions.
 */

#include "la_compiler.hpp"

#include <iomanip>
#include <Eigen/Eigenvalues> 
// #include <type_traits>

// using namespace lampc;

#include "gtest/gtest.h"

TEST(BSTest, CopySimple) {
    // Test copying a sub-matrix into a larger one with the same sparsity structure

    using S = int;
    using T = Eigen::Triplet<S>;

    Eigen::SparseMatrix<S> target(10,8);
    {
      std::vector<T> trip = {T{0,0,1},{7,0,1},{9,0,1},{3,1,1},{5,1,1},{3,2,1},{4,2,1},{8,2,1},{9,2,1},{0,3,1},{1,3,1},{2,3,1},{3,3,1},{7,3,1},{8,3,1},{1,4,1},{3,4,1},{5,4,1},{6,4,1},{7,4,1},{8,4,1},{0,5,1},{1,5,1},{2,5,1},{3,5,1},{4,5,1},{7,5,1},{0,6,1},{1,6,1},{4,6,1},{6,6,1},{7,6,1},{0,7,1},{1,7,1},{2,7,1},{4,7,1},{5,7,1},{6,7,1},{7,7,1}};
      target.setFromTriplets(trip.begin(), trip.end());
    }
    Eigen::SparseMatrix<S> target_orig(target);

    Eigen::SparseMatrix<S> source(5,4);
    {
      std::vector<T> trip = {T{1,0,84},{2,0,81},{0,1,86},{1,1,14},{1,2,59},{3,2,45},{4,2,4},{0,3,28},{1,3,37},{2,3,50}};
      source.setFromTriplets(trip.begin(), trip.end());
    }

    std::vector<sparseblock_info<int>> blocks;
    int target_row = 2; // Place we want to put the sub-matrix
    int target_col = 2;
    for(int c=0; c<source.cols(); c++)
        build_copy_sequence(target, source, target_row, target_col+c, c, blocks);

    copy_submatrix<int>(target, source, &blocks[0], blocks.size());

    std::cout << "source = \n" << Eigen::MatrixX<S>(source) << std::endl;
    std::cout << "target = \n" << Eigen::MatrixX<S>(target) << std::endl;

    // Confirm that we got the result we wanted
    for(int r=0; r<target.rows(); r++)
    {
        for(int c=0; c<target.cols(); c++)
        {
            if(r < target_row || 
               r >= target_row + source.rows() ||
               c >= target_col + source.cols() ||
               c < target_col) 
                EXPECT_EQ(target_orig.coeff(r,c), target.coeff(r,c));
            else
                EXPECT_EQ(target.coeff(r,c), source.coeff(r - target_row, c - target_col));
        }
    }
}

TEST(BSTest, CopyComplex) {
    // Test copying a sub-matrix into a larger one with a different sparsity structure

    using S = int;
    using T = Eigen::Triplet<S>;

    Eigen::SparseMatrix<S> target(10,8);
    {
      std::vector<T> trip = {T{0,0,1},{7,0,1},{9,0,1},{3,1,1},{5,1,1},{3,2,1},{4,2,1},{8,2,1},{9,2,1},{0,3,1},{1,3,1},{2,3,1},{3,3,1},{7,3,1},{8,3,1},{1,4,1},{3,4,1},{5,4,1},{6,4,1},{7,4,1},{8,4,1},{0,5,1},{1,5,1},{2,5,1},{3,5,1},{4,5,1},{7,5,1},{0,6,1},{1,6,1},{4,6,1},{6,6,1},{7,6,1},{0,7,1},{1,7,1},{2,7,1},{4,7,1},{5,7,1},{6,7,1},{7,7,1}};
      target.setFromTriplets(trip.begin(), trip.end());
    }
    Eigen::SparseMatrix<S> target_orig(target);

    Eigen::SparseMatrix<S> source(4,5);
    {
      std::vector<T> trip = {T{0,0,41},{1,0,3},{2,0,29},{3,0,80},{0,2,35},{2,2,8},{0,3,51},{1,3,37},{0,4,74},{1,4,52},{2,4,80}};
      source.setFromTriplets(trip.begin(), trip.end());
    }

    std::vector<sparseblock_info<int>> blocks;
    int target_row = 0; // Place we want to put the sub-matrix
    int target_col = 3;
    for(int c=0; c<source.cols(); c++)
        build_copy_sequence(target, source, target_row, target_col+c, c, blocks);

    copy_submatrix<int>(target, source, &blocks[0], blocks.size());

    std::cout << "source = \n" << Eigen::MatrixX<S>(source) << std::endl;
    std::cout << "target = \n" << Eigen::MatrixX<S>(target) << std::endl;

    // Confirm that we got the result we wanted
    for(int r=0; r<target.rows(); r++)
    {
        for(int c=0; c<target.cols(); c++)
        {
            if(r < target_row || 
               r >= target_row + source.rows() ||
               c >= target_col + source.cols() ||
               c < target_col) 
                EXPECT_EQ(target_orig.coeff(r,c), target.coeff(r,c));
            else
            {
                if(source.coeff(r - target_row, c - target_col) == 0)
                    EXPECT_EQ(target_orig.coeff(r,c), target.coeff(r,c));
                else
                    EXPECT_EQ(target.coeff(r,c), source.coeff(r - target_row, c - target_col));
            }
        }
    }
}
