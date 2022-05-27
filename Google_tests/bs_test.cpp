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

    std::cout << "sparsity = \n" << sparsity.get_sparsity() << std::endl;

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
    BS.allocate_memory(S);

    // Run the copy
    BS.set_zero();
    F(BS);

    {
        auto ground = triplet_to_sparse<double>(6,8,{{0,0,1},{1,0,4},{0,1,2},{1,1,5},{0,2,3},{1,2,6},{2,3,7},{3,3,12},{4,3,17},{5,3,22},{2,4,8},{3,4,13},{4,4,18},{5,4,23},{2,5,9},{3,5,14},{4,5,19},{5,5,24},{2,6,10},{3,6,15},{4,6,20},{5,6,25},{2,7,11},{3,7,16},{4,7,21},{5,7,26}});
        EXPECT_TRUE(ground.isApprox(S, 0));
    }

    // std::cout << "resizeing..." << std::endl;
    // for(int j=0; j<1000; j++)
    // for(int i=0; i<100; i++)
    // {
    //     sparsity.resize(10,10);
    //     sparsity.resize(10+i,10+i);
    // }
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
        tape(seqN(10,A.rows()),concantenate_indices(Eigen::Vector<int,2>{10,11},Eigen::Vector<int,1>{5})) = A;
        tape(concantenate_indices(Eigen::Vector<int,2>{6,7}, Eigen::Vector<int,3>{0,1,2}),
             concantenate_indices(Eigen::Vector<int,1>{7},Eigen::Vector<int,1>{5},Eigen::Vector<int,1>{3},Eigen::Vector<int,1>{1},Eigen::Vector<int,1>{0})) = B;
    };

    auto BS = lampc::template makeBSMatrix<scalar_t>(F, 20,20);
    Eigen::SparseMatrix<scalar_t> S;
    BS.allocate_memory(S);

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
    BS.allocate_memory(S);

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


/**
 * Test the simple dense version.
 */
TEST(BSMatrixDense, Construction)
{
    auto F = [](auto& D)
    {
        // std::cout << "D.shape = " << D.rows() << " x " << D.cols() << std::endl;
        D.resize(2,3);
        // std::cout << "D.shape = " << D.rows() << " x " << D.cols() << std::endl;
        // std::cout << "D.buffer shape = " << D.buffer_rows() << " x " << D.buffer_cols() << std::endl;

        // std::cout << "D = \n" << D.value() << std::endl;
        D.value() << 1,2,3,4,5,6;
        // std::cout << "D = \n" << D.value() << std::endl;

        // std::cout << "D(1,all) = " << D(1,all) << std::endl;
        D.resize(2,2);
        // std::cout << "D.shape = " << D.rows() << " x " << D.cols() << std::endl;
        // std::cout << "D.buffer shape = " << D.buffer_rows() << " x " << D.buffer_cols() << std::endl;
        // std::cout << "D(1,all) = " << D(1,all) << std::endl;
    };

    using scalar_t = double;
    lampc::BSMatrixDenseConstruction<scalar_t> D;
    F(D);
    Eigen::MatrixX<scalar_t> ground(2,2);
    ground << 1,2,4,5;
    EXPECT_EQ(D.value(), ground);

    lampc::BSMatrixDenseDeployment<scalar_t> DD;
    ASSERT_DEATH(F(DD), "");
    
    Eigen::MatrixX<scalar_t> mat(10,10);
    DD.set_buffer(mat);
    F(DD);
    EXPECT_EQ(DD.value(), ground);


    Eigen::MatrixX<scalar_t> mat2(7,7);
    DD.set_buffer(mat2);
    F(DD);
    EXPECT_EQ(DD.value(), ground);

    Eigen::MatrixX<scalar_t> mat3(2,2);
    DD.set_buffer(mat3);
    ASSERT_DEATH(F(DD), "");
}

template<typename scalar_t, template<int,int> class Matrix>
void test_assignment_speeds()
{
    const std::array<int, 5> rows{6,7,0,1,2};
    const std::array<int, 5> cols{7,5,3,1,0};

    const lampc::Segment row1{.index=6, .length=2};
    const lampc::Segment row2{.index=0, .length=3};

    const lampc::Segment col1{.index=7, .length=1};
    const lampc::Segment col2{.index=5, .length=1};
    const lampc::Segment col3{.index=3, .length=1};
    const lampc::Segment col4{.index=1, .length=1};
    const lampc::Segment col5{.index=0, .length=1};

    lampc::BSMatrixDenseDeployment<scalar_t> M;
    Matrix<10,10> buffer(10,10);
    M.set_buffer(buffer);
    M.resize(10,10);

    Matrix<5,5> target(5,5);
    // Eigen::Matrix<scalar_t, 20, 20> target_buffer;
    // auto target = target_buffer(seqN(3,fix<5>), seqN(4,fix<5>));

    std::cout << "type(target) = " << type_name<decltype(target)>() << std::endl;

    Matrix<5,5> source(5,5);
    Eigen::Map<Eigen::Matrix<scalar_t,5,5>> map(buffer.data());
    Eigen::Map<Eigen::Matrix<scalar_t,10,10>> map_buffer(buffer.data());

    auto start = std::chrono::steady_clock::now();
    auto end = std::chrono::steady_clock::now();
    std::size_t NUM_EXP = 1000000;

    double acc;
    acc = 0; buffer.array() = 0; target.array() = 0; source.array() = 0;
    start = std::chrono::steady_clock::now();
    for(int i = 0; i < NUM_EXP; ++i)
    {
        buffer(6,7) = i+1;
        target = source(rows, cols);
        acc += target(0,0);
    }
    end = std::chrono::steady_clock::now();
    // std::cout << " acc = " << acc << std::endl;
    std::cout << std::setw (40) << "source(rows, cols): "
      << std::setw(10) << std::fixed << std::setprecision(2) << std::right << std::chrono::duration_cast<std::chrono::nanoseconds>((end - start)).count()/(double)NUM_EXP
      << " ns" << " (" << acc << ")" << std::endl;

    // acc = 0; buffer.array() = 0; target.array() = 0; source.array() = 0;
    // start = std::chrono::steady_clock::now();
    // for(int i = 0; i < NUM_EXP; ++i)
    // {
    //     buffer(6,7) = i+1;
    //     target = source(multiSeq(seqN(6,2),seqN(0,3)), multiSeq(seqN(7,1),seqN(5,1),seqN(3,1),seqN(1,1),seqN(0,1)));
    //     acc += target(0,0);
    // }
    // end = std::chrono::steady_clock::now();
    // // std::cout << " acc = " << acc << std::endl;
    // std::cout << std::setw (40) << "source(multiSeq, multiSeq): "
    //   << std::setw(10) << std::fixed << std::setprecision(2) << std::right << std::chrono::duration_cast<std::chrono::nanoseconds>((end - start)).count()/(double)NUM_EXP
    //   << " ns" << " (" << acc << ")" << std::endl;

    // acc = 0; buffer.array() = 0; target.array() = 0; source.array() = 0;
    // start = std::chrono::steady_clock::now();
    // for(int i = 0; i < NUM_EXP; ++i)
    // {
    //     buffer(6,7) = i+1;
    //     target = source(lampc::multiSeq_to_index<5>({row1, row2}), lampc::multiSeq_to_index<5>({col1, col2, col3, col4, col5}));
    //     acc += target(0,0);
    // }
    // end = std::chrono::steady_clock::now();
    // // std::cout << " acc = " << acc << std::endl;
    // std::cout << std::setw (40) << "source(multiSeq array, multiSeq array): "
    //   << std::setw(10) << std::fixed << std::setprecision(2) << std::right << std::chrono::duration_cast<std::chrono::nanoseconds>((end - start)).count()/(double)NUM_EXP
    //   << " ns" << " (" << acc << ")" << std::endl;

    // acc = 0; buffer.array() = 0; target.array() = 0; source.array() = 0;
    // start = std::chrono::steady_clock::now();
    // for(int i = 0; i < NUM_EXP; ++i)
    // {
    //     source(0,0) = i+1;
    //     target = source;
    //     acc += target(0,0);
    // }
    // end = std::chrono::steady_clock::now();
    // // std::cout << " acc = " << acc << std::endl;
    // std::cout << std::setw (40) << "source: "
    //   << std::setw(10) << std::fixed << std::setprecision(2) << std::right << std::chrono::duration_cast<std::chrono::nanoseconds>((end - start)).count()/(double)NUM_EXP
    //   << " ns" << " (" << acc << ")" << std::endl;

    acc = 0; buffer.array() = 0; target.array() = 0; source.array() = 0;
    start = std::chrono::steady_clock::now();
    for(int i = 0; i < NUM_EXP; ++i)
    {
        buffer(6,7) = i+1;
        target = buffer(rows, cols);
        acc += target(0,0);
    }
    end = std::chrono::steady_clock::now();
    // std::cout << " acc = " << acc << std::endl;
    std::cout << std::setw (40) << "buffer(rows, cols): "
      << std::setw(10) << std::fixed << std::setprecision(2) << std::right << std::chrono::duration_cast<std::chrono::nanoseconds>((end - start)).count()/(double)NUM_EXP
      << " ns" << " (" << acc << ")" << std::endl;

    acc = 0; buffer.array() = 0; target.array() = 0; source.array() = 0;
    start = std::chrono::steady_clock::now();
    for(int i = 0; i < NUM_EXP; ++i)
    {
        source(0,0) = i+1;
        target = map;
        acc += target(0,0);
    }
    end = std::chrono::steady_clock::now();
    // std::cout << " acc = " << acc << std::endl;
    std::cout << std::setw (40) << "fixed map: "
      << std::setw(10) << std::fixed << std::setprecision(2) << std::right << std::chrono::duration_cast<std::chrono::nanoseconds>((end - start)).count()/(double)NUM_EXP
      << " ns" << " (" << acc << ")" << std::endl;

    acc = 0; buffer.array() = 0; target.array() = 0; source.array() = 0;
    start = std::chrono::steady_clock::now();
    for(int i = 0; i < NUM_EXP; ++i)
    {
        map_buffer(6,7) = i+1;
        target = map_buffer(rows, cols);
        acc += target(0,0);
    }
    end = std::chrono::steady_clock::now();
    // std::cout << " acc = " << acc << std::endl;
    std::cout << std::setw (40) << "fixed map(rows, cols): "
      << std::setw(10) << std::fixed << std::setprecision(2) << std::right << std::chrono::duration_cast<std::chrono::nanoseconds>((end - start)).count()/(double)NUM_EXP
      << " ns" << " (" << acc << ")" << std::endl;
}


template<int Rows, int Cols>
using MatrixDynamic = Eigen::MatrixX<double>;
template<int Rows, int Cols>
using MatrixStatic = Eigen::Matrix<double,Rows,Cols>;

template<int Rows, int Cols>
using Matrix = MatrixStatic<Rows,Cols>;

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


// /**
//  * Test the weighted sum
//  */
// TEST(WeightedSum, Simple) {
//     WeightedSumTape<double> wsum_tape;

//     Eigen::VectorX<scalar_t> A(2);
//     Eigen::VectorX<scalar_t> B(4);

//     Eigen::MatrixX<scalar_t> JA(2,3);
//     Eigen::MatrixX<scalar_t> JB(4,5);

//     Eigen::MatrixX<scalar_t> HA(3,3);
//     Eigen::MatrixX<scalar_t> HB(5,5);

//     Index obj_index; 

//      // Create the matrix blkdiag(A,B)
//     auto F = [&](auto& wsum){
//         wsum += wsum.weight(rows) * std::make_tuple(A, JA, HA);
//         wsum += wsum.weight(-1) * std::make_tuple(B, JB, HB);
//     };


//     auto F = [&](auto& wsum){
//         wsum(0)  += wsum.weight(-1) * std::make_tuple(A, JA, HA);
//         wsum(-1) += wsum.weight(-1) * std::make_tuple(B, JB, HB);
//     };


//     F(wsum_tape);
//     wsum_tape.finalize_structure();
//     F(wsum_tape);
 
//   template<typename D, typename T>
//   void objective(T& lag)
//   {
//     // for(int i=0; i<x().cols()-1; i++)
//     //   obj({x[i],u[i]}) += obj.w(-1) * stage_cost(D(),x(i),u(i));
//     // obj({x[i],u[i],xs,us}) += obj.w(-1) * terminal_cost(D(),x(i),u(i),xs,us);

//     // for(int i=0; i<x().cols()-1; i++)
//     //   lagrangian({x[i],u[i]}) += lagrangian.w(-1) * stage_cost(D(),x(i),u(i));
//     // lagrangian({x[i],u[i],xs,us}) += lagrangian.w(-1) * terminal_cost(D(),x(i),u(i),xs,us);

//     // lag = obj + lag.w(-1) * equalities + lag.w(-1) * inequalities;
//     lag = obj + dual_eq * equalities + dual_ineq * inequalities;
//   };

//   template<typename D, typename T>
//   void constraints(T& g)
//   {
//     for(int i=0; i<x().cols()-1; i++)
//       g(-1,{x[i+1],x[i],u[i]}).con(i,"name") = this->dsys_equality(D(),x(i+1),x(i),u(i)); // Sets both the value and the jacobian
//   };


//     auto dual_eq = lag.w(-1);
//     auto dual_ineq = lag.w(-1);

// }


}