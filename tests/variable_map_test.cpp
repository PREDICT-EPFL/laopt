/**
 * Unit test for the construction and computation of indexed.
 */
#include <iostream>
#include "laopt/variable_map.hpp"
#include "gtest/gtest.h"

TEST(VariableMapTest, Map) {
    Eigen::Vector<double, 20> var;
    for (int i = 0; i < 20; i++) var[i] = i;

    laopt::Variable<double, 8> map(var.data() + 3);
    map.index_offset() = 5;

    testing::internal::CaptureStdout();
    std::cout << "map = " << map.transpose();
    EXPECT_EQ(testing::internal::GetCapturedStdout(), "map =  3  4  5  6  7  8  9 10");
    testing::internal::CaptureStdout();
    std::cout << "map.indices() = " << variable_indices(map).transpose();
    EXPECT_EQ(testing::internal::GetCapturedStdout(), "map.indices() =  5  6  7  8  9 10 11 12");

    testing::internal::CaptureStdout();
    std::cout << "map(2) = " << map(Eigen::seqN(2, Eigen::fix<1>)).transpose();
    EXPECT_EQ(testing::internal::GetCapturedStdout(), "map(2) = 5");
    testing::internal::CaptureStdout();
    std::cout << "map(2).indices() = " << variable_indices(map(Eigen::seqN(2, Eigen::fix<1>))).transpose();
    EXPECT_EQ(testing::internal::GetCapturedStdout(), "map(2).indices() = 7");

    testing::internal::CaptureStdout();
    std::cout << "map({3,2,1}) = " << map(std::array<int, 3>({3, 2, 1})).transpose();
    EXPECT_EQ(testing::internal::GetCapturedStdout(), "map({3,2,1}) = 6 5 4");
    testing::internal::CaptureStdout();
    std::cout << "map({3,2,1}).indices() = " << variable_indices(map(std::array<int, 3>({3, 2, 1}))).transpose();
    EXPECT_EQ(testing::internal::GetCapturedStdout(), "map({3,2,1}).indices() = 8 7 6");

    auto block = map(Eigen::seq(Eigen::fix<2>, Eigen::fix<6>));
    auto sub_block = block(Eigen::seq(Eigen::fix<2>, Eigen::fix<3>));
    testing::internal::CaptureStdout();
    std::cout << "map(seq(2,3)) = " << sub_block.transpose();
    EXPECT_EQ(testing::internal::GetCapturedStdout(), "map(seq(2,3)) = 7 8");
    testing::internal::CaptureStdout();
    std::cout << "map(seq(2,3)).indices() = " << variable_indices(sub_block).transpose();
    EXPECT_EQ(testing::internal::GetCapturedStdout(), "map(seq(2,3)).indices() =  9 10");

    testing::internal::CaptureStdout();
    std::cout << "map({3,2,1}) = " << map({3, 2, 1}).transpose();
    EXPECT_EQ(testing::internal::GetCapturedStdout(), "map({3,2,1}) = 6 5 4");
    testing::internal::CaptureStdout();
    std::cout << "map({3,2,1}).indices() = " << variable_indices(map({3, 2, 1})).transpose();
    EXPECT_EQ(testing::internal::GetCapturedStdout(), "map({3,2,1}).indices() = 8 7 6");

    testing::internal::CaptureStdout();
    std::cout << "map({3,2,1})({0,2}) = " << map({3, 2, 1})({0, 2}).transpose();
    EXPECT_EQ(testing::internal::GetCapturedStdout(), "map({3,2,1})({0,2}) = 6 4");
    testing::internal::CaptureStdout();
    std::cout << "map({3,2,1})({0,2}).indices() = " << variable_indices(map({3, 2, 1})({0, 2})).transpose();
    EXPECT_EQ(testing::internal::GetCapturedStdout(), "map({3,2,1})({0,2}).indices() = 8 6");
}

template<typename Derived>
decltype(auto) cast_to_matrix_base(const Eigen::MatrixBase<Derived>& mat)
{
    return mat;
}

TEST(VariableMapTest, VariableInfo) {
    laopt::Variable<double, 11> var;
    static_assert(laopt::variable_info<Eigen::Vector<double, 11>>::size == 0, "");
    static_assert(laopt::variable_info<decltype(var)>::size == 11, "");
    static_assert(laopt::variable_info<decltype(cast_to_matrix_base(var))>::size == 11, "");
}
