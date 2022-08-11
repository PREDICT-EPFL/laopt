/**
 * Unit test for the construction and computation of indexed.
 */
#include <iostream>
#include "indexed_vector.hpp"
#include "gtest/gtest.h"

TEST(IndexedVectorTest, Map) {
    Eigen::Vector<double, 20> var;
    for (int i = 0; i < 20; i++) var[i] = i;

    lampc::IndexedVector<Eigen::Map<Eigen::Vector<double, 8>>> map(var.data() + 3);
    map.set_offset(5);

    testing::internal::CaptureStdout();
    std::cout << "map = " << map.transpose();
    EXPECT_EQ(testing::internal::GetCapturedStdout(), "map =  3  4  5  6  7  8  9 10");
    testing::internal::CaptureStdout();
    std::cout << "map.indices() = " << map.indices().transpose();
    EXPECT_EQ(testing::internal::GetCapturedStdout(), "map.indices() =  5  6  7  8  9 10 11 12");

    testing::internal::CaptureStdout();
    std::cout << "map({3,2,1}) = " << map(std::array<int, 3>({3, 2, 1})).transpose();
    EXPECT_EQ(testing::internal::GetCapturedStdout(), "map({3,2,1}) = 6 5 4");
    testing::internal::CaptureStdout();
    std::cout << "map({3,2,1}).indices() = " << map(std::array<int, 3>({3, 2, 1})).indices().transpose();
    EXPECT_EQ(testing::internal::GetCapturedStdout(), "map({3,2,1}).indices() = 8 7 6");

    testing::internal::CaptureStdout();
    std::cout << "map(seq(2,3)) = " << map(Eigen::seq(2, 3)).transpose();
    EXPECT_EQ(testing::internal::GetCapturedStdout(), "map(seq(2,3)) = 5 6");
    testing::internal::CaptureStdout();
    std::cout << "map(seq(2,3)).indices() = " << map(Eigen::seq(2, 3)).indices().transpose();
    EXPECT_EQ(testing::internal::GetCapturedStdout(), "map(seq(2,3)).indices() = 7 8");

    testing::internal::CaptureStdout();
    std::cout << "map({3,2,1}) = " << map({3, 2, 1}).transpose();
    EXPECT_EQ(testing::internal::GetCapturedStdout(), "map({3,2,1}) = 6 5 4");
    testing::internal::CaptureStdout();
    std::cout << "map({3,2,1}).indices() = " << map({3, 2, 1}).indices().transpose();
    EXPECT_EQ(testing::internal::GetCapturedStdout(), "map({3,2,1}).indices() = 8 7 6");

    testing::internal::CaptureStdout();
    std::cout << "map({3,2,1})({0,2}) = " << map({3, 2, 1})({0, 2}).transpose();
    EXPECT_EQ(testing::internal::GetCapturedStdout(), "map({3,2,1})({0,2}) = 6 4");
    testing::internal::CaptureStdout();
    std::cout << "map({3,2,1})({0,2}).indices() = " << map({3, 2, 1})({0, 2}).indices().transpose();
    EXPECT_EQ(testing::internal::GetCapturedStdout(), "map({3,2,1})({0,2}).indices() = 8 6");
}

TEST(IndexedVectorTest, Vector) {

    Eigen::Vector<double, 3> x;
    x << 1, 2, 3;
    lampc::IndexedVector<Eigen::Vector<double, 6>> vec;
    vec << x, 2 * x;
    vec.set_offset(12);

    testing::internal::CaptureStdout();
    std::cout << "vec = " << vec.transpose();
    EXPECT_EQ(testing::internal::GetCapturedStdout(), "vec = 1 2 3 2 4 6");
    testing::internal::CaptureStdout();
    std::cout << "vec.indices() = " << vec.indices().transpose();
    EXPECT_EQ(testing::internal::GetCapturedStdout(), "vec.indices() = 12 13 14 15 16 17");

    testing::internal::CaptureStdout();
    std::cout << "vec(seqN(1,4))(seqN(1,2)) = " << vec(Eigen::seqN(1, 4))(Eigen::seqN(1, 2)).transpose();
    EXPECT_EQ(testing::internal::GetCapturedStdout(), "vec(seqN(1,4))(seqN(1,2)) = 3 2");
    testing::internal::CaptureStdout();
    std::cout << "vec(seqN(1,4))(seqN(1,2)).indices() = " << vec(Eigen::seqN(1, 4))(Eigen::seqN(1, 2)).indices().transpose();
    EXPECT_EQ(testing::internal::GetCapturedStdout(), "vec(seqN(1,4))(seqN(1,2)).indices() = 14 15");
}

TEST(IndexedVectorTest, Assignment) {

    Eigen::Vector<double, 10> y;
    lampc::IndexedVector<Eigen::Map<Eigen::Vector<double, 10>>> x(y.data());
    x.set_offset(10);
    x << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9;

    x({9, 8, 4, 2}).array() = -4;

    testing::internal::CaptureStdout();
    std::cout << "x = " << x.transpose();
    EXPECT_EQ(testing::internal::GetCapturedStdout(), "x =  0  1 -4  3 -4  5  6  7 -4 -4");
    testing::internal::CaptureStdout();
    std::cout << "x.indices() = " << x.indices().transpose();
    EXPECT_EQ(testing::internal::GetCapturedStdout(), "x.indices() = 10 11 12 13 14 15 16 17 18 19");
}

TEST(IndexedVectorTest, NullMap) {

    using map_t = lampc::IndexedVector<Eigen::Map<Eigen::Vector<double, 3>>>;
    map_t vec(nullptr);
    Eigen::Vector<double, 3> x;
    x << 1, 2, 3;

    new(&vec) map_t(x.data());

    testing::internal::CaptureStdout();
    std::cout << "vec = " << vec.transpose();
    EXPECT_EQ(testing::internal::GetCapturedStdout(), "vec = 1 2 3");
    testing::internal::CaptureStdout();
    std::cout << "vec.indices() = " << vec.indices().transpose();
    EXPECT_EQ(testing::internal::GetCapturedStdout(), "vec.indices() = -1 -1 -1");
}
