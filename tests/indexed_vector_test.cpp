/**
 * Unit test for the construction and computation of indexed.
 */
#include <iostream>
#include "laopt/indexed_vector.hpp"
#include "laopt/variable.hpp"
#include "gtest/gtest.h"

#include "laopt/utility.hpp"

/**
 * Given an array of N Eigen vectors, return a vector from first of size number
 * 
 * stack_variables([x1,x2,x3,x4,5], 1,2) = [x2,x3]
 */
template<int number_elements, typename array_t>
auto stack_variables(array_t& vars, int first)
{
    // Eigen type of the elements of the array
    using element_t = typename array_t::value_type; //typename std::decay<decltype(std::declval<T>()[0])>::type;

    // Get the element at the given index in the concantenation
    auto get = [vars=std::move(vars),first](int row, int col)
    {
        auto ind = std::div(row, element_t::RowsAtCompileTime);
        return vars[ind.quot + first].cast_base()(ind.rem, 0);
    };

    constexpr int NumRows = element_t::RowsAtCompileTime * number_elements;
    auto stack_vector = Eigen::Vector<decltype(get(0,0)), NumRows>::NullaryExpr(get);
    laopt::IndexedVector<decltype(stack_vector)> stack(NumRows, 1, get);

    // Set the indices
    for(int i=0; i<number_elements; i++)
    {
        constexpr int N = element_t::RowsAtCompileTime;
        stack.indices()(Eigen::seqN(i*N,N)) = vars[i+first].indices();
    }

    return stack;
};

TEST(VariableTest, StackTest) {
    constexpr int N = 10;
    constexpr int n = 2;

    // Create our array of variables
    std::array<laopt::Variable<double, n>, N> var_array;

    // Register the variables to the master_var
    Eigen::Vector<double, N*n> master_var;
    for (int i = 0; i < N*n; i++) master_var[i] = i;
    for(int i=0; i<N; i++) var_array[i].register_variable(i*n)(master_var.data());

    // Create a stack of 4 variables starting at element 3
    auto stack = stack_variables<4>(var_array, 3);
    std::cout << "stack = " << stack.transpose() << std::endl;
    std::cout << "stack.indices = " << stack.indices().transpose() << std::endl;    
};




// Given a Variable of length N*n, return an array
// of length N of Variables of length n partitioning
// X.

template<int n, int N, typename T, int... ints>
auto make_variable_array_helper(T& X, std::integer_sequence<int, ints...>)
{
    using element_t = decltype(X(Eigen::seqN(0 * n,n)));
    return std::array<element_t, N> {X(Eigen::seqN(ints * n,n))...};
};

template<int n, int N, typename T>
auto make_variable_array(T& X)
{
    return make_variable_array_helper<n,N>(X, std::make_integer_sequence<int, N>{});
};


// Option 2
// Define a large X-variable and then decompose it
TEST(VariableTest, VarTest) {
    constexpr int N = 10;
    constexpr int n = 2;

    // Create our state variable
    laopt::Variable<double, N*n> X;

    // Register X to the master_var
    Eigen::Vector<double, N*n> master_var;
    for (int i = 0; i < N*n; i++) master_var[i] = i;
    X.register_variable(0)(master_var.data());

    std::cout << "X = " << X.transpose() << std::endl;

    auto x = make_variable_array<n, N>(X);

    for(int i=0; i<N; i++)
        std::cout << "x[i] = " << x[i].transpose() << "  indices = " << x[i].indices().transpose() << std::endl;

    // Confirm that these things share memory
    std::cout << "Changing the master variable" << std::endl;
    X(Eigen::seqN(4,1)) << 100;
    std::cout << "X = " << X.transpose() << std::endl;
    for(int i=0; i<N; i++)
        std::cout << "x[i] = " << x[i].transpose() << "  indices = " << x[i].indices().transpose() << std::endl;

    std::cout << "Changing the array of variables" << std::endl;
    x[4] << 100, 200;
    std::cout << "X = " << X.transpose() << std::endl;
    for(int i=0; i<N; i++)
        std::cout << "x[i] = " << x[i].transpose() << "  indices = " << x[i].indices().transpose() << std::endl;

};



TEST(IndexedVectorTest, Map) {
    Eigen::Vector<double, 20> var;
    for (int i = 0; i < 20; i++) var[i] = i;

    laopt::IndexedVector<Eigen::Map<Eigen::Vector<double, 8>>> map(var.data() + 3);
    map.set_offset(5);

    testing::internal::CaptureStdout();
    std::cout << "map = " << map.transpose();
    EXPECT_EQ(testing::internal::GetCapturedStdout(), "map =  3  4  5  6  7  8  9 10");
    testing::internal::CaptureStdout();
    std::cout << "map.indices() = " << map.indices().transpose();
    EXPECT_EQ(testing::internal::GetCapturedStdout(), "map.indices() =  5  6  7  8  9 10 11 12");

    // auto tmp = map(std::array<int, 3>({3, 2, 1}));
    // std::cout << "tmp = " << tmp.transpose() << std::endl;
    // for (int i = 0; i < 20; i++) var[i] = 2*i;
    // std::cout << "tmp = " << tmp.transpose() << std::endl;
    // std::cout << laopt::type_name<decltype(tmp)>() << std::endl;

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
    laopt::IndexedVector<Eigen::Vector<double, 6>> vec;
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
    laopt::IndexedVector<Eigen::Map<Eigen::Vector<double, 10>>> x(y.data());
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

    using map_t = laopt::IndexedVector<Eigen::Map<Eigen::Vector<double, 3>>>;
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
