/**
 * Unit test for the construction and computation of LAMPC functions.
 */

#include "bsmatrix.hpp"
// #include "lampc_function.hpp"

#include <iomanip>
#include <Eigen/Eigenvalues> 
#include <type_traits>

// using namespace lampc;

// #include "gtest/gtest.h"

namespace {
using namespace BS;


// TEST(BSTest, FunctionMatrix) {

int main()
{
    const int blockRows = 5;
    std::array<int, blockRows> rowSizes{1,2,3,4,5};

    const int blockColumns = 4;
    std::array<int, blockColumns> colSizes{3,4,5,6};

    BSMatrix<double, blockRows, blockColumns, 20, 30> M(rowSizes, colSizes);

    // EXPECT_EQ(17, 17);
// }
    return 0;
}
}