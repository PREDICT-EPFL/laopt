#ifndef __BSMATRIX_HPP
#define __BSMATRIX_HPP

#include "bsmatrixcore.hpp"
#include "bsdenseblock.hpp"
#include "bssparseblock.hpp"
#include "bszeroblock.hpp"

namespace BS {

// Helper to produce a row of all dense blocks
template<typename Scalar, int len, typename... vars>
struct DenseRow : BS::Row<Scalar, len, DenseBlock<Scalar, len, vars>...>
{};


// Helper to produce a row of all dense blocks
template<typename Scalar, int len, typename... vars>
struct SparseRow : BS::Row<Scalar, len, SparseBlock<Scalar, len, vars>...>
{};


};

#endif // __BSMATRIX_HPP