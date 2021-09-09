#ifndef __BSMATRIX_HPP
#define __BSMATRIX_HPP

#include "bsmatrixcore.hpp"

template<int size_, typename... Variables>
struct DenseConstraint : Constraint<size_, Block<Variables>...> {};


#endif 