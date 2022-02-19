#ifndef __LAMPC__HPP
#define __LAMPC__HPP

#include <iostream>
#include <tuple>
#include <array>

#include "map.hpp"

#include "Eigen/Dense"
#include <Eigen/Sparse>

#include "utils/helpers.hpp"

// #include "lampc_utility.hpp"
#include "lampc_function.hpp"
// #include "lampc_impl.hpp"
#include "lampc_functionset.hpp"

/**
 * Information about location of a variable in a vector
 */
struct variable_info_t
{
	int offset;
	int size;
	std::string name;
};

#endif // __LAMPC__HPP