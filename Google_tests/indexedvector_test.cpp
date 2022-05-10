/**
 * Unit test for the construction and computation of LAMPC variables.
 */

#include <iostream>
#include "lampc.hpp"

#include "indexedvector.hpp"

#include "test_utils.hpp"
#include "gtest/gtest.h"

template <class T, std::size_t N>
std::ostream& operator<<(std::ostream& o, const std::array<T, N>& arr)
{
    copy(arr.cbegin(), arr.cend(), std::ostream_iterator<T>(o, " "));
    return o;
}

namespace {

TEST(Problem, Variable) {
	using scalar_t = double;

	Eigen::Vector<scalar_t, 20> var;
	for(int i=0; i<20; i++) var[i] = i;

	lampc::IndexedVector<Eigen::Map<Eigen::Vector<scalar_t,8>>> map(var.data() + 3);
	map.set_offset(5);
	std::cout << "map         = " << map.transpose() << std::endl;
	std::cout << "map.indices = " << map.indices().transpose() << std::endl;

{
	auto tmp = map(std::array<int,3>({3,2,1}));
	std::cout << "map({3,2,1}) = " << tmp.transpose() << std::endl;
	std::cout << "map({3,2,1}).indices = " << tmp.indices().transpose() << std::endl;

}
{
	auto tmp = map(seq(2,3));
	std::cout << "map(seq(2,3)) = " << tmp.transpose() << std::endl;
	std::cout << "map(seq(2,3)).indices = " << tmp.indices().transpose() << std::endl;

}
{
	auto tmp = map({3,2,1});
	std::cout << "map({3,2,1}) = " << tmp.transpose() << std::endl;
	std::cout << "map({3,2,1}).indices = " << tmp.indices().transpose() << std::endl;

}
{
	auto tmp = map({3,2,1})({0,2});
	std::cout << "map({3,2,1})({0,2}) = " << tmp.transpose() << std::endl;
	std::cout << "map({3,2,1})({0,2}).indices = " << tmp.indices().transpose() << std::endl;

}

std::cout << "==========================" << std::endl;
{
	Eigen::Vector<scalar_t,3> x;
	x << 1,2,3;
	lampc::IndexedVector<Eigen::Vector<scalar_t,6>> vec;
	vec << x,2*x;
	vec.set_offset(12);

	std::cout << "x = " << x.transpose() << std::endl;
	std::cout << "vec = " << vec.transpose() << std::endl;
	std::cout << "vec.indices() = " << vec.indices().transpose() << std::endl;

	std::cout << "vec(seqN(1,4))(seqN(1,2)) = " << vec(seqN(1,4))(seqN(1,2)).transpose() << " (" << vec(seqN(1,4))(seqN(1,2)).indices().transpose() << ")" << std::endl;
}

// Test assignment
std::cout << "==========================" << std::endl;
{
	Eigen::Vector<scalar_t,10> y;
	lampc::IndexedVector<Eigen::Map<Eigen::Vector<scalar_t,10>>> x(y.data());
	x.set_offset(10);
	x << 0,1,2,3,4,5,6,7,8,9;

	x({9,8,4,2}).array() = -4;

	std::cout << "x = " << x.transpose() << " (" << x.indices().transpose() << ")" << std::endl;
}

// Test an initially NULL map
std::cout << "==========================" << std::endl;
{
	using map_t = lampc::IndexedVector<Eigen::Map<Eigen::Vector<scalar_t,3>>>;
	map_t vec(NULL);
	Eigen::Vector<scalar_t,3> x;
	x << 1,2,3;

	new (&vec) map_t(x.data());

	std::cout << "x = " << x.transpose() << std::endl;
	std::cout << "vec = " << vec.transpose() << std::endl;
	std::cout << "vec.indices() = " << vec.indices().transpose() << std::endl;
}

}

}