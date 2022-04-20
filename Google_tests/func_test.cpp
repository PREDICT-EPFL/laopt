/**
 * Unit test for the construction and computation of LAMPC functions.
 */

#include <iostream>

#include "lampc.hpp"

#include "gtest/gtest.h"

namespace {


template<typename Info>
struct _TestFunction : public lampc::Hessian<_TestFunction, Info>
{
    typename Info::param_t& param;
    _TestFunction(typename Info::param_t& param) : param(param) {}

    template<typename diff_t>
    EIGEN_STRONG_INLINE Eigen::Vector<diff_t, 2> 
    impl( const Eigen::Ref< const Eigen::Vector<diff_t, 2> >& x, 
          const Eigen::Ref< const Eigen::Vector<diff_t, 1> >& u) noexcept        
    {
        return x(0) * (param.A.template cast<diff_t>() * x + param.B.template cast<diff_t>() * u);
    }
};
template<typename scalar_t, typename param_t>
using TestFunction = _TestFunction<lampc::FuncInfo<scalar_t,param_t, 2, 2,1>>;

template<typename scalar_t>
struct param_t {
  Eigen::Matrix<scalar_t, 2, 2> A{{0,1},{0,0}};
  Eigen::Matrix<scalar_t, 2, 1> B{{0},{1}};
};


/**
 * Compute the jacobian and hessian of a function
 */
TEST(FunctionTest, Construction) {

    using scalar_t = double;
    using param_t = param_t<scalar_t>;

    param_t param;
    TestFunction<scalar_t,param_t> test(param);

    Eigen::Vector<scalar_t, 2> xp;
    Eigen::Vector<scalar_t, 2> x;
    Eigen::Vector<scalar_t, 1> u;
    x << 1,2;
    u << 3;
    xp = test.eval(x, u);   
    std::cout << "xp = " << xp.transpose() << std::endl;

    param.A(0,0) = 7;
    test.eval_jacobian(x,u);
    std::cout << "value = " << test.value.transpose() << std::endl;
    std::cout << "jacobian = \n" << test.jacobian << std::endl;

    param.A(0,0) = 70;
    test.eval_hessian(x,u);
    std::cout << "value = " << test.value.transpose() << std::endl;
    std::cout << "jacobian = \n" << test.jacobian << std::endl;
    for(int i=0; i<2; i++)
        std::cout << "hessian[" << i << "] = \n" << test.hessian[i] << std::endl;


}

}