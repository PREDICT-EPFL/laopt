#ifndef __QP_FUNCTIONS_HPP
#define __QP_FUNCTIONS_HPP

#include "make_yaml_info.hpp"

template<typename scalar_t_>
struct MyFunctions
{
    using scalar_t = scalar_t_;

    struct param_t_
    {
        int blank;
    };
    using param_t = param_t_;

    FUNCTION(ineq, scalar_t, param_t, (val, 1), (x1, 1), (x2, 1), (x3, 1), (x4, 1))
    {
        val = x1 * x2 * x3 * x4;
    }

    FUNCTION(eq, scalar_t, param_t, (val, 1), (x1, 1), (x2, 1), (x3, 1), (x4, 1))
    {
        val[0] = x1[0]*x1[0] + x2[0]*x2[0] + x3[0]*x3[0] + x4[0]*x4[0];
    }

    FUNCTION(obj, scalar_t, param_t, (val, 1), (x1, 1), (x2, 1), (x3, 1), (x4, 1))
    {
        val = x1*x4*(x1+x2+x3)+x3;
    }

    static std::string saveFunctionInfo()
    {
        return saveInfoYAML<scalar_t, ineq, eq, obj>();
    }
};

#endif