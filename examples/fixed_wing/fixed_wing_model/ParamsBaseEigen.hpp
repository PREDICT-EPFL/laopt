#ifndef SRC_PARAMSBASEEIGEN_HPP
#define SRC_PARAMSBASEEIGEN_HPP

#include <iostream>
#include <vector>

#include "flight_model_utils.hpp"

namespace flight_model {
namespace eigen_model {

template<typename param_t = double>
struct ParamsBase
{
public:
    using Param = param_t;
    void load_params_from_yaml(const std::string &yaml_filepath)
    {
        std::cout << "load_params_from_yaml(): Implement me!" << "\n";
    }
};

} //namespace eigen_model
} //namespace flight_model

#endif //SRC_PARAMSBASEEIGEN_HPP
