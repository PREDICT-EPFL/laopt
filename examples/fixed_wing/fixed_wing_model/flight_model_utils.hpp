#ifndef SRC_FLIGHT_MODEL_UTILS_HPP
#define SRC_FLIGHT_MODEL_UTILS_HPP

#include <iostream>
#include <cassert>
#include <cmath>

#include <yaml-cpp/yaml.h>

namespace flight_model {
enum StateRepresentation
{
    Undefined,
    AttQuat, AttEuler, LongitudinalEulerAoa, LongitudinalUW, LongitudinalFlightPath
};

double get_value(const YAML::Node &node, const std::string &name)
{
    bool property_found{false};
    double property_value = 0.0;
    for (YAML::const_iterator it = node.begin(); it != node.end(); ++it)
    {
        std::string key = it->first.as<std::string>();
        switch (it->second.Type())
        {
            case YAML::NodeType::Scalar :
                if (key == name)
                {
                    property_value = it->second.as<double>();
                    return property_value;
                }
                break;
            case YAML::NodeType::Map :
                /** iterate over the map */
                for (auto iter = it->second.begin(); iter != it->second.end(); ++iter)
                {
                    if ((iter->first).as<std::string>() == name)
                    {
                        property_value = iter->second.as<double>();
                        return property_value;
                    }
                }
                break;
            case YAML::NodeType::Null :
                std::cout << "real lox \n";
                break;
            default :
                std::cout << "lox \n";
                break;
        }
    }
    std::cout << "Error: YAML does not contain " << name << "\n";
    assert(property_found);
    return property_value;
}

inline double deg2rad(const double &deg) { return (M_PI / 180.0) * deg; }

}

#endif //SRC_FLIGHT_MODEL_UTILS_HPP
