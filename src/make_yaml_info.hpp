#ifdef MAKE_YAML_INFO

#include <iostream>
#include "lampc_function.hpp"
#include "yaml-cpp/yaml.h"

struct Info
{
    std::string signature;
    std::string name;
    int num_input_vars;
    int num_outputs;
    std::vector<int> input_sizes;

    Eigen::SparseMatrix<int> jacobianStructure;
    std::vector<Eigen::SparseMatrix<int>> hessianStructure;

    // Pull out the required info from the template type F for the callable
    template<typename F>
    Info(F)
    {
        signature = type_name<F>();
        name = F::name;
        num_input_vars = F::num_input_vars;
        num_outputs = F::num_outputs;

        jacobianStructure = F::jacobianStructure();

        for(int i=0; i<F::num_outputs; i++)
            hessianStructure.push_back(F::hessianStructure(i));

        input_sizes = F::get_input_sizes();
    }
};

/**
 * Save only the sparsity structure from a sparse matrix
 */
template<>
struct YAML::convert<Eigen::SparseMatrix<int>>
{
    static Node encode(const Eigen::SparseMatrix<int> & rhs)
    {
        Node node;
        std::vector<std::pair<int,int>> nonZeros;
        for (int k=0; k<rhs.outerSize(); ++k)
          for (SparseMatrix<int>::InnerIterator it(rhs,k); it; ++it)
            nonZeros.push_back({it.row(), it.col()});
        node = nonZeros;

        return node;
    }
};

template<>
struct YAML::convert<Info>
{
    static Node encode(const Info & rhs)
    {
        Node node;
        node["signature"] = rhs.signature;
        node["name"] = rhs.name;
        node["num_input_vars"] = rhs.num_input_vars;
        node["num_outputs"] = rhs.num_outputs;
        node["input_sizes"] = rhs.input_sizes;
        node["input_sizes"].SetStyle(YAML::EmitterStyle::Flow);

        node["jacobianStructure"] = rhs.jacobianStructure;
        node["jacobianStructure"].SetStyle(YAML::EmitterStyle::Flow);

        for(int i=0; i<rhs.num_outputs; i++)
        {
            node["hessianStructure"].push_back(rhs.hessianStructure[i]);
            node["hessianStructure"][i].SetStyle(YAML::EmitterStyle::Flow);
        }

        return node;
    }
};


template<typename F>
inline YAML::Node get_info()
{
    YAML::Node node;  // starts out as null
    node["signature"] = std::string(type_name<F>());
    node["name"] = F::name;
    node["num_input_vars"] = F::num_input_vars;
    node["num_outputs"] = F::num_outputs;
    node["input_sizes"] = F::get_input_sizes();
    node["input_sizes"].SetStyle(YAML::EmitterStyle::Flow);

    node["jacobianStructure"] = F::jacobianStructure();
    node["jacobianStructure"].SetStyle(YAML::EmitterStyle::Flow);

    for(int i=0; i<F::num_outputs; i++)
    {
        node["hessianStructure"].push_back(F::hessianStructure(i));
        node["hessianStructure"][i].SetStyle(YAML::EmitterStyle::Flow);
    }

    return node;
}

template<typename scalar_t, typename... Funcs>
inline std::string saveInfoYAML()
{
    YAML::Node node;
    std::array<Info, sizeof...(Funcs)> funcs = {Info(Funcs())...};
    for(int i=0; i<sizeof...(Funcs); i++)
        node[funcs[i].name] = funcs[i];

    YAML::Emitter out;
    out << node;
    return out.c_str();
}

#else
template<typename scalar_t, typename... Funcs>
inline std::string saveInfoYAML() {return "";}
#endif

