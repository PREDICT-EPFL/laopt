#define MAKE_YAML_INFO

// #include "qp_functions.hpp"
#include "ipopt_test_nlp_functions.hpp"

#include <fstream>

int main()
{
    std::ofstream f;
    // f.open("qp_functions.yml", std::ios::trunc);
    f.open("ipopt_functions.yml", std::ios::trunc);
    f << MyFunctions<double>::saveFunctionInfo() << std::endl;
    f.close();
    return 0;
}