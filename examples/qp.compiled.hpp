struct QP
{
  using scalar_t = double;
  using param_t = typename MyFunctions<double>::param_t_;
  using functions = std::tuple<
    lampc::Jacobian<MyFunctions<double>::dynamics_eq_, double, MyFunctions<double>::param_t_, 2, 2, 2, 1>,
    lampc::Jacobian<MyFunctions<double>::dynamics_ss_, double, MyFunctions<double>::param_t_, 2, 2, 1>,
    lampc::Jacobian<MyFunctions<double>::stage_cost_, double, MyFunctions<double>::param_t_, 1, 2, 1, 2, 1>>;

struct variables_info
{
  static const int inputSize = 32;
  static const int numVariables = 21;
  static constexpr LA::variable_info_t variable_info[]
  {
    {.offset = 0, .size = 2}, // x_0
    {.offset = 2, .size = 2}, // x_1
    {.offset = 4, .size = 2}, // x_2
    {.offset = 6, .size = 2}, // x_3
    {.offset = 8, .size = 2}, // x_4
    {.offset = 10, .size = 2}, // x_5
    {.offset = 12, .size = 2}, // x_6
    {.offset = 14, .size = 2}, // x_7
    {.offset = 16, .size = 2}, // x_8
    {.offset = 18, .size = 2}, // x_9
    {.offset = 20, .size = 1}, // u_0
    {.offset = 21, .size = 1}, // u_1
    {.offset = 22, .size = 1}, // u_2
    {.offset = 23, .size = 1}, // u_3
    {.offset = 24, .size = 1}, // u_4
    {.offset = 25, .size = 1}, // u_5
    {.offset = 26, .size = 1}, // u_6
    {.offset = 27, .size = 1}, // u_7
    {.offset = 28, .size = 1}, // u_8
    {.offset = 29, .size = 2}, // xss
    {.offset = 31, .size = 1}  // uss
  };
};

using variables_t = Eigen::Vector<scalar_t, variables_info::inputSize>;

struct equalities_info
{
  static constexpr int numFunctionCalls = 10;
  
  static constexpr int totalNumArgs = 29;
  static constexpr int functionArguments[] = {
    1, 0, 10,   // x_1, x_0, u_0, 
    2, 1, 11,   // x_2, x_1, u_1, 
    3, 2, 12,   // x_3, x_2, u_2, 
    4, 3, 13,   // x_4, x_3, u_3, 
    5, 4, 14,   // x_5, x_4, u_4, 
    6, 5, 15,   // x_6, x_5, u_5, 
    7, 6, 16,   // x_7, x_6, u_6, 
    8, 7, 17,   // x_8, x_7, u_7, 
    9, 8, 18,   // x_9, x_8, u_8, 
    19, 20  // xss, uss, 
  };
};
using equalities_t = LA::Function<scalar_t, param_t, functions,
  variables_info, equalities_info,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 1>; // Function call sequence
struct objective_info
{
  static constexpr int numFunctionCalls = 9;
  
  static constexpr int totalNumArgs = 36;
  static constexpr int functionArguments[] = {
    0, 10, 19, 20,   // x_0, u_0, xss, uss, 
    1, 11, 19, 20,   // x_1, u_1, xss, uss, 
    2, 12, 19, 20,   // x_2, u_2, xss, uss, 
    3, 13, 19, 20,   // x_3, u_3, xss, uss, 
    4, 14, 19, 20,   // x_4, u_4, xss, uss, 
    5, 15, 19, 20,   // x_5, u_5, xss, uss, 
    6, 16, 19, 20,   // x_6, u_6, xss, uss, 
    7, 17, 19, 20,   // x_7, u_7, xss, uss, 
    8, 18, 19, 20  // x_8, u_8, xss, uss, 
  };
};
using objective_t = LA::Function<scalar_t, param_t, functions,
  variables_info, objective_info,
  2, 2, 2, 2, 2, 2, 2, 2, 2>; // Function call sequence
};
