struct Test
{
  using scalar_t = double;
  using param_t = typename MyFunctions<double>::param_t_;
  using functions = std::tuple<
    lampc::Jacobian<MyFunctions<double>::dynamics_eq_, double, MyFunctions<double>::param_t_, 2, 2, 2, 1>,
    lampc::Jacobian<MyFunctions<double>::dynamics_ss_, double, MyFunctions<double>::param_t_, 2, 2, 1>,
    lampc::Jacobian<MyFunctions<double>::test_func_, double, MyFunctions<double>::param_t_, 1, 2, 1>,
    lampc::Jacobian<MyFunctions<double>::dynamics_, double, MyFunctions<double>::param_t_, 2, 2, 1>>;

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

struct equalities_info
{
  static constexpr int numFunctionCalls = 12;
  
  static constexpr int totalNumArgs = 33;
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
    8, 18,   // x_8, u_8, 
    3, 20,   // x_3, uss, 
    19, 20  // xss, uss, 
  };
};
using equalities_t = LA::Function<scalar_t, param_t, functions,
  variables_info, equalities_info,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1>; // Function call sequence
struct inequalities_info
{
  static constexpr int numFunctionCalls = 9;
  
  static constexpr int totalNumArgs = 18;
  static constexpr int functionArguments[] = {
    19, 10,   // xss, u_0, 
    19, 11,   // xss, u_1, 
    19, 12,   // xss, u_2, 
    19, 13,   // xss, u_3, 
    19, 14,   // xss, u_4, 
    19, 15,   // xss, u_5, 
    19, 16,   // xss, u_6, 
    19, 17,   // xss, u_7, 
    19, 18  // xss, u_8, 
  };
};
using inequalities_t = LA::Function<scalar_t, param_t, functions,
  variables_info, inequalities_info,
  3, 3, 3, 3, 3, 3, 3, 3, 3>; // Function call sequence
};
