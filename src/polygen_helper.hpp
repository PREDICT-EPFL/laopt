// #include <iostream>
// #include <typeinfo>
// #include <vector>
// #include <tuple>
// #include <functional>
// #include <string>
// #include <sstream>
// #include <array>
// #include <chrono>
// #include <utility>
// using namespace std;

// Eigen includes
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>
using namespace Eigen;

#include "unsupported/Eigen/AutoDiff"

#include <string_view>

template <typename T>
constexpr auto type_name() noexcept {
  std::string_view name = "Error: unsupported compiler", prefix, suffix;
#ifdef __clang__
  name = __PRETTY_FUNCTION__;
  prefix = "auto type_name() [T = ";
  suffix = "]";
#elif defined(__GNUC__)
  name = __PRETTY_FUNCTION__;
  prefix = "constexpr auto type_name() [with T = ";
  suffix = "]";
#elif defined(_MSC_VER)
  name = __FUNCSIG__;
  prefix = "auto __cdecl type_name<";
  suffix = ">(void) noexcept";
#endif
  name.remove_prefix(prefix.size());
  name.remove_suffix(suffix.size());
  return name;
}



// #define SEG(size, offset) template segment<size>(offset) // Segment of an Eigen vector
// #define BLK(x_size, y_size, x_offset, y_offset) template block<x_size, y_size>(x_offset, y_offset) // Block of an eigen matrix

// #define MatX Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>

template<typename scalar_t, int Size>
using RCVec = const Ref<const Eigen::Matrix<scalar_t, Size, 1>>;
// template<typename scalar_t, int Rows, int Cols>
// using RCMat = const Ref<const Eigen::Matrix<scalar_t, Rows, Cols>>;
// template<typename scalar_t, int Size>
// using Vec = Eigen::Matrix<scalar_t, Size, 1>;
// template<typename scalar_t, int Size>
// using RVec = Eigen::Ref<Eigen::Matrix<scalar_t, Size, 1>>;

/**
 * Macro to declare a templated variable type
 */
#define DECLARE_VAR_TYPE(name, length) \
    static constexpr int name##_size = length; \
    template<typename T> using name = Eigen::Matrix<T, name##_size, 1>;

/**
 * Macro to define a variable
 * 
 * Args:
 *  name - variable name
 *  offset - offset of the start of this variable into the 
 *  size - size of a vector of this variable type
 *  number [optional] - number of vectors
 */
#define DECLARE_VAR3(name, offset, size) \
    static constexpr auto name##_offset() {return offset;} /* Offset */ \
    template<typename T> \
    constexpr auto name(T x) \
        {return x.template segment<size>(offset);} /* Offset */
    // constexpr auto name(const Eigen::Ref<const Eigen::Matrix<scalar_t, VAR_SIZE, 1>> x) \
    //     {return x.template segment<size>(offset);} /* Offset */ \
    // template<typename T> \
    // constexpr auto name##_d(const Eigen::Ref<const Eigen::Matrix<T, VAR_SIZE, 1>> x) \
    //     {return x.template segment<size>(offset);} /* Offset */

    // constexpr auto name(const Ref<const variable_t<scalar_t>>& x) \
    //     {return x.template segment<s##name>(offset)} /* Offset */

    // static constexpr auto s##name = size; /* Size of the variable */ \
    // template<typename T> using name##_t = Eigen::Matrix<T, s##name, 1>; /* Templated type of the variable */ \
    // constexpr auto o##name() {return offset;} /* Offset */ \
    // constexpr auto name(RCVec<scalar_t, VAR_SIZE> x) \
    //     {return x.template segment<s##name>(o##name());} /* Offset */

#define DECLARE_VAR4(name, offset, size, number) \
    static constexpr auto name##_mat_size = size*number; /* Size of the vectorized variable */ \
    static constexpr auto name##_numvecs = number; /* Number of vectors */ \
    static constexpr auto name##_offset(int col) {return offset + size * col;} /* Offset */ \
    template<typename T> \
    constexpr auto name(T x, int col) \
        {return x.template segment<size>(offset + size * col);} /* Offset */
    // constexpr auto name(const Eigen::Ref<const Eigen::Matrix<scalar_t, VAR_SIZE, 1>> x, int col) \
    //     {return x.template segment<size>(offset + size * col);} /* Offset */ \
    // template<typename T> \
    // constexpr auto name##_d(const Eigen::Ref<const Eigen::Matrix<T, VAR_SIZE, 1>> x, int col) \
    //     {return x.template segment<size>(offset + size * col);} /* Offset */

    // constexpr auto name(RCVec<scalar_t, VAR_SIZE> x, int col) \
    //     {return x.template segment<size>(offset + size * col);} /* Offset */
    // static constexpr auto s##name = size; /* Size of the variable */ \
    // template<typename T> using name##_t = Eigen::Matrix<T, s##name, 1>; /* Templated type of the variable */ \
    // static constexpr auto num##name = number; /* Number of vectors */ \
    // constexpr auto o##name(int col) {return offset + size * col;} /* Offset */ \
    // constexpr auto name(RCVec<scalar_t, VAR_SIZE> x, int col) \
    //     {return x.template segment<s##name>(o##name(col));} /* Offset */

#define GET_MACRO(_1,_2,_3,_4,NAME,...) NAME
#define DECLARE_VAR(...) GET_MACRO(__VA_ARGS__, DECLARE_VAR4, DECLARE_VAR3)(__VA_ARGS__)

/**
 * Macro to define a constraint
 * 
 * Args:
 *  name - constraint name
 *  offset - offset from the start of the constraint vector
 *  size - size of a vector of this variable type
 *  number [optional] - number of vectors
 */
#define DECLARE_CON3(name, offset, size) \
    static constexpr auto name##_size = size; \
    constexpr auto name##_offset() {return offset;}; \
    template<typename T> \
    constexpr auto name(T g) {return g.template segment<name##_size>(name##_offset());};

#define DECLARE_CON4(name, offset, size, number) \
    static constexpr auto name##_size = size; \
    constexpr auto name##_offset(int ind) {return offset + size * ind;}; \
    template<typename T> \
    constexpr auto name(T g, int ind) {return g.template segment<name##_size>(name##_offset(ind));};

#define DECLARE_CONSTRAINT(...) GET_MACRO(__VA_ARGS__, DECLARE_CON4, DECLARE_CON3)(__VA_ARGS__)




template<typename scalar_t, int num_inputs>
using AD_scalar = AutoDiffScalar<Eigen::Matrix<scalar_t, num_inputs, 1>>;


template <typename vec>
constexpr int AD_Seed(vec &x, int offset)
{
    for (int i=0; i<x.rows(); i++) {
        x[i].derivatives().coeffRef(i + offset) = 1;
    }
    return offset + x.rows();
}

// template<typename scalar_t, int num_outputs, int num_inputs, int... input_sizes, 
//         std::size_t... Is, // Index sequence for the inputs (0,1,2,3..., number of inputs)
//         typename Functor>
// EIGEN_STRONG_INLINE void jacobian_impl(Functor func, 
//     std::tuple<const Eigen::Ref<const Eigen::Matrix<scalar_t, input_sizes, 1>>&...> inputs, // Note: forward_as_tuple forwards a reference
//     Eigen::Ref<Eigen::Matrix<scalar_t,num_outputs,1>> out, 
//     std::tuple<Eigen::Ref<Eigen::Matrix<scalar_t, num_outputs, input_sizes>>&...> J, 
//     std::index_sequence<Is...>)
// {
//     // AD variables for block-inputs and output
//     using AD_scalar = AD_scalar<scalar_t, num_inputs>;
//     using AD_input_t = std::tuple<Matrix<AD_scalar, input_sizes, 1>...>;
//     using AD_output_t = Matrix<AD_scalar, num_outputs, 1>;  

//     AD_input_t _x;    // Tuple of dual variables as inputs
//     AD_output_t _out; // Vector of dual variables as outputs

    // std::get<0>(_x) = std::get<0>(inputs);
    // std::get<1>(_x) = std::get<1>(inputs);
    // std::get<2>(_x) = std::get<2>(inputs);

    // int offset = 0;
    // offset = AD_Seed(std::get<0>(_x), offset); // Set to unit vectors
    // offset = AD_Seed(std::get<1>(_x), offset); // Set to unit vectors
    // offset = AD_Seed(std::get<2>(_x), offset); // Set to unit vectors

    // Copy the current value of the inputs into the AD variable
    // and set derivative equal to identity
    // int offset = 0;
    // (void)std::initializer_list<int>{ 
    //     (
    //         std::get<Is>(_x) = std::get<Is>(inputs),    // Copy inputs
    //         offset = AD_Seed(std::get<Is>(_x), offset), // Set to unit vectors
    //         0
    //     )...
    // };

    // Call our function
    // func(std::get<Is>(_x)..., _out);

    // // Copy Jacobian into output variables
    // for(int i=0; i<num_outputs; i++)
    // {
    //     out(i) = _out[i].value();
    //     Eigen::Ref<Eigen::Matrix<scalar_t, num_inputs, 1>> deriv = _out[i].derivatives();
        
    //     // Copy gradients to Jacobian matrices
    //     offset = 0;
    //     (void)std::initializer_list<int>{ 
    //         (
    //             std::get<Is>(J).row(i) = deriv.template segment<input_sizes>(offset), 
    //             offset += input_sizes,
    //             0
    //         )... 
    //     };
    // }
// }

// Take a vector input and return a AD version of the vector
template<typename scalar_t, int num_inputs, int n>
EIGEN_STRONG_INLINE Matrix<AD_scalar<scalar_t, num_inputs>, n, 1> 
    make_ad(const Ref<const Matrix<scalar_t, n, 1>> x)
{
    Matrix<AD_scalar<scalar_t, num_inputs>, n, 1> y;
    y = x;
    for (int i=0; i<y.rows(); i++) {
        y[i].derivatives().setZero();
    }
    return y;
}

template<typename scalar_t, int num_outputs, int num_inputs, int... input_sizes, typename Functor>
EIGEN_STRONG_INLINE void jacobian_impl(
        Functor func, // Function with AD template
        Matrix<AD_scalar<scalar_t, num_inputs>, input_sizes, 1>... inputs,
        Eigen::Ref<Eigen::Matrix<scalar_t, num_outputs, 1>> out, 
        Eigen::Ref<Eigen::Matrix<scalar_t, num_outputs, input_sizes>>... J)
{
    // Set derivative equal to identity
    int offset = 0;
    (void)std::initializer_list<int>{ 
        (
            offset = AD_Seed(inputs, offset), // Set to unit vectors
            0
        )...
    };

    // AD output
    using AD_scalar = AD_scalar<scalar_t, num_inputs>;
    using AD_output_t = Matrix<AD_scalar, num_outputs, 1>;  
    AD_output_t _out;

    // Call our function
    func(inputs..., _out);

    // Copy Jacobian into output variables
    for(int i=0; i<num_outputs; i++)
    {
        out(i) = _out[i].value();
        Eigen::Ref<Eigen::Matrix<scalar_t, num_inputs, 1>> deriv = _out[i].derivatives();
        
        // Copy gradients to Jacobian matrices
        offset = 0;
        (void)std::initializer_list<int>{ 
            (
                J.row(i) = deriv.template segment<input_sizes>(offset), 
                offset += input_sizes,
                0
            )... 
        };
    }
}

template<typename scalar_t, int num_outputs, int num_inputs, int... input_sizes, typename Functor>
EIGEN_STRONG_INLINE void call_jacobian(
        Functor func, // Function with AD template
        const Eigen::Ref<const Eigen::Matrix<scalar_t, input_sizes, 1>>... inputs,
        Eigen::Ref<Eigen::Matrix<scalar_t,num_outputs,1>> out, 
        Eigen::Ref<Eigen::Matrix<scalar_t, num_outputs, input_sizes>>... J
)
{
    jacobian_impl<scalar_t, num_outputs, num_inputs, input_sizes...>(
        func, 
        make_ad<scalar_t, num_inputs, input_sizes>(inputs)..., // Convert inputs to AD variables
        out, J...);
}
