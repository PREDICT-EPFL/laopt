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

#define DECLARE_VAR4(name, offset, size, number) \
    static constexpr auto name##_mat_size = size*number; /* Size of the vectorized variable */ \
    static constexpr auto name##_numvecs = number; /* Number of vectors */ \
    static constexpr auto name##_offset(int col) {return offset + size * col;} /* Offset */ \
    template<typename T> \
    constexpr auto name(T x, int col) \
        {return x.template segment<size>(offset + size * col);} /* Offset */

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



// Declares a functor for member function in the derived class
#define DECLARE_FUNCTION(function_name, num_outputs, input_sizes...) \
    template<int nOutputs, int... inSizes> \
    struct function_name##_wrap \
    { \
        template<typename T> \
        static EIGEN_STRONG_INLINE void eval( \
            const Derived *obj,  \
            const Ref<const Matrix<T, inSizes, 1>>&... args, \
            Ref<Matrix<T, nOutputs, 1>> out) noexcept \
        { \
            obj->template function_name<T>(std::forward<decltype(args)>(args)...,  \
                                           std::forward<decltype(out)>(out)); \
        } \
    }; \
    Jacobian<function_name##_wrap, Derived, scalar_t, num_outputs, input_sizes> function_name;



// Sum the inputs to get total number of inputs
template<int... S>
constexpr int sum_template() {
    int result = 0;
    for(auto s : { S... }) result += s;
    return result;
}

template<template <int, int...> class _Functor, // Wrapped function from DECLARE_FUNCTION
        typename Object, // Object type of class that contains the wrapped function
        typename scalar_t,
        int num_outputs, int... input_sizes>
struct Jacobian
{
    static constexpr int num_inputs = sum_template<input_sizes...>();

    using AD_scalar = AutoDiffScalar<Eigen::Matrix<scalar_t, num_inputs, 1>>;
    using AD_output_t = Matrix<AD_scalar, num_outputs, 1>;  

    using Functor = _Functor<num_outputs, input_sizes...>;

    const Object* obj;
    template<typename T>
    Jacobian(const T *_obj) : obj(static_cast<const Object*>(_obj)) {}

    EIGEN_STRONG_INLINE void operator()(
        const Ref<const Matrix<scalar_t, input_sizes, 1>>&... args,
        Ref<Matrix<scalar_t, num_outputs, 1>> out) const noexcept
    {
        Functor::template eval<scalar_t>(
                    obj,
                    std::forward<decltype(args)>(args)..., 
                    std::forward<decltype(out)>(out));
    }

    EIGEN_STRONG_INLINE void operator()(
        const Ref<const Matrix<scalar_t, input_sizes, 1>>... args,
        Ref<Matrix<scalar_t, num_outputs, 1>> out,
        Ref<Matrix<scalar_t, num_outputs, input_sizes>>... J
        ) const noexcept
    {
        AD_output_t _out;

        // Convert to AD variables for the inputs and call our function
        seed_and_call(make_ad<input_sizes>(args)..., _out);

        // Copy Jacobian into output variables
        for(int i=0; i<num_outputs; i++)
        {
            out(i) = _out[i].value();
            Eigen::Ref<Eigen::Matrix<scalar_t, num_inputs, 1>> deriv = _out[i].derivatives();
            
            // Copy gradients to Jacobian matrices
            int offset = 0;
            (void)std::initializer_list<int>{ 
                (
                    J.row(i) = deriv.template segment<input_sizes>(offset), 
                    offset += input_sizes,
                    0
                )... 
            };
        }
    }

private:
    EIGEN_STRONG_INLINE void seed_and_call(
            Matrix<AD_scalar, input_sizes, 1>... args,
            Eigen::Ref<Eigen::Matrix<AD_scalar, num_outputs, 1>> out) const
    {
        // Set derivative equal to identity
        int offset = 0;
        (void)std::initializer_list<int>{ 
            (
                offset = AD_Seed(args, offset), // Set to unit vectors
                0
            )...
        };

        // Call our function
        Functor::template eval<AD_scalar>(obj, args..., out);
    }


    // Take a vector input and return a AD version of the vector
    template<int n>
    EIGEN_STRONG_INLINE Matrix<AD_scalar, n, 1> 
        make_ad(const Ref<const Matrix<scalar_t, n, 1>> x) const
    {
        Matrix<AD_scalar, n, 1> y;
        y = x;
        for (int i=0; i<y.rows(); i++) {
            y[i].derivatives().setZero();
        }
        return y;
    }

    // Sets the input derivatives to the identity. 
    // Assumes that the derivative matrix is initially zero
    template <typename vec>
    constexpr int AD_Seed(vec &x, int offset) const
    {
        for (int i=0; i<x.rows(); i++)
            x[i].derivatives().coeffRef(i + offset) = 1;
        return offset + x.rows();
    }
};




// // How to build a functor to a member function
// template <typename Member, typename Obj> 
// struct MemberFunctor
// {
//     Obj _obj;
//     Member _member;
//     MemberFunctor(Obj obj, Member member) : _obj(obj), _member(member) {};

//     template<typename... T>
//     EIGEN_STRONG_INLINE auto operator()(T&&... args) noexcept
//     {
//         return _member(_obj, std::forward<T>(args)...);
//     }
// };

// template<typename Functor>
// EIGEN_STRONG_INLINE auto get_functor(Functor f) noexcept
// {
//     return MemberFunctor<decltype(std::mem_fn(f)), Derived*> 
//         (static_cast<Derived*>(this), std::mem_fn(f));
// }
