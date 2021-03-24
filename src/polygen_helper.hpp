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
#include <array>

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
    static constexpr auto name##_o() {return offset;} /* Offset */ \
    template<typename T> \
    constexpr auto name(T x) {return x.template segment<size>(offset);} /* Offset */ \
    static constexpr int name##_len = size;

#define DECLARE_VAR4(name, offset, size, number) \
    static constexpr auto name##_mat_size = size*number; /* Size of the vectorized variable */ \
    static constexpr auto name##_numvecs = number; /* Number of vectors */ \
    static constexpr auto name##_o(int col) {return offset + size * col;} /* Offset */ \
    template<typename T> \
    constexpr auto name(T x, int col) \
        {return x.template segment<size>(offset + size * col);} /* Offset */ \
    static constexpr int name##_len = size;

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
    static constexpr auto name##_len = size; \
    static constexpr auto name##_o() {return offset;}; \
    template<typename T> \
    constexpr auto name(T g) {return g.template segment<name##_len>(name##_o());};

#define DECLARE_CON4(name, offset, size, number) \
    static constexpr auto name##_len = size; \
    constexpr auto name##_o(int ind) {return offset + size * ind;}; \
    template<typename T> \
    constexpr auto name(T g, int ind) {return g.template segment<name##_len>(name##_o(ind));};

#define DECLARE_CONSTRAINT(...) GET_MACRO(__VA_ARGS__, DECLARE_CON4, DECLARE_CON3)(__VA_ARGS__)



// Declares a functor for member function in the derived class
#define MAKE_JACOBIAN(function_name, Derived, num_outputs, input_sizes...) \
    template<int nOutputs, int... inSizes> \
    struct function_name##_wrap \
    { \
        template<typename T> \
        static EIGEN_STRONG_INLINE void eval( \
            const Derived *obj,  \
            const Ref<const Matrix<T, inSizes, 1>>&... args, \
            Ref<Matrix<T, nOutputs, 1>> out) noexcept \
        { \
            obj->template _##function_name<T>(std::forward<decltype(args)>(args)...,  \
                                            std::forward<decltype(out)>(out)); \
        } \
    }; \
    Jacobian<function_name##_wrap, Derived, scalar_t, num_outputs, input_sizes> function_name;
    // using function_name##_fill_dense = decltype({con.function.name})::Fill_Dense; \
    // using function_name##_fill_sparse = decltype({con.function.name})::Fill_Sparse; \
    // using function_name##_fill_sparse_values = decltype({con.function.name})::Fill_Sparse_Values; \


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
    static constexpr int num_inputs = sum_template<input_sizes...>();  // Total number of inputs
    static constexpr int num_input_vars = sizeof...(input_sizes);  // Number of input vector variables

    using AD_scalar = AutoDiffScalar<Eigen::Matrix<scalar_t, num_inputs, 1>>;
    using AD_output_t = Matrix<AD_scalar, num_outputs, 1>;  

    using Functor = _Functor<num_outputs, input_sizes...>;

    const Object* obj;
    template<typename T>
    Jacobian(const T *_obj) : obj(static_cast<const Object*>(_obj)) {}

    // Call format:
    //   function(arg1, arg2, ..., argN, out)
    EIGEN_STRONG_INLINE void operator()(
        const Ref<const Matrix<scalar_t, input_sizes, 1>>&... args,
        Ref<Matrix<scalar_t, num_outputs, 1>> out) const noexcept
    {
        Functor::template eval<scalar_t>(
                    obj,
                    std::forward<decltype(args)>(args)..., 
                    std::forward<decltype(out)>(out));
    }

    // Call format:
    //   function({offset1, offset2, ...}, variable, out)
    template<typename variable_t, typename out_t,
             typename Indices = std::make_index_sequence<num_input_vars>>
    EIGEN_STRONG_INLINE void operator()(
        const std::array<int, num_input_vars> var_offsets,  // Offsets into the variable where the start of each input are stored
        const int eq_offset,  // Offset into constraint vector where we should start writing the output
        const Eigen::MatrixBase<variable_t>& var,
        Eigen::MatrixBase<out_t>& out) const noexcept
    {
        call_array(var_offsets, var, Indices{}, out.template segment<num_outputs>(eq_offset));
    }

    // Call format:
    //   function(arg1, arg2, ..., argN, out, Jac1, Jac2, ..., JacN)
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

    // Call format:
    //   function(arg1, arg2, ..., argN, out, callback)
    //   callback in a function that is passed the jacobian rowwise
    template<typename Function>
    EIGEN_STRONG_INLINE void operator()(
        const Ref<const Matrix<scalar_t, input_sizes, 1>>... args,
        Ref<Matrix<scalar_t, num_outputs, 1>> out,
        Function jacobian_callback
        ) const noexcept
    {
        AD_output_t _out;

        // Convert to AD variables for the inputs and call our function
        seed_and_call(make_ad<input_sizes>(args)..., _out);

        // Copy Jacobian into output variables
        for(int i=0; i<num_outputs; i++)
        {
            out(i) = _out[i].value();
            // std::cout << "Size of row  " << type_name<decltype(_out[i].derivatives())>() << std::endl;
            Eigen::Ref<Eigen::Matrix<scalar_t, num_inputs, 1>> deriv = _out[i].derivatives();
            // std::cout << "Size of row  " << type_name<decltype(deriv)>() << std::endl;
            jacobian_callback.template set_jacobian<decltype(deriv), input_sizes...>(deriv, i);

            // Eigen::Ref<Eigen::Matrix<scalar_t, num_inputs, 1>> deriv = _out[i].derivatives();            
            // // Copy gradients to Jacobian matrices
            // int offset = 0;
            // (void)std::initializer_list<int>{ 
            //     (
            //         J.row(i) = deriv.template segment<input_sizes>(offset), 
            //         offset += input_sizes,
            //         0
            //     )... 
            // };
        }
    }

    // Call format:
    //   function({offset1, offset2, ..., offsetN}, eq_offset, variable, out, jacobian)
    //   Takes the args from variable(offset1), variable(offset2), ...
    //   and writes the output into out(eq_offset)
    // template<typename variable_t, typename out_t, typename jacobian_t, 
    //          typename Indices = std::make_index_sequence<num_inputs>>
    // EIGEN_STRONG_INLINE void operator()(
    //     const std::array<int, num_inputs> var_offsets,  // Offsets into the variable where the start of each input are stored
    //     const int eq_offset,  // Offset into constraint vector where we should start writing the output
    //     const Eigen::MatrixBase<const variable_t>& var,
    //     Eigen::MatrixBase<out_t>& out,
    //     Eigen::MatrixBase<jacobian_t>& jacobian) const noexcept
    // {
    //     AD_output_t _out;
    //     seed_and_call_array(var_offsets, var, Indices{}, _out);

    //     // Copy Jacobian into output variables
    //     for(int i=0; i<num_outputs; i++)
    //     {
    //         out(i) = _out[i].value();
    //         Eigen::Ref<Eigen::Matrix<scalar_t, num_inputs, 1>> deriv = _out[i].derivatives();
    //         jacobian_callback.template set_jacobian<decltype(deriv), input_sizes...>(deriv, i);

    //         // Eigen::Ref<Eigen::Matrix<scalar_t, num_inputs, 1>> deriv = _out[i].derivatives();            
    //         // // Copy gradients to Jacobian matrices
    //         // int offset = 0;
    //         // (void)std::initializer_list<int>{ 
    //         //     (
    //         //         J.row(i) = deriv.template segment<input_sizes>(offset), 
    //         //         offset += input_sizes,
    //         //         0
    //         //     )... 
    //         // };
    //     }
    // }

    // template<int m, std::size_t N, std::size_t... I>
    // EIGEN_STRONG_INLINE void call_impl(
    //     const Ref<const Eigen::Matrix<scalar_t, m, 1>>& var,
    //     const std::array<int, N> offsets,
    //     Ref<Matrix<scalar_t, num_outputs, 1>> out,
    //     std::index_sequence<I...>
    //     ) const noexcept
    // {
    //     Functor::template eval<scalar_t>(
    //                 obj,
    //                 var.template segment<input_sizes>(offsets[I])...,
    //                 std::forward<decltype(out)>(out));
    // }


    // template<int m, std::size_t N, typename Indices = std::make_index_sequence<N>>
    // EIGEN_STRONG_INLINE void call(
    //     const Ref<const Eigen::Matrix<scalar_t, m, 1>>& var,
    //     const std::array<int, N> offsets,
    //     Ref<Matrix<scalar_t, num_outputs, 1>> out
    //     ) const noexcept
    // {
    //     static_assert(sizeof...(input_sizes) == N, "Array size is the wrong length");
    //     call_impl<m, N>(std::forward<decltype(var)>(var),
    //         std::forward<decltype(offsets)>(offsets),
    //         std::forward<decltype(out)>(out),
    //         Indices{}
    //     );
    // }


    // // Functor to copy rows of a jacobian into a larger jacobian matrix at the given var_offsets locations
    // template<typename T, int eq_offset, int... var_offsets>
    // struct Fill_Dense
    // {
    //     template<typename D, int... var_sizes>
    //     EIGEN_STRONG_INLINE void set_jacobian(Eigen::Ref<D> row, int iRow)
    //     {
    //         // Copy gradients into Jacobian matrix
    //         int offset = 0;
    //         auto Jrow = J.row(eq_offset + iRow);
    //         (void)std::initializer_list<int>{ 
    //             (
    //                 Jrow.template segment<var_sizes>(var_offsets) = row.template segment<var_sizes>(offset), 
    //                 offset += var_sizes,
    //                 0
    //             )... 
    //         };
    //     }

    //     Fill_Dense(Eigen::Ref<T> _J) : J(_J) {}
    //     Eigen::Ref<T> J;
    // };

    // // Factory
    // template<typename T, int eq_offset, int... var_offsets>
    // EIGEN_STRONG_INLINE Fill_Dense<T, eq_offset, var_offsets...> fill_dense(Eigen::Ref<T> J)
    // {
    //     return Fill_Dense<T, eq_offset, var_offsets...>(J);
    // }


    // // Functor to copy rows of a jacobian into a larger jacobian matrix at the given var_offsets locations
    // template<int eq_offset, int... var_offsets>
    // struct Fill_Sparse
    // {
    //     template<typename D>
    //     EIGEN_STRONG_INLINE int fill_var(Eigen::Ref<D> row, int iRow, 
    //                                     int var_size,   // Column (variable) offset and size for J
    //                                     int var_offset,
    //                                     int offset // Initial column offset for row
    //                                     )
    //     {
    //         for(int col=0; col<var_size; col++)
    //            J.coeffRef(iRow, var_offset + col) = row(offset++);
    //         return offset;
    //     }    

    //     template<typename D, int... var_sizes>
    //     EIGEN_STRONG_INLINE void set_jacobian(Eigen::Ref<D> row, int iRow)
    //     {
    //         // Copy gradients into Jacobian matrix
    //         int offset = 0;
    //         (void)std::initializer_list<int>{ 
    //             (
    //                 offset = fill_var<D>(row, eq_offset + iRow, var_sizes, var_offsets, offset),
    //                 0
    //             )... 
    //         };
    //     }

    //     Fill_Sparse(Eigen::SparseMatrix<scalar_t>& _J) : J(_J) {}
    //     Eigen::SparseMatrix<scalar_t>& J;
    // };
    
    // // Factory
    // template<int eq_offset, int... var_offsets>
    // EIGEN_STRONG_INLINE Fill_Sparse<eq_offset, var_offsets...> fill_sparse(Eigen::SparseMatrix<scalar_t>& J)
    // {
    //     return Fill_Sparse<eq_offset, var_offsets...>(J);
    // }


    // // Functor to copy rows of a jacobian into a sparse jacobian matrix formed by constraints_sparse_initialize
    // // This functor only takes the value array of the sparse matrix and fills it in in the order defined in 
    // // constraints_sparse_initialize (i.e., as ipopt requires)
    // struct Fill_Sparse_Values
    // {
    //     template<typename D, int... var_sizes>
    //     EIGEN_STRONG_INLINE void set_jacobian(Eigen::Ref<D> row, int iRow)
    //     {
    //         J.row(iRow) = row;
    //     }

    //     // Pass in a vector of length num_outpus * num_inputs where the derivative will be placed rowwise
    //     Fill_Sparse_Values(Eigen::Ref<Eigen::Matrix<scalar_t, num_outputs*num_inputs, 1>> _J) : 
    //         J(_J) {}
    //     Eigen::Map<Eigen::Matrix<scalar_t, num_outputs, num_inputs>> J;
    // };

    // // Factory
    // EIGEN_STRONG_INLINE Fill_Sparse_Values fill_sparse_values(Eigen::Ref<Eigen::Matrix<scalar_t, num_outputs*num_inputs, 1>> J)
    // {
    //     return Fill_Sparse_Values(J);
    // }



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

    // Takes an array of offsets, and then calls seed_and_call with the args as offsets into the var vector
    template<typename variable_t, typename out_t, int... ind>
    EIGEN_STRONG_INLINE void seed_and_call_array(
        const std::array<int, num_input_vars> var_offsets,  // Offsets into the variable where the start of each input is stored
        const Eigen::MatrixBase<const variable_t>& var,
        std::integer_sequence<int, ind...> int_seq,
        Eigen::Ref<Eigen::Matrix<AD_scalar, num_outputs, 1>> out
        ) const noexcept
    {
        seed_and_call(var.template segment<input_sizes>(var_offsets[ind])..., out);
    }

    // Takes an array of offsets, and then calls our function with the args as offsets into the var vector
    template<typename variable_t, typename T, T... ind>
    EIGEN_STRONG_INLINE void call_array(
        const std::array<int, num_input_vars> var_offsets,  // Offsets into the variable where the start of each input is stored
        const Eigen::MatrixBase<variable_t>& var,
        std::integer_sequence<T, ind...> int_seq,
        Eigen::Ref<Eigen::Matrix<scalar_t, num_outputs, 1>> out
        ) const noexcept
    {
        Functor::template eval<scalar_t>(
                    obj,
                    var.template segment<input_sizes>(var_offsets[ind])...,
                    out);
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



/******************************************************
 * Sparse Matrices
 ******************************************************/
using BlockInfo = std::tuple<int,int,int,int>; // row, col, num_rows, num_cols

/* Sets the sparsity structure of J to match the blocks in BlockInfo */
template<typename scalar_t>
void set_nonzero_blocks(Eigen::SparseMatrix<scalar_t>& J, 
                        std::initializer_list<BlockInfo> blocks)
{
    std::vector<Eigen::Triplet<scalar_t>> trip;

    int row, col, num_rows, num_cols;
    for (auto blk : blocks)
    {
        std::tie(row, col, num_rows, num_cols) = blk;
        for (int i=row; i<row+num_rows; i++)
            for (int j=col; j<col+num_cols; j++)
                trip.emplace_back(i, j, -1.0);    
    }

    J.setFromTriplets(trip.begin(), trip.end());
    J.makeCompressed();    
}




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



