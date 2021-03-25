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
// #define DECLARE_VAR_TYPE(name, length) \
//     static constexpr int name##_size = length; \
//     template<typename T> using name = Eigen::Matrix<T, name##_size, 1>;

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
    static constexpr auto name = offset; /* Offset */ \
    template<typename T> \
    constexpr auto name##_get(T x) {return x.template segment<size>(offset);} /* Offset */ \
    static constexpr int name##_len = size;

#define DECLARE_VAR4(name, offset, size, number) \
    static constexpr auto name##_mat_size = size*number; /* Size of the vectorized variable */ \
    static constexpr auto name##_numvecs = number; /* Number of vectors */ \
    static constexpr auto name(int col) {return offset + size * col;} /* Offset */ \
    template<typename T> \
    constexpr auto name##_get(T x, int col) \
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
    static constexpr auto name = offset; \
    template<typename T> \
    constexpr auto name##_get(T g) {return g.template segment<name##_len>(name);};

#define DECLARE_CON4(name, offset, size, number) \
    static constexpr auto name##_len = size; \
    constexpr auto name(int ind) {return offset + size * ind;}; \
    template<typename T> \
    constexpr auto name##_get(T g, int ind) {return g.template segment<name##_len>(name(ind));};

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
            Eigen::Ref<Eigen::Matrix<scalar_t, num_inputs, 1>> deriv = _out[i].derivatives();
            jacobian_callback.template set_jacobian<decltype(deriv), input_sizes...>(deriv, i);
        }
    }

    /********************************************************
     * Call formats taking a variable vector + offsets
     ********************************************************/

    // Call format:
    //   function({offset1, offset2, ...}, variable, out)
    template<typename variable_t, typename out_t,
             typename Indices = std::make_index_sequence<num_input_vars>>
    EIGEN_STRONG_INLINE void operator()(
        const std::array<Eigen::Index, num_input_vars> var_offsets,  // Offsets into the variable where the start of each input are stored
        const std::array<Eigen::Index, num_input_vars> col_offsets,  // Offset into the compressed column for each variable
        const int eq_offset,  // Offset into constraint vector where we should start writing the output
        const Eigen::MatrixBase<variable_t>& var,
        Eigen::MatrixBase<out_t>& out,
        int jacobian  // Ignored
        ) const noexcept
    {
        call_array(var_offsets, var, Indices{}, out.template segment<num_outputs>(eq_offset));
    }

    // Call format:
    //   function({offset1, offset2, ..., offsetN}, eq_offset, variable, out, jacobian)
    //   Takes the args from variable(offset1), variable(offset2), ...
    //   and writes the output into out(eq_offset)
    template<typename variable_t, typename out_t, 
             typename Indices = std::make_index_sequence<num_input_vars>>
    EIGEN_STRONG_INLINE void operator()(
        const std::array<Eigen::Index, num_input_vars> var_offsets,  // Offsets into the variable where the start of each input are stored
        const std::array<Eigen::Index, num_input_vars> col_offsets,  // Offset into the compressed column for each variable
        const int eq_offset,  // Offset into constraint vector where we should start writing the output
        const Eigen::MatrixBase<variable_t>& var,
        Eigen::MatrixBase<out_t>& out,
        Eigen::Ref<Eigen::Matrix<scalar_t, out_t::RowsAtCompileTime, variable_t::RowsAtCompileTime>> jacobian
        ) const noexcept
    {
        AD_output_t _out;
        seed_and_call_array(var_offsets, var, Indices{}, _out);
        copy_jacobian_dense_array<decltype(jacobian)>(var_offsets, eq_offset, out, jacobian, _out, Indices{});
    }

    // Helper function to fill in the jacobian
    template<typename J_t, typename out_t, typename T, T... ind>
    EIGEN_STRONG_INLINE void copy_jacobian_dense_array(
        const std::array<Eigen::Index, num_input_vars> var_offsets,  // Offsets into the variable where the start of each input are stored
        const int eq_offset,  // Offset into constraint vector where we should start writing the output
        Eigen::MatrixBase<out_t>& out,
        Eigen::Ref<J_t> jacobian,
        AD_output_t& _out,
        std::integer_sequence<T, ind...> int_seq
        ) const noexcept
    {
        // Copy Jacobian into output variables
        for(int i=0; i<num_outputs; i++)
        {
            out(i + eq_offset) = _out[i].value();
            Eigen::Ref<Eigen::Matrix<scalar_t, num_inputs, 1>> deriv = _out[i].derivatives();

            // Copy gradients to Jacobian matrix
            int offset = 0;
            (void)std::initializer_list<int>{ 
                (
                    jacobian.row(i+eq_offset).template segment<input_sizes>(var_offsets[ind])
                        = deriv.template segment<input_sizes>(offset), 
                    offset += input_sizes,
                    0
                )... 
            };
        }
    }


    // Call format:
    //   function({offset1, offset2, ..., offsetN}, eq_offset, variable, out, jacobian, ...)
    //   Takes the args from variable(offset1), variable(offset2), ...
    //   and writes the output into out(eq_offset)
    //   jacobian here is a vector of non-zeros of length nnz_constraints_jacobian
    template<typename variable_t, typename out_t,
             typename Indices = std::make_index_sequence<num_input_vars>>
    EIGEN_STRONG_INLINE void operator()(
        const std::array<Eigen::Index, num_input_vars> var_offsets,  // Offsets into the variable where the start of each input are stored
        const std::array<Eigen::Index, num_input_vars> col_offsets,  // Offset into the compressed column for each variable
        const int eq_offset,  // Offset into constraint vector where we should start writing the output
        const Eigen::MatrixBase<variable_t>& var,
        Eigen::MatrixBase<out_t>& out,
        Eigen::Ref<Eigen::SparseMatrix<scalar_t>> jacobian
        ) const noexcept
    {
        AD_output_t _out;
        seed_and_call_array(var_offsets, var, Indices{}, _out);
        copy_jacobian_sparse_array<decltype(jacobian)>(var_offsets, col_offsets, eq_offset, out, jacobian, _out, Indices{});
    }

    // Helper function to fill in a sparse jacobian
    template<typename J_t, typename out_t, typename T, T... ind>
    EIGEN_STRONG_INLINE void copy_jacobian_sparse_array(
        const std::array<Eigen::Index, num_input_vars> var_offsets,  // Offsets into the variable where the start of each input are stored
        const std::array<Eigen::Index, num_input_vars> col_offsets,  // Offset into the compressed column for each variable
        const int eq_offset,  // Offset into constraint vector where we should start writing the output
        Eigen::MatrixBase<out_t>& out,
        Eigen::Ref<Eigen::SparseMatrix<scalar_t>> jacobian,
        AD_output_t& _out,
        std::integer_sequence<T, ind...> int_seq
        ) const noexcept
    {

        // We know that the jacobian is blockwise sparse, meaning that it's either zero
        // for a given constraint / variable, or it's dense
        // 
        // We have the form
        //         var1 | var2 | var3 | ... | varN
        //  con1 |      |      |      |     |
        //  con2 |      |      |      |     |
        //  con3 |      |      |      |     |
        //  con4 |      |      |      |     |
        //  con5 |      |      |      |     |
        //
        //
        // We take the dense Values vector of the sparse jacobian and reshape it into a 
        // dense matrix of size nnz_var x var_len
        // We then write into the row eq_offset. Note that eq_offset needs to refer to the 
        // row in the compressed matrix
        //
        // OuterIndexPtr[var_offset] => Index into Values for the start of this column.
        // nnz_var = OuterStarts[j+1]-OuterStarts[j] => Number of non-zero elements 
        //
        // 

        using MatrixScalar = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;
        using Map_t = Eigen::Map<MatrixScalar, 0, Eigen::Stride<Eigen::Dynamic, 1>>;
        using Stride_t = Eigen::Stride<Eigen::Dynamic, 1>;

        // Copy Jacobian into output variables
        for(int i=0; i<num_outputs; i++)
        {
            out(i + eq_offset) = _out[i].value();
            Eigen::Ref<Eigen::Matrix<scalar_t, num_inputs, 1>> deriv = _out[i].derivatives();

            // Copy gradients to Jacobian matrix
            int offset = 0;
            int nnz = 0;
            scalar_t* data;
            (void)std::initializer_list<int>{ 
                (
                    // We use map to produce a dense view into the constraint/variable block and then copy over the right
                    nnz = jacobian.outerIndexPtr()[var_offsets[ind]+1] - jacobian.outerIndexPtr()[var_offsets[ind]],
                    data = jacobian.valuePtr() + jacobian.outerIndexPtr()[var_offsets[ind]] + col_offsets[ind],
                    Map_t(data, num_outputs, input_sizes, Stride_t(nnz, 1)).row(i) = deriv.template segment<input_sizes>(offset),
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

    // Takes an array of offsets, and then calls seed_and_call with the args as offsets into the var vector
    template<typename variable_t, typename T, T... ind>
    EIGEN_STRONG_INLINE void seed_and_call_array(
        const std::array<Eigen::Index, num_input_vars> var_offsets,  // Offsets into the variable where the start of each input is stored
        const Eigen::MatrixBase<variable_t>& var,
        std::integer_sequence<T, ind...> int_seq,
        Eigen::Ref<Eigen::Matrix<AD_scalar, num_outputs, 1>> out
        ) const noexcept
    {
        seed_and_call(make_ad<input_sizes>(var.template segment<input_sizes>(var_offsets[ind]))..., out);
    }

    // Takes an array of offsets, and then calls our function with the args as offsets into the var vector
    template<typename variable_t, typename T, T... ind>
    EIGEN_STRONG_INLINE void call_array(
        const std::array<Eigen::Index, num_input_vars> var_offsets,  // Offsets into the variable where the start of each input is stored
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



