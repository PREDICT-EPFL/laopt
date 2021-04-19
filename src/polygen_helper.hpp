// Eigen includes
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>
using namespace Eigen;

#include "unsupported/Eigen/AutoDiff"

#include <string_view>
#include <array>
#include <tuple>
#include <type_traits>

/************************************************************************
    Optimizations / TODO
 ************************************************************************/
// - Bring jacobian / hessian into alignment
// - 



/************************************************************************
    Helper functions
 ************************************************************************/

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

// Helper function to print out the argument types 
template <typename... Args>
void type_args(Args... args)
{
    int i=0;
    (void)std::initializer_list<int>{ 
        (
            std::cout << "arg " << i << " " << type_name<Args>() << std::endl,
            i++,
            0
        )...
    };
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
    static constexpr auto name##_o = offset; /* Offset */ \
    template<typename T> \
    constexpr auto name##_get(T x) {return x.template segment<size>(offset);} /* Offset */ \
    static constexpr int name##_len = size;

// template<typename T, typename = std::enable_if_t<std::is_const<T>::value>>
// constexpr auto _map_matrix(map_from x)
// {
//     return Eigen::Map<const Eigen::Matrix<scalar_t, name##_len, name##_numvecs>>((x.template segment<size>(offset)).data());} /* Matrix representation */
// }

// Helper function to reshape an eigen matrix while maintaining const'ness
template <typename map_to, typename scalar_t>
constexpr auto _map_matrix(const scalar_t* x) {
    return Eigen::Map<const map_to>(x);
}

template <typename map_to, typename scalar_t>
constexpr auto _map_matrix(scalar_t* x) {
    return Eigen::Map<map_to>(x);
}


#define DECLARE_VAR4(name, offset, size, number) \
    static constexpr auto name##_mat_size = size*number; /* Size of the vectorized variable */ \
    static constexpr auto name##_numvecs = number; /* Number of vectors */ \
    static constexpr auto name##_o(int col) {return offset + size * col;} /* Offset */ \
    static constexpr int name##_len = size; \
    template <typename T> \
    constexpr auto name##_get(T& x, int col) \
        {return x.template segment<size>(offset + size * col);} /* Offset */ \
    template <typename T> \
    constexpr auto name##_get_matrix(T& x) \
        {return _map_matrix<Eigen::Matrix<scalar_t, name##_len, name##_numvecs>>(x.template segment<size*number>(offset).data());}

    // template<typename T> \
    // constexpr auto name##_get_matrix(T& x) \
    //     {return Eigen::Map<const Eigen::Matrix<scalar_t, name##_len, name##_numvecs>>((x.template segment<size>(offset)).data());} /* Matrix representation */ 


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



/*************************************************************
    Declares a templated functor for member function in the 
    derived class
    
    This needs to be done in a macro because we can't take a
    pointer to a templated function, and we need to be
    able to call this wrapped function with a variety
    of scalar types.
 *************************************************************/
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


/*************************************************************
    Meta-programming helper functions
 *************************************************************/

// Sum the inputs to get total number of inputs
template<int... S>
constexpr int sum_template() {
    int result = 0;
    for(auto s : { S... }) result += s;
    return result;
}

// Build an cumulative array at compile time
template <typename T, T... args>
constexpr std::array<T, sizeof...(args)> make_cumarray()
{
    T offset = 0;
    std::array<T, sizeof...(args)> cum = 
    {
        (
            offset += args,
            offset - args
        )...
    };
    return cum;
}

// Build an array at compile time
template <typename T, typename... Args>
constexpr std::array<T, sizeof...(Args)> make_array(Args... args)
{
    return {args...};
}

#define CALL_MEMBER_FN(object,ptrToMember) ((object).*(ptrToMember))





/*************************************************************
    The main Jacobian class

    Called as:

    Evaluate the function
        function(arg1, arg2, ..., argN, out)

    Return the jacobian blockwise:
        jacobian_blockwise(arg1, arg2, ..., argN, out, Jac1, Jac2, ..., JacN)

    Return the jacobian in a dense matrix
        function(arg1, arg2, ..., argN, out, jacobian)

    Read the input arguments from variable at the given offset locations
        function({offset1, offset2, ...}, variable, out)

    Read the input arguments from variable at the given offset locations
    and write the output into out(eq_offset) and the jacobian into the blocks
    jacobian(eq_offset, offset_i) of the dense jacobian matrix
        function({offset1, offset2, ..., offsetN}, eq_offset, variable, out, jacobian)

    Read the input arguments from variable at the given offset locations
    and write the output into out(eq_offset) and the jacobian into the blocks
    jacobian(eq_offset, col_offset_i). jacobian here is a sparse matrix
        function({offset1, offset2, ..., offsetN}, {col_offset1, col_offset2, ...}, eq_offset, variable, out, jacobian)

 *************************************************************/

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
    EIGEN_STRONG_INLINE void jacobian_blockwise(
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

    // // Call format:
    // //   function(arg1, arg2, ..., argN, out, callback)
    // //   callback in a function that is passed the jacobian rowwise
    // template<typename Function>
    // EIGEN_STRONG_INLINE void operator()(
    //     const Ref<const Matrix<scalar_t, input_sizes, 1>>... args,
    //     Ref<Matrix<scalar_t, num_outputs, 1>> out,
    //     Function jacobian_callback
    //     ) const noexcept
    // {
    //     AD_output_t _out;

    //     // Convert to AD variables for the inputs and call our function
    //     seed_and_call(make_ad<input_sizes>(args)..., _out);

    //     // Copy Jacobian into output variables
    //     for(int i=0; i<num_outputs; i++)
    //     {
    //         out(i) = _out[i].value();
    //         Eigen::Ref<Eigen::Matrix<scalar_t, num_inputs, 1>> deriv = _out[i].derivatives();
    //         jacobian_callback.template set_jacobian<decltype(deriv), input_sizes...>(deriv, i);
    //     }
    // }

    // Call format:
    //   function(arg1, arg2, ..., argN, out, jacobian)
    //   Write jacobian into a dense matrix [Jarg1, Jarg2, Jarg3...]
    EIGEN_STRONG_INLINE void operator()(
        const Ref<const Matrix<scalar_t, input_sizes, 1>>... args,
        Ref<Matrix<scalar_t, num_outputs, 1>> out,
        Ref<Matrix<scalar_t, num_outputs, num_inputs>> jacobian
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

            int offset = 0;
            (void)std::initializer_list<int>{ 
                (
                    jacobian.row(i).template segment<input_sizes>(offset)
                        = deriv.template segment<input_sizes>(offset), 
                    offset += input_sizes,
                    0
                )... 
            };
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


/*************************************************************************
    Hessian
 *************************************************************************/

// Macro to wrap a member function and then create a hessian object
//   class_name,  Name of class containing the function to wrap \
//   func_name,  Name of member function (without _) to call \
//   _input_sizes...  Sizes of the function inputs
#define MAKE_HESSIAN(class_name, func_name) \
template<int... input_sizes> \
struct func_name##Hessian : public Hessian<func_name##Hessian, scalar_t, input_sizes...> \
{ \
    const class_name<scalar_t>* obj; \
    func_name##Hessian(const class_name<scalar_t>* parent) : obj(parent) {}; \
    template<typename T> \
    EIGEN_STRONG_INLINE void eval(const Eigen::Ref<const Eigen::Matrix<T, input_sizes, 1>>... x, T &y) const \
    { \
        obj->template _##func_name<T>(x..., y); \
    } \
}


template<template <int...> class Derived, typename scalar_t, int... input_sizes>
struct Hessian
{
public:
    using Child = Derived<input_sizes...>;

    // Total number of inputs
    static constexpr int num_inputs = sum_template<input_sizes...>();
    // Number of input vector variables
    static constexpr int num_input_vars = sizeof...(input_sizes);

    // Second order derivative
    using Derivatives = Eigen::Matrix<scalar_t, num_inputs, 1>;
    using ADScalar = Eigen::AutoDiffScalar<Derivatives>;
    using outerDerivatives = Eigen::Matrix<ADScalar, num_inputs, 1>;
    using outerADScalar = Eigen::AutoDiffScalar<outerDerivatives>;
    using Hessian_t = Eigen::Matrix<scalar_t, num_inputs, num_inputs>;

    /****************************************************************************
     Calls for dense, contiguous outputs
     ****************************************************************************/

    // Evaluate the function
    EIGEN_STRONG_INLINE scalar_t operator()(
        const Eigen::Ref<const Eigen::Matrix<scalar_t, input_sizes, 1>>... x) const noexcept
    {
        scalar_t y;
        static_cast<const Child*>(this)->template eval<scalar_t>(x..., y);
        return y;
    }    

    // Compute the gradient
    EIGEN_STRONG_INLINE scalar_t operator()(
        const Eigen::Ref<const Eigen::Matrix<scalar_t, input_sizes, 1>>... x,
        Eigen::Ref<Eigen::Matrix<scalar_t, num_inputs, 1>> gradient) const noexcept
    {
        return gradient_impl(make_ad<input_sizes>(x)..., gradient);
    }

    // Compute the gradient and the hessian
    EIGEN_STRONG_INLINE scalar_t operator()(
        const Eigen::Ref<const Eigen::Matrix<scalar_t, input_sizes, 1>>... x,
        Eigen::Ref<Eigen::Matrix<scalar_t, num_inputs, 1>> gradient,
        Eigen::Ref<Eigen::Matrix<scalar_t, num_inputs, num_inputs>> hessian) const noexcept
    {
        return hessian_impl(make_ad2<input_sizes>(x)..., gradient, hessian);
    }

    /********************************************************
     * Call formats taking a variable vector + offsets
     ********************************************************/

    // Call format:
    //   out = function({offset1, offset2, ...}, variable)
    // Inputs taken from variable[offest_i]
    template<typename variable_t, 
             typename Indices = std::make_integer_sequence<Eigen::Index, num_input_vars>>
    EIGEN_STRONG_INLINE scalar_t operator()(
        const std::array<Eigen::Index, num_input_vars> var_offsets,  // Offsets into the variable where the start of each input are stored
        const Eigen::Ref<variable_t>& var) const noexcept
    {
        return call_array<variable_t>(var_offsets, var, Indices{});
    }

    // Call format:
    //   out = function({offset1, offset2, ...}, variable, gradient)
    // Inputs taken from variable[offest_i]
    // Gradient written to gradient[offset_i]
    template<typename variable_t, typename gradient_t,
             typename Indices = std::make_integer_sequence<Eigen::Index, num_input_vars>>
    EIGEN_STRONG_INLINE scalar_t operator()(
        const std::array<Eigen::Index, num_input_vars> var_offsets,  // Offsets into the variable where the start of each input are stored
        const Eigen::Ref<variable_t>& var,
        Eigen::Ref<gradient_t>& gradient,
        const bool addValue = false ) // If true, doesn't overwrite the gradient, but adds to it
        const noexcept
    {
        return call_array<variable_t, gradient_t>(var_offsets, var, gradient, Indices{}, addValue);
    }

    // Call format:
    //   out = function({offset1, offset2, ...}, variable, gradient, hessian)
    // Inputs taken from variable[offest_i]
    // Gradient written to gradient
    // Hessian written to hessian
    template<typename variable_t, typename Indices = std::make_integer_sequence<Eigen::Index, num_input_vars>>
    EIGEN_STRONG_INLINE scalar_t operator()(
        const std::array<Eigen::Index, num_input_vars> var_offsets,  // Offsets into the variable where the start of each input are stored
        const Eigen::MatrixBase<variable_t>& var,
        Eigen::Ref<Eigen::Matrix<scalar_t, num_inputs, 1>> gradient,
        Eigen::Ref<Eigen::Matrix<scalar_t, num_inputs, num_inputs>> hessian) const noexcept
    {
        return call_array_dense(var_offsets, var, gradient, hessian, Indices{});
    }


    // Call format:
    //   out = function({offset1, offset2, ...}, variable, gradient, {h_offset1, h_offset2, ...}, hessian)
    // Inputs taken from variable[offest_i]
    // Gradient written to gradient[offset_i]
    // Hessian written to the block of the sparse hessian starting at h_offset_i
    template<typename variable_t, typename gradient_t, 
             typename Indices = std::make_integer_sequence<Eigen::Index, num_input_vars>>
    EIGEN_STRONG_INLINE scalar_t operator()(
        const std::array<Eigen::Index, num_input_vars> var_offsets,  // Offsets into the variable where the start of each input are stored
        const Eigen::Ref<variable_t>& var,
        Eigen::Ref<gradient_t>& gradient,
        const Eigen::Ref<const Eigen::Matrix<Eigen::Index, num_input_vars, num_input_vars>>& hess_offsets, // Offsets into the hessian value array
        Eigen::Ref<Eigen::SparseMatrix<scalar_t>> hessian,
        const bool addValue = false // If true, adds to gradient, rather than overwriting
        ) const noexcept
    {
        Eigen::Matrix<scalar_t, num_inputs, 1> _gradient;
        Eigen::Matrix<scalar_t, num_inputs, num_inputs> _hessian;
        scalar_t val = call_array_dense(var_offsets, var, _gradient, _hessian, Indices{});

        // Copy the contiguous gradient and hessian into the right locations
        contiguous_to_array(var_offsets, _gradient, gradient, Indices{});

        // Copy hessian into output locations
        using MatrixScalar = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;
        using Map_t = Eigen::Map<MatrixScalar, 0, Eigen::Stride<Eigen::Dynamic, 1>>;
        using Stride_t = Eigen::Stride<Eigen::Dynamic, 1>;

        Eigen::Index var_sizes[sizeof...(input_sizes)] = {input_sizes...};
        Eigen::Index row_offset = 0;
        Eigen::Index col_offset = 0;
        Eigen::Index nnz = 0;
        scalar_t* data;

        for(int row=0; row<num_input_vars; row++)
        {
            col_offset = 0;
            for(int col=0; col<num_input_vars; col++)
            {
                // We use map to produce a dense view into the variable/variable block and then copy over from the dense hessian
                data = hessian.valuePtr() + hess_offsets(row, col);  // Pre-computed offset into the valueptr
                nnz = hessian.outerIndexPtr()[var_offsets[col]+1] - hessian.outerIndexPtr()[var_offsets[col]]; // nnz of this column
                if(addValue)
                    Map_t(data, var_sizes[row], var_sizes[col], Stride_t(nnz, 1)) 
                        += _hessian.block(row_offset, col_offset, var_sizes[row], var_sizes[col]);
                else
                    Map_t(data, var_sizes[row], var_sizes[col], Stride_t(nnz, 1)) 
                        = _hessian.block(row_offset, col_offset, var_sizes[row], var_sizes[col]);

                col_offset += var_sizes[col];
            }
            row_offset += var_sizes[row];
        }
        // monkey is the best

        return val;
    }

    // Copy a continuous array to the given block locations
    template<typename variable_t, Eigen::Index... ind>
    EIGEN_STRONG_INLINE void contiguous_to_array(
        const std::array<Eigen::Index, num_input_vars> var_offsets,
        const Eigen::Ref<const Eigen::Matrix<scalar_t, num_inputs, 1>>& contiguous,
        Eigen::MatrixBase<variable_t>& target,
        std::integer_sequence<Eigen::Index, ind...>,
        const bool addValue = false // If true, adds rather than overwriting
        ) const noexcept
    {
        if(addValue)
        {
            int offset = 0;
            (void)std::initializer_list<int>{ 
                (
                    target.template segment<input_sizes>(var_offsets[ind]) += contiguous.template segment<input_sizes>(offset),
                    offset += input_sizes,
                    0
                )...
            };
        } else
        {
            int offset = 0;
            (void)std::initializer_list<int>{ 
                (
                    target.template segment<input_sizes>(var_offsets[ind]) = contiguous.template segment<input_sizes>(offset),
                    offset += input_sizes,
                    0
                )...
            };
        }
    }


    // Call format:
    //   out = function({offset1, offset2, ...}, variable, gradient, hessian)
    // Inputs taken from variable[offest_i]
    // Gradient written to gradient[offset_i]
    // Hessian written to hessian[offset_i, offset_j]
    //
    // hessian must be a sparse matrix
    // template<typename variable_t, typename hessian_t,
    //          typename Indices = std::make_integer_sequence<Eigen::Index, num_input_vars>>
    // EIGEN_STRONG_INLINE scalar_t operator()(
    //     const std::array<Eigen::Index, num_input_vars> var_offsets,  // Offsets into the variable where the start of each input are stored
    //     const Eigen::MatrixBase<variable_t>& var,
    //     Eigen::MatrixBase<variable_t>& gradient,
    //     Eigen::Ref<Eigen::SparseMatrix<scalar_t>> hessian) const noexcept
    // {
    //     return call_array(var_offsets, var, gradient, hessian, Indices{});
    // }


private:
    // Read inputs from variable[var_offsets[ind]] and return a dense gradient and hessian
    template<typename variable_t, Eigen::Index... ind>
    EIGEN_STRONG_INLINE scalar_t call_array_dense(
        const std::array<Eigen::Index, num_input_vars> var_offsets,  // Offsets into the variable where the start of each input is stored
        const Eigen::MatrixBase<variable_t>& var,
        Eigen::Ref<Eigen::Matrix<scalar_t, num_inputs, 1>> gradient,
        Eigen::Ref<Eigen::Matrix<scalar_t, num_inputs, num_inputs>> hessian,
        std::integer_sequence<Eigen::Index, ind...>) const noexcept
    {
        return this->operator()(var.template segment<input_sizes>(var_offsets[ind])..., gradient, hessian);
    }


    template<typename variable_t, Eigen::Index... ind>
    EIGEN_STRONG_INLINE typename variable_t::Scalar call_array(
        const std::array<Eigen::Index, num_input_vars> var_offsets,  // Offsets into the variable where the start of each input is stored
        const Eigen::Ref<variable_t>& var,
        std::integer_sequence<Eigen::Index, ind...>) const noexcept
    {
        return this->operator()(var.template segment<input_sizes>(var_offsets[ind])...);
    }

    template<typename variable_t, typename gradient_t, Eigen::Index... ind>
    EIGEN_STRONG_INLINE typename variable_t::Scalar call_array(
        const std::array<Eigen::Index, num_input_vars> var_offsets,  // Offsets into the variable where the start of each input is stored
        const Eigen::Ref<variable_t>& var,
        Eigen::Ref<gradient_t>& gradient,
        std::integer_sequence<Eigen::Index, ind...>,
        const bool addValue = false // If true, adds to gradient, rather than overwriting
        ) const noexcept
    {
        Eigen::Matrix<scalar_t, num_inputs, 1> _gradient;
        scalar_t y = this->operator()(var.template segment<input_sizes>(var_offsets[ind])..., _gradient);

        // Copy gradient into output locations
        int offset = 0;
        if(addValue)
        {
            (void)std::initializer_list<int>{ 
                (
                    gradient.template segment<input_sizes>(var_offsets[ind]) += _gradient.template segment<input_sizes>(offset),
                    offset += input_sizes,
                    0
                )...
            };            
        } else
        {
            (void)std::initializer_list<int>{ 
                (
                    gradient.template segment<input_sizes>(var_offsets[ind]) = _gradient.template segment<input_sizes>(offset),
                    offset += input_sizes,
                    0
                )...
            };            
        }

        return y;
    }

    template<typename variable_t, typename hessian_t, Eigen::Index... ind>
    EIGEN_STRONG_INLINE typename variable_t::Scalar call_array(
        const std::array<Eigen::Index, num_input_vars> var_offsets,  // Offsets into the variable where the start of each input is stored
        const Eigen::MatrixBase<variable_t>& var,
        Eigen::MatrixBase<variable_t>& gradient,
        Eigen::MatrixBase<hessian_t>& hessian,
        std::integer_sequence<Eigen::Index, ind...>) const noexcept
    {
        Eigen::Matrix<scalar_t, num_inputs, 1> _gradient;
        Eigen::Matrix<scalar_t, num_inputs, num_inputs> _hessian;
        scalar_t y = this->operator()(var.template segment<input_sizes>(var_offsets[ind])..., _gradient, _hessian);

        // Copy gradient into output locations
        int offset = 0;
        (void)std::initializer_list<int>{ 
            (
                gradient.template segment<input_sizes>(var_offsets[ind]) = _gradient.template segment<input_sizes>(offset),
                offset += input_sizes,
                0
            )...
        };

        // Copy hessian into output locations
        Eigen::Index var_sizes[sizeof...(input_sizes)] = {input_sizes...};
        Eigen::Index row_offset = 0;
        Eigen::Index col_offset = 0;
        for(int row=0; row<num_input_vars; row++)
        {
            col_offset = 0;
            for(int col=0; col<num_input_vars; col++)
            {
                hessian.block(var_offsets[row], var_offsets[col], var_sizes[row], var_sizes[col])
                    = _hessian.block(row_offset, col_offset, var_sizes[row], var_sizes[col]);
                col_offset += var_sizes[col];
            }
            row_offset += var_sizes[row];
        }

        return y;
    }









private:

    // Output functions: How to copy the gradient and hessian into various output forms
    //   Hessian
    //   - block offsets into sparse matrix

    // Take a vector input and return a AD version of the vector
    template<int n>
    EIGEN_STRONG_INLINE Matrix<ADScalar, n, 1> 
        make_ad(const Ref<const Matrix<scalar_t, n, 1>> x) const
    {
        Matrix<ADScalar, n, 1> y;
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

    // Take a vector input and return a AD version of the vector
    template<int n>
    EIGEN_STRONG_INLINE Eigen::Matrix<outerADScalar, n, 1> 
        make_ad2(const Eigen::Ref<const Eigen::Matrix<scalar_t, n, 1>> x) const
    {
        Eigen::Matrix<outerADScalar, n, 1> y;
        // y = x;
        for (int i=0; i<n; i++) {
            y(i).value().value() = x(i);
            y(i).value().derivatives().setZero();
            y(i).derivatives().setZero();
            for (int j = 0; j < n; j++) {
                y(i).derivatives()(j).derivatives().setZero();
            }
        }
        return y;
    }

    // Sets the input derivatives to the identity. 
    // Assumes that the derivative matrix is initially zero
    template <typename vec>
    constexpr int AD_Seed2(vec &x, int offset) const
    {
        for (int i=0; i<x.rows(); i++)
        {
            x(i).value().derivatives().coeffRef(i + offset) = 1;
            x(i).derivatives().coeffRef(i + offset) = 1;
        }

        return offset + x.rows();
    }

    EIGEN_STRONG_INLINE scalar_t hessian_impl(
        Eigen::Matrix<outerADScalar, input_sizes, 1>... x,
        Eigen::Ref<Eigen::Matrix<scalar_t, num_inputs, 1>> gradient,
        Eigen::Ref<Eigen::Matrix<scalar_t, num_inputs, num_inputs>> hessian) const
    {
        // Set derivative equal to identity
        int offset = 0;
        (void)std::initializer_list<int>{ 
            (
                offset = AD_Seed2(x, offset), // Set to unit vectors
                0
            )...
        };

        outerADScalar y;
        static_cast<const Child*>(this)->template eval<outerADScalar>(x..., y);

        gradient = y.value().derivatives();

        for (int i = 0; i < num_inputs; i++) {
            hessian.template middleRows<1>(i) = y.derivatives()(i).derivatives().transpose();
        }

        return y.value().value();
    }


    EIGEN_STRONG_INLINE scalar_t gradient_impl(
        Eigen::Matrix<ADScalar, input_sizes, 1>... x,
        Eigen::Ref<Eigen::Matrix<scalar_t, num_inputs, 1>> gradient) const
    {
        // Set derivative equal to identity
        int offset = 0;
        (void)std::initializer_list<int>{ 
            (
                offset = AD_Seed(x, offset), // Set to unit vectors
                0
            )...
        };

        ADScalar y;
        static_cast<const Child*>(this)->template eval<ADScalar>(x..., y);

        gradient = y.derivatives();
        return y.value();
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
                trip.emplace_back(i, j, 1.0);    
    }

    J.setFromTriplets(trip.begin(), trip.end());
    J.makeCompressed();    
}



