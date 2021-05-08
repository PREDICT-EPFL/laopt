#include <iostream>
#include <tuple>
#include <array>

#include "Eigen/Dense"
#include <Eigen/Sparse>
#include "unsupported/Eigen/AutoDiff"

using namespace Eigen;

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

// Build an array at compile time
template <typename T, typename... Args>
constexpr std::array<T, sizeof...(Args)> make_array(Args... args)
{
    return {args...};
}

// Helper function to reshape an eigen matrix while maintaining const'ness
template <typename map_to, typename scalar_t>
constexpr auto _map_matrix(const scalar_t* x) {
    return Eigen::Map<const map_to>(x);
}

template <typename map_to, typename scalar_t>
constexpr auto _map_matrix(scalar_t* x) {
    return Eigen::Map<map_to>(x);
}


/*************************************************************
   Short names to pass constant vectors and writable vectors
 *************************************************************/
template<typename T, std::size_t n>
using cVec = const Eigen::Ref<const Eigen::Matrix<T, n, 1>>;

template<typename T, std::size_t n>
using Vec = Eigen::Ref<Eigen::Matrix<T, n, 1>>;


/*************************************************************
   Jacobian computation
 *************************************************************/

// template<typename func,
//       typename scalar_t,
//       typename param_t, 
//       int num_outputs_, 
//       int... input_sizes>
// struct FunctionTraits
// {
//  static const auto num_outputs = num_outputs_;
// };

template<typename scalar_t, int num_outputs, int num_inputs>
struct jacobian_return_t
{
    Eigen::Matrix<scalar_t, num_outputs, 1> val;
    Eigen::Matrix<scalar_t, num_outputs, num_inputs> jacobian;
};

template<typename scalar_t, int num_outputs, int num_inputs>
struct hessian_return_t
{
    Eigen::Matrix<scalar_t, num_outputs, 1> val;
    Eigen::Matrix<scalar_t, num_outputs, num_inputs> jacobian;
    using hessian_t = Eigen::Matrix<scalar_t, num_inputs, num_inputs>;
    std::array<hessian_t, num_outputs> hessian;
};

// template<typename>
// struct Jacobian;

template<typename Func, typename scalar_t, typename param_t, int num_outputs_, int... input_sizes>
struct Jacobian // < FunctionTraits<Func, scalar_t, param_t, num_outputs, input_sizes...> >
{
    static constexpr int num_inputs = sum_template<input_sizes...>();  // Total number of inputs
    static constexpr int num_input_vars = sizeof...(input_sizes);  // Number of input vector variables
    static constexpr int num_outputs = num_outputs_;

    // First order derivative
    using AD_scalar = Eigen::AutoDiffScalar<Eigen::Matrix<scalar_t, num_inputs, 1>>;
    using AD_output_t = Eigen::Matrix<AD_scalar, num_outputs, 1>;  

    // Second order derivative
    using outerDerivatives = Eigen::Matrix<AD_scalar, num_inputs, 1>;
    using outerADScalar = Eigen::AutoDiffScalar<outerDerivatives>;
    using outerAD_t = Eigen::Matrix<outerADScalar, num_outputs, 1>;  

    /*
        Evaluate the function
     */
    static EIGEN_STRONG_INLINE auto eval(
        const param_t& param,
        const Eigen::Ref<const Eigen::Matrix<scalar_t, input_sizes, 1>>&... args) 
        noexcept
    {
        Eigen::Matrix<scalar_t, num_outputs, 1> out;
        Func::template eval<scalar_t>(param, out, args...);
        return out;
    }

    /*
        Evaluate the function, and its jacobian
     */
    static EIGEN_STRONG_INLINE auto jac(
        const param_t& param,
        const Ref<const Matrix<scalar_t, input_sizes, 1>>&... args) 
        noexcept
    {
        AD_output_t _out;

        // Convert to AD variables for the inputs and call our function
        seed_and_call(make_ad<input_sizes>(args)..., _out, param);

        // Copy Jacobian into output variables
        jacobian_return_t<scalar_t, num_outputs, num_inputs> ret;
        for(int i=0; i<num_outputs; i++)
        {
            ret.val(i) = _out[i].value();
            ret.jacobian.row(i) = _out[i].derivatives();
        }

        return ret;
    }

    /*
        Evaluate the function, its jacobian and hessian
     */
    static EIGEN_STRONG_INLINE auto hessian(
        const param_t& param,
        const Ref<const Matrix<scalar_t, input_sizes, 1>>&... args) 
        noexcept
    {
        outerAD_t _out;

        // Convert to AD variables for the inputs and call our function
        seed_and_call2(make_ad2<input_sizes>(args)..., _out, param);

        // Copy Hessian into output variables
        hessian_return_t<scalar_t, num_outputs, num_inputs> ret;
        for(int i=0; i<num_outputs; i++)
        {
            ret.val(i) = _out[i].value().value();
            ret.jacobian.row(i) = _out[i].value().derivatives();
            for (int j = 0; j < num_inputs; j++) {
                ret.hessian[i].template middleRows<1>(j) = _out[i].derivatives()(j).derivatives().transpose();
            }
        }

        return ret;
    }


private:

    /*********
     Jacobians 
     *********/

    // Take a vector input and return a AD version of the vector
    template<int n>
    static EIGEN_STRONG_INLINE Matrix<AD_scalar, n, 1> 
        make_ad(const Ref<const Matrix<scalar_t, n, 1>> x)
    {
        Matrix<AD_scalar, n, 1> y;
        y = x;
        for (int i=0; i<y.rows(); i++) {
            y[i].derivatives().setZero();
        }
        return y;
    }

    static EIGEN_STRONG_INLINE void seed_and_call(
            Matrix<AD_scalar, input_sizes, 1>... args,
            Eigen::Ref<Eigen::Matrix<AD_scalar, num_outputs, 1>> out,
            const param_t& param)
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
        Func::template eval<AD_scalar>(param, out, args...);
    }

    // Sets the input derivatives to the identity. 
    // Assumes that the derivative matrix is initially zero
    template <typename vec>
    static constexpr int AD_Seed(vec &x, int offset)
    {
        for (int i=0; i<x.rows(); i++)
            x[i].derivatives().coeffRef(i + offset) = 1;
        return offset + x.rows();
    }


    /********
     Hessians 
     ********/

    // Take a vector input and return a AD version of the vector
    template<int n>
    static EIGEN_STRONG_INLINE Eigen::Matrix<outerADScalar, n, 1> 
        make_ad2(const Eigen::Ref<const Eigen::Matrix<scalar_t, n, 1>> x)
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
    static constexpr int AD_Seed2(vec &x, int offset)
    {
        for (int i=0; i<x.rows(); i++)
        {
            x(i).value().derivatives().coeffRef(i + offset) = 1;
            x(i).derivatives().coeffRef(i + offset) = 1;
        }

        return offset + x.rows();
    }

    static EIGEN_STRONG_INLINE void seed_and_call2(
        Eigen::Matrix<outerADScalar, input_sizes, 1>... args,
        Eigen::Ref<outerAD_t> out,
        const param_t& param)
    {
        // Set derivative equal to identity
        int offset = 0;
        (void)std::initializer_list<int>{ 
            (
                offset = AD_Seed2(args, offset), // Set to unit vectors
                0
            )...
        };

        // Call our function
        Func::template eval<outerADScalar>(param, out, args...);
    }

};


/*************************************************************
   Variables
 *************************************************************/

struct ZeroVariable_ // Fake type to indicate the start of the variable sequence
{
    static const std::size_t offset = 0;
    static const std::size_t len = 0;
    static const std::size_t num_vars = 0;
    static const std::size_t next = 0;

    static constexpr std::size_t var_index = 0;
};

template<std::size_t _len, std::size_t _num_vars, 
         typename Prev = ZeroVariable_> // Variable defined before this one (specifies ordering)
struct var_t
{
    static constexpr std::size_t len      = _len;      // Length of a variable of this type
    static constexpr std::size_t num_vars = _num_vars; // Number of variables in this set

    // Location of this variable in the NLP optimizer
    // var = [x1; x2; x3; ...]
    // var[offset] = this variable
    //
    // var{0} = x1, var{1} = x2, var{2} = x3 ...
    // var{index} = this variable

    static constexpr std::size_t offset    = Prev::next;               // Offset into the main variable
    static constexpr std::size_t next      = offset + len * num_vars;  // Offset where next variable should go
    static constexpr std::size_t var_index = Prev::var_index + Prev::num_vars; // Variable index

    // Get indexed offset
    static constexpr std::size_t o(int ind = 0) 
    {
        assert(ind < num);
        return offset + ind * len;
    }

    // Static programatic interface

    // Return ind variable segment of var
    template<typename Var>
    static EIGEN_STRONG_INLINE constexpr auto get(Var& var, int ind = 0)
    {
        return var.template segment<len>(offset + ind * len);
    }

    // User convenience interface

    // Return ind variable segment of var
    template<typename Var>
    EIGEN_STRONG_INLINE constexpr auto operator()(Var& var, int ind) const
    {
        return var.template segment<len>(offset + ind * len);
    }

    template<typename Var>
    EIGEN_STRONG_INLINE constexpr auto operator()(Var& var) const
    {
        return _map_matrix<Eigen::Matrix<typename Var::Scalar, len, num_vars>>(var.template segment<len * num_vars>(offset).data());
    }
};

// Represents an iterator over a var_t
template<typename variable_set, int start=0, int step=0>
struct iterator
{
    static constexpr std::size_t var_len = variable_set::len;

    // Return ind variable segment of var
    template<typename Var>
    static EIGEN_STRONG_INLINE constexpr auto get(Var& var, int ind = 0)
    {
        return variable_set::get(var, start + ind * step);
    }

    static EIGEN_STRONG_INLINE constexpr std::size_t offset(int ind = 0)
    {
        return variable_set::o(start + ind * step);
    }
};

// Sum the inputs to get total number of inputs
template<typename... Var>
constexpr std::size_t sum_variable_size() {
    int sum = 0;
    for(std::size_t num : { Var::len * Var::num_vars... })
        sum += num;
    return sum;
}



/*************************************************************
   Constraints
 *************************************************************/

/*************************************************************
    A single constraint, or collection of constraints of the same type.
    A collection is a constraint that's calling the same function 
    with the same variable sets, but with diffent indices
 *************************************************************/
template<typename, int, typename...>
struct con_t;

template<typename Func,      // The function to be called
         typename scalar_t_,
         typename param_t, 
         int num_outputs,    // Output size of the function
         int... input_sizes, // Input sizes of the function
         int num_iterations, // Number of iterations in the constraint collection
         typename... Var_t>  // Variable Index's (i.e., range objects)
struct con_t< Jacobian<Func, scalar_t_, param_t, num_outputs, input_sizes...>, num_iterations, Var_t...>
{
    using scalar_t = scalar_t_;
    using jacFunc = Jacobian<Func, scalar_t, param_t, num_outputs, input_sizes...>;

    // Total vector length required to store this constraint collection
    static constexpr std::size_t constraint_vector_length = num_iterations * num_outputs;

    // Index into the constraint vector where this constraint starts
    int constraint_offset = 0; 

    // Index into a compressed sparse jacobian where each (constraint, variable) block starts
    Eigen::Matrix<std::size_t, num_iterations, sizeof...(input_sizes)> jacobian_offsets;

    // Index into a compressed sparse hessian where each (variable, variable) block starts
    using hessian_offset_t = Eigen::Matrix<std::size_t, sizeof...(input_sizes), sizeof...(input_sizes)>;
    std::array<hessian_offset_t, num_iterations> hessian_offsets;

    con_t()
    {}

private:
    /*
        Functions to copy the constraints, jacobian and hessian into the global
        variables in dense and sparse formats
     */
    template<typename ret_t, typename constraint_t>
    EIGEN_STRONG_INLINE void copy_to_constraint(int iteration, ret_t& ret, constraint_t& con) const noexcept
    {
        con.template segment<num_outputs>(offset(iteration)) = ret.val;
    }

    template<typename ret_t, typename jacobian_t>
    EIGEN_STRONG_INLINE void copy_to_jacobian_dense(int iteration, ret_t& ret, jacobian_t& jac) const noexcept
    {
        // Write the jacobian into the dense jacobian matrix
        int var_offset = 0;
        (void)std::initializer_list<int>{ 
            (
                jac.template block<num_outputs, input_sizes>(offset(iteration), Var_t::offset(iteration))
                    = ret.jacobian.template block<num_outputs, input_sizes>(0, var_offset),
                var_offset += input_sizes,
                0
            )...
        };
    }

    template<typename ret_t>
    EIGEN_STRONG_INLINE void copy_to_jacobian_sparse(int iteration, ret_t& ret, Eigen::Ref<Eigen::SparseMatrix<scalar_t>> jac) const noexcept
    {
        // Write the jacobian into the sparse jacobian matrix
        int var_offset = 0;
        int var_num = 0;
        (void)std::initializer_list<int>{ 
            (
                get_sparse_block<num_outputs, Var_t::var_len>(jac, Var_t::offset(iteration), jacobian_offsets(iteration, var_num))
                    = ret.jacobian.template block<num_outputs, input_sizes>(0, var_offset),
                var_num++,
                var_offset += input_sizes,
                0
            )...
        };
    }

    template<typename ret_t, typename hessian_t, typename hessian_multiplier_t>
    EIGEN_STRONG_INLINE void copy_to_hessian_dense(int iteration, ret_t& ret, hessian_t& hessian, const hessian_multiplier_t& hessian_multiplier)
     const noexcept
    {
        // static std::array<std::size_t, sizeof...(Var_t)> var_offsets = {Var_t::offset(iteration)...};
        // static std::array<std::size_t, sizeof...(Var_t)> var_sizes = {Var_t::var_len...};

        // // Write the hessian into the dense hessian matrix
        // for(int output_index=0; output_index<num_outputs; output_index++)
        // {
        //     ret.hessian[output_index] *= hessian_multiplier(offset(iteration) + output_index);

        //     std::size_t row_offset = 0;
        //     for(int row=0; row<sizeof...(Var_t); row++)
        //     {
        //         std::size_t col_offset = 0;
        //         for(int col=0; col<sizeof...(Var_t); col++)
        //         {
        //             hessian.block(var_offsets[row], var_offsets[col], var_sizes[row], var_sizes[col])
        //                 += ret.hessian[output_index].block(row_offset, col_offset, var_sizes[row], var_sizes[col]);
        //             col_offset += var_sizes[col];
        //         }
        //         row_offset += var_sizes[row];
        //     }
        // }



        // Write the hessian into the dense hessian matrix
        for(int output_index=0; output_index<num_outputs; output_index++)
        {
            ret.hessian[output_index] *= hessian_multiplier(offset(iteration) + output_index);

            int var_offset = 0;
            (void)std::initializer_list<int>{ 
                (
                    // copy_dense_hessian_blockrow<input_sizes, variable_t::RowsAtCompileTime>(
                    copy_dense_hessian_blockrow(
                        // Rows of the global hessian for this variable
                        hessian.template block<input_sizes, hessian_t::ColsAtCompileTime>(Var_t::offset(iteration), 0), 
                        // Rows of the function hessian for this variable
                        ret.hessian[output_index].template block<input_sizes, jacFunc::num_inputs>(var_offset, 0),
                        iteration // Iteration number
                    ),
                    var_offset += input_sizes,
                    0
                )...
            };
        }
    }

    template<typename ret_t, typename hessian_multiplier_t>
    EIGEN_STRONG_INLINE void copy_to_hessian_sparse(int iteration, ret_t& ret, 
                                                    Eigen::Ref<Eigen::SparseMatrix<scalar_t>> hessian, 
                                                    const hessian_multiplier_t& hessian_multiplier) const noexcept
    {
        std::array<std::size_t, sizeof...(Var_t)> var_offsets = {Var_t::offset(iteration)...};
        std::array<std::size_t, sizeof...(Var_t)> var_sizes = {Var_t::var_len...};

        // Write the hessian into the sparse hessian matrix
        for(int output_index=0; output_index<num_outputs; output_index++)
        {
            ret.hessian[output_index] *= hessian_multiplier(offset(iteration) + output_index);

            std::size_t row_offset = 0;
            for(int row=0; row<sizeof...(Var_t); row++)
            {
                std::size_t col_offset = 0;
                for(int col=0; col<sizeof...(Var_t); col++)
                {
                     get_sparse_block_dynamic(hessian, var_sizes[row], var_sizes[col], var_offsets[col], hessian_offsets[iteration](row, col))
                        += ret.hessian[output_index].block(row_offset, col_offset, var_sizes[row], var_sizes[col]);
                    col_offset += var_sizes[col];
                }
                row_offset += var_sizes[row];
            }
        }
    }


public:
    /*
        Evaluate the constraint
     */
    template<typename variable_t, typename constraint_t>
    EIGEN_STRONG_INLINE void operator()(
        const param_t& param,
        const variable_t& var,
        constraint_t& con)
        const noexcept
    {
        for(int i=0; i<num_iterations; i++)
        {
            Func::template eval<scalar_t>(
                param, 
                con.template segment<num_outputs>(offset(i)), // Output
                Var_t::get(var, i) ...);  // Inputs
        }
    }

    /*
        Evaluate the constraint, and its jacobian in dense format
     */
    template<typename variable_t, typename constraint_t>
    EIGEN_STRONG_INLINE auto operator()(
        const param_t& param,
        const variable_t& var,
        constraint_t& con,
        Eigen::Ref<Eigen::Matrix<scalar_t, constraint_t::RowsAtCompileTime, variable_t::RowsAtCompileTime>> jac)
        const noexcept
    {
        for(int i=0; i<num_iterations; i++)
        {
            auto ret = jacFunc::jac(param, Var_t::get(var, i)...);
            copy_to_constraint(i, ret, con);
            copy_to_jacobian_dense(i, ret, jac);
        }
    }

    /*
        Evaluate the constraint, and its jacobian in sparse format
     */
    template<typename variable_t, typename constraint_t>
    EIGEN_STRONG_INLINE auto operator()(
        const param_t& param,
        const variable_t& var,
        constraint_t& con,
        Eigen::Ref<Eigen::SparseMatrix<scalar_t>> jac)
        const noexcept
    {
        for(int i=0; i<num_iterations; i++)
        {
            auto ret = jacFunc::jac(param, Var_t::get(var, i)...);
            copy_to_constraint(i, ret, con);
            copy_to_jacobian_sparse(i, ret, jac);
        }
    }

    /*
        Evaluate the constraint, its jacobian and hessian in dense format

        The hessian is added into the hessian matrix multiplied by 
        hessian_multiplier. (i.e., the Lagrange multipliers)
     */
    template<typename variable_t, typename constraint_t>
    EIGEN_STRONG_INLINE auto operator()(
        const param_t& param,
        const variable_t& var,
        constraint_t& con,
        Eigen::Ref<Eigen::Matrix<scalar_t, constraint_t::RowsAtCompileTime, variable_t::RowsAtCompileTime>> jac,
        Eigen::Ref<Eigen::Matrix<scalar_t, variable_t::RowsAtCompileTime, variable_t::RowsAtCompileTime>> hessian,
        const Eigen::Ref<const Eigen::Matrix<scalar_t, constraint_t::RowsAtCompileTime, 1>>& hessian_multiplier)
        const noexcept
    {
        for(int i=0; i<num_iterations; i++)
        {
            auto ret = jacFunc::hessian(param, Var_t::get(var, i)...);
            copy_to_constraint(i, ret, con);
            copy_to_jacobian_dense(i, ret, jac);
            copy_to_hessian_dense(i, ret, hessian, hessian_multiplier);
        }
    }

    /*
        Evaluate the constraint, its jacobian and hessian in sparse format

        The hessian is added into the hessian matrix multiplied by 
        hessian_multiplier. (i.e., the Lagrange multipliers)
     */
    template<typename variable_t, typename constraint_t>
    EIGEN_STRONG_INLINE auto operator()(
        const param_t& param,
        const variable_t& var,
        constraint_t& con,
        Eigen::Ref<Eigen::SparseMatrix<scalar_t>> jac,
        Eigen::Ref<Eigen::SparseMatrix<scalar_t>> hessian,
        const Eigen::Ref<const Eigen::Matrix<scalar_t, constraint_t::RowsAtCompileTime, 1>>& hessian_multiplier)
        const noexcept
    {
        for(int i=0; i<num_iterations; i++)
        {
            auto ret = jacFunc::hessian(param, Var_t::get(var, i)...);
            copy_to_constraint(i, ret, con);
            copy_to_jacobian_sparse(i, ret, jac);
            copy_to_hessian_sparse(i, ret, hessian, hessian_multiplier);
        }
    }





    /*
        Sets the offset, and returns the new offset
     */
    std::size_t set_offset(std::size_t offset_)
    {
        constraint_offset = offset_;
        return constraint_offset + num_outputs * num_iterations;
    }

    /*
        Return the offset into the constraint vector for the 
        ind'th constraint
     */
    std::size_t offset(int ind = 0) const
    {
        return constraint_offset + ind * num_outputs;
    }

    /*
        Adds the non-zero elements to the triplet vector for the given constraint
    */
    void jac_get_sparsity_triplet(std::vector<Eigen::Triplet<scalar_t>>& trip) const
    {
        for(int i=0; i<num_iterations; i++)
        {
            (void)std::initializer_list<int>{ 
                (
                    get_sparsity_triplet_block(trip, offset(i), num_outputs, Var_t::offset(i), Var_t::var_len),
                    0
                )...
            };
        }
    }

    /*
        Adds the non-zero elements to the triplet vector for the given constraint
    */
    void hessian_get_sparsity_triplet(std::vector<Eigen::Triplet<scalar_t>>& trip) const
    {
        for(int i=0; i<num_iterations; i++)
        {
            std::array<std::size_t, sizeof...(Var_t)> var_offsets = {Var_t::offset(i)...};
            std::array<std::size_t, sizeof...(Var_t)> var_sizes = {Var_t::var_len...};

            for(int row=0; row<sizeof...(Var_t); row++)
                for(int col=0; col<sizeof...(Var_t); col++)
                    get_sparsity_triplet_block(trip, var_offsets[row], var_sizes[row], var_offsets[col], var_sizes[col]);
        }
    }

    /*
        Set sparsity offsets. This can only be called once the global constraint structure is known 
        from constraints_t

        J is a compressed matrix whose structure was set by constraints_impl.initialize_sparse_jacobian

        !! This function is only called from the constraints constructor !!
     */
    void jac_set_sparsity_offsets(Eigen::SparseMatrix<scalar_t>& J)
    {
        for(int i=0; i<num_iterations; i++)
        {
            // Get the index at the start of each constraint / variable block
            int var_ind = 0;
            (void)std::initializer_list<int>{ 
                (
                    jacobian_offsets(i, var_ind) = J.coeff(offset(i), Var_t::offset(i)),
                    var_ind++,
                    0
                )...
            };
        }
    }

    /*
        Set sparsity offsets. This can only be called once the global constraint structure is known 
        from constraints_t

        H is a compressed matrix whose structure was set by constraints_impl.initialize_sparse_hessian

        !! This function is only called from the constraints constructor !!
     */
    void hessian_set_sparsity_offsets(Eigen::SparseMatrix<scalar_t>& H)
    {
        for(int i=0; i<num_iterations; i++)
        {
            std::array<std::size_t, sizeof...(Var_t)> var_offsets = {Var_t::offset(i)...};

            // Get the index at the start of each constraint / variable block
            for(int row=0; row<sizeof...(Var_t); row++)
                for(int col=0; col<sizeof...(Var_t); col++)
                {
                    hessian_offsets[i](row, col) = H.coeff(var_offsets[row], var_offsets[col]);
                }
        }
    }

    /*
        Print out the elements of the jacobian that this constraint 
        will impact for debugging
     */
    void jac_print_sparsity_structure(Eigen::SparseMatrix<scalar_t>& J)
    {
        scalar_t* values = J.valuePtr();
        for(int i=0; i<J.nonZeros(); i++)
            values[i] = -1.0;

        for(int i=0; i<num_iterations; i++)
        {
            // Write the jacobian into the sparse jacobian matrix
            int var_num = 0;
            (void)std::initializer_list<int>{ 
                (
                    get_sparse_block<num_outputs, Var_t::var_len>(J, Var_t::offset(i), jacobian_offsets(i, var_num)).array()
                        = (double)i + (double)(var_num+1) / 10.0,
                    var_num++,
                    0
                )...
            };
        }
    }


private:
    void get_sparsity_triplet_block(std::vector<Eigen::Triplet<scalar_t>>& trip, int start_row, int row_len, int start_col, int col_len) const
    {
        for(int row=start_row; row<start_row + row_len; row++)
            for(int col=start_col; col<start_col + col_len; col++)
                trip.emplace_back(row, col, 1.0);
    }

    /*
        Returns a dense map to a block of the hessian for a particular pair of variables
     */
    template<std::size_t row_len, std::size_t col_len>
    Eigen::Map<Eigen::Matrix<scalar_t, row_len, col_len>, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
         get_sparse_block(Eigen::Ref<Eigen::SparseMatrix<scalar_t>> sparse_matrix,
                            int col, int valueStart) const
    {
        using Map_t = Eigen::Map<Eigen::Matrix<scalar_t, row_len, col_len>, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;
        using Stride_t = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;

        int nnz = sparse_matrix.outerIndexPtr()[col+1] - sparse_matrix.outerIndexPtr()[col];
        scalar_t* data = sparse_matrix.valuePtr() + valueStart;

        // Terrible fix: Eigen seems to mixup inner and outer strides for row vectors
        int outerStride, innerStride;
        if(row_len == 1)
        {
            outerStride = 1;
            innerStride = nnz;
        } else
        {
            innerStride = 1;
            outerStride = nnz;
        }

        // return Map_t(data, Stride_t(nnz, 2));
        return Map_t(data, Stride_t(outerStride, innerStride));
    }

    Eigen::Map<Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
         get_sparse_block_dynamic(Eigen::Ref<Eigen::SparseMatrix<scalar_t>> sparse_matrix,
                            std::size_t row_len, std::size_t col_len, int col, int valueStart) const
    {
        using Map_t = Eigen::Map<Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;
        using Stride_t = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;

        int nnz = sparse_matrix.outerIndexPtr()[col+1] - sparse_matrix.outerIndexPtr()[col];
        scalar_t* data = sparse_matrix.valuePtr() + valueStart;

        // Terrible fix: Eigen seems to mixup inner and outer strides for row vectors
        int outerStride, innerStride;
        // if(row_len == 1)
        // {
        //     outerStride = 1;
        //     innerStride = nnz;
        // } else
        // {
            innerStride = 1;
            outerStride = nnz;
        // }

        // return Map_t(data, Stride_t(nnz, 2));
        // if(row_len == 1)
        // {
            // std::cout << "------------------------ get_sparse_block_dynamic ---------------------\n";
            // std::cout << "Map = \n" << Map_t(data, row_len, col_len, Stride_t(outerStride, innerStride)) << std::endl;
            // std::cout << "row_len = " << row_len << " col_len = " << col_len << " col = " << col << " nnz = " << nnz << " valueStart = " << valueStart << std::endl;
            // std::cout << Eigen::MatrixXd(sparse_matrix) << std::endl;
            // std::cout << "------------------------ ------------------------ ---------------------\n";
        // }
        return Map_t(data, row_len, col_len, Stride_t(outerStride, innerStride));
    }


    /*
        Copies a block-row of a dense hessian into the right location of the full hessian
     */
    template<typename target_t,   // Length of the variable row being copied
             typename source_t> // Total number of variables in the target hessian
    EIGEN_STRONG_INLINE void copy_dense_hessian_blockrow(
        target_t&& target_hessian_row,
        const source_t& source_hessian_row,
        // Eigen::Ref<Eigen::Matrix<scalar_t, num_rows, num_inputs>> target_hessian_row,
        // const Eigen::Ref<const Eigen::Matrix<scalar_t, num_rows, jacFunc::num_inputs>>& source_hessian_row,
        const int iteration_number) const
    {
        constexpr auto num_rows = target_t::RowsAtCompileTime;

        // Write the hessian into the dense hessian matrix
        int var_offset = 0;
        (void)std::initializer_list<int>{ 
            (
                target_hessian_row.template block<num_rows, input_sizes>(0, Var_t::offset(iteration_number))
                    += source_hessian_row.template block<num_rows, input_sizes>(0, var_offset),
                var_offset += input_sizes,
                0
            )...
        };        
    }

};

/*************************************************************
    A set of constraints of different types
 *************************************************************/
template<std::size_t num_variables, typename cons_t, std::size_t... ind>
struct constraints_impl
{
    // Number of constraint collections. NOT the length of the constraint vector.
    static const std::size_t num_constraints = std::tuple_size<cons_t>::value; 

    cons_t cons; // Tuple of constraints

    using scalar_t = typename std::tuple_element_t<0, cons_t>::scalar_t;

    constraints_impl(const cons_t cons_) : cons(cons_)
    {
        // Set the offsets / ordering
        int offset = 0;
        (void)std::initializer_list<int>{ 
            (
                offset = std::get<ind>(cons).set_offset(offset),
                0
            )...
        };

        // Store the offset locations for each block of the jacobian
        SparseMatrix<scalar_t> J(constraint_vector_length(), num_variables);
        initialize_sparse_jacobian(J);

        scalar_t* values = J.valuePtr();
        for(int i=0; i<J.nonZeros(); i++)
            values[i] = i;

        (void)std::initializer_list<int>{ 
            (
                std::get<ind>(cons).jac_set_sparsity_offsets(J),
                0
            )...
        };


        // Store the offset locations for each block of the hessian
        SparseMatrix<scalar_t> H(num_variables, num_variables);
        initialize_sparse_hessian(H);

        values = H.valuePtr();
        for(int i=0; i<H.nonZeros(); i++)
            values[i] = i;

        (void)std::initializer_list<int>{ 
            (
                std::get<ind>(cons).hessian_set_sparsity_offsets(H),
                0
            )...
        };
    }

    /*
        Evaluate all constraints
     */
    template<typename param_t, typename variable_t, typename constraint_t>
    EIGEN_STRONG_INLINE void operator()(
        const param_t& param,
        const variable_t& var,
        constraint_t& con)
        const noexcept
    {
        (void)std::initializer_list<int>{ 
            (
                std::get<ind>(cons)(param, var, con),
                0
            )...
        };
    }

    /*
        Evaluate all constraints and dense jacobians
     */
    template<typename param_t, typename variable_t, typename constraint_t>
    EIGEN_STRONG_INLINE auto operator()(
        const param_t& param,
        const variable_t& var,
        constraint_t& con,
        Eigen::Ref<Eigen::Matrix<typename variable_t::Scalar, constraint_t::RowsAtCompileTime, variable_t::RowsAtCompileTime>> jac)
        const noexcept
    {
        (void)std::initializer_list<int>{ 
            (
                std::get<ind>(cons)(param, var, con, jac),
                0
            )...
        };
    }

    /*
        Evaluate all constraints and their jacobian in sparse format
     */
    template<typename param_t, typename variable_t, typename constraint_t>
    EIGEN_STRONG_INLINE auto operator()(
        const param_t& param,
        const variable_t& var,
        constraint_t& con,
        Eigen::Ref<Eigen::SparseMatrix<scalar_t>> jac)
        const noexcept
    {
        (void)std::initializer_list<int>{ 
            (
                std::get<ind>(cons)(param, var, con, jac),
                0
            )...
        };
    }

    /*
        Set non-zeros of J to match the sparsity structure of the constraint Jacobian
     */
    EIGEN_STRONG_INLINE void initialize_sparse_jacobian(SparseMatrix<scalar_t>& J)
    {
        std::vector<Eigen::Triplet<scalar_t>> trip;

        (void)std::initializer_list<int>{ 
            (
                std::get<ind>(cons).jac_get_sparsity_triplet(trip),
                0
            )...
        };

        J.setFromTriplets(trip.begin(), trip.end());
        J.makeCompressed();
    }

    /*
        Set non-zeros of H to match the sparsity structure of the constraint Jacobian
     */
    EIGEN_STRONG_INLINE void initialize_sparse_hessian(SparseMatrix<scalar_t>& H)
    {
        std::vector<Eigen::Triplet<scalar_t>> trip;

        (void)std::initializer_list<int>{ 
            (
                std::get<ind>(cons).hessian_get_sparsity_triplet(trip),
                0
            )...
        };

        H.setFromTriplets(trip.begin(), trip.end());
        H.makeCompressed();
    }


    /*
        Evaluate all constraints and dense jacobians and hessians
     */
    template<typename param_t, typename variable_t, typename constraint_t>
    EIGEN_STRONG_INLINE auto operator()(
        const param_t& param,
        const variable_t& var,
        constraint_t& con,
        Eigen::Ref<Eigen::Matrix<typename variable_t::Scalar, constraint_t::RowsAtCompileTime, variable_t::RowsAtCompileTime>> jac,
        Eigen::Ref<Eigen::Matrix<typename variable_t::Scalar, variable_t::RowsAtCompileTime, variable_t::RowsAtCompileTime>> hessian,
        const Eigen::Ref<Eigen::Matrix<typename variable_t::Scalar, constraint_t::RowsAtCompileTime, 1>> hessian_multiplier)
        const noexcept
    {
        hessian.array() = 0.0;
        (void)std::initializer_list<int>{ 
            (
                std::get<ind>(cons)(param, var, con, jac, hessian, hessian_multiplier),
                0
            )...
        };
    }

    /*
        Evaluate all constraints and sparse jacobians and hessians
     */
    template<typename param_t, typename variable_t, typename constraint_t>
    EIGEN_STRONG_INLINE auto operator()(
        const param_t& param,
        const variable_t& var,
        constraint_t& con,
        Eigen::Ref<Eigen::SparseMatrix<scalar_t>> jac,
        Eigen::Ref<Eigen::SparseMatrix<scalar_t>> hessian,
        const Eigen::Ref<Eigen::Matrix<typename variable_t::Scalar, constraint_t::RowsAtCompileTime, 1>> hessian_multiplier)
        const noexcept
    {
        scalar_t* data = hessian.valuePtr();
        for(std::size_t i=0; i<hessian.nonZeros(); i++)
            data[i] = 0.0;

        (void)std::initializer_list<int>{ 
            (
                std::get<ind>(cons)(param, var, con, jac, hessian, hessian_multiplier),
                0
            )...
        };
    }


    /****************************************************
        Static functions to construct the constraint set
     ****************************************************/

    // Return an initalized object of this type
    // This is just a helper so that the user doesn't have to re-enter the constraint types
    static constexpr auto make_constraints()
    {
        return constraints_impl<num_variables, cons_t, ind...>({std::tuple_element_t<ind, cons_t>()...});
    }

    // Compute the total vector length required to store all constraints
    static constexpr std::size_t constraint_vector_length()
    {
        int sum = 0;
        for(std::size_t num : {std::tuple_element_t<ind, cons_t>::constraint_vector_length...})
            sum += num;
        return sum;
    }
};

// Helper function to create a constraint list with an index sequence to iterate over
template<std::size_t num_variables, typename cons_t, std::size_t... ind>
constexpr auto make_constraints_helper(cons_t& cons, std::index_sequence<ind...>)
{
    return constraints_impl<num_variables, cons_t, ind...>(cons);  
}

template<std::size_t num_variables, typename... cons_t>
constexpr auto get_constraint_type()
{
    std::tuple<cons_t...> cons = std::make_tuple(cons_t()...);
    return make_constraints_helper<num_variables, decltype(cons)>(cons, std::index_sequence_for<cons_t...>{});
}

template<std::size_t num_variables, typename... cons_t>
using make_constraints = decltype(get_constraint_type<num_variables, cons_t...>());
