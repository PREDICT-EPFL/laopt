#ifndef LAOPT_DIFFERENTIABLE_EIGEN_HPP
#define LAOPT_DIFFERENTIABLE_EIGEN_HPP

#include <Eigen/Dense>
#include "laopt/autodiff/autodiff_scalar.hpp"
#include "laopt/autodiff/differentiable_options.hpp"

namespace laopt
{

template<typename Derived, int Options>
class DifferentiableBaseEigen
{
protected:
    // Compute the jacobian using eigen autodiff
    template<typename Tag, typename OutJacobian, typename AScalar, typename... Args, int Opts = Options>
    EIGEN_STRONG_INLINE typename std::enable_if<(has_tag_override<Tag>::type::value && has_tag_eigen<Tag>::value) ||
                                                (!has_tag_override<Tag>::type::value && (Opts & CASADI_JACOBIAN) == 0)>::type
    jacobian_impl_autodiff_eval(
        const Tag& tag, // Function to call
        OutJacobian& out_jacobian, // Outputs
        const AScalar& alpha, // Scaling factor
        const Args&... args) noexcept // Function arguments
    {
        using Info = typename Derived::template FuncInfo<Tag, Args...>;
        using Scalar = typename Info::scalar_t;

        // First order derivative
        using ADScalar = laopt::AutoDiffScalar<Eigen::Vector<Scalar, Info::n_inputs>>;
        using ADOutput = Eigen::Vector<ADScalar, Info::n_outputs>;

        // Convert the arguments to AD variables, and call the function
        ADOutput out = seed_and_call(tag, make_ad<ADScalar>(args)...);

        // Copy out into output variables
        for(int i = 0; i < out.rows(); i++)
        {
            out_jacobian(i, Eigen::all) += alpha * out[i].derivatives().transpose();
        }
    }

    // Compute sparsity pattern of the jacobian using eigen autodiff
    template<typename Tag, typename SparsityNullMat, typename AScalar, typename... Args, int Opts = Options>
    EIGEN_STRONG_INLINE typename std::enable_if<(has_tag_override<Tag>::type::value && has_tag_eigen<Tag>::value) ||
                                                (!has_tag_override<Tag>::type::value && (Opts & CASADI_JACOBIAN) == 0)>::type
    jacobian_impl_autodiff_sparsity(
        const Tag& tag, // Function to call
        BSSliceSparsity<BSMatrixSparsity, SparsityNullMat>& out_jacobian, // Outputs
        const AScalar& alpha, // Scaling factor
        const Args&... args) noexcept // Function arguments
    {
        using Info = typename Derived::template FuncInfo<Tag, Args...>;
        using Scalar = typename Info::scalar_t;

        // First order derivative
        using ADScalar = laopt::AutoDiffScalar<Eigen::Vector<TouchableDerivative<Scalar>, Info::n_inputs>>;
        using ADOutput = Eigen::Vector<ADScalar, Info::n_outputs>;

        // Convert the arguments to AD variables, and call the function
        ADOutput out = seed_and_call(tag, make_ad_touchable<ADScalar>(args)...);

        for(int i = 0; i < out.rows(); i++)
        {
            for (int j = 0; j < out[i].derivatives().rows(); j++)
            {
                if (out[i].derivatives()(j).value() != 0)
                {
                    out_jacobian(i, j) = 1;
                }
            }
        }
    }

    // Compute the hessian using eigen autodiff
    template<typename Tag, typename OutHessian, typename Weight, typename... Args, int Opts = Options>
    EIGEN_STRONG_INLINE typename std::enable_if<(has_tag_override<Tag>::type::value && has_tag_eigen<Tag>::value) ||
                                                (!has_tag_override<Tag>::type::value && (Opts & CASADI_HESSIAN) == 0)>::type
    hessian_impl_autodiff_eval(
        const Tag& tag,
        OutHessian& out_hessian,
        const Eigen::MatrixBase<Weight>& weight,
        const Args&... args) noexcept
    {
        using Info = typename Derived::template FuncInfo<Tag, Args...>;
        using Scalar = typename Info::scalar_t;

        // First order derivative
        using ADScalar = laopt::AutoDiffScalar<Eigen::Vector<Scalar, Info::n_inputs>>;
        // only calculate one col of the hessian at once to reduce memory requirements
        using outerDerivatives = Eigen::Vector<ADScalar, 1>;
        // Second order derivative
        using outerADScalar = laopt::AutoDiffScalar<outerDerivatives>;
        using ADOutput = Eigen::Vector<outerADScalar, Info::n_outputs>;

        ADOutput out;
        for (size_t i = 0; i < Info::n_inputs; i++) {
            // Convert to AD variables for the inputs and call our function for ith row of hessian
            out = seed_and_call2(i, tag, make_ad2<outerADScalar>(args)...);
            for(size_t j = 0; j < Info::n_outputs; j++) {
                out_hessian(Eigen::all, i) += weight(j) * out[j].derivatives()(0).derivatives();
            }
        }
    }

    // Compute sparsity pattern of the hessian using eigen autodiff
    template<typename Tag, typename SparsityNullMat, typename Weight, typename... Args, int Opts = Options>
    EIGEN_STRONG_INLINE typename std::enable_if<(has_tag_override<Tag>::type::value && has_tag_eigen<Tag>::value) ||
                                                (!has_tag_override<Tag>::type::value && (Opts & CASADI_HESSIAN) == 0)>::type
    hessian_impl_autodiff_sparsity(
        const Tag& tag,
        BSSliceSparsity<BSMatrixSparsity, SparsityNullMat>& out_hessian,
        const Eigen::MatrixBase<Weight>& weight,
        const Args&... args) noexcept
    {
        using Info = typename Derived::template FuncInfo<Tag, Args...>;
        using Scalar = typename Info::scalar_t;

        // First order derivative
        using ADScalar = laopt::AutoDiffScalar<Eigen::Vector<TouchableDerivative<Scalar>, Info::n_inputs>>;
        // only calculate one row of the hessian at once to reduce memory requirements
        using outerDerivatives = Eigen::Vector<ADScalar, 1>;
        // Second order derivative
        using outerADScalar = laopt::AutoDiffScalar<outerDerivatives>;
        using ADOutput = Eigen::Vector<outerADScalar, Info::n_outputs>;

        ADOutput out;
        for (size_t i = 0; i < Info::n_inputs; i++) {
            // Convert to AD variables for the inputs and call our function for ith row of hessian
            out = seed_and_call2(i, tag, make_ad2_touchable<outerADScalar>(args)...);
            for(size_t j = 0; j < Info::n_outputs; j++) {
                for (int k = 0; k < out[j].derivatives()(0).derivatives().rows(); k++) {
                    if (out[j].derivatives()(0).derivatives()(k).value() != 0)
                    {
                        out_hessian(k, i) = 1;
                    }
                }
            }
        }
    }

    // Take a vector input and return an AD version of the vector
    template<typename ADScalar, typename Base>
    EIGEN_STRONG_INLINE auto
    make_ad(const IndexedVector<Base>& x) noexcept
    {
        constexpr size_t n = Base::RowsAtCompileTime;
        Eigen::Vector<ADScalar, n> y;
        y = x;
        for (int i = 0; i < y.rows(); i++) {
            y[i].derivatives().setZero();
        }
        return y;
    }

    template<typename ADScalar, typename T>
    EIGEN_STRONG_INLINE const T&
    make_ad(const T& x) noexcept
    {
        return x;
    }

    // Take a vector input and return an AD version of the vector
    template<typename ADScalar, typename Base>
    EIGEN_STRONG_INLINE auto
    make_ad_touchable(const IndexedVector<Base>& x) noexcept
    {
        constexpr size_t n = Base::RowsAtCompileTime;
        Eigen::Vector<ADScalar, n> y;
        for (int i = 0; i < y.rows(); i++) {
            y[i].value() = 1;
        }
        return y;
    }

    template<typename ADScalar, typename T>
    EIGEN_STRONG_INLINE const T&
    make_ad_touchable(const T& x) noexcept
    {
        return x;
    }

    template<typename Tag, typename... Args>
    EIGEN_STRONG_INLINE auto
    seed_and_call(const Tag& tag, Args&&... args) noexcept
    {
        // Set derivative equal to identity
        int offset = 0;
        (void) std::initializer_list<int>{
            (
                offset = ad_seed(args, offset), // Set to unit vectors
                0
            )...
        };

        return static_cast<Derived*>(this)->function(tag, std::forward<Args>(args)...);
    }

    // Sets the input derivatives to the identity.
    // Assumes that the derivative matrix is initially zero
    template<typename X, int n>
    EIGEN_STRONG_INLINE int
    ad_seed(Eigen::Vector<laopt::AutoDiffScalar<X>, n>& x, int offset) noexcept
    {
        for (int i = 0; i < x.rows(); i++)
        {
            x[i].derivatives().coeffRef(i + offset) = 1;
        }
        return offset + x.rows();
    }

    template<typename T>
    EIGEN_STRONG_INLINE int
    ad_seed(const T, int offset) noexcept
    {
        return offset;
    }

    // Take a vector input and return an AD version of the vector
    template<typename outerADScalar, typename Base>
    EIGEN_STRONG_INLINE auto
    make_ad2(const IndexedVector<Base>& x) noexcept
    {
        constexpr size_t n = Base::RowsAtCompileTime;
        Eigen::Vector<outerADScalar, n> y;
        // y = x;
        for (size_t i = 0; i < n; i++) {
            y(i).value().value() = x.cast_base()(i);
            y(i).value().derivatives().setZero();
            y(i).derivatives().setZero();
            y(i).derivatives()(0).derivatives().setZero();
        }
        return y;
    }

    template<typename outerADScalar, typename T>
    EIGEN_STRONG_INLINE const T&
    make_ad2(const T& x) noexcept
    {
        return x;
    }

    // Take a vector input and return an AD version of the vector
    template<typename outerADScalar, typename Base>
    EIGEN_STRONG_INLINE auto
    make_ad2_touchable(const IndexedVector<Base>& x) noexcept
    {
        constexpr size_t n = Base::RowsAtCompileTime;
        Eigen::Vector<outerADScalar, n> y;
        // y = x;
        for (size_t i = 0; i < n; i++) {
            y(i).value().value() = 1;
        }
        return y;
    }

    template<typename outerADScalar, typename T>
    EIGEN_STRONG_INLINE const T&
    make_ad2_touchable(const T& x) noexcept
    {
        return x;
    }

    template<typename Tag, typename... Args>
    EIGEN_STRONG_INLINE auto seed_and_call2(int outer_index, const Tag& tag, Args&&... args) noexcept
    {
        // Set derivative equal to identity
        int inner_offset = 0;
        (void) std::initializer_list<int>{
            (
                inner_offset = ad_seed2(args, outer_index, inner_offset), // Set to unit vectors
                0
            )...
        };

        // Call our function
        return static_cast<Derived*>(this)->function(tag, std::forward<Args>(args)...);
    }

    // Sets the input derivatives to the identity.
    // Assumes that the derivative matrix is initially zero
    template <typename X, int n>
    EIGEN_STRONG_INLINE int
    ad_seed2(Eigen::Vector<laopt::AutoDiffScalar<X>, n>& x, int outer_index, int inner_offset) noexcept
    {
        for (int i = 0; i < x.rows(); i++)
        {
            x(i).value().derivatives().coeffRef(i + inner_offset) = 1;
            // Seed only the relevant value for the current row of the hessian
            if (i + inner_offset == outer_index) {
                x(i).derivatives().coeffRef(0).value() = 1;
            }
        }

        return inner_offset + x.rows();
    }

    template<typename T>
    EIGEN_STRONG_INLINE int
    ad_seed2(const T, int outer_index, int inner_offset) noexcept
    {
        return inner_offset;
    }
};

} // namespace laopt

#endif // LAOPT_DIFFERENTIABLE_EIGEN_HPP
