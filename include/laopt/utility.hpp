#ifndef LAOPT_UTILITY_HPP
#define LAOPT_UTILITY_HPP

#include <iostream>
#include <limits>
#include <type_traits>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "laopt/expressions/expr_base.hpp"

namespace laopt
{
    namespace meta
    {
        // Sum the inputs to get total number of inputs
        template<int... S>
        constexpr int sum_template() {
            int result = 0;
            for (int s : std::initializer_list<int>{ S... }) result += s;
            return result;
        }

        /** Return the index of What in Args, or -1
         */
        template<typename What, typename ... Args>
        constexpr int get_index()
        {
            int ind = -1;
            int i = 0;
            auto l = {(
                ind = std::is_same<What, Args>::value ? i : ind,
                i++,
                0)...};
            return ind;
        }

        /**
         * Extracts compile time info about an Eigen matrix.
         * If the type passed in is not a matrix (i.e., it's a scalar),
         * then its shape is 1 x 1 and is_matrix is false.
         */

        // Tags telling us if a return value is a scalar or an Eigen Matrix
        struct is_eigen_base {};
        struct is_expr_base {};

        template<typename Type, typename TypeBase = typename std::conditional<std::is_base_of<Eigen::MatrixBase<Type>, Type>::value, is_eigen_base,
                                                    typename std::conditional<std::is_base_of<ExprBase<Type>, Type>::value, is_expr_base, std::false_type>::type>::type>
        struct matrix_info;

        template <typename Type>
        struct matrix_info<Type, std::false_type>
        {
            using Scalar = Type;
            static constexpr int RowsAtCompileTime = 1;
            static constexpr int ColsAtCompileTime = 1;
        };

        template<typename Type>
        struct matrix_info<Type, is_eigen_base>
        {
            using Scalar = typename Type::Scalar;
            static constexpr int RowsAtCompileTime = Type::RowsAtCompileTime;
            static constexpr int ColsAtCompileTime = Type::ColsAtCompileTime;
        };

        template<class Type>
        struct matrix_info<Type, is_expr_base>
        {
            using Scalar = typename Type::Scalar;
            static constexpr int RowsAtCompileTime = Type::n_outputs;
            static constexpr int ColsAtCompileTime = 1;
        };

        /**
         * Compute the scalar type of the arguments
         *
         * Note: We take the scalar type of the first argument, and assume that the
         * rest are the same, or can be auto-cast to be the same.
         * We should likely test them all at compile time...
         */
        template<typename Arg, typename...>
        struct get_scalar
        {
            using type = typename Arg::Scalar;
        };
        template<typename... Args>
        using get_scalar_t = typename get_scalar<Args...>::type;

        // Helper to extract compile-time size from ArithmeticSequence
        template<typename T>
        struct seq_size {
            static constexpr int value = T::SizeAtCompileTime; // won't work for ArithmeticSequence
        };

        // Specialization for ArithmeticSequence
        template<typename First, typename Size, typename Incr>
        struct seq_size<Eigen::ArithmeticSequence<First, Size, Incr>> {
            static constexpr int value = Eigen::internal::get_fixed_value<Size>::value;
        };

    } // end namespace meta

    template<typename T>
    typename std::enable_if<!std::is_base_of<Eigen::MatrixBase<T>, T>::value, Eigen::Matrix<T, 1, 1>>::type
    to_matrix_type(const T& value)
    {
        Eigen::Matrix<T, 1, 1> res;
        res(0) = value;
        return res;
    }

    template<typename Derived>
    const Eigen::MatrixBase<Derived>& to_matrix_type(const Eigen::MatrixBase<Derived>& value)
    {
        return value;
    }

    /**
     * Takes a parameter pack of Eigen::Vector's and concatenates
     * them into a single Eigen::Vector.
     * Everything must be fixed-size.
     */
    template<int... n>
    Eigen::Vector<int, meta::sum_template<n...>()>
    concatenate_indices(const Eigen::Vector<int, n>&... args)
    {
        Eigen::Vector<int, meta::sum_template<n...>()> out;
        int offset = 0;
        auto l = {
            (
                out(Eigen::seqN(offset, Eigen::fix<n>)) = args,
                offset += n,
                0
            )...
        };
        (void) l; // get rid of unused variable warning
        return out;
    }

    constexpr bool is_all_positive(std::initializer_list<int> values)
    {
        for (auto i: values) {
            if (i < 0) {
                return false;
            }
        }
        return true;
    }

    template<typename InputIt, typename Size, typename Scalar, int Options, typename StorageIndex>
    EIGEN_STRONG_INLINE void
    copy_n_into_sparse_matrix(InputIt first, Size count, Eigen::SparseMatrix<Scalar, Options, StorageIndex>& dst, const Eigen::Index &col, const Eigen::Index &offset)
    {
        // assert col exists
        eigen_assert(col < dst.outerSize());
        // uncompressed
        if (dst.innerNonZeroPtr() != 0)
        {
            // assert not writing outside column
            eigen_assert(offset + count <= dst.innerNonZeroPtr()[col]);
        }
        // compressed
        else
        {
            // assert not writing outside column
            eigen_assert(dst.outerIndexPtr()[col] + offset + count <= dst.outerIndexPtr()[col + 1]);
        }
        std::copy_n(first, count, dst.valuePtr() + dst.outerIndexPtr()[col] + offset);
    }

    template<typename InputIt, typename Size, typename Scalar>
    EIGEN_STRONG_INLINE void
    add_n_into_sparse_matrix(InputIt first, Size count, Eigen::SparseMatrix<Scalar>& dst, const Eigen::Index &col, const Eigen::Index &offset)
    {
        // assert col exists
        eigen_assert(col < dst.outerSize());
        // uncompressed
        if (dst.innerNonZeroPtr() != 0)
        {
            // assert not writing outside column
            eigen_assert(offset + count <= dst.innerNonZeroPtr()[col]);
        }
        // compressed
        else
        {
            // assert not writing outside column
            eigen_assert(dst.outerIndexPtr()[col] + offset + count <= dst.outerIndexPtr()[col + 1]);
        }
        std::transform(dst.valuePtr() + dst.outerIndexPtr()[col] + offset,
                       dst.valuePtr() + dst.outerIndexPtr()[col] + offset + count,
                       first, dst.valuePtr() + dst.outerIndexPtr()[col] + offset,
                       std::plus<Scalar>());
    }

    template<typename... T, int n = meta::sum_template<meta::seq_size<T>::value...>()>
    inline Eigen::Vector<int, n> multiSeq_to_index(T... segments)
    {
        static_assert(is_all_positive({meta::seq_size<T>::value...}), "SIZES OF ARITHMETIC SEQUENCES MUST BE FIXED");

        Eigen::Vector<int, n> ret;
        int i = 0;

        auto fill_me = [&i, &ret](auto seg) {
            for (int j = 0; j < meta::seq_size<decltype(seg)>::value; j++) {
                ret[i++] = seg[j];
            }
        };

        auto l = {
            (fill_me(segments), 0)...
        };
        (void) l; // get rid of unused variable warning

        return ret;
    }

    template <typename T>
    auto type_name() noexcept {
#ifdef __clang__
        std::string name = __PRETTY_FUNCTION__;
        std::string prefix = "auto laopt::type_name() [T = ";
        std::string suffix = "]";
#elif defined(__GNUC__)
        std::string name = __PRETTY_FUNCTION__;
        std::string prefix = "auto laopt::type_name() [with T = ";
        std::string suffix = "]";
#elif defined(_MSC_VER)
        std::string name = __FUNCSIG__;
        std::string prefix = "auto __cdecl laopt::type_name<";
        std::string suffix = ">(void) noexcept";
#else
        std::string name = "Error: unsupported compiler";
        std::string prefix = "";
        std::string suffix = "";
#endif
        return name.substr(prefix.length(), name.length() - prefix.length() - suffix.length());
    }

}

#endif // LAOPT_UTILITY_HPP