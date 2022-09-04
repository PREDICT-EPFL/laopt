#ifndef LAOPT_UTILITY_HPP
#define LAOPT_UTILITY_HPP

#include <iostream>
#include <limits>
#include <type_traits>
#include "Eigen/Dense"

#include "indexed_vector.hpp"

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
        struct IsScalar {};
        struct IsMatrix {};

        template<class Type, bool is_eigen = std::is_base_of<Eigen::MatrixBase<Type>, Type>::value>
        struct matrix_info;

        template <class Type>
        struct matrix_info<Type, false>
        {
            using Scalar = Type;
            static constexpr int RowsAtCompileTime = 1;
            static constexpr int ColsAtCompileTime = 1;

            using is_matrix_t = IsScalar;
        };

        template<class Type>
        struct matrix_info<Type, true>
        {
            using Scalar = typename Type::Scalar;
            static constexpr int RowsAtCompileTime = Type::RowsAtCompileTime;
            static constexpr int ColsAtCompileTime = Type::ColsAtCompileTime;

            using is_matrix_t = IsMatrix;
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


        /**
         * Checks if arguments contain a variable.
         */
        template <typename...>
        struct contains_variable : std::true_type {};

        template <typename Arg, typename... Args>
        struct contains_variable<Arg, Args...>
                : std::conditional<std::is_base_of<VariableBase<Arg>, Arg>::value,
                        std::true_type,
                        contains_variable<Args...>
                >::type {};

        template<>
        struct contains_variable<> : std::false_type {};

        /**
         * Used to get information about variables.
         */
        template<typename Derived>
        struct variable_info
        {
            // If it's not a variable we ignore it
            static constexpr int size = 0;
        };
        template<typename Base>
        struct variable_info<IndexedVector<Base>>
        {
            static constexpr int size = IndexedVector<Base>::RowsAtCompileTime;
        };

    } // end namespace meta

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

    template<typename... T, int n = meta::sum_template<T::SizeAtCompileTime...>()>
    inline Eigen::Vector<int, n> multiSeq_to_index(T... segments)
    {
        static_assert(is_all_positive({T::SizeAtCompileTime...}), "SIZES OF ARITHMETIC SEQUENCES MUST BE FIXED");

        Eigen::Vector<int, n> ret;
        int i = 0;

        auto fill_me = [&i, &ret](auto seg) {
            for (int j = 0; j < decltype(seg)::SizeAtCompileTime; j++) {
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