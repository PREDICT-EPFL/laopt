#ifndef LAMPC_UTILITY
#define LAMPC_UTILITY

#include <limits>
#include "Eigen/Dense"
#include <type_traits>

namespace lampc
{
	// template<typename scalar_t>
	// static constexpr auto INF = std::numeric_limits<scalar_t>::infinity();

	/// /todo Problem with M1 mac - limits doesn't seem to be defined yet.
	template<typename scalar_t>
	static constexpr auto INF = 1e20;

    namespace meta
    {
        // Sum the inputs to get total number of inputs
        template<int... S>
        constexpr int sum_template() {
            int result = 0;
            for(auto s : { S... }) result += s;
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

        /**
         * has_eval<T>::value == true if and only if T has a member function "eval"
         * We use this to test if T is an Eigen expression or matrix
         */
        template <typename T>
        struct has_eval
        {
          template<typename U> static decltype(std::declval<U>().eval(), std::true_type{}) test (std::remove_reference_t<U>*); 
          template<typename U> static std::false_type test (...);
          using  type = decltype(test<T>(nullptr));
          static constexpr bool value { type::value };
        };

        template < class Type, bool is_eigen = has_eval<Type>::value > struct matrix_info;

        template < class Type >
        struct matrix_info<Type, false> {
            using Scalar = Type;
            static constexpr int RowsAtCompileTime = 1;
            static constexpr int ColsAtCompileTime = 1;

            using is_matrix_t = IsScalar;
        };

        template < class Type >
        struct matrix_info<Type, true> {
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
        template<typename Arg, typename... Args>
        struct get_scalar
        {
            using type = typename Arg::Scalar;
        };
        template<typename... Args>
        using get_scalar_t = typename get_scalar<Args...>::type;
    };

}


// /*************************************************************
//     Meta-programming helper functions
//  *************************************************************/

// // Sum the inputs to get total number of inputs
// template<int... S>
// constexpr int sum_template() {
//     int result = 0;
//     for(auto s : { S... }) result += s;
//     return result;
// }

// // Build an array at compile time
// template <typename T, typename... Args>
// constexpr std::array<T, sizeof...(Args)> make_array(Args... args)
// {
//     return {args...};
// }

// // Helper function to reshape an eigen matrix while maintaining const'ness
// template <typename map_to, typename scalar_t>
// constexpr auto _map_matrix(const scalar_t* x) {
//     return Eigen::Map<const map_to>(x);
// }

// template <typename map_to, typename scalar_t>
// constexpr auto _map_matrix(scalar_t* x) {
//     return Eigen::Map<map_to>(x);
// }

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

// /*
//  * Extracts the return-type of a function
//  * 
//  * Use like this:     
//  *  std::cout << type_name<decltype(ret(f))>() << std::endl;
//  */
 
// template<typename R, typename... A>
// R ret(R(*)(A...)) ;

// template<typename C, typename R, typename... A>
// R ret(R(C::*)(A...));


#endif // LAMPC_UTILITY