#ifndef LAMPC_UTILITY
#define LAMPC_UTILITY

#include <limits>
#include "Eigen/Dense"

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