#ifndef __BSUTIL_HPP
#define __BSUTIL_HPP

/**
 * Useful meta-functions
 */

// Implementation of conjunction and disjunction in C++11 
namespace meta {

template<bool...> struct bool_pack{};

template<bool... Bs>
using conjunction = std::is_same<bool_pack<true,Bs...>, bool_pack<Bs..., true>>;

template <bool B>
using bool_constant = std::integral_constant<bool, B>; // redefining C++17 bool_constant helper

template<bool... Bs>
struct disjunction : bool_constant<!conjunction<!Bs...>::value>{};

template<typename T, typename... Ts>
struct contains : disjunction<std::is_same<T, Ts>::value...>
{};

/** Return the index into the tuple where type matches x
 * Returns -1 if x is not found.
 */
template<typename x, typename... elements>
constexpr int get_index(int value_on_fail = -1)
{
	int ind = 0;
	int out = value_on_fail;
    (void)std::initializer_list<int>{
        (
			out = std::is_same<x, elements>::value ? ind : out,
			ind++,
            0
        )...
    };
    return out;
}

};
#endif // __BSUTIL_HPP