#include <iostream>
#include <type_traits>
#include <tuple>

#include "lampc_utility.hpp"

struct Eigen_autodiff{};
struct CUSTOM {};

struct Tag
{
  // Overload if custom internal versions are defined
  using jacobian = Eigen_autodiff;
  using wsum     = CUSTOM; // inner product <w, f>
  using gradient = CUSTOM;
  using hessian  = Eigen_autodiff;
};


template <typename Handler0, typename... Handlers>
struct TheCaller : Handler0, TheCaller<Handlers...>
{
    using Handler0::function_impl;
    using TheCaller<Handlers...>::function_impl;
};

template <typename Handler>
struct TheCaller<Handler> : Handler
{
    using Handler::function_impl;
};

template <typename Derived>
struct BaseFunctions
{
  template<typename T, typename... Args>
  int function_impl(CUSTOM, Args... args)
  {
    std::cout << "In MidLayer calling function_impl" << std::endl;
    return 1;
  }
};

template<typename Derived, template<typename> class... Libraries>
struct MidLayer : public Libraries<Derived>...
{
  // Builds a class recursively that has all the function_impl's in scope
  TheCaller<BaseFunctions<Derived>, Libraries<Derived>...> caller;

  template<typename T, typename Tag, typename... Args>
  static auto test_user_function(int) -> decltype(std::declval<T>().template function_impl<int>(std::declval<Tag>(), std::declval<Args>()...), std::true_type{});
  template<typename T, typename... Args>
  static auto test_user_function(long) -> std::false_type;
  template<typename T, typename Tag, typename... Args>
  struct has_user_function : decltype(test_user_function<T, Tag, Args...>(0)){};

  template<typename Tag, typename... Args>
  typename std::enable_if<has_user_function<Derived, Tag, Args...>() == true, int>::type
  function(Tag t, Args... args)
  {
    std::cout << "In function calling function_impl from Derived" << std::endl;
    return static_cast<Derived*>(this)->template function_impl<int>(t, args...);
  }

  template<typename Tag, typename... Args>
  typename std::enable_if<has_user_function<Derived, Tag, Args...>() == false, int>::type
  function(Tag t, Args... args)
  {
    std::cout << "In function calling function_impl from MidLayer" << std::endl;
    return caller.template function_impl<int>(t, args...);
  }

};


// A library with some useful functions in it
struct LibraryTag : Tag {};
template<typename Derived>
struct MyLibrary
{
  template<typename T, typename... Args>
  int function_impl(LibraryTag, Args... args)
  {
    std::cout << "In MyLibrary calling function_impl" << std::endl;
    return 1;
  }
};


// User-level code
struct User : public MidLayer<User, MyLibrary>
{
  struct TagUser : Tag {};
  template<typename T, typename... Args>
  int function_impl(TagUser, Args... args)
  {
    std::cout << "In user TagUser" << std::endl;
    return 1;
  }
};


int main()
{
  User user;

  std::cout << "In main calling function with TagUser" << std::endl;
  user.function(User::TagUser{});

  std::cout << std::endl << std::endl;  

  std::cout << "In main calling function with CUSTOM" << std::endl;
  user.function(CUSTOM{});

  std::cout << std::endl << std::endl;  

  std::cout << "In main calling function with LibraryTag" << std::endl;
  user.function(LibraryTag{});

}
