#include <iostream>
#include <type_traits>
#include <tuple>

#include "lampc_utility.hpp"



/**
 * Base class to clean up CRTP definitions with multiple
 * base types
 * 
 * This is from https://www.fluentcpp.com/2017/05/19/crtp-helper/
 * 
 */
template <typename T, template<typename> class crtpType>
struct crtp
{
    constexpr T& call() { return static_cast<T&>(*this); }
    constexpr T const& call() const { return static_cast<T const&>(*this); }
private:
    crtp(){}
    friend crtpType<T>;
};

/**
 * This is a recursive way of calling "using" on each 
 * of the list of arguments and pulling them all into
 * scope.
 */
template <typename Handler0, typename... Handlers>
struct TheCaller : Handler0, TheCaller<Handlers...>
{
    using Handler0::function;
    using TheCaller<Handlers...>::function;

    using Handler0::jacobian;
    using TheCaller<Handlers...>::jacobian;
};

template <typename Handler>
struct TheCaller<Handler> : Handler
{
    using Handler::function;
    using Handler::jacobian;
};


/**
 * This is the main class the the user would see
 * It just collects all the given crtp classes together
 * and brings their functions into scope.
 */
template<template<typename> class... Libraries>
struct Differentiator : public TheCaller<Libraries<Differentiator<Libraries...>>...>
{
	using TheCaller<Libraries<Differentiator<Libraries...>>...>::function;
	using TheCaller<Libraries<Differentiator<Libraries...>>...>::jacobian;

	// template<typename Tag>
	// void jacobian(Tag)
	// {
	// 	std::cout << "In Default jacobian - catches all tags" << std::endl;
	// 	function(Tag{});
	// }
};

struct EigenTag{};
struct IgnoreTag{};
struct Tag
{
  // Overload if custom internal versions are defined
  using jacobian = EigenTag;
};


/**
 * Library code is the same as user-level code.
 * 
 * The Library is crtp derived from the Differentiator class
 */
struct Lib2Tag : Tag {};
struct LibraryTag : Tag {};
template <typename T>
struct Library : crtp<T, Library>
{
	/**
	 * We define function directly - no more impl
	 */
  void function(LibraryTag)
  {
  	std::cout << "in library. Calling another library function." << std::endl;
  	this->call().function(Lib2Tag{});
  }

  /**
   * If the function is not defined, then an ignore call needs
   * to be added (also for user-level code).
   * We could perhaps get rid of this with some meta-foo
   */
  struct IgnoreTag{};
  void jacobian(IgnoreTag) {};
};

/**
 * A second library that we could swap in if desired
 */
template <typename T>
struct Library2 : crtp<T, Library2>
{
  void function(Lib2Tag)
  {
  	std::cout << "in library 2" << std::endl;
  }

  void jacobian(Lib2Tag)
  {
  	std::cout << "jacobian in library 2" << std::endl;
  }
};

/**
 * Library defining default autodiff via Eigen
 */
template <typename T>
struct EigenDiff : crtp<T, EigenDiff>
{
	void function(IgnoreTag) {}

	template<typename Tag>
  void jacobian(Tag t)
  {
  	std::cout << "In EigenAutoDiff. Calling function." << std::endl;
  	this->call().function(t);
  }
};

/**
 * Library defining default autodiff via some other method
 */
template <typename T>
struct OtherDiff : crtp<T, OtherDiff>
{
	struct IgnoreTag{};
	void function(IgnoreTag) {}

	template<typename Tag>
  void jacobian(Tag t)
  {
  	std::cout << "In OtherDiff. Calling function." << std::endl;
  	this->call().function(t);
  }
};



/**
 * This is the user-level code.
 *
 * It's the same as library code.
 */

struct UserTagDefault : Tag {};
struct UserTagCustom : Tag {};

template <typename T>
struct User : crtp<T, User>
{
    void function(UserTagDefault)
    {
    	std::cout << "in user. Calling library." << std::endl;
    	// This is the one downside of this approach... 
    	// This is how a user calls library "function" or "jacobian" 
    	// from within their own class.
      this->call().function(LibraryTag{});
    }

    /**
     * Overload the jacobian just by defining it at the user level.
     */
    void jacobian(UserTagCustom)
    {
    	std::cout << "User specified jacobian" << std::endl;
    }

    void function(UserTagCustom)
    {
    	std::cout << "In UserTagCustom function." << std::endl;
    }
};


int main()
{
	std::cout << "========= Calling with Eigen Differentiator =========" << std::endl;
	Differentiator<User, Library, Library2, EigenDiff>	s;

	std::cout << std::endl << "Calling function(LibraryTag)" << std::endl;
	s.function(LibraryTag{});

	std::cout << std::endl << "Calling jacobian(LibraryTag)" << std::endl;
	s.jacobian(LibraryTag{});

	std::cout << std::endl << "Calling function(UserTagDefault)" << std::endl;
	s.function(UserTagDefault{});

	std::cout << std::endl << "Calling jacobian(UserTagDefault)" << std::endl;
	s.jacobian(UserTagDefault{});

	std::cout << std::endl << "Calling function(UserTagCustom)" << std::endl;
	s.function(UserTagCustom{});

	std::cout << std::endl << "Calling jacobian(UserTagCustom)" << std::endl;
	s.jacobian(UserTagCustom{});


	std::cout << std::endl << std::endl;
	std::cout << "========= Calling with Other Differentiator =========" << std::endl;
	Differentiator<User, Library, Library2, OtherDiff>	s2;

	std::cout << std::endl << "Calling function(LibraryTag)" << std::endl;
	s2.function(LibraryTag{});

	std::cout << std::endl << "Calling jacobian(LibraryTag)" << std::endl;
	s2.jacobian(LibraryTag{});

	std::cout << std::endl << "Calling function(UserTagDefault)" << std::endl;
	s2.function(UserTagDefault{});

	std::cout << std::endl << "Calling jacobian(UserTagDefault)" << std::endl;
	s2.jacobian(UserTagDefault{});

	std::cout << std::endl << "Calling function(UserTagCustom)" << std::endl;
	s2.function(UserTagCustom{});

	std::cout << std::endl << "Calling jacobian(UserTagCustom)" << std::endl;
	s2.jacobian(UserTagCustom{});
}
