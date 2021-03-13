#include <iostream>
// Eigen includes
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>
using namespace Eigen;

#include "unsupported/Eigen/AutoDiff"


// COMMON FILE
template<typename Derived> struct nlp_traits;
template<typename T> struct nlp_traits<const T> : nlp_traits<T> {};

template <typename Derived>
struct ProblemBase 
{
    enum
    {
        /** problem dimensions */
        VAR_SIZE  = nlp_traits<Derived>::NX,
        NUM_EQ    = nlp_traits<Derived>::NE,
        NUM_INEQ  = nlp_traits<Derived>::NI,
        NUM_BOX   = nlp_traits<Derived>::NX,
        DUAL_SIZE = NUM_EQ + NUM_INEQ + NUM_BOX,
    };

	// using nlp_traits<Derived>::b;
	using scalar_t = typename nlp_traits<Derived>::scalar_t;

    /** optimisation variable */
    template<typename T>
    using variable_t = Eigen::Matrix<T, VAR_SIZE, 1>;

    template<typename T>
    using constraint_t = Eigen::Matrix<T, NUM_EQ, 1>;

    /** parameters */
    template<typename T>
    using parameter_t = Eigen::Matrix<T, nlp_traits<Derived>::NP, 1>;

    using static_parameter_t = Eigen::Matrix<scalar_t, nlp_traits<Derived>::NP, 1>;

    template<typename T>
    EIGEN_STRONG_INLINE void cost(const Eigen::Ref<const variable_t<T>>& x, const Eigen::Ref<const static_parameter_t>& p, T& cost) const noexcept
    {
        static_cast<const Derived*>(this)->cost_impl(x, p, cost);
    }
};


// GENERATED FILE

// Define traits class
template<typename Derived>
struct UserBase;

template<typename Derived>
struct nlp_traits<UserBase<Derived>> 
{
	int b = 4;
	using scalar_t = double;

    enum { NX = 10, NE = 5, NI = 2, NP = 3};
};


template <typename Derived>
struct UserBase : public ProblemBase<UserBase<Derived>>
{
	using Base = ProblemBase<UserBase<Derived>>;
	using typename Base::scalar_t;

    template<typename T>
    using variable_t = typename Base::template variable_t<T>;
    using static_parameter_t = typename Base::static_parameter_t;

    variable_t<scalar_t> bill;

    template<typename T>
    using X_t = Eigen::Matrix<T, 3, 1>;

    template<typename T>
    EIGEN_STRONG_INLINE void cost_impl(const Eigen::Ref<const variable_t<T>>& x, const Eigen::Ref<const static_parameter_t>& p, T& cost) const noexcept
    {
    	// Add constexpressions to evaluate offsets
    	T cost1;
        static_cast<const Derived*>(this)->template user_function1<T>(x.template segment<3>(0), cost1);
        // static_cast<const Derived*>(this)->template user_function1<T>(q, cost1);
        // static_cast<const Derived*>(this)->user_function2(x.segment<1>(5), cost2);
    }

};


// USER FILE

struct User : public UserBase<User>
{
	double bob;

    variable_t<scalar_t> jill;

    template<typename T>
    EIGEN_STRONG_INLINE void user_function1(const Eigen::Ref<const X_t<T>>& x, T& cost) const noexcept
    {
    	std::cout << "user_function1" << std::endl;
    }
};

int main(void)
{
	User x;
	x.bob = 4.5;
	// x.bill = 5.6;

	std::cout << x.bob << std::endl;
	// std::cout << x.bill << std::endl;

	nlp_traits<UserBase<User>> t;
	std::cout << "t.b = " << t.b << std::endl;

	User::X_t<double> y;
	double cost;
	x.user_function1<double>(y, cost);

	User::static_parameter_t p;

	User::variable_t<double> var;

    x.cost<double>(var, p, cost);

	return 0;
}
