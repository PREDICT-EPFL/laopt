#ifndef PROBLEMBASE
#define PROBLEMBASE

// Eigen includes
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>
using namespace Eigen;


/** define derived class traits */
template<typename Derived> struct nlp_traits;
template<typename T> struct nlp_traits<const T> : nlp_traits<T> {};

/** forward declare base class */
template<typename Derived> class ProblemBase;

template<typename Derived>
class ProblemBase
{
public:

    ProblemBase()
    {
    }

    ~ProblemBase() = default;

    enum
    {
        /** problem dimensions */
        VAR_SIZE  = nlp_traits<Derived>::NX,
        NUM_EQ    = nlp_traits<Derived>::NE,
        NUM_INEQ  = nlp_traits<Derived>::NI,
        NUM_BOX   = nlp_traits<Derived>::NX,
        DUAL_SIZE = NUM_EQ + NUM_INEQ + NUM_BOX,
    };

    using scalar_t = typename nlp_traits<Derived>::scalar_t;

    /** parameters */
    template<typename T>
    using parameter_t = Eigen::Matrix<T, nlp_traits<Derived>::NP, 1>;

    /** Parameterized variables */
    template<typename T>
    using variable_t = Eigen::Matrix<T, VAR_SIZE, 1>;

    template<typename T>
    using constraint_t = Eigen::Matrix<T, NUM_EQ, 1>;


    /** NLP variables */
    using nlp_variable_t    = Matrix<scalar_t, VAR_SIZE, 1>;
    using nlp_constraints_t = Matrix<scalar_t, NUM_EQ + NUM_INEQ, 1>;
    // choose to allocate sparse or dense jacoabian and hessian
    using nlp_eq_jacobian_t = Matrix<scalar_t, NUM_EQ + NUM_INEQ, VAR_SIZE>;
    using nlp_hessian_t     = Matrix<scalar_t, VAR_SIZE, VAR_SIZE>;
    using nlp_cost_t        = scalar_t;
    using nlp_dual_t        = Matrix<scalar_t, DUAL_SIZE, 1>;
    using static_parameter_t = Matrix<scalar_t, nlp_traits<Derived>::NP, 1>;

    
    /**  NLP interface functions */
    EIGEN_STRONG_INLINE void cost(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p, scalar_t &cost) noexcept
    {
        static_cast<const Derived*>(this)->cost_impl(var, p, cost);
    }

    EIGEN_STRONG_INLINE void cost_gradient(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p,
                                           scalar_t &_cost, Eigen::Ref<nlp_variable_t> cost_gradient) noexcept
    {
        static_cast<const Derived*>(this)->cost_gradient_impl(var, p, _cost, cost_gradient);
    }

    EIGEN_STRONG_INLINE void cost_gradient_hessian(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p,
                                                   scalar_t &_cost, Eigen::Ref<nlp_variable_t> _cost_gradient, Eigen::Ref<nlp_hessian_t> hessian) noexcept
    {
        static_cast<const Derived*>(this)->cost_gradient_hessian_impl(var, p, _cost, _cost_gradient, hessian);
    }

    EIGEN_STRONG_INLINE void equalities(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p,
                                        Eigen::Ref<nlp_constraints_t> equalities) noexcept
    {
        static_cast<const Derived*>(this)->template equalities_impl<scalar_t>(var, p, equalities);
    }

    EIGEN_STRONG_INLINE void equalities_linearised(const Eigen::Ref<const nlp_variable_t>& var,
												   const Eigen::Ref<const static_parameter_t>& p,
												   Eigen::Ref<nlp_constraints_t> equalities,
												   Eigen::Ref<nlp_eq_jacobian_t> jacobian) noexcept
    {
        static_cast<const Derived*>(this)->equalities_linearised(var, p, equalities, jacobian);
    }

    
    // /**  NLP interface functions */
    // EIGEN_STRONG_INLINE void cost(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p, scalar_t &cost) noexcept
    // {
    //     static_cast<const Derived*>(this)->cost_impl(var, p, cost);
    // }

    // EIGEN_STRONG_INLINE void cost_gradient(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p,
    //                                        scalar_t &_cost, Eigen::Ref<nlp_variable_t> cost_gradient) noexcept
    // {
    //     static_cast<const Derived*>(this)->cost_gradient_impl(var, p, _cost, cost_gradient);
    // }

    // EIGEN_STRONG_INLINE void cost_gradient_hessian(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p,
    //                                                scalar_t &_cost, Eigen::Ref<nlp_variable_t> _cost_gradient, Eigen::Ref<nlp_hessian_t> hessian) noexcept
    // {
    //     static_cast<const Derived*>(this)->cost_gradient_hessian_impl(var, p, _cost, _cost_gradient, hessian);
    // }

    // EIGEN_STRONG_INLINE void equalities(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p,
    //                                     Eigen::Ref<nlp_constraints_t> equalities) noexcept
    // {
    //     static_cast<const Derived*>(this)->template equalities_impl<scalar_t>(var, p, equalities);
    // }

    // EIGEN_STRONG_INLINE void equalities_linearised(const Eigen::Ref<const nlp_variable_t>& var,
				// 								   const Eigen::Ref<const static_parameter_t>& p,
				// 								   Eigen::Ref<nlp_constraints_t> equalities,
				// 								   Eigen::Ref<nlp_eq_jacobian_t> jacobian) noexcept
    // {
    //     static_cast<const Derived*>(this)->equalities_linearised(var, p, equalities, jacobian);
    // }

};

#endif