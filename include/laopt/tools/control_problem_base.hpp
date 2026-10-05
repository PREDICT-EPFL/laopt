#ifndef LAOPT_CONTROL_PROBLEM_BASE_HPP
#define LAOPT_CONTROL_PROBLEM_BASE_HPP

#include <iostream>
#include <iomanip>

#include <Eigen/Dense>
#include "laopt/laopt.hpp"
#include "laopt/tools/constants.hpp"

namespace laopt_tools {

/* ControlProblemBase template parameters:
 * cNumeric:           Numeric scalar type
 * cNX, cNU, cNP:     Length of state, input, optimization parameter
 * cNG cNG0, cNGF:    Number of inequality constraints (elsewhere, initial, final)
 * cOptions:          Problem options (free/fixed end time, discrete dynamics)
 * */
template<typename cNumeric,
         int cNX, int cNU, int cNP = 0,
         int cNG = 0, int cNG0 = 0, int cNGF = 0,
         int cOptions = FixedEndTime>
class ControlProblemBase
{
public:
    /* Wrap template parameters */
    using Numeric = cNumeric;
    /* Deprecated alias of Numeric, kept for backwards compatibility. */
    using Scalar [[deprecated("Use Numeric instead.")]] = Numeric;
    static const int NX = cNX;
    static const int NU = cNU;
    static const int NP = cNP;

    static const int NG = cNG;
    static const int NG0 = cNG0;
    static const int NGF = cNGF;

    static const int Options = cOptions;

    static_assert((cOptions & (DiscreteDynamics | FreeEndTime)) != (DiscreteDynamics | FreeEndTime),
                  "ControlProblemBase: DiscreteDynamics is not compatible with FreeEndTime.");

    /* Define state and input types */
    template<typename T> using state_t = Eigen::Vector<T, NX>;
    template<typename T> using input_t = Eigen::Vector<T, NU>;
    template<typename T> using param_t = Eigen::Vector<T, NP>;

    template<typename T> using ineq_constr_t = Eigen::Vector<T, NG>;
    template<typename T> using ineq_constr0_t = Eigen::Vector<T, NG0>;
    template<typename T> using ineq_constrf_t = Eigen::Vector<T, NGF>;

    /* Numeric state and input types */
    using State = state_t<Numeric>;
    using Input = input_t<Numeric>;
    using Param = param_t<Numeric>;
    using IneqBound = ineq_constr_t<Numeric>;
    using Ineq0Bound = ineq_constr0_t<Numeric>;
    using IneqfBound = ineq_constrf_t<Numeric>;

    /* Static parameters */
    Numeric t0 = 0;

    /* Bounds on state and input */
    Input u_ub = Input::Constant(std::numeric_limits<Numeric>::infinity());
    Input u_lb = -u_ub;
    State x_ub = State::Constant(std::numeric_limits<Numeric>::infinity());
    State x_lb = -x_ub;

    State x0_ub = State::Constant(std::numeric_limits<Numeric>::infinity());
    State x0_lb = -x0_ub;
    State xf_ub = State::Constant(std::numeric_limits<Numeric>::infinity());
    State xf_lb = -xf_ub;

    /* Bounds on inequality constraints */
    IneqBound g_ub = IneqBound::Zero();
    IneqBound g_lb = IneqBound::Constant(-std::numeric_limits<Numeric>::infinity());
    Ineq0Bound g0_ub = Ineq0Bound::Zero();
    Ineq0Bound g0_lb = Ineq0Bound::Constant(-std::numeric_limits<Numeric>::infinity());
    IneqfBound gf_ub = IneqfBound::Zero();
    IneqfBound gf_lb = IneqfBound::Constant(-std::numeric_limits<Numeric>::infinity());

    /* Final time bounds */
    Numeric tf_lb{1}, tf_ub{1};

    /* Additional decision variables (optimized parameters) bound */
    Param p_lb = Param::Constant(std::numeric_limits<Numeric>::infinity());
    Param p_ub = -p_lb;

    /* Convenience setters for zero-range bounds */
    void set_x0(const State &x0) { x0_lb = x0_ub = x0; }
    void set_xf(const State &xf) { xf_lb = xf_ub = xf; }
    void set_tf(const Numeric &tf) { tf_lb = tf_ub = tf; }

    /* Diagnosis */
    void print_problem_dimension() const
    {
        std::cout << std::setprecision(4) << std::defaultfloat;
        std::cout << "Diagnostics: ControlProblem with \n"
                  << "NX = " << NX << ", NU = " << NU << ", NP = " << NP << "\n"
                  << "NG = " << NG << ", NG0 = " << NG0 << ", NGF = " << NGF << "\n"
                  << "End time: " << ((Options & FreeEndTime) ? "free" : "fixed") << "\n";
    }
    void print_diagnostics() const
    {
        print_problem_dimension();

        std::cout << std::setprecision(4) << std::defaultfloat;
        std::cout << "ubu: " << u_ub.transpose() << "\n"
                  << "lbu: " << u_lb.transpose() << "\n"
                  << "ubx: " << x_ub.transpose() << "\n"
                  << "lbx: " << x_lb.transpose() << "\n";

        if (x0_lb == x0_ub) { std::cout << "x0: " << x0_lb.transpose() << "\n"; }
        else
        {
            std::cout << "x0_ub: " << x0_ub.transpose() << "\n"
                      << "x0_lb: " << x0_lb.transpose() << "\n";
        }
        if (xf_lb == xf_ub) { std::cout << "xf: " << xf_lb.transpose() << "\n"; }
        else
        {
            std::cout << "xf_ub: " << xf_ub.transpose() << "\n"
                      << "xf_lb: " << xf_lb.transpose() << "\n";
        }
        if (tf_lb == tf_ub) { std::cout << "tf: " << tf_lb << "\n"; }
        else { std::cout << "tf: [" << tf_lb << ", " << tf_ub << "]\n"; }
    }

    /*
     * Templates for problem formulation
     */
    /* Convenience function to silence unused parameter compiler warnings */
    template <typename... Ts>
    constexpr void unused(Ts&&...) noexcept {}

    template<typename x_t, typename u_t, typename p_t, typename t0_t, typename tf_t, typename tau_t,
             typename T = typename x_t::Scalar> // T is scalar type
    T lagrange_term_impl(const Eigen::MatrixBase<x_t>& x,
                         const Eigen::MatrixBase<u_t>& u,
                         const Eigen::MatrixBase<p_t>& p,
                         const Eigen::MatrixBase<t0_t>& t0,
                         const Eigen::MatrixBase<tf_t>& tf,
                         const tau_t& tau)
    {
        unused(x, u, p, t0, tf, tau);
        return static_cast<T>(0);
    }

    template<typename x_tf, typename p_t, typename t0_t, typename tf_t,
             typename T = typename x_tf::Scalar> // T is scalar type
    T mayer_term_impl(const Eigen::MatrixBase<x_tf>& xf,
                      const Eigen::MatrixBase<p_t>& p,
                      const Eigen::MatrixBase<t0_t>& t0,
                      const Eigen::MatrixBase<tf_t>& tf)
    {
        unused(xf, p, t0, tf);
        return static_cast<T>(0);
    }

    template<typename x_t, typename u_t, typename p_t, typename t0_t, typename tf_t, typename tau_t,
             typename T = typename x_t::Scalar> // T is scalar type
    state_t<T> dynamics_impl(const Eigen::MatrixBase<x_t>& x,
                             const Eigen::MatrixBase<u_t>& u,
                             const Eigen::MatrixBase<p_t>& p,
                             const Eigen::MatrixBase<t0_t>& t0,
                             const Eigen::MatrixBase<tf_t>& tf,
                             const tau_t& tau)
    {
        // Only ever instantiated if this is actually called and not overridden by user ControlProblem.
        static_assert(sizeof(x_t) == 0, "dynamics_impl() not implemented.");
        unused(x, u, p, t0, tf, tau);
        return state_t<T>();
    }

    template<typename x_t, typename u_t, typename p_t, typename t0_t, typename tf_t, typename tau_t,
             typename T = typename x_t::Scalar> // T is scalar type
    state_t<T> discrete_dynamics_impl(const Eigen::MatrixBase<x_t>& x,
                                      const Eigen::MatrixBase<u_t>& u,
                                      const Eigen::MatrixBase<p_t>& p,
                                      const Eigen::MatrixBase<t0_t>& t0,
                                      const Eigen::MatrixBase<tf_t>& tf,
                                      const tau_t& tau)
    {
        // Only ever instantiated if this is actually called and not overridden by user ControlProblem.
        static_assert(sizeof(x_t) == 0, "discrete_dynamics_impl() not implemented, but selected via DiscreteDynamics option.");
        unused(x, u, p, t0, tf, tau);
        return state_t<T>();
    }

    /* Inequality constraints */
    template<typename x_t, typename u_t, typename p_t, typename t0_t, typename tf_t, typename tau_t,
             typename T = typename x_t::Scalar> // T is scalar type
    ineq_constr_t<T> inequality_constraints_impl(const Eigen::MatrixBase<x_t>& x,
                                                 const Eigen::MatrixBase<u_t>& u,
                                                 const Eigen::MatrixBase<p_t>& p,
                                                 const Eigen::MatrixBase<t0_t>& t0,
                                                 const Eigen::MatrixBase<tf_t>& tf,
                                                 const tau_t& tau)
    {
        // Only ever instantiated if this is actually called and not overridden by user ControlProblem.
        static_assert(NG == 0, "NG > 0 but inequality_constraints_impl() not implemented.");
        unused(x, u, p, t0, tf, tau);
        return {};
    }

    template<typename x_t, typename u_t, typename p_t, typename t0_t,
             typename T = typename x_t::Scalar> // T is scalar type
    ineq_constr0_t<T> inequality_constraints0_impl(const Eigen::MatrixBase<x_t>& x0,
                                                   const Eigen::MatrixBase<u_t>& u0,
                                                   const Eigen::MatrixBase<p_t>& p,
                                                   const Eigen::MatrixBase<t0_t>& t0)
    {
        // Only ever instantiated if this is actually called and not overridden by user ControlProblem.
        static_assert(NG0 == 0, "NG0 > 0 but inequality_constraints0_impl() not implemented.");
        unused(x0, u0, p, t0);
        return {};
    }

    template<typename x_tf, typename p_t, typename t0_t, typename tf_t,
             typename T = typename x_tf::Scalar> // T is scalar type
    ineq_constrf_t<T> inequality_constraintsf_impl(const Eigen::MatrixBase<x_tf>& xf,
                                                   const Eigen::MatrixBase<p_t>& p,
                                                   const Eigen::MatrixBase<t0_t>& t0,
                                                   const Eigen::MatrixBase<tf_t>& tf)
    {
        // Only ever instantiated if this is actually called and not overridden by user ControlProblem.
        static_assert(NGF == 0, "NGF > 0 but inequality_constraintsf_impl() not implemented.");
        unused(xf, p, t0, tf);
        return {};
    }
};

} // namespace laopt_tools

#endif // LAOPT_CONTROL_PROBLEM_BASE_HPP
