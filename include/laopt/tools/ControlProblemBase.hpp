#ifndef LAOPT_CTRLPROBLEMBASE_HPP
#define LAOPT_CTRLPROBLEMBASE_HPP

#include <Eigen/Dense>
#include "laopt/laopt.hpp"

namespace laopt_tools {

template<typename cScalar, int cNX, int cNU, int cNP = 0>
class ControlProblemBase
{
public:
    /* Wrap template parameters */
    using Scalar = cScalar;
    static const int NX = cNX;
    static const int NU = cNU;
    static const int NP = cNP;

    /* Define state and input types */
    template<typename T> using state_t = Eigen::Vector<T, NX>;
    template<typename T> using input_t = Eigen::Vector<T, NU>;
    template<typename T> using param_t = Eigen::Vector<T, NP>;
    using State = state_t<Scalar>;
    using Input = input_t<Scalar>;
    using Param = param_t<Scalar>;

    /* Static parameters */
    Scalar t0 = 0;

    /* Bounds on state and input */
    Input u_ub = Input::Constant(std::numeric_limits<Scalar>::infinity());
    Input u_lb = -u_ub;
    State x_ub = State::Constant(std::numeric_limits<Scalar>::infinity());
    State x_lb = -x_ub;

    State x0_ub = State::Constant(std::numeric_limits<Scalar>::infinity());
    State x0_lb = -x0_ub;
    State xf_ub = State::Constant(std::numeric_limits<Scalar>::infinity());
    State xf_lb = -xf_ub;

    /* Final time bounds */
    Scalar tf_lb{1}, tf_ub{1};

    /* Additional decision variables (optimized parameters) */
    class OptParam
    {
    public:
        /* Method allows to extract a reference to a (writeable) segment of the parameter vector */
        template<unsigned LEN>
        Eigen::Ref<Eigen::Vector<Scalar, LEN>>
        get_parameter(unsigned index) { return m_param_vector.template segment<LEN>(index); }

        /* Methods to write and read parameter vector */
        Param &vector() { return m_param_vector; }
        const Param &vector() const { return m_param_vector; }

        template<unsigned LEN>
        using VecRef = Eigen::Ref<Eigen::Vector<Scalar, LEN>>;
    private:
        Param m_param_vector;
    };
    OptParam opt_params_lb, opt_params_ub; // Serve as placeholder in case child class does not define them

    /* Convenience setters for zero-range bounds */
    void set_x0(const State &x0) { x0_lb = x0_ub = x0; }
    void set_xf(const State &xf) { xf_lb = xf_ub = xf; }
    void set_tf(const Scalar &tf) { tf_lb = tf_ub = tf; }

    /*
     * Templates for problem formulation
     */
    template<typename T>
    // T is scalar type
    T lagrange_term_impl(const Eigen::Ref<const state_t<T>> &x,
                         const Eigen::Ref<const input_t<T>> &u,
                         const Eigen::Ref<const param_t<T>> &p) { return static_cast<T>(0); }

    template<typename T>
    // T is scalar type
    T mayer_term_impl(const Eigen::Ref<const state_t<T>> &x,
                      const Eigen::Ref<const param_t<T>> &p,
                      const Eigen::Ref<const T> &tf) { return static_cast<T>(0); }

    template<typename T>
    // T is scalar type
    state_t<T> dynamics_impl(const Eigen::Ref<const state_t<T>> &x,
                             const Eigen::Ref<const input_t<T>> &u,
                             const Eigen::Ref<const param_t<T>> &p)
    {
        assert(false && "dynamics must be implemented");
        return state_t<T>();
    }

    // TODO: Add template for box constraints / nonlinear constraints?
};

} // namespace laopt_tools

#endif //LAOPT_CTRLPROBLEMBASE_HPP
