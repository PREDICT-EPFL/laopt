#ifndef LAOPT_CTRLPROBLEMBASE_HPP
#define LAOPT_CTRLPROBLEMBASE_HPP

#include <Eigen/Dense>
#include "laopt/laopt.hpp"

template<typename cScalar, int cNX, int cNU, int cNP = 1>
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

    /* Member variables */
    Scalar t0 = 0;
    Scalar tf_lb{1}, tf_ub{1};
    void set_tf(const Scalar &tf) { tf_lb = tf_ub = tf;}

    Input ubu = Input::Constant(std::numeric_limits<Scalar>::infinity());
    Input lbu = -ubu;
    State ubx = State::Constant(std::numeric_limits<Scalar>::infinity());
    State lbx = -ubx;

    State x0_ub = State::Constant(std::numeric_limits<Scalar>::infinity());
    State x0_lb = -x0_ub;
    void set_x0(const State &x0) { x0_lb = x0_ub = x0; }

    State xf_ub = State::Constant(std::numeric_limits<Scalar>::infinity());
    State xf_lb = -xf_ub;
    void set_xf(const State &xf) { xf_lb = xf_ub = xf; }

    /* Templates for problem formulation */
    template<typename T> // T is scalar type
    T lagrange_term_impl(const Eigen::Ref<const state_t<T>> &x,
                         const Eigen::Ref<const input_t<T>> &u,
                         const Eigen::Ref<const param_t<T>> &p) { return static_cast<T>(0); }

    template<typename T> // T is scalar type
    T mayer_term_impl(const Eigen::Ref<const state_t<T>> &x,
                      const Eigen::Ref<const param_t<T>> &p,
                      const Eigen::Ref<const T> &tf) { return static_cast<T>(0); }

    template<typename T> // T is scalar type
    state_t<T> dynamics_impl(const Eigen::Ref<const state_t<T>> &x,
                             const Eigen::Ref<const input_t<T>> &u,
                             const Eigen::Ref<const param_t<T>> &p)
    {
        assert(false && "dynamics must be implemented");
        return state_t<T>();
    }

    // TODO: Add template for box constraints / nonlinear constraints?
};

#endif //LAOPT_CTRLPROBLEMBASE_HPP
