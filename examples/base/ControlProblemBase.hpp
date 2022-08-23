#ifndef LAOPT_CTRLPROBLEMBASE_HPP
#define LAOPT_CTRLPROBLEMBASE_HPP

#include <Eigen/Dense>
#include "laopt/laopt.hpp"

template<typename cScalar, int cNX, int cNU>
class ControlProblemBase
{
public:
    /* Wrap template parameters */
    using Scalar = cScalar;
    static const int NX = cNX;
    static const int NU = cNU;

    template<int n>
    using variable_t = laopt::Variable<Scalar, n>;

    /* Define state and input types */
    template<typename T> using state_t = Eigen::Vector<T, NX>;
    template<typename T> using input_t = Eigen::Vector<T, NU>;
    using State = state_t<Scalar>;
    using Input = input_t<Scalar>;

    /* Member variables */
    Scalar t0 = 0;
    Scalar tf = 1;

//    variable_t<1> tf_var;
//    double w_tf{0};

    Input ubu = Input::Constant(std::numeric_limits<Scalar>::infinity());
    Input lbu = -ubu;
    State ubx = State::Constant(std::numeric_limits<Scalar>::infinity());
    State lbx = -ubx;

    State x0_ub = State::Constant(std::numeric_limits<Scalar>::infinity());
    State x0_lb = -x0_ub;
    void set_x0(const State &x0_) { x0_lb = x0_ub = x0_; }

    State xf_ub = State::Constant(std::numeric_limits<Scalar>::infinity());
    State xf_lb = -xf_ub;
    void set_xf(const State &xf_) { xf_lb = xf_ub = xf_; }

    /* Templates for problem formulation */
    template<typename T>
    T lagrange_term_impl(const Eigen::Ref<const state_t<T>> &x,
                         const Eigen::Ref<const input_t<T>> &u) { return static_cast<T>(0); }

    template<typename T>
    T mayer_term_impl(const Eigen::Ref<const state_t<T>> &x) { return static_cast<T>(0); }

    template<typename T>
    state_t<T> dynamics_impl(const Eigen::Ref<const state_t<T>> &x,
                             const Eigen::Ref<const input_t<T>> &u)
    {
        assert(false && "dynamics must be implemented");
        return state_t<T>();
    }

    // TODO: Add template for box constraints / nonlinear constraints?
};

#endif //LAOPT_CTRLPROBLEMBASE_HPP
