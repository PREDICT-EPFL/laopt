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
    template<typename T> using constref_t = const Eigen::Ref<const T>;
    template<typename T> using ref_t = Eigen::Ref<T>;

    /* Member variables */
    Scalar t0 = 0;
    Scalar tf = 1;

    Input ubu = Input::Constant(std::numeric_limits<Scalar>::infinity());
    Input lbu = -ubu;
    State ubx = State::Constant(std::numeric_limits<Scalar>::infinity());
    State lbx = -ubx;

    State x0_ub = State::Constant(std::numeric_limits<Scalar>::infinity());
    State x0_lb = -x0_ub;
    void set_x0(const State &x0_) { x0_lb = x0_ub = x0_; }
    void set_x0(const State &lbx0_, const State &ubx0_)
    {
        x0_lb = lbx0_;
        x0_ub = ubx0_;
    }

    State xf_ub = State::Constant(std::numeric_limits<Scalar>::infinity());
    State xf_lb = -xf_ub;
    void set_xf(const State &xf_) { xf_lb = xf_ub = xf_; }
    void set_xf(const State &lbxf_, const State &ubxf_)
    {
        xf_lb = lbxf_;
        xf_ub = ubxf_;
    }

    /* Templates for problem formulation */
    template<typename T>
    void lagrange_term_impl(T &lagrange,
                            constref_t<state_t<T>> &x,
                            constref_t<input_t<T>> &u)
    {
        lagrange = static_cast<T>(0);
    }

    template<typename T>
    void mayer_term_impl(T &mayer,
                         constref_t<state_t<T>> &x)
    {
        mayer = static_cast<T>(0);
    }

    template<typename T>
    void dynamics_impl(ref_t<state_t<T>> x_dot,
                       constref_t<state_t<T>> &x,
                       constref_t<input_t<T>> &u)
    {
        assert(false && "dynamics must be implemented");
    }

    // TODO: Add template for box constraints / nonlinear constraints?
};

#endif //LAOPT_CTRLPROBLEMBASE_HPP
