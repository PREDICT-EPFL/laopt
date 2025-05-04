#ifndef LAOPT_ROCKET_OCP_HPP
#define LAOPT_ROCKET_OCP_HPP

// End user (level 1)

#include <limits>
#include <cmath>
#include <Eigen/Dense>

#include "laopt/laopt.hpp"
#include "laopt/tools/control_problem_base.hpp"

class RocketOcp : public laopt_tools::ControlProblemBase</*Scalar*/ double, /*NX*/ 12, /*NU*/ 4>
{
public:
    using Base = laopt_tools::ControlProblemBase<double, 12, 4>;
    using Scalar = typename Base::Scalar;
    using State = typename Base::State;
    using Input = typename Base::Input;

    template<typename T> using state_t = typename Base::template state_t<T>;
    template<typename T> using input_t = typename Base::template input_t<T>;
    template<typename T> using param_t = typename Base::template param_t<T>;

    const double mass = 1.7;                                                   // mass
    const double g = 9.81;                                                     // gravitational acceleration
    const Eigen::DiagonalMatrix<Scalar, 3> J{0.0644, 0.0644, 0.0128};          // inertia tensor
    const Eigen::DiagonalMatrix<Scalar, 3> J_inv{J.diagonal().cwiseInverse()}; // inverse of inertia tensor

    const Eigen::Vector<Scalar, 3> thrust_coeff{0, 0.03, 0}; // experimentally identified
    const Scalar torque_coeff = -0.1040;                     // experimentally identified
    const Eigen::Vector<Scalar, 3> r_F{0, 0, -0.215};        // thruster position in body frame

    Eigen::DiagonalMatrix<Scalar, Base::NX> Q;
    Eigen::Matrix<Scalar, Base::NX, Base::NX> P;
    Eigen::DiagonalMatrix<Scalar, Base::NU> R;

    Eigen::Vector<Scalar, 4> ref = Eigen::Vector<Scalar, 4>::Zero();

    static inline Scalar deg2rad(Scalar deg) {
        return deg * M_PI / 180.0;
    }

    RocketOcp()
    {
        u_ub << deg2rad(15), deg2rad(15), 80, 20;
        u_lb << deg2rad(-15), deg2rad(-15), 50, -20;
        x_ub(3) = deg2rad(7);
        x_ub(4) = deg2rad(7);
        x_lb = -x_ub;

        Eigen::Vector<Scalar, Base::NX> Sx_diag; // x scaling
        Sx_diag(Eigen::seqN(0, Eigen::fix<3>)).array() = deg2rad(60);
        Sx_diag(Eigen::seqN(3, Eigen::fix<3>)).array() = deg2rad(30);
        Sx_diag(Eigen::seqN(6, Eigen::fix<3>)).array() = 4;
        Sx_diag(Eigen::seqN(9, Eigen::fix<3>)).array() = 2;

        Eigen::Vector<Scalar, Base::NU> Su_diag; // u scaling
        Su_diag << u_ub(Eigen::seqN(0, Eigen::fix<2>)), u_ub(2) - u_lb(2), u_ub(3);

        Q.diagonal().array() = 1;
        // x, y tuning
        Q.diagonal()({6, 7}).array() = 10; // vx, vy
        Q.diagonal()({9, 10}).array() = 50; // x, y
        // z tuning
        Q.diagonal()(8) = 30; // vz
        Q.diagonal()(11) = 50; // z
        // roll tuning
        Q.diagonal()(5) = 10; // gamma

        R.diagonal() << 20, 20, 0.01, 0.01;

        Q.diagonal().array() /= Sx_diag.array();
        R.diagonal().array() /= Su_diag.array();

        P << 15.4821, -0.0000,  0.0000,   84.0947,  -0.0000,   0.0000,   0.0000,  -29.5967,  0.0000,   0.0000,  -42.2689,   0.0000,
             -0.0000, 15.4821, -0.0000,   -0.0000,  84.0947,  -0.0000,  29.5967,   -0.0000,  0.0000,  42.2689,   -0.0000,   0.0000,
             0.0000, -0.0000,  7.7941,    0.0000,  -0.0000,  18.9133,  -0.0000,   -0.0000, -0.0000,  -0.0000,    0.0000,   0.0000,
             84.0947, -0.0000,  0.0000,  599.0064,   0.0000,   0.0000,   0.0000, -212.5523,  0.0000,   0.0000, -306.7417,   0.0000,
             -0.0000, 84.0947, -0.0000,    0.0000, 599.0064,  -0.0000, 212.5523,   -0.0000,  0.0000, 306.7417,   -0.0000,   0.0000,
             0.0000, -0.0000, 18.9133,    0.0000,  -0.0000, 157.3161,  -0.0000,   -0.0000, -0.0000,  -0.0000,   -0.0000,  -0.0000,
             0.0000, 29.5967, -0.0000,    0.0000, 212.5523,  -0.0000,  93.0610,   -0.0000,  0.0000, 147.9222,   -0.0000,   0.0000,
             -29.5967, -0.0000, -0.0000, -212.5523,  -0.0000,  -0.0000,  -0.0000,   93.0610, -0.0000,  -0.0000,  147.9222,  -0.0000,
             0.0000,  0.0000, -0.0000,    0.0000,   0.0000,  -0.0000,   0.0000,   -0.0000, 11.1985,   0.0000,    0.0000,  12.5737,
             0.0000, 42.2689, -0.0000,    0.0000, 306.7417,  -0.0000, 147.9222,   -0.0000,  0.0000, 428.5270,   -0.0000,   0.0000,
             -42.2689, -0.0000,  0.0000, -306.7417,  -0.0000,  -0.0000,  -0.0000,  147.9222,  0.0000,  -0.0000,  428.5270,  -0.0000,
             0.0000,  0.0000,  0.0000,    0.0000,   0.0000,  -0.0000,   0.0000,   -0.0000, 12.5737,   0.0000,   -0.0000, 308.6925;
    }

    /* Override function implementations from base class ------------------------------ */
    template<typename x_t, typename u_t, typename p_t, typename t0_t, typename tf_t, typename tau_t,
            typename T = typename x_t::Scalar> // T is scalar type
    T lagrange_term_impl(const Eigen::MatrixBase<x_t>& x,
                         const Eigen::MatrixBase<u_t>& u,
                         const Eigen::MatrixBase<p_t>& p,
                         const Eigen::MatrixBase<t0_t>& t0,
                         const Eigen::MatrixBase<tf_t>& tf,
                         const tau_t& tau)
    {
        State x_ref = State::Zero();
        x_ref(Eigen::seqN(9, Eigen::fix<3>)) = ref(Eigen::seqN(0, Eigen::fix<3>));
        x_ref(5) = ref(3);

        state_t<T> x_err = (x_ref - x);
        return x_err.dot(Q * x_err) + u.dot(R * u);
    }

    template<typename xf_t, typename p_t, typename t0_t, typename tf_t,
            typename T = typename xf_t::Scalar> // T is scalar type
    T mayer_term_impl(const Eigen::MatrixBase<xf_t>& xf,
                      const Eigen::MatrixBase<p_t>& p,
                      const Eigen::MatrixBase<t0_t>& t0,
                      const Eigen::MatrixBase<tf_t>& tf)
    {
        State x_ref = State::Zero();
        x_ref(Eigen::seqN(9, Eigen::fix<3>)) = ref(Eigen::seqN(0, Eigen::fix<3>));
        x_ref(5) = ref(3);

        state_t<T> x_err = (x_ref - xf);
        return x_err.dot(P * x_err);
    }

    template<typename x_t, typename u_t, typename p_t,
            typename T = typename x_t::Scalar> // T is scalar type
    state_t<T> dynamics_impl(const Eigen::MatrixBase<x_t>& x,
                             const Eigen::MatrixBase<u_t>& u,
                             const Eigen::MatrixBase<p_t>& p)
    {
        Eigen::Vector<T, 3> w   = x(Eigen::seqN(0, Eigen::fix<3>));
        Eigen::Vector<T, 3> phi = x(Eigen::seqN(3, Eigen::fix<3>));
        Eigen::Vector<T, 3> v   = x(Eigen::seqN(6, Eigen::fix<3>));

        Eigen::Matrix<T, 3, 3> Twb = eul2mat(phi);

        Eigen::Vector<T, 6> b_F_and_b_M = get_force_and_moment_from_thrust(u);
        Eigen::Vector<T, 3> b_F = b_F_and_b_M(Eigen::seqN(0, Eigen::fix<3>));
        Eigen::Vector<T, 3> b_M = b_F_and_b_M(Eigen::seqN(3, Eigen::fix<3>));

        T bet = phi(1);
        T gam = phi(2);

        Eigen::Matrix<T, 3, 3> E_inv;
        E_inv << cos(gam),             -sin(gam),           T(0),
                 sin(gam) * cos(bet),  cos(gam) * cos(bet), T(0),
                 -cos(gam) * sin(bet), sin(gam) * sin(bet), cos(bet);
        E_inv *= 1.0 / cos(bet);

        Eigen::Vector<T, 3> w_dot = J_inv * (b_M - w.template cross(J * w));

        Eigen::Vector<T, 3> phi_dot = E_inv * w;

        Eigen::Vector<T, 3> v_dot = Twb * b_F / mass;
        v_dot(2) -= g;

        Eigen::Vector<T, 3> p_dot = v;

        state_t<T> x_dot;
        x_dot << w_dot, phi_dot, v_dot, p_dot;
        return x_dot;
    }

    template<typename T>
    EIGEN_STRONG_INLINE Eigen::Matrix<T, 3, 3> eul2mat(const Eigen::Vector<T, 3>& phi)
    {
        T alp = phi(0);
        T bet = phi(1);
        T gam = phi(2);

        Eigen::Matrix<T, 3, 3> T1;
        T1 << T(1), T(0),       T(0),
              T(0), cos(-alp),  sin(-alp),
              T(0), -sin(-alp), cos(-alp);

        Eigen::Matrix<T, 3, 3> T2;
        T2 << cos(-bet), T(0), -sin(-bet),
              T(0),      T(1), T(0),
              sin(-bet), T(0), cos(-bet);

        Eigen::Matrix<T, 3, 3> T3;
        T3 << cos(-gam),  sin(-gam), T(0),
              -sin(-gam), cos(-gam), T(0),
              T(0),       T(0),      T(1);

        return T1 * T2 * T3;
    }

    template<typename U, typename T = typename U::Scalar> // T is scalar type
    EIGEN_STRONG_INLINE Eigen::Vector<T, 6> get_force_and_moment_from_thrust(const Eigen::MatrixBase<U>& u)
    {
        T thrust = g * (thrust_coeff(0) * u(2) * u(2) + thrust_coeff(1) * u(2) + thrust_coeff(2));
        T torque = torque_coeff * J.diagonal()(2) * u(3);

        Eigen::Vector<T, 3> b_eF;
        b_eF << sin(u(1)), -sin(u(0)) * cos(u(1)), cos(u(1));

        Eigen::Vector<T, 3> b_F = thrust * b_eF;
        Eigen::Vector<T, 3> b_M = torque * b_eF + r_F.template cross(b_F);

        Eigen::Vector<T, 6> b_F_and_b_M;
        b_F_and_b_M << b_F, b_M;
        return b_F_and_b_M;
    }
};

#endif //LAOPT_ROCKET_OCP_HPP
