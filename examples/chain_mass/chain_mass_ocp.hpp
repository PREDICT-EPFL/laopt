#ifndef LAOPT_CHAIN_MASS_OCP_HPP
#define LAOPT_CHAIN_MASS_OCP_HPP

// End user (level 1)

#include <limits>
#include <Eigen/Dense>

#include "laopt/laopt.hpp"
#include "laopt/tools/control_problem_base.hpp"
#include "laopt/differentiable_functions/quadratic_cost.hpp"

template<int cM = 5, int cNX = 3 * (2 * (cM - 2) + 1), int cNU = 3>
class ChainMassOcp : public laopt_tools::ControlProblemBase</*Scalar*/ double, cNX, cNU>
{
public:
    using Base = laopt_tools::ControlProblemBase<double, cNX, cNU>;
    using Scalar = typename Base::Scalar;
    using State = typename Base::State;

    template<typename T> using state_t = typename Base::template state_t<T>;
    template<typename T> using input_t = typename Base::template input_t<T>;
    template<typename T> using param_t = typename Base::template param_t<T>;

    static const int M = cM;
    const double m = 0.033; // mass of the balls
    const double D = 1.0; // spring constant
    const double L = 0.033; // rest length of spring

    const Eigen::Vector<double, 3> x0{0, 0, 0}; // fix mass (at wall)

    laopt::QuadraticCost<Scalar, Base::NX> state_cost;
    laopt::QuadraticCost<Scalar, Base::NU> input_cost;
    laopt::QuadraticCost<Scalar, Base::NX> final_state_cost;

    ChainMassOcp()
    {
        state_cost.x_ref.setZero();
        state_cost.x_ref(3 * (M - 2)) = 7.5;
        final_state_cost.x_ref.setZero();
        final_state_cost.x_ref(3 * (M - 2)) = 7.5;

        state_cost.Q.setZero();
        state_cost.Q.diagonal()(Eigen::seqN(3 * (M - 2), Eigen::fix<3>)).array() = 2.5;
        state_cost.Q.diagonal()(Eigen::lastN(Eigen::fix<3 * (M - 2)>)).array() = 25;

        input_cost.Q.setZero();
        input_cost.Q.diagonal().array() = 0.1;

        final_state_cost.Q.setZero();
        final_state_cost.Q.diagonal()(Eigen::seqN(3 * (M - 2), Eigen::fix<3>)).array() = 10;
    }

    /* Override function implementations from base class ------------------------------ */
    template<typename X, typename U, typename P>
    auto lagrange_term_impl(const Eigen::MatrixBase<X>& x,
                            const Eigen::MatrixBase<U>& u,
                            const Eigen::MatrixBase<P>& p)
    {
        return state_cost(x) + input_cost(u);
    }

    template<typename Xf, typename P, typename Ttf>
    auto mayer_term_impl(const Eigen::MatrixBase<Xf>& xf,
                         const Eigen::MatrixBase<P>& p,
                         const Ttf& tf)
    {
        return final_state_cost(xf);
    }

    template<typename X, typename U, typename P, typename T = typename X::Scalar> // T is scalar type
    state_t<T> dynamics_impl(const Eigen::MatrixBase<X>& x,
                             const Eigen::MatrixBase<U>& u,
                             const Eigen::MatrixBase<P>& p)
    {
        // x = (x_pos((M-1)*3), x_vel((M-2)*3))

        Eigen::Vector<T, 3 * (M - 2)> f; // force on intermediate masses
        f.setZero();

        for (Eigen::Index i = 0; i < M - 2; i++) {
            f(3 * i + 2) = -9.81;
        }

        for (Eigen::Index i = 0; i < M - 1; i++) {
            Eigen::Vector<T, 3> dist;
            if (i == 0) {
                dist = x(Eigen::seq(i*3, (i+1)*3-1)) - x0;
            } else {
                dist = x(Eigen::seq(i*3, (i+1)*3-1)) - x(Eigen::seq((i-1)*3, i*3-1));
            }
            T scale = D / m * (1 - L / dist.norm());
            Eigen::Vector<T, 3> F = scale * dist;

            // mass on the right
            if (i < M - 2) {
                f(Eigen::seq(i*3, (i+1)*3-1)) -= F;
            }

            // mass on the left
            if (i > 0) {
                f(Eigen::seq((i-1)*3, i*3-1)) += F;
            }
        }

        state_t<T> x_dot;
        x_dot << x(Eigen::lastN((M-2)*3)), u, f;
        return x_dot;
    }
};

#endif //LAOPT_CHAIN_MASS_OCP_HPP
