#ifndef LAMPC_DOUBLE_INTEGRATOR_OCP_HPP
#define LAMPC_DOUBLE_INTEGRATOR_OCP_HPP

#include <limits>
#include <Eigen/Dense>

// Simple user level

// Provided my lampc
template<typename cScalar, int cNX, int cNU>
class OCPBase
{
public:
    using scalar_t = cScalar;
    static const int NX = cNX;
    static const int NU = cNU;

    template<typename T>
    using state_t = Eigen::Vector<T, NX>;
    template<typename T>
    using control_t = Eigen::Vector<T, NU>;

    scalar_t tf = 1;
    state_t<scalar_t> x0;

    state_t<scalar_t> lbx = state_t<scalar_t>::Constant(-std::numeric_limits<scalar_t>::infinity());
    state_t<scalar_t> ubx = state_t<scalar_t>::Constant(std::numeric_limits<scalar_t>::infinity());
    control_t<scalar_t> lbu = control_t<scalar_t>::Constant(-std::numeric_limits<scalar_t>::infinity());
    control_t<scalar_t> ubu = control_t<scalar_t>::Constant(std::numeric_limits<scalar_t>::infinity());

    template<typename T>
    inline void dynamics_impl(const Eigen::Ref<const state_t<T>> &x, const Eigen::Ref<const control_t<T>> &u,
                              Eigen::Ref<state_t<T>> x_dot) const noexcept
    {
        assert(false && "dynamics must be implemented");
    }

    template<typename T>
    inline void lagrange_term_impl(const Eigen::Ref<const state_t<T>> &x, const Eigen::Ref<const control_t<T>> &u,
                                   T &lagrange) noexcept
    {
        lagrange = (T) 0;
    }

    template<typename T>
    inline void mayer_term_impl(const Eigen::Ref<const state_t<T>> &x, T &mayer) noexcept
    {
        mayer = (T) 0;
    }
};


class DoubleIntegratorOCP : public OCPBase<double, 2, 1>
{
public:
    Eigen::Matrix<scalar_t, NX, NX> A{{0,1},{0,0}};
    Eigen::Matrix<scalar_t, NX, NU> B{{0},{1}};

    Eigen::Matrix<scalar_t, NX, NX> Q{{10,0},{0,1}};
    Eigen::Matrix<scalar_t, NU, NU> R{{0.01}};

    scalar_t tf = 2;
    state_t<scalar_t> ref{3, 0};

    state_t<scalar_t> lbx{-1, -2};
    control_t<scalar_t> lbu{-3};
    control_t<scalar_t> ubu{5};

    template<typename T>
    inline void dynamics_impl(const Eigen::Ref<const state_t<T>> &x, const Eigen::Ref<const control_t<T>> &u,
                              Eigen::Ref<state_t<T>> x_dot) const noexcept
    {
        x_dot = A * x + B * u;
    }

    template<typename T>
    inline void lagrange_term_impl(const Eigen::Ref<const state_t<T>> &x, const Eigen::Ref<const control_t<T>> &u,
                                   T &lagrange) noexcept
    {
        lagrange = (ref - x).dot(Q * (ref - x)) + u.dot(R * u);
    }

    template<typename T>
    inline void mayer_term_impl(const Eigen::Ref<const state_t<T>> &x, T &mayer) noexcept
    {
        mayer = (ref - x).dot(Q * (ref - x));
    }
};

#endif //LAMPC_DOUBLE_INTEGRATOR_OCP_HPP
