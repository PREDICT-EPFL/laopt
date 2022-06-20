#ifndef LAMPC_OCP_BASE_HPP
#define LAMPC_OCP_BASE_HPP

#include <Eigen/Dense>

namespace lampc{

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

}

#endif //LAMPC_OCP_BASE_HPP
