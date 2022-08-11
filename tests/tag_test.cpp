#include <iostream>

#include "lampc.hpp"
#include "lampc_function_tag.hpp"
#include "lampc_functor.hpp"
#include "lampc_function_library.hpp"

template<typename scalar_t>
struct User : public laopt::Differentiable<User<scalar_t>>
{
    static constexpr int nx = 2;
    static constexpr int nu = 1;

    template<typename T> using state_t = Eigen::Vector<T, nx>;
    template<typename T> using input_t = Eigen::Vector<T, nu>;

    Eigen::Matrix<scalar_t, 2, 2> A{{1,2},{3,4}};
    Eigen::Matrix<scalar_t, 2, 1> B{{1},{3}};

    // Example of function with Eigen autodiff for jacobian
    struct Sys {};
    template<typename X, typename U, typename Scalar = typename Eigen::MatrixBase<X>::Scalar>
    EIGEN_STRONG_INLINE state_t<Scalar>
    function_impl(Sys, const Eigen::MatrixBase<X> &x, const Eigen::MatrixBase<U> &u) noexcept
    {
        return A * x + B * u;
    }

    // Example of function with user-defined jacobian
    struct SysX {};
    template<typename X, typename U, typename Scalar = typename Eigen::MatrixBase<X>::Scalar>
    EIGEN_STRONG_INLINE state_t<Scalar>
    function_impl(SysX, const Eigen::MatrixBase<X> &x, const Eigen::MatrixBase<U> &u) noexcept
    {
        return A * x + B * u;
    }

    template<typename OutValue, typename OutJacobian, typename X, typename U>
    inline void jacobian_impl(SysX, OutValue &out, OutJacobian &jac, const Eigen::MatrixBase<X> &x, const Eigen::MatrixBase<U> &u) noexcept
    {
        std::cout << "jacobian_impl called" << std::endl;

        out = this->function(SysX{}, x, u);

        jac(Eigen::all, Eigen::seqN(0, nx)) = A;
        jac(Eigen::all, Eigen::seqN(nx, nu)) = B;
    }

    // Example of using RK4 in a user function
    struct SysRK4 {};
    template<typename X, typename U, typename Scalar = typename Eigen::MatrixBase<X>::Scalar>
    EIGEN_STRONG_INLINE state_t<Scalar>
    function_impl(SysRK4, const Eigen::MatrixBase<X> &x, const Eigen::MatrixBase<U> &u) noexcept
    {
        return rk4_sys.function(x, u);
    }

    struct NLSys {};
    template<typename X, typename U, typename Scalar = typename Eigen::MatrixBase<X>::Scalar>
    EIGEN_STRONG_INLINE state_t<Scalar>
    function_impl(NLSys, const Eigen::MatrixBase<X> &x, const Eigen::MatrixBase<U> &u) noexcept
    {
        return x(0) * u(0) * (A * x + B * u);
    }

    laopt::functions::RK4<User<scalar_t>, double, Sys> rk4_sys;
    laopt::Function<User<scalar_t>, Sys> sys;

    User() : rk4_sys(*this, 0.2), sys(*this) {}

};


int main()
{
    using User = User<double>;

    User user;

    User::state_t<double> x;
    User::input_t<double> u;
    x << 1, 2;
    u << 3;
    User::state_t<double> val;
    Eigen::Matrix<double, 2, 3> J;
    Eigen::Vector<double, 3> p;
    p << 1, 2, 3;

    std::cout << "\n=== CALLING Function for SYS ===" << std::endl;
    val = user.function(User::Sys{}, x, u);
    std::cout << "user(Sys, x,u) = " << val.transpose() << std::endl;

    std::cout << "\n=== CALLING Jacobian FOR SYS ===" << std::endl;
    user.jacobian(User::Sys{}, val, J, x, u);
    std::cout << "Jacobian = \n" << J << std::endl;
    std::cout << "val = " << val.transpose() << std::endl;

    std::cout << "\n=== CALLING Function for SYS FUNCTOR ===" << std::endl;
    val = user.sys.function(x, u);
    std::cout << "user(Sys, x,u) = " << val.transpose() << std::endl;

    std::cout << "\n=== CALLING Jacobian FOR SYS FUNCTOR ===" << std::endl;
    user.sys.jacobian(val, J, x, u);
    std::cout << "Jacobian = \n" << J << std::endl;
    std::cout << "val = " << val.transpose() << std::endl;

    std::cout << "\n=== CALLING Function on SYSX ===" << std::endl;
    val = user.function(User::SysX{}, x, u);
    std::cout << "val = " << val.transpose() << std::endl;

    std::cout << "\n=== CALLING custom Jacobian on SYSX ===" << std::endl;
    user.jacobian(User::SysX{}, val, J, x, u);
    std::cout << "val = " << val.transpose() << std::endl;
    std::cout << "Jacobian = \n" << J << std::endl;

    std::cout << "\n=== CALLING Function for SYSRK4 ===" << std::endl;
    val = user.function(User::SysRK4{}, x, u);
    std::cout << "user(SysRK4, x,u) = " << val.transpose() << std::endl;

    std::cout << "\n=== CALLING Jacobian FOR SYSRK4 ===" << std::endl;
    user.jacobian(User::SysRK4{}, val, J, x, u);
    std::cout << "Jacobian = \n" << J << std::endl;
    std::cout << "val = " << val.transpose() << std::endl;

    std::cout << "\n=== TESTING RK4 ===" << std::endl;
    laopt::functions::RK4<User, double, User::Sys> rk4_sys(user, 0.2);

    std::cout << "--- Evaluation" << std::endl;
    val = rk4_sys.function(x, u);
    std::cout << "val = " << val.transpose() << std::endl;

    std::cout << "--- Jacobian" << std::endl;
    rk4_sys.jacobian(val, J, x, u);
    std::cout << "Jacobian = \n" << J << std::endl;
    std::cout << "val = " << val.transpose() << std::endl;

    std::cout << "\n=== TESTING RK4 + RK4 + RK4 ===" << std::endl;
    laopt::functions::RK4<laopt::functions::RK4<User, double, User::Sys>, double> rk4_rk4_sys(rk4_sys, 0.3);
    laopt::functions::RK4<laopt::functions::RK4<laopt::functions::RK4<User, double, User::Sys>, double>, double> rk4_rk4_rk4_sys(rk4_rk4_sys, 0.4);

    std::cout << "--- Evaluation" << std::endl;
    val = rk4_rk4_rk4_sys.function(x, u);
    std::cout << "val = " << val.transpose() << std::endl;

    std::cout << "--- Jacobian" << std::endl;
    rk4_rk4_rk4_sys.jacobian(val, J, x, u);
    std::cout << "Jacobian = \n" << J << std::endl;

    std::cout << "\n=== TESTING Weighted sum ===" << std::endl;
    Eigen::Vector<double, 2> w;
    w << 1, 2;
    std::cout << "wsum = " << user.wsum(User::NLSys{}, w, x, u) << std::endl;

    Eigen::Vector<double, 3> grad;
    grad.array() = 0;
    std::cout << "wsum = " << user.gradient(User::NLSys{}, grad, w, x, u) << std::endl;
    std::cout << "grad = " << grad.transpose() << std::endl;

    Eigen::Matrix<double, 3, 3> H;
    H.array() = 0;
    grad.array() = 0;
    std::cout << "wsum = " << user.hessian(User::NLSys{}, grad, H, w, x, u) << std::endl;
    std::cout << "grad = " << grad.transpose() << std::endl;
    std::cout << "hessian = \n" << H << std::endl;

    return 0;
}