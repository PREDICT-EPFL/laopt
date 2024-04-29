#include <iostream>

#include "laopt/laopt.hpp"
#include "laopt/differentiable_functions/integrators.hpp"

template<typename scalar_t>
struct User : public laopt::Differentiable<User<scalar_t>, laopt::TAGGED>
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

    template<typename OutJacobian, typename AScalar, typename X, typename U>
    inline void jacobian_impl(SysX, OutJacobian &jac, const AScalar& alpha, const Eigen::MatrixBase<X> &x, const Eigen::MatrixBase<U> &u) noexcept
    {
        std::cout << "jacobian_impl called" << std::endl;

        jac(Eigen::all, Eigen::seqN(0, Eigen::fix<nx>)) = alpha * A;
        jac(Eigen::all, Eigen::seqN(Eigen::fix<nx>, Eigen::fix<nu>)) = alpha * B;
    }

    // Example of using RK4 in a user function
    struct SysRK4 {};
    template<typename X, typename U, typename Scalar = typename Eigen::MatrixBase<X>::Scalar>
    EIGEN_STRONG_INLINE auto
    function_impl(SysRK4, const Eigen::MatrixBase<X> &x, const Eigen::MatrixBase<U> &u) noexcept
    {
        return rk4_sys.integrator(x, u);
    }

    struct NLSys {};
    template<typename X, typename U, typename Scalar = typename Eigen::MatrixBase<X>::Scalar>
    EIGEN_STRONG_INLINE state_t<Scalar>
    function_impl(NLSys, const Eigen::MatrixBase<X> &x, const Eigen::MatrixBase<U> &u) noexcept
    {
        return x(0) * u(0) * (A * x + B * u);
    }

    laopt::ERK4<User<scalar_t>, double, Sys> rk4_sys;
    laopt::Functor<User<scalar_t>, Sys> sys;

    User() : rk4_sys(*this, 0.2), sys(*this) {}

};


int main()
{
    using User = User<double>;

    User user;

    User::state_t<double> x_data; x_data << 1, 2;
    User::input_t<double> u_data; u_data << 3;

    laopt::VariableMap<User::state_t<double>> x(x_data.data()); x.index_offset() = 0;
    laopt::VariableMap<User::input_t<double>> u(u_data.data()); u.index_offset() = 2;

    User::state_t<double> val;
    Eigen::Matrix<double, 2, 3> J;
    Eigen::Vector<double, 3> p;
    p << 1, 2, 3;

    std::cout << "\n=== CALLING Function for SYS ===" << std::endl;
    val = user.function(User::Sys{}, x, u);
    std::cout << "user(Sys, x,u) = " << val.transpose() << std::endl;

    std::cout << "\n=== CALLING Jacobian FOR SYS ===" << std::endl;
    J.setZero();
    user.jacobian(User::Sys{}, J, 1, x, u);
    std::cout << "Jacobian = \n" << J << std::endl;

    std::cout << "\n=== CALLING Function for SYS FUNCTOR ===" << std::endl;
    val = user.sys.function(x, u);
    std::cout << "user(Sys, x,u) = " << val.transpose() << std::endl;

    std::cout << "\n=== CALLING Jacobian FOR SYS FUNCTOR ===" << std::endl;
    J.setZero();
    user.sys.jacobian(J, 1, x, u);
    std::cout << "Jacobian = \n" << J << std::endl;

    std::cout << "\n=== CALLING Function on SYSX ===" << std::endl;
    val = user.function(User::SysX{}, x, u);
    std::cout << "val = " << val.transpose() << std::endl;

    std::cout << "\n=== CALLING custom Jacobian on SYSX ===" << std::endl;
    J.setZero();
    user.jacobian(User::SysX{}, J, 1, x, u);
    std::cout << "Jacobian = \n" << J << std::endl;

    std::cout << "\n=== CALLING Function for SYSRK4 ===" << std::endl;
    val = user.function(User::SysRK4{}, x, u);
    std::cout << "user(SysRK4, x,u) = " << val.transpose() << std::endl;

    std::cout << "\n=== CALLING Jacobian FOR SYSRK4 ===" << std::endl;
    J.setZero();
    user.jacobian(User::SysRK4{}, J, 1, x, u);
    std::cout << "Jacobian = \n" << J << std::endl;

    std::cout << "\n=== TESTING RK4 ===" << std::endl;
    laopt::ERK4<User, double, User::Sys> rk4_sys(user, 0.2);

    std::cout << "--- Evaluation" << std::endl;
    val = rk4_sys.integrator.function(x, u);
    std::cout << "val = " << val.transpose() << std::endl;

    std::cout << "--- Jacobian" << std::endl;
    J.setZero();
    rk4_sys.integrator.jacobian(J, 1, x, u);
    std::cout << "Jacobian = \n" << J << std::endl;

    std::cout << "\n=== TESTING Weighted sum ===" << std::endl;
    Eigen::Vector<double, 2> w;
    w << 1, 2;
    std::cout << "wsum = " << user.wsum(User::NLSys{}, w, x, u) << std::endl;

    Eigen::Vector<double, 3> grad;
    grad.setZero();
    user.gradient(User::NLSys{}, grad, w, x, u);
    std::cout << "grad = " << grad.transpose() << std::endl;

    Eigen::Matrix<double, 3, 3> H;
    H.setZero();
    user.hessian(User::NLSys{}, H, w, x, u);
    std::cout << "hessian = \n" << H << std::endl;

    return 0;
}