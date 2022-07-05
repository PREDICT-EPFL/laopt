#include <iostream>

#include "lampc.hpp"
#include "lampc_function_tag.hpp"
#include "lampc_function_library.hpp"

using namespace lampc::tags;

template<typename scalar_t>
struct User : public lampc::Differentiable<User<scalar_t>>
{
  static constexpr int nx = 2;
  static constexpr int nu = 1;

  template<typename T> using state_t = Eigen::Vector<T, nx>;
  template<typename T> using input_t = Eigen::Vector<T, nu>;

  Eigen::Matrix<scalar_t, 2, 2> A{{1,2},{3,4}};
  Eigen::Matrix<scalar_t, 2, 1> B{{1},{3}};

  // Example of function with Eigen autodiff for jacobian
  struct Sys {};
  template<typename T>
  inline void 
  function_impl(Sys, Eigen::Ref<state_t<T>> x_dot,
           const Eigen::Ref<const state_t<T>>& x, const Eigen::Ref<const input_t<T>>& u) noexcept
  {
    x_dot = A * x + B * u;
  }

  // Example of function with user-defined jacobian
  struct SysX {};
  template<typename T>
  inline void 
  function_impl(SysX, Eigen::Ref<state_t<T>> x_dot,
                const Eigen::Ref<const state_t<T>>& x, const Eigen::Ref<const input_t<T>>& u) noexcept
  {
    x_dot = A * x + B * u;
  }

  template<typename OutValue, typename OutJacobian>
  inline void jacobian_impl(SysX, OutValue& out, OutJacobian& jac,
           const Eigen::Ref<const state_t<scalar_t>>& x, const Eigen::Ref<const input_t<scalar_t>>& u) noexcept
  {
    this->function(SysX{}, out, x,u);

    jac(all,seqN(0, nx)) = A;
    jac(all,seqN(nx,nu)) = B;
  }

  // Example of using RK4 in a user function
  struct SysRK4 {};
  template<typename T>
  inline void
  function_impl(SysRK4, Eigen::Ref<state_t<T>> x_dot,
                const Eigen::Ref<const state_t<T>>& x, const Eigen::Ref<const input_t<T>>& u) noexcept
  {
    lampc::RK4<User<scalar_t>, double, Sys> rk4_sys(*this, 0.2);
    rk4_sys.function(x_dot, x, u);
  }

  struct NLSys {};
  template<typename T>
  inline void 
  function_impl(NLSys, Eigen::Ref<state_t<T>> x_dot,
           const Eigen::Ref<const state_t<T>>& x, const Eigen::Ref<const input_t<T>>& u) noexcept
  {
    x_dot = x(0)*u(0)*(A * x + B * u);
  }

};


int main()
{
  using User = User<double>;

  User user;

  User::state_t<double> x;
  User::input_t<double> u;
  x << 1,2;
  u << 3;
  User::state_t<double> val;
  Eigen::Matrix<double, 2, 3> J;
  Eigen::Vector<double,3> p;
  p << 1,2,3;

  std::cout << "\n=== CALLING Function for SYS ===" << std::endl;
  user.function(User::Sys{}, val, x, u);
  std::cout << "user(Sys, x,u) = " << val.transpose() << std::endl;

  std::cout << "\n=== CALLING Jacobian FOR SYS ===" << std::endl;
  user.jacobian(User::Sys{}, val, J, x, u);
  std::cout << "Jacobian = \n" << J << std::endl;
  std::cout << "val = " << val.transpose() << std::endl;

  std::cout << "\n=== CALLING Function on SYSX ===" << std::endl;
  user.function(User::SysX{}, val, x,u);
  std::cout << "val = " << val.transpose() << std::endl;

  std::cout << "\n=== CALLING custom Jacobian on SYSX ===" << std::endl;
  user.jacobian(User::SysX{}, val,J, x,u);
  std::cout << "val = " << val.transpose() << std::endl;
  std::cout << "Jacobian = \n" << J << std::endl;

    std::cout << "\n=== CALLING Function for SYSRK4 ===" << std::endl;
    user.function(User::SysRK4{}, val, x, u);
    std::cout << "user(SysRK4, x,u) = " << val.transpose() << std::endl;

    std::cout << "\n=== CALLING Jacobian FOR SYSRK4 ===" << std::endl;
    user.jacobian(User::SysRK4{}, val, J, x, u);
    std::cout << "Jacobian = \n" << J << std::endl;
    std::cout << "val = " << val.transpose() << std::endl;

  std::cout << "\n\n=== TESTING EQ ===" << std::endl;
  lampc::EQ<User, User::Sys> eq_sys(user);

  std::cout << "--- Calling function ---" << std::endl;
  User::state_t<double> xp;
  xp << 1,2;
  eq_sys.function(val, xp, x, u);
  std::cout << "val = " << val.transpose() << std::endl;

  std::cout << "--- Testing jacobian ---" << std::endl;
  Eigen::Matrix<double, 2, 5> Jeq;
  Jeq.array() = 0;
  eq_sys.jacobian(val, Jeq, xp, x, u);
  std::cout << "val = " << val.transpose() << std::endl;
  std::cout << "Jeq = \n" << Jeq << std::endl;

  std::cout << "\n=== TESTING RK4 ===" << std::endl;
  lampc::RK4<User, double, User::Sys> rk4_sys(user, 0.2);

  std::cout << "--- Evaluation" << std::endl;
  rk4_sys.function(val, x, u);
  std::cout << "val = " << val.transpose() << std::endl;

  std::cout << "--- Jacobian" << std::endl;
  rk4_sys.jacobian(val,J, x, u);
  std::cout << "Jacobian = \n" << J << std::endl;
  std::cout << "val = " << val.transpose() << std::endl;

  std::cout << "\n=== TESTING RK4 + EQ ===" << std::endl;
  lampc::EQ<lampc::RK4<User, double, User::Sys>> eq_rk4_sys(rk4_sys);

  std::cout << "--- Evaluation" << std::endl;
  rk4_sys.function(val, x, u);
  std::cout << "integrated value = " << val.transpose() << std::endl;
  std::cout << "xp = " << xp.transpose() << std::endl;
  eq_rk4_sys.function(val, xp, x, u);
  std::cout << "eq val = " << val.transpose() << std::endl;

  std::cout << "--- Jacobian" << std::endl;
  eq_rk4_sys.jacobian(val, Jeq, xp, x, u);
  std::cout << "Jeq = \n" << Jeq << std::endl;

  std::cout << "\n=== TESTING RK4 + RK4 + RK4 ===" << std::endl;
  lampc::RK4<lampc::RK4<User, double, User::Sys>, double> rk4_rk4_sys(rk4_sys, 0.3);
  lampc::RK4<lampc::RK4<lampc::RK4<User, double, User::Sys>, double>, double> rk4_rk4_rk4_sys(rk4_rk4_sys, 0.4);

  std::cout << "--- Evaluation" << std::endl;
  rk4_rk4_rk4_sys.function(val, x, u);
  std::cout << "val = " << val.transpose() << std::endl;

  std::cout << "--- Jacobian" << std::endl;
  rk4_rk4_rk4_sys.jacobian(val, J, x, u);
  std::cout << "Jacobian = \n" << J << std::endl;

  std::cout << "\n=== TESTING Weighted sum ===" << std::endl;
  Eigen::Vector<double, 2> w;
  w << 1,2;
  std::cout << "wsum = " << user.wsum(User::NLSys{}, w, x,u) << std::endl;

  Eigen::Vector<double, 3> grad; grad.array() = 0;
  std::cout << "wsum = " << user.gradient(User::NLSys{}, grad, w,x,u) << std::endl;
  std::cout << "grad = " << grad.transpose() << std::endl;

  Eigen::Matrix<double, 3,3> H; 
  H.array() = 0; grad.array() = 0;
  std::cout << "wsum = " << user.hessian(User::NLSys{}, grad,H, w, x, u) << std::endl;
  std::cout << "grad = " << grad.transpose() << std::endl;
  std::cout << "hessian = \n" << H << std::endl;

  return 0;
}