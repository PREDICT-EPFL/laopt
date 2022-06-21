#include <iostream>
#include <type_traits>
#include <tuple>

#include "lampc.hpp"
#include "lampc_function_tag.hpp"

template<typename scalar_t>
struct User : public Differentiable<User<scalar_t>>
{
  static constexpr int nx = 2;
  static constexpr int nu = 1;

  template<typename T> using state_t = Eigen::Vector<T, nx>;
  template<typename T> using input_t = Eigen::Vector<T, nu>;

  Eigen::Matrix<scalar_t, 2, 2> A{{1,2},{3,4}};
  Eigen::Matrix<scalar_t, 2, 1> B{{1},{3}};

  // Example of function with Eigen autodiff for jacobian
  struct Sys : Tag {};
  template<typename T, typename OutValue>
  inline void 
  function_impl(Sys, OutValue&& dotx,
           const Eigen::Ref<const state_t<T>>& x, const Eigen::Ref<const input_t<T>>& u) noexcept
  {
    dotx = A * x + B * u;
  }

  // Example of function with parameters
  struct SysP : Tag {
    // Add any parameters that the problem may have
    Eigen::Ref<Eigen::Vector<scalar_t,3>> p;
    double h;

    SysP(Eigen::Ref<Eigen::Vector<scalar_t,3>> p, double&& h) : 
      p(p), h(h) {}
  };

  template<typename T>
  auto function_impl(SysP&& p, 
    Eigen::Ref<state_t<T>> dotx, // Output
    const Eigen::Ref<const state_t<T>>& x, const Eigen::Ref<const input_t<T>>& u) noexcept
  {
    p.p << 6,7,8;
    dotx = p.h*(A*x + B*u);
  }

  // Example of function with user-defined jacobian
  struct SysX : Tag {
    using jacobian = USER;
  };
  template<typename T, typename OutValue>
  inline void 
  function_impl(SysX, OutValue&& dotx,
                const Eigen::Ref<const state_t<T>>& x, const Eigen::Ref<const input_t<T>>& u) noexcept
  {
    dotx = A * x + B * u;
  }

  template<typename OutValue, typename OutJacobian>
  inline void jacobian_impl(SysX, USER, OutValue&& out, OutJacobian&& jac,
           const Eigen::Ref<const state_t<scalar_t>>& x, const Eigen::Ref<const input_t<scalar_t>>& u) noexcept
  {
    function_impl(SysX{}, out, x,u);

    jac(all,seqN(0, nx)) = A;
    jac(all,seqN(nx,nu)) = B;
  }
};


int main()
{
  using User = User<double>;
  using Sys = User::Sys;
  // using SysX = User::SysX;

  User user;

  User::state_t<double> x;
  User::input_t<double> u;
  x << 1,2;
  u << 3;
  User::state_t<double> val;
  Eigen::Matrix<double, 2, 3> J;
  Eigen::Vector<double,3> p;
  p << 1,2,3;

  std::cout << "\n=== CALLING Function for sys ===" << std::endl;
  user.function(Sys{}, val, x,u);
  std::cout << "user(Sys, x,u) = " << val.transpose() << std::endl;

  std::cout << "\n=== CALLING Function for sysP ===" << std::endl;
  user.function(User::SysP{p,0.5}, val, x,u);
  std::cout << "user(SysP, x,u) = " << val.transpose() << std::endl;

  std::cout << "\n=== CALLING Jacobian FOR SYS ===" << std::endl;
  user.jacobian(Sys{}, val,J, x,u);
  std::cout << "Jacobian = \n" << J << std::endl;
  std::cout << "val = " << val.transpose() << std::endl;

  std::cout << "\n=== CALLING Jacobian FOR SYSP ===" << std::endl;
  p << 1,2,3;
  user.jacobian(User::SysP{p,0.5}, val,J, x,u);
  std::cout << "Jacobian = \n" << J << std::endl;
  std::cout << "val = " << val.transpose() << std::endl;
  std::cout << "p = " << p.transpose() << std::endl; // Check that there was no copy

  std::cout << "\n=== CALLING Function on SYSX ===" << std::endl;
  user.function(User::SysX{}, val, x,u);
  std::cout << "val = " << val.transpose() << std::endl;

  std::cout << "\n=== CALLING custom Jacobian on SYSX ===" << std::endl;
  user.jacobian(User::SysX{}, val,J, x,u);
  std::cout << "val = " << val.transpose() << std::endl;
  std::cout << "Jacobian = \n" << J << std::endl;

  std::cout << "\n\n=== TESTING EQ ===" << std::endl;

  std::cout << "--- Calling function ---" << std::endl;
  User::state_t<double> xp;
  xp << 1,2;
  user.function(EQ<User::Sys>{}, val, xp,x,u);
  std::cout << "val = " << val.transpose() << std::endl;

  std::cout << "--- Testing jacobian ---" << std::endl;
  Eigen::Matrix<double, 2,5> Jeq;
  Jeq.array() = 0;
  user.jacobian(EQ<User::Sys>{}, val,Jeq, xp,x,u);
  std::cout << "val = " << val.transpose() << std::endl;
  std::cout << "Jeq = \n" << Jeq << std::endl;

  std::cout << "--- Calling function with parameters ---" << std::endl;
  x << 1,2;
  u << 3;
  xp << 1,2;
  user.function(EQ<User::SysP>{p,0.5}, val, xp,x,u);
  std::cout << "EQ<SysP>(x,u) = " << val.transpose() << std::endl;
  Jeq.array() = 0;
  user.jacobian(EQ<User::SysP>{p,0.5}, val,Jeq, xp,x,u);
  std::cout << "val = " << val.transpose() << std::endl;
  std::cout << "Jeq = \n" << Jeq << std::endl;

  std::cout << "\n=== TESTING RK4 ===" << std::endl;

  std::cout << "--- Evaluation" << std::endl;
  user.function(RK4<User::Sys, double>{0.2}, val, x,u);
  std::cout << "val = " << val.transpose() << std::endl;

  std::cout << "--- Jacobian" << std::endl;
  user.jacobian(RK4<User::Sys, double>{0.2}, val,J, x,u);
  std::cout << "Jacobian = \n" << J << std::endl;
  std::cout << "val = " << val.transpose() << std::endl;

  std::cout << "\n=== TESTING RK4 + EQ ===" << std::endl;

  std::cout << "--- Evaluation" << std::endl;
  user.function(RK4<User::Sys, double>{0.2}, val, x,u);
  std::cout << "integrated value = " << val.transpose() << std::endl;
  std::cout << "xp = " << xp.transpose() << std::endl;
  user.function(EQ<RK4<User::Sys, double>>{0.2}, val, xp,x,u);
  std::cout << "eq val = " << val.transpose() << std::endl;

  std::cout << "--- Jacobian" << std::endl;
  user.jacobian(EQ<RK4<User::Sys, double>>{0.2}, val,Jeq, xp,x,u);
  std::cout << "Jeq = \n" << Jeq << std::endl;

  std::cout << "\n=== TESTING RK4 + EQ + Parameters ===" << std::endl;
  std::cout << "--- Evaluation" << std::endl;
  user.function(EQ<RK4<User::SysP, double>>{0.2,p,0.4}, val, xp,x,u);
  std::cout << "eq val = " << val.transpose() << std::endl;

  std::cout << "--- Jacobian" << std::endl;
  user.jacobian(EQ<RK4<User::SysP, double>>{0.2,p,0.4}, val,Jeq, xp,x,u);
  std::cout << "Jeq = \n" << Jeq << std::endl;

  std::cout << "\n=== TESTING RK4 + RK4 + RK4 ===" << std::endl;
  std::cout << "--- Evaluation" << std::endl;
  user.function(RK4<RK4<RK4<User::Sys, double>,double>,double>{0.2,0.3,0.4}, val, x,u);
  std::cout << "val = " << val.transpose() << std::endl;

  std::cout << "--- Jacobian" << std::endl;
  user.jacobian(RK4<RK4<RK4<User::Sys, double>,double>,double>{0.2,0.3,0.4}, val,J, x,u);
  std::cout << "Jeq = \n" << Jeq << std::endl;


  return 0;
}