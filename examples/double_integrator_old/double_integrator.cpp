/**
 * Double Integrator example calling IPOpt
 */

#include <iostream>
#include <iomanip>
#include "laopt/laopt.hpp"
#include "laopt/ipopt_interface.hpp"

#include <casadi/casadi.hpp>
using namespace casadi;

#include "double_integrator.hpp"
#include "double_integrator_tape.hpp"

/**
 * Solve a linear double integrator problem discretized via RK4
 */

// Continuous time dynamics
MX sys(const MX& x, const MX& u) {
  return vertcat(x(1), u);
}

// RK4 Discretization
MX rk4_sys(const MX& x, const MX& u, const double dt) {
  auto k1 = sys(x,         u);
  auto k2 = sys(x+dt/2*k1, u);
  auto k3 = sys(x+dt/2*k2, u);
  auto k4 = sys(x+dt*k3,   u);
  return x + dt/6*(k1+2*k2+2*k3+k4);
}

void solve_casadi(const double dt, const Eigen::Ref<const Eigen::Vector<double, 2>>& x0, const double ref)
{
  // Compute the solution with CASADI to compare
  auto opti = casadi::Opti();

  auto X = opti.variable(2,N);
  auto U = opti.variable(1,N-1);
  auto xss = opti.variable(2);
  auto uss = opti.variable(1);

  Slice all;

  // ---- dynamic constraints --------
  for(int i=0; i<N-1; i++)
    opti.subject_to(X(all,i+1) == rk4_sys(X(all,i),U(all,i),dt));

  opti.subject_to(xss == rk4_sys(xss, uss, dt));
  opti.subject_to(xss(0) == ref);

  // ---- path constraints -----------
  opti.subject_to(-1 <= U <= 1);
  opti.subject_to(-50 <= X <= 50);
  opti.subject_to(-2 <= X(1,all) <= 2);
  opti.subject_to(-50 <= xss <= 50);
  opti.subject_to(-1 <= uss <= 1);

  // ---- boundary conditions --------
  opti.subject_to(X(0,0) == x0(0));
  opti.subject_to(X(1,0) == x0(1));

  // ---- cost function ----
  auto obj = 0*X(0,0);
  for(int i=0; i<N-1; i++)
    obj += dot(X(all,i)-xss,X(all,i)-xss) + 0.01*dot(U(all,i)-uss,U(all,i)-uss);
  obj += dot(X(all,N-1)-xss,X(all,N-1)-xss)*100;

  opti.minimize(obj);

  Dict opts_dict=Dict();
  opts_dict["ipopt.print_level"] = 0;
  opts_dict["ipopt.sb"] = "yes";
  opts_dict["print_time"] = 0;
  opti.solver("ipopt",opts_dict);
  auto sol = opti.solve();

  std::cout << std::endl << std::endl << std::endl;

  Eigen::Matrix<double, 2, N> sol_X(sol.value(X).ptr());
  Eigen::Matrix<double, 1, N-1> sol_U(sol.value(U).ptr());
  Eigen::Vector<double, 2> sol_xss(sol.value(xss).ptr());
  Eigen::Vector<double, 1> sol_uss(sol.value(uss).ptr());

  std::cout << "==== CASADI SOLUTION ====" << std::endl;
  std::cout << std::setprecision(2) << std::defaultfloat;
  std::cout << "X   = \n" << sol_X << std::endl;
  std::cout << "U   = \n" << sol_U << std::endl;
  std::cout << "xss = \n" << sol_xss << std::endl;
  std::cout << "uss = \n" << sol_uss << std::endl;
  
  std::cout << "obj = " << sol.value(obj) << std::endl;

  std::cout << std::endl;
  
}


int main()
{
  using scalar_t = double;

  constexpr double step_size = 0.5;
  OCP ocp(step_size);
  laopt::TapeInfo<OCP> tape = double_integrator_tape(ocp);
  Problem prob = tape_to_problem(tape, ocp);

  // Create the IPOpt solver
  SmartPtr<laopt::Solver_IPOpt<Problem>> mynlp = new laopt::Solver_IPOpt<Problem>(prob);
  SmartPtr<IpoptApplication> app = IpoptApplicationFactory();

  // app->Options()->SetStringValue("hessian_approximation", "limited-memory");
  app->Options()->SetIntegerValue("print_level", 0);

  // Initialize the IpoptApplication and process the options
  ApplicationReturnStatus status;
  status = app->Initialize();
  if( status != Solve_Succeeded )
  {
    std::cout << std::endl << std::endl << "*** Error during initialization!" << std::endl;
  }

  // Set the initial primal variable
  prob.set_decision_variable(mynlp->init_primal);
  ocp.x0 << -3,-1;
  ocp.ref = 5;

  // Solve the problem
  status = app->OptimizeTNLP(mynlp);

  std::cout << std::endl << std::endl << std::endl << std::endl;

  // Print out the solution
  std::cout << std::setprecision(2) << std::defaultfloat;
 
  Eigen::Matrix<scalar_t, 2, N> X;
  Eigen::Matrix<scalar_t, 1, N-1> U;
  int i=0; for(auto& x: ocp.X) X.col(i++) << x;
      i=0; for(auto& u: ocp.U) U.col(i++) << u;

  std::cout << "X = \n" << X << std::endl;
  std::cout << "U = \n" << U << std::endl;
  std::cout << "xss = " << ocp.xss.transpose() << std::endl;
  std::cout << "uss = " << ocp.uss << std::endl;
  std::cout << "obj = " << prob.eval_objective(laopt::Eval(), mynlp->sol_primal) << std::endl;




  // // Print out the solution
  // std::cout << std::setprecision(6) << std::defaultfloat;
 
  // // Compute the value and jacobian of the constraints at the optimal point
  // laopt::ProblemMemory<scalar_t> mem(prob);

  // std::cout << std::endl;
  // std::cout << "=================== SOLUTION ===================" << std::endl;
  // std::cout << "primal = " << mynlp->sol_primal.transpose() << std::endl;
  // std::cout << "dual = " << mynlp->sol_dual.transpose() << std::endl;

  // std::cout << std::endl;
  // std::cout << "=================== VARIABLE BOUNDS ===================" << std::endl;  
  // prob.eval_variable_bounds(mem.variable_bounds);
  // std::cout << "lb_x = " << mem.variable_bounds.lb.transpose() << std::endl;
  // std::cout << "ub_x = " << mem.variable_bounds.ub.transpose() << std::endl;

  // std::cout << std::endl;
  // std::cout << "=================== CONSTRAINTS ===================" << std::endl;
  // prob.eval_constraints(laopt::Jacobian(), mynlp->sol_primal, mem.constraints);
  // std::cout << "lb = " << mem.constraints.lb.transpose() << std::endl;
  // std::cout << "ub = " << mem.constraints.ub.transpose() << std::endl;
  // std::cout << " con = " << mem.constraints.value.transpose() << std::endl;
  // std::cout << "_con = " << g(prob71.x).transpose() << std::endl;
  // std::cout << " jacobian =\n" << Eigen::MatrixX<scalar_t>(mem.constraints.jacobian) << std::endl;
  // std::cout << "_jacobian =\n" << jac_g(prob71.x) << std::endl;

  // std::cout << std::endl;
  // std::cout << "=================== OBJECTIVE ===================" << std::endl;  
  // std::cout << " obj = " << prob.eval_objective(laopt::Hessian(), mynlp->sol_primal, mem.objective) << std::endl;
  // std::cout << "_obj = " << f(prob71.x) << std::endl;
  // std::cout << "gradient  = " << mem.objective.gradient.transpose() << std::endl;
  // std::cout << "_gradient = " << grad_f(prob71.x).transpose() << std::endl;
  // std::cout << "hessian = \n" << Eigen::MatrixX<scalar_t>(mem.objective.hessian) << std::endl;

  // std::cout << std::endl;
  // std::cout << "=================== LAGRANGIAN ===================" << std::endl;  
  // std::cout << "obj = " << prob.eval_lagrangian(laopt::Hessian(), mynlp->sol_primal, 1.0, mynlp->sol_dual, mem.lagrangian) << std::endl;
  // std::cout << "gradient = " << mem.lagrangian.gradient.transpose() << std::endl;
  // std::cout << " hessian = \n" << Eigen::MatrixX<scalar_t>(mem.lagrangian.hessian) << std::endl;
  // std::cout << "dual = " << mynlp->sol_dual.transpose() << std::endl;
  // std::cout << "_hessian = \n" << hessian_l(prob71.x, mynlp->sol_dual, 1.0) << std::endl;


  solve_casadi(step_size, ocp.x0, ocp.ref);
  return 0;
}
