/**
 * Double Integrator example calling IPOpt
 */

#include <iostream>
#include <iomanip>
#include "laopt/lampc.hpp"
#include "laopt/ipopt_interface.hpp"

#include <casadi/casadi.hpp>
using namespace casadi;

/**
 * Simple linear MPC problem
 */
template<typename _scalar_t, int N>
struct OCP_DoubleIntegrator
{  
  using scalar_t = _scalar_t; // laopt::Problem assumes that the user code will contain a scalar_t

  /**
   * Discrete-time dynamics - double integrator
   */
  struct sys_t : public laopt::MakeDifferentiable<sys_t>
  {
    Eigen::Matrix<scalar_t, 2, 2> A;
    Eigen::Matrix<scalar_t, 2, 1> B;

    sys_t(scalar_t h)
    {
      A << 1, h, 0, 1;
      B << h*h/2.0, h;
    }

    template<typename X, typename U, typename Scalar = typename Eigen::MatrixBase<X>::Scalar>
    EIGEN_STRONG_INLINE Eigen::Vector<Scalar, 2>
    impl( const Eigen::MatrixBase<X>& x, const Eigen::MatrixBase<U>& u) noexcept        
    {
      return A.template cast<Scalar>() * x + B.template cast<Scalar>() * u;
    }
  };

  struct stage_cost_t : public laopt::MakeDifferentiable<stage_cost_t>
  {
    Eigen::Matrix<scalar_t, 2, 2> Q{{1,0},{0,1}};
    Eigen::Matrix<scalar_t, 1, 1> R{0.01};

    template<typename X, typename U, typename Scalar = typename Eigen::MatrixBase<X>::Scalar>
    EIGEN_STRONG_INLINE Scalar 
    impl(const Eigen::MatrixBase<X>& x, const Eigen::MatrixBase<U>& u) noexcept
    {
      return x.dot(Q.template cast<Scalar>()*x) + u.dot(R.template cast<Scalar>()*u);
    }
  };

  static constexpr int nx = 2;
  static constexpr int nu = 1;

  template<size_t n>
  using Variable = laopt::Variable<scalar_t, n>;
  std::array<Variable<nx>, N>   X;
  std::array<Variable<nu>, N-1> U;

  Eigen::Vector<scalar_t, 2> x0; // Initial state

  // Functions we use to define the problem
  sys_t sys;
  laopt::functions::eq<sys_t> sys_eq; // sys(x,u) - xp
  stage_cost_t stage_cost;

  laopt::functions::id id; // Identity

  OCP_DoubleIntegrator(scalar_t step_size) :
                      sys(step_size),
                      sys_eq(sys)
  {};

  template<typename OptProblem>
  void define_variables(OptProblem& problem)
  {
    for(int i=0; i<N-1; i++)
    {
      problem.add_variable(X[i]);
      problem.add_variable(U[i]);
    }
    problem.add_variable(X[N-1]);
  }

  template<typename Bounds>
  void eval_variable_bounds(Bounds& bnd)
  {
    // Add constraints on the variables
    for(auto& u : U) -1 <= bnd.add(u) <= 1;
    for(auto& x : X) -50 <= bnd.add(x) <= 50;
  }

  template<typename Constraints, typename Dtype>
  void eval_constraints(Dtype dtype, Constraints& con)
  {
    // Initial state
    con.add(dtype, id, X[0]) == x0;

    // Dynamics
    for(int i=0; i<N-1; i++) con.add(dtype, sys_eq, X[i+1], X[i], U[i]) == 0;
  }

  template<typename Objective, typename Dtype>
  void eval_objective(Dtype dtype, Objective& obj)
  {
    for(int i=0; i<X.size()-1; i++)
      obj.add(dtype, stage_cost, X[i], U[i]);
  }
};

using scalar_t = double;
constexpr int N = 10;
using OCP = OCP_DoubleIntegrator<scalar_t, N>;
using Problem = laopt::Problem<OCP>;


/**
 * Solve the problem using casadi
 */

// Discretized double integrator
MX dsys(const MX& x, const MX& u, const double h) {
  return vertcat(x(0)+h*x(1)+h*h/2.0*u, x(1)+h*u);
}

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

void solve_casadi(const double dt, const Eigen::Ref<const Eigen::Vector<double, 2>>& x0)
{
  // Compute the solution with CASADI to compare
  auto opti = casadi::Opti();

  auto X = opti.variable(2,N);
  auto U = opti.variable(1,N-1);

  Slice all;

  // ---- dynamic constraints --------
  for(int i=0; i<N-1; i++)
    // opti.subject_to(X(all,i+1) == dsys(X(all,i),U(all,i),dt));
    opti.subject_to(X(all,i+1) == rk4_sys(X(all,i),U(all,i),dt));

  // opti.subject_to(xss == dsys(xss, uss, dt));

  // ---- path constraints -----------
  opti.subject_to(-1 <= U <= 1);
  opti.subject_to(-50 <= X <= 50);

  // ---- boundary conditions --------
  opti.subject_to(X(0,0) == x0(0));
  opti.subject_to(X(1,0) == x0(1));

  // ---- cost function ----
  auto obj = 0*X(0,0);
  for(int i=0; i<N-1; i++)
    obj += dot(X(all,i),X(all,i)) + 0.01*dot(U(all,i),U(all,i));

  opti.minimize(obj);

  Dict opts_dict=Dict();
  opts_dict["ipopt.print_level"] = 0;
  opts_dict["ipopt.sb"] = "yes";
  opts_dict["print_time"] = 0;
  opti.solver("ipopt",opts_dict);
  auto sol = opti.solve();

  std::cout << std::endl << std::endl << std::endl;

  Eigen::Matrix<double, 2, N> solX(sol.value(X).ptr());
  Eigen::Matrix<double, 1, N-1> solU(sol.value(U).ptr());

  std::cout << "==== CASADI SOLUTION ====" << std::endl;
  std::cout << std::setprecision(2) << std::defaultfloat;
  std::cout << "X = \n" << solX << std::endl;
  std::cout << "U = \n" << solU << std::endl;
  
  std::cout << "obj = " << sol.value(obj) << std::endl;

  std::cout << std::endl;
  
}




int main()
{
  constexpr double step_size = 1.0;
  OCP ocp(step_size);
  auto prob = laopt::generate(ocp);

  // Initial state
  ocp.x0 << 10,0;

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

  for(auto& x : ocp.X) x.array() = 0;
  for(auto& u : ocp.U) u.array() = 0;

  // Solve the problem
  status = app->OptimizeTNLP(mynlp);

  // Print out the solution
  std::cout << std::setprecision(2) << std::defaultfloat;
 
  Eigen::Matrix<scalar_t, 2, N> X;
  Eigen::Matrix<scalar_t, 1, N-1> U;
  int i=0; for(auto& x: ocp.X) X.col(i++) << x;
      i=0; for(auto& u: ocp.U) U.col(i++) << u;

  std::cout << "X = \n" << X << std::endl;
  std::cout << "U = \n" << U << std::endl;

  solve_casadi(step_size, ocp.x0);

  // Compute the value and jacobian of the constraints at the optimal point
  laopt::ProblemMemory<scalar_t> mem(prob);

  std::cout << std::endl;
  std::cout << "=================== SOLUTION ===================" << std::endl;
  std::cout << "primal = " << mynlp->sol_primal.transpose() << std::endl;
  std::cout << "dual = " << mynlp->sol_dual.transpose() << std::endl;

  std::cout << std::endl;
  std::cout << "=================== VARIABLE BOUNDS ===================" << std::endl;  
  prob.eval_variable_bounds(mem.variable_bounds);
  std::cout << "lb_x = " << mem.variable_bounds.lb.transpose() << std::endl;
  std::cout << "ub_x = " << mem.variable_bounds.ub.transpose() << std::endl;

  std::cout << std::endl;
  std::cout << "=================== CONSTRAINTS ===================" << std::endl;
  prob.eval_constraints(laopt::Jacobian(), mynlp->sol_primal, mem.constraints);
  std::cout << "lb = " << mem.constraints.lb.transpose() << std::endl;
  std::cout << "ub = " << mem.constraints.ub.transpose() << std::endl;
  std::cout << "con = " << mem.constraints.value.transpose() << std::endl;
  std::cout << "con.jacobian =\n" << Eigen::MatrixX<scalar_t>(mem.constraints.jacobian) << std::endl;

  std::cout << std::endl;
  std::cout << "=================== OBJECTIVE ===================" << std::endl;  
  std::cout << "obj = " << prob.eval_objective(laopt::Hessian(), mynlp->sol_primal, mem.objective) << std::endl;
  std::cout << "gradient = " << mem.objective.gradient.transpose() << std::endl;
  std::cout << "hessian = \n" << Eigen::MatrixX<scalar_t>(mem.objective.hessian) << std::endl;

  std::cout << std::endl;
  std::cout << "=================== LAGRANGIAN ===================" << std::endl;  
  std::cout << "obj = " << prob.eval_lagrangian(laopt::Hessian(), mynlp->sol_primal, 1, mynlp->sol_dual, mem.lagrangian) << std::endl;
  std::cout << "gradient = " << mem.lagrangian.gradient.transpose() << std::endl;
  std::cout << "hessian = \n" << Eigen::MatrixX<scalar_t>(mem.lagrangian.hessian) << std::endl;

};
