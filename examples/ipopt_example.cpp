/**
 * Double Integrator example calling IPOpt
 */

#include <iostream>
#include <iomanip>
#include "lampc.hpp"
#include "ipopt_interface.hpp"

#include <casadi/casadi.hpp>
using namespace casadi;


/**
 * Solve problem 71 from the Hock-Schittkowsky test suite [6],
 * 
 * min x1*x4*(x1+x2+x3)+x3
 * s.t. x1*x2*x3*x4 >= 25
 *      x1^2 + x2^2 + x3^2 + x4^2 = 40
 *      1 <= x1,x2,x3,x4 <= 5
 */

/**
 * Simple linear MPC problem
 */
template<typename _scalar_t>
struct Prob71
{  
  using scalar_t = _scalar_t; // laopt::Problem assumes that the user code will contain a scalar_t

  /**
   * x1*x2*x3*x4 >= 25
   * x1^2 + x2^2 + x3^2 + x4^2 = 40
   */
  struct constraints_t : public laopt::MakeDifferentiable<constraints_t>
  {
    template<typename X, typename Scalar = typename Eigen::MatrixBase<X>::Scalar>
    EIGEN_STRONG_INLINE Eigen::Vector<Scalar, 2>
    impl( const Eigen::MatrixBase<X>& x ) noexcept        
    {
      Eigen::Vector<Scalar,2> out;
      out << x[0]*x[1]*x[2]*x[3],
             x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3];
      return out;
    }
  };

  /**
   * Obj = x1*x4*(x1+x2+x3)+x3
   */
  struct obj_t : public laopt::MakeDifferentiable<obj_t>
  {
    template<typename X, typename Scalar = typename Eigen::MatrixBase<X>::Scalar>
    EIGEN_STRONG_INLINE Scalar
    impl( const Eigen::MatrixBase<X>& x ) noexcept        
    {
      return x[0]*x[3]*(x[0]+x[1]+x[2])+x[2];
    }
  };

  // Define our variables
  laopt::Variable<scalar_t, 4> x;

  // Define our functions
  constraints_t constraints;
  obj_t objective;
  laopt::functions::id id;

  template<typename OptProblem>
  void define_variables(OptProblem& problem)
  {
    problem.add_variable(x);
  }

  template<typename Bounds>
  void eval_variable_bounds(Bounds& bnd)
  {
    1 <= bnd.add(x) <= 5;
    // -std::numeric_limits<scalar_t>::infinity() <= bnd.add(x) <= std::numeric_limits<scalar_t>::infinity();
  }

  template<typename Constraints, typename Dtype>
  void eval_constraints(Dtype dtype, Constraints& con)
  {
    Eigen::Vector<scalar_t,2>{25,40} <= con.add(dtype, constraints, x) <= Eigen::Vector<scalar_t,2>{std::numeric_limits<scalar_t>::infinity(),40};
    // 1 <= con.add(dtype, id, x) <= 5;
  }

  template<typename Objective, typename Dtype>
  void eval_objective(Dtype dtype, Objective& obj)
  {
    obj.add(dtype, objective, x);
  }  
};


/**
 * Symbolic values of the various elements we need
 */

// Objective function
template<typename Derived, typename scalar_t = typename Eigen::MatrixBase<Derived>::Scalar>
scalar_t f(const Eigen::MatrixBase<Derived>& x)
{
  scalar_t x1 = x[0]; scalar_t x2 = x[1]; scalar_t x3 = x[2]; scalar_t x4 = x[3];  

  return x1*x4*(x1+x2+x3)+x3;
}

// Gradient of objective function
template<typename Derived, typename scalar_t = typename Eigen::MatrixBase<Derived>::Scalar>
Eigen::Vector<scalar_t, 4> grad_f(const Eigen::MatrixBase<Derived>& x)
{
  scalar_t x1 = x[0]; scalar_t x2 = x[1]; scalar_t x3 = x[2]; scalar_t x4 = x[3];  
  Eigen::Vector<scalar_t, 4> gf;
  gf << x1*x4 + x4*(x1+x2+x3),
        x1*x4,
        x1*x4+1,
        x1*(x1+x2+x3);
  return gf;
}

// Constraints
template<typename Derived, typename scalar_t = typename Eigen::MatrixBase<Derived>::Scalar>
Eigen::Vector<scalar_t, 2> g(const Eigen::MatrixBase<Derived>& x)
{
  scalar_t x1 = x[0]; scalar_t x2 = x[1]; scalar_t x3 = x[2]; scalar_t x4 = x[3];  
  Eigen::Vector<scalar_t, 2> g;
  g << x1*x2*x3*x4,
       x1*x1 + x2*x2 + x3*x3 + x4*x4;
  return g;
}

// Jacobian of constraints
template<typename Derived, typename scalar_t = typename Eigen::MatrixBase<Derived>::Scalar>
Eigen::Matrix<scalar_t, 2, 4> jac_g(const Eigen::MatrixBase<Derived>& x)
{
  scalar_t x1 = x[0]; scalar_t x2 = x[1]; scalar_t x3 = x[2]; scalar_t x4 = x[3];  
  Eigen::Matrix<scalar_t, 2, 4> jac_g;
  jac_g << x2*x3*x4, x1*x3*x4, x1*x2*x4, x1*x2*x3,
           2*x1, 2*x2, 2*x3, 2*x4;
  return jac_g;
}

// Hessian of lagrangian
template<typename Derived, typename L, typename scalar_t = typename Eigen::MatrixBase<Derived>::Scalar>
Eigen::Matrix<scalar_t, 4, 4> hessian_l(const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<L>& dual, const scalar_t obj_factor)
{
  scalar_t x1 = x[0]; scalar_t x2 = x[1]; scalar_t x3 = x[2]; scalar_t x4 = x[3];  
  scalar_t l1 = dual[0]; scalar_t l2 = dual[1];

  Eigen::Matrix<scalar_t, 4, 4> O;
  O << 2*x4, x4, x4, 2*x1+x2+x3,
         x4,  0,  0, x1,
         x4,  0,  0, x1,
       2*x1+x2+x3, x1, x1, 0;

  Eigen::Matrix<scalar_t, 4, 4> L1;
  L1 << 0, x3*x4, x2*x4, x2*x3,
        x3*x4, 0, x1*x4, x1*x3,
        x2*x4, x1*x4, 0, x1*x2,
        x2*x3, x1*x3, x1*x2, 0;

  Eigen::Matrix<scalar_t, 4, 4> L2;
  L2 << 2, 0, 0, 0,
        0, 2, 0, 0,
        0, 0, 2, 0,
        0, 0, 0, 2;

  Eigen::Matrix<scalar_t, 4, 4> H;
  H = obj_factor*O + l1*L1 + l2*L2;

  return H;
}


int main()
{
  using scalar_t = double;
  Prob71<scalar_t> prob71;
  using Problem = laopt::Problem<Prob71<scalar_t>>;
  Problem prob = laopt::generate(prob71);

  // Create the IPOpt solver
  SmartPtr<laopt::Solver_IPOpt<Problem>> mynlp = new laopt::Solver_IPOpt<Problem>(prob);
  SmartPtr<IpoptApplication> app = IpoptApplicationFactory();

  // app->Options()->SetStringValue("hessian_approximation", "limited-memory");
  app->Options()->SetIntegerValue("print_level", 5);

  // Initialize the IpoptApplication and process the options
  ApplicationReturnStatus status;
  status = app->Initialize();
  if( status != Solve_Succeeded )
  {
    std::cout << std::endl << std::endl << "*** Error during initialization!" << std::endl;
  }

  // Set the initial primal variable
  prob.set_decision_variable(mynlp->init_primal);
  prob71.x << 1,5,5,1;

  // Solve the problem
  status = app->OptimizeTNLP(mynlp);

  std::cout << std::endl << std::endl << std::endl << std::endl;


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


  // Compute the solution with CASADI to compare
  auto opti = casadi::Opti();

  auto x = opti.variable(4);

  opti.minimize(x(0)*x(3)*(x(0)+x(1)+x(2))+x(2));

  opti.subject_to(x(0)*x(1)*x(2)*x(3) >= 25);
  opti.subject_to(x(0)*x(0) + x(1)*x(1) + x(2)*x(2) + x(3)*x(3) == 40);
  opti.subject_to(1 <= x <= 5);

  opti.set_initial(x(0),1.0);
  opti.set_initial(x(1),5.0);
  opti.set_initial(x(2),5.0);
  opti.set_initial(x(3),1.0);

  opti.solver("ipopt");
  auto sol = opti.solve();


  std::cout << std::endl << std::endl << std::endl;

  std::cout << "==== CASADI SOLUTION ====" << std::endl;
  std::cout << sol.value(x) << std::endl;
  
  std::cout << std::endl;

  std::cout << "==== LAMPC SOLUTION ====" << std::endl;
  std::cout << prob71.x.transpose() << std::endl;
  
  return 0;
}
