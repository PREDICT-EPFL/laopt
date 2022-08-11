/**
 * Unit test for the computation of IpOpt problems.
 */

#include <iostream>
#include "lampc.hpp"

#include "ipopt_interface.hpp"

#include "test_utils.hpp"
#include "gtest/gtest.h"

namespace {

/**
 * Example used in the IpOpt C++ example
 * 
 * min_x f(x) = -(x2-2)^2
 *  s.t.
 *       0 = x1^2 + x2 - 1
 *       -1 <= x1 <= 1
 */
template<typename scalar_t_>
struct IpOpt_example : public laopt::Differentiable<IpOpt_example<scalar_t_>>
{
  using scalar_t = scalar_t_;

  template<typename FunctionTag>
  using Function = laopt::Function<IpOpt_example<scalar_t>, FunctionTag>;

  template<size_t n>
  using Variable = laopt::Variable<scalar_t, n>;

  /**
   * Equalities
   * 
   * x1^2 + x2 - 1
   */
  struct Equality {};
  template<typename X, typename Scalar = typename Eigen::MatrixBase<X>::Scalar>
  EIGEN_STRONG_INLINE Eigen::Vector<Scalar, 1>
  function_impl(Equality, const Eigen::MatrixBase<X>& x1, const Eigen::MatrixBase<X>& x2) noexcept
  {
      return (-(x1*x1 + x2).array() + 1.0).matrix();
  }
  Function<Equality> equality;

  struct Cost {};
  template<typename X, typename Scalar = typename Eigen::MatrixBase<X>::Scalar>
  EIGEN_STRONG_INLINE Scalar 
  function_impl(Cost, const Eigen::MatrixBase<X>& x1, const Eigen::MatrixBase<X>& x2) noexcept
  {
    return -(x2(0)-2.0) * (x2(0)-2.0);
  }
  Function<Cost> cost;

  Variable<1> x1;
  Variable<1> x2;

  IpOpt_example() : equality(*this), cost(*this)
  {};

  template<typename Problem>
  void define_problem(Problem& problem) {
    problem.add_variable(x1);
    problem.add_variable(x2);

    problem.add_constr(-1 <= x1 <= 1);
    problem.add_constr(-2.0e19 <= x2 <= 2.0e19);

    problem.add_constr(equality(x1, x2) == 0);

    problem.add_obj(cost(x1,x2));
  }  
};



TEST(IpOptTest, Construction) {
  using scalar_t = double;

  using UserCode = IpOpt_example<scalar_t>;
  UserCode my_problem;
  auto sparsity = laopt::generate_sparsity(my_problem);
  auto tape = laopt::generate_tape(my_problem, sparsity);

  std::cout << sparsity << std::endl;
  std::cout << tape << std::endl;

  using Problem = laopt::Problem<UserCode>;
  Problem prob(my_problem, tape);

  // Create the IPOpt solver
  auto mynlp = new laopt::Solver_IPOpt<Problem>(prob);
  SmartPtr<IpoptApplication> app = IpoptApplicationFactory();

  // app->Options()->SetStringValue("hessian_approximation", "limited-memory");
  app->Options()->SetIntegerValue("print_level", 0);

  // Initialize the IpoptApplication and process the options
  ApplicationReturnStatus status;
  status = app->Initialize();
  if( status != Solve_Succeeded ) std::cout << std::endl << std::endl << "*** Error during initialization!" << std::endl;

  // Set the initial primal variable
  prob.set_decision_variable(mynlp->init_primal);
  my_problem.x1 << 0.5;
  my_problem.x2 << 1.5;

  status = app->OptimizeTNLP(mynlp);
  if( status != Solve_Succeeded ) { std::cout << std::endl << std::endl << "*** Error during solution!" << std::endl; }
};


// /**
//  * Simple linear MPC problem
//  */
// template<typename _scalar_t, int N>
// struct OCP_DoubleIntegrator
// {  
//   using scalar_t = _scalar_t; // laopt::Problem assumes that the user code will contain a scalar_t

//   /**
//    * Continuous-time dynamics - double integrator
//    */
//   struct sys_t
//   {
//     Eigen::Matrix<scalar_t, 2, 2> A{{0,1},{0,0}};
//     Eigen::Matrix<scalar_t, 2, 1> B{{0},{1}};

//     template<typename X, typename U, typename Scalar = typename Eigen::MatrixBase<X>::Scalar>
//     EIGEN_STRONG_INLINE Eigen::Vector<Scalar, 2>
//     impl( const Eigen::MatrixBase<X>& x, const Eigen::MatrixBase<U>& u) noexcept        
//     {
//       return A * x + B * u;
//     }
//   };

//   // Discretized dynamics
//   using dsys_t = laopt::functions::RK4<sys_t, scalar_t>;

//   struct stage_cost_t : public laopt::MakeDifferentiable<stage_cost_t>
//   {
//     OCP_DoubleIntegrator& p;
//     stage_cost_t(OCP_DoubleIntegrator& p) : p(p) {}

//     template<typename X, typename U, typename Scalar = typename Eigen::MatrixBase<X>::Scalar>
//     EIGEN_STRONG_INLINE Scalar impl(
//       const Eigen::MatrixBase<X>& x, const Eigen::MatrixBase<X>& xss,
//       const Eigen::MatrixBase<U>& u, const Eigen::MatrixBase<U>& uss) noexcept
//     {
//       return (x-xss).dot(p.Q*(x-xss)) + (u-uss).dot(p.R*(u-uss));
//     }

//     // // Bring all the calls that we're not overloading into scope
//     // using laopt::MakeDifferentiable<stage_cost_t>::weightedsum;

//     // template<typename OutGradient, typename OutHessian, typename Weight, typename X, typename U>
//     // EIGEN_STRONG_INLINE scalar_t
//     // weightedsum(laopt::Hessian,
//     //             OutGradient&& outgradient, OutHessian&& outhessian,
//     //             const Eigen::MatrixBase<Weight>& weight,
//     //             const Eigen::MatrixBase<X>& x, const Eigen::MatrixBase<X>& xss,
//     //             const Eigen::MatrixBase<U>& u, const Eigen::MatrixBase<U>& uss) noexcept
//     // {
//     //   outgradient(seqN(0,2)) +=  2*p.Q*(x-xss);
//     //   outgradient(seqN(2,2)) += -2*p.Q*(x-xss);
//     //   outgradient(seqN(4,1)) +=  2*p.R*(u-uss);
//     //   outgradient(seqN(5,1)) += -2*p.R*(u-uss);

//     //   outhessian(seqN(0,2),seqN(0,2)) +=  2*p.Q;
//     //   outhessian(seqN(0,2),seqN(2,2)) += -2*p.Q;
//     //   outhessian(seqN(2,2),seqN(2,2)) +=  2*p.Q;
//     //   outhessian(seqN(2,2),seqN(0,2)) += -2*p.Q;
//     //   outhessian(seqN(4,1),seqN(4,1)) +=  2*p.R;
//     //   outhessian(seqN(5,1),seqN(4,1)) += -2*p.R;
//     //   outhessian(seqN(5,1),seqN(5,1)) +=  2*p.R;
//     //   outhessian(seqN(4,1),seqN(5,1)) += -2*p.R;

//     //   return impl(x,xss,u,uss);
//     // }

//   };

//   struct terminal_cost_t : public laopt::MakeDifferentiable<terminal_cost_t>
//   {
//     OCP_DoubleIntegrator& p;
//     terminal_cost_t(OCP_DoubleIntegrator& p) : p(p) {}

//     template<typename X, typename Scalar = typename Eigen::MatrixBase<X>::Scalar>
//     EIGEN_STRONG_INLINE Scalar impl(
//       const Eigen::MatrixBase<X>& x, const Eigen::MatrixBase<X>& xss) noexcept
//     {
//       return 20*(x-xss).dot(p.Q*(x-xss));
//     }
//   };

//   static constexpr int nx = 2;
//   static constexpr int nu = 1;

//   template<size_t n>
//   using Variable = laopt::Variable<scalar_t, n>;
//   std::array<Variable<nx>, N>   X;
//   std::array<Variable<nu>, N-1> U;
//   Variable<nx> xss;
//   Variable<nu> uss;

//   // Parameters shared across different elements of the cost function
//   Eigen::Matrix<scalar_t, 2, 2> Q{{1,0},{0,1}};
//   Eigen::Matrix<scalar_t, 1, 1> R{2};

//   Eigen::Vector<scalar_t, 2> x0; // Initial state

//   // Functions we use to define the problem
//   sys_t sys; // Continuous-time dynamics
//   dsys_t dsys; // Discrete-time dynamics

//   laopt::functions::eq<dsys_t> dsys_eq; // dsys(x,u) - xp

//   // Tuning parameters
//   stage_cost_t stage_cost;
//   terminal_cost_t terminal_cost;

//   laopt::functions::id id; // Identity

//   OCP_DoubleIntegrator(scalar_t step_size) :
//                       dsys(sys, step_size),
//                       dsys_eq(dsys),
//                       stage_cost(*this),
//                       terminal_cost(*this)
//   {};

//   template<typename OptProblem>
//   void define_variables(OptProblem& problem)
//   {
//     for(int i=0; i<N-1; i++)
//     {
//       problem.add_variable(X[i]);
//       problem.add_variable(U[i]);
//     }
//     problem.add_variable(X[N-1]);
//     problem.add_variable(xss);
//     problem.add_variable(uss);
//   }

//   template<typename Bounds>
//   void eval_variable_bounds(Bounds& bnd)
//   {
//     // Add constraints on the variables
//     for(auto& u : U) -1 <= bnd.add(u) <= 1;
//     for(auto& x : X) -50 <= bnd.add(x) <= 50;
//     -50 <= bnd.add(xss) <= 50;
//     -30 <= bnd.add(uss) <= 30;
//   }

//   template<typename Constraints, typename Dtype>
//   void eval_constraints(Dtype dtype, Constraints& con)
//   {
//     // Initial state
//     con.add(dtype, id, X[0]) == x0;

//     // Dynamics
//     for(int i=0; i<N-1; i++) con.add(dtype, dsys_eq, X[i+1], X[i], U[i]) == 0;
//     con.add(dtype, id, xss) == 0;
//     con.add(dtype, id, uss) == 0;
//   }

//   template<typename Objective, typename Dtype>
//   void eval_objective(Dtype dtype, Objective& obj)
//   {
//     for(int i=0; i<X.size()-1; i++)
//       obj.add(dtype, stage_cost, X[i], xss, U[i], uss);
//     obj.add(dtype, terminal_cost, X[N-1], xss);
//   }
// };


// TEST(ProblemTest, DoubleIntegrator) {
//   using scalar_t = double;
//   const int N = 100;

//   using OCP = OCP_DoubleIntegrator<scalar_t, N>;
//   using Problem = laopt::Problem<OCP>;

//   OCP ocp(0.5);
//   Problem prob = laopt::generate(ocp);

//   ocp.x0 << 1,0;

//   // Create the IPOpt solver
//   SmartPtr<laopt::Solver_IPOpt<Problem>> mynlp = new laopt::Solver_IPOpt<Problem>(prob);
//   SmartPtr<IpoptApplication> app = IpoptApplicationFactory();

//   // app->Options()->SetStringValue("hessian_approximation", "limited-memory");
//   app->Options()->SetIntegerValue("print_level", 5);

//   // Initialize the IpoptApplication and process the options
//   ApplicationReturnStatus status;
//   status = app->Initialize();
//   if( status != Solve_Succeeded )
//   {
//     std::cout << std::endl << std::endl << "*** Error during initialization!" << std::endl;
//   }

//   // Set the initial primal variable
//   prob.set_decision_variable(mynlp->init_primal);

//   for(auto& x : ocp.X) x.array() = 0;
//   for(auto& u : ocp.U) u.array() = 0;
//   ocp.xss.array() = 0;
//   ocp.uss.array() = 0;

//   status = app->OptimizeTNLP(mynlp);

//   std::cout << std::setprecision(2) << std::defaultfloat;

//   std::cout << "X = " << std::endl;
//   for(auto& x: ocp.X) std::cout << x.transpose() << std::endl;

//   std::cout << "U = " << std::endl;
//   for(auto& u: ocp.U) std::cout << u.transpose() << std::endl;

//   std::cout << "xss = " << ocp.xss.transpose() << std::endl;
//   std::cout << "uss = " << ocp.uss.transpose() << std::endl;

//   // std::cout << "Solution = " << mynlp->sol_primal.transpose() << std::endl;
//   // prob.set_decision_variable(mynlp->sol_primal);
// };



/*
 * Solve problem 71 from the Hock-Schittkowsky test suite [6],
 * 
 * min x1*x4*(x1+x2+x3)+x3
 * s.t. x1*x2*x3*x4 >= 25
 *      x1^2 + x2^2 + x3^2 + x4^2 = 40
 *      1 <= x1,x2,x3,x4 <= 5
 */ 
template<typename _scalar_t>
struct Prob71 : public laopt::Differentiable<Prob71<_scalar_t>>
{  
  using scalar_t = _scalar_t; // laopt::Problem assumes that the user code will contain a scalar_t

  /**
   * x1*x2*x3*x4 >= 25
   * x1^2 + x2^2 + x3^2 + x4^2 = 40
   */
  struct Constraints {};
  template<typename X, typename Scalar = typename Eigen::MatrixBase<X>::Scalar>
  EIGEN_STRONG_INLINE Eigen::Vector<Scalar, 2>
  function_impl(Constraints, const Eigen::MatrixBase<X>& x ) noexcept        
  {
    Eigen::Vector<Scalar,2> out;
    out << x[0]*x[1]*x[2]*x[3],
           x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3];
    return out;
  }

  /**
   * Obj = x1*x4*(x1+x2+x3)+x3
   */
  struct Obj {};
  template<typename X, typename Scalar = typename Eigen::MatrixBase<X>::Scalar>
  EIGEN_STRONG_INLINE Scalar
  function_impl(Obj, const Eigen::MatrixBase<X>& x ) noexcept        
  {
    return x[0]*x[3]*(x[0]+x[1]+x[2])+x[2];
  }

  // Define our variables
  laopt::Variable<scalar_t, 4> x;

  // Define our functions
  laopt::Function<Prob71<scalar_t>, Constraints> constraints;
  laopt::Function<Prob71<scalar_t>, Obj> objective;

  explicit Prob71() : constraints(*this), objective(*this) {};

  template<typename Problem>
  void define_problem(Problem& problem) {
    problem.add_variable(x);

    problem.add_constr(1 <= x <= 5);
    problem.add_constr(Eigen::Vector<scalar_t,2>{25,40} <= constraints(x) <= Eigen::Vector<scalar_t,2>{2e9,40});
    problem.add_obj(objective(x));
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



TEST(ProblemTest, Prob71) {
  using scalar_t = double;
  Prob71<scalar_t> prob71;
  using Problem = laopt::Problem<Prob71<scalar_t>>;
  Problem prob = laopt::generate(prob71);

  // Create the IPOpt solver
  SmartPtr<laopt::Solver_IPOpt<Problem>> mynlp = new laopt::Solver_IPOpt<Problem>(prob);
  SmartPtr<IpoptApplication> app = IpoptApplicationFactory();

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

  EXPECT_TRUE(mynlp->init_primal.isApprox(prob71.x, 0));

  // Solve the problem
  status = app->OptimizeTNLP(mynlp);

  // Print out the solution
  std::cout << std::setprecision(6) << std::defaultfloat;
 
  // Check the solution
  Eigen::Vector<scalar_t,4> sol_primal{ 1, 4.743, 3.82115, 1.37941 };
  EXPECT_TRUE(mynlp->sol_primal.isApprox(sol_primal, 1e-4));

  Eigen::Vector<scalar_t,2> sol_dual{ -0.552294, 0.161469 };
  EXPECT_TRUE(mynlp->sol_dual.isApprox(sol_dual, 1e-4));

  // Compute the value and jacobian of the constraints at the optimal point
  laopt::ProblemMemory<scalar_t> mem(prob);

  // Check the computation of variable bounds
  prob.eval_variable_bounds(mem.variable_bounds);
  EXPECT_TRUE(mem.variable_bounds.lb.isApprox(Eigen::Vector<scalar_t,4>{1,1,1,1}, 1e-4));
  EXPECT_TRUE(mem.variable_bounds.ub.isApprox(Eigen::Vector<scalar_t,4>{5,5,5,5}, 1e-4));

  // Check the computation of the constraints
  prob.eval_constraints(laopt::Jacobian(), mynlp->sol_primal, mem.constraints);
  EXPECT_TRUE(mem.constraints.lb.isApprox(Eigen::Vector<scalar_t,2>{25,40}, 1e-4));
  EXPECT_TRUE(mem.constraints.ub.isApprox(Eigen::Vector<scalar_t,2>{2e9,40}, 1e-4));
  EXPECT_TRUE(mem.constraints.value.isApprox(g(prob71.x), 1e-4));
  EXPECT_TRUE(Eigen::MatrixX<scalar_t>(mem.constraints.jacobian).isApprox(jac_g(prob71.x), 1e-4));

  // Check the computation of the objective
  scalar_t obj = prob.eval_objective(laopt::Hessian(), mynlp->sol_primal, mem.objective);
  EXPECT_NEAR(obj, f(prob71.x), 1e-4);
  EXPECT_TRUE(mem.objective.gradient.isApprox(grad_f(prob71.x), 1e-4));

  // Check the computation of the lagrangian
  prob.eval_lagrangian(laopt::Hessian(), mynlp->sol_primal, 1.0, mynlp->sol_dual, mem.lagrangian);
  EXPECT_TRUE(Eigen::MatrixX<scalar_t>(mem.lagrangian.hessian).isApprox(hessian_l(prob71.x, mynlp->sol_dual, 1.0), 1e-4));
};




}
