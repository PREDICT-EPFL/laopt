#include "solvers/sqp_base.hpp"
#include "polynomials/ebyshev.hpp"
#include "control/continuous_ocp.hpp"
#include "control/mpc_wrapper.hpp"
#include "polynomials/splines.hpp"

#include <iomanip>
#include <iostream>
#include <chrono>

#include "control/simple_robot_model.hpp"
#include "solvers/box_admm.hpp"
#include "solvers/admm.hpp"
#include "solvers/qp_preconditioners.hpp"

#include "lampc.hpp"
#include "polympc_interface.hpp"


#define test_POLY_ORDER 5
#define test_NUM_SEG    2
#define test_NUM_EXP    1

/** benchmark the new collocation class */
using Polynomial = polympc::Chebyshev<test_POLY_ORDER, polympc::GAUSS_LOBATTO, double>;
using Approximation = polympc::Spline<Polynomial, test_NUM_SEG>;

POLYMPC_FORWARD_DECLARATION(/*Name*/ RobotOCP, /*NX*/ 3, /*NU*/ 2, /*NP*/ 0, /*ND*/ 1, /*NG*/0, /*TYPE*/ double)

using namespace Eigen;

class RobotOCP : public ContinuousOCP<RobotOCP, Approximation, DENSE>
{
public:
    ~RobotOCP() = default;

    Eigen::DiagonalMatrix<scalar_t, 3> Q{1,1,1};
    Eigen::DiagonalMatrix<scalar_t, 2> R{1,1};
    Eigen::DiagonalMatrix<scalar_t, 3> QN{1,1,1};

    template<typename T>
    inline void dynamics_impl(const Eigen::Ref<const state_t<T>> x, const Eigen::Ref<const control_t<T>> u,
                              const Eigen::Ref<const parameter_t<T>> p, const Eigen::Ref<const static_parameter_t> &d,
                              const T &t, Eigen::Ref<state_t<T>> xdot) const noexcept
    {
        xdot(0) = u(0) * cos(x(2)) * cos(u(1));
        xdot(1) = u(0) * sin(x(2)) * cos(u(1));
        xdot(2) = u(0) * sin(u(1)) / d(0);
    }

    template<typename T>
    inline void lagrange_term_impl(const Eigen::Ref<const state_t<T>> x, const Eigen::Ref<const control_t<T>> u,
                                   const Eigen::Ref<const parameter_t<T>> p, const Eigen::Ref<const static_parameter_t> d,
                                   const scalar_t &t, T &lagrange) noexcept
    {
        Eigen::Matrix<T,3,3> Qm = Q.toDenseMatrix().template cast<T>();
        Eigen::Matrix<T,2,2> Rm = R.toDenseMatrix().template cast<T>();

        lagrange = x.dot(Qm * x) + u.dot(Rm * u);
    }

    template<typename T>
    inline void mayer_term_impl(const Eigen::Ref<const state_t<T>> x, const Eigen::Ref<const control_t<T>> u,
                                const Eigen::Ref<const parameter_t<T>> p, const Eigen::Ref<const static_parameter_t> d,
                                const scalar_t &t, T &mayer) noexcept
    {
        Eigen::Matrix<T,3,3> Qm = Q.toDenseMatrix().template cast<T>();
        mayer = x.dot(Qm * x);
    }

    void set_Q_coeff(const scalar_t& coeff)
    {
        Q.diagonal() << coeff, coeff, coeff;
    }
};

/** create solver */
template<typename Problem, typename QPSolver> class MySolver;

template<typename Problem, typename QPSolver = boxADMM<Problem::VAR_SIZE, Problem::NUM_EQ + Problem::NUM_INEQ,
                                               typename Problem::scalar_t, Problem::MATRIXFMT, linear_solver_traits<RobotOCP::MATRIXFMT>::default_solver>>
class MySolver : public SQPBase<MySolver<Problem, QPSolver>, Problem, QPSolver>
{
public:
    using Base = SQPBase<MySolver<Problem, QPSolver>, Problem, QPSolver>;
    using typename Base::scalar_t;
    using typename Base::nlp_variable_t;
    using typename Base::nlp_hessian_t;


    /** change Hessian update algorithm to the one provided by ContinuousOCP*/
    EIGEN_STRONG_INLINE void hessian_update_impl(Eigen::Ref<nlp_hessian_t> hessian, const Eigen::Ref<const nlp_variable_t>& x_step,
                                                 const Eigen::Ref<const nlp_variable_t>& grad_step) noexcept
    {
        this->problem.hessian_update_impl(hessian, x_step, grad_step);
    }
};




// LAMPC version

template<typename scalar_t>
struct Opt_t
{
    static constexpr auto INF = std::numeric_limits<scalar_t>::infinity();

    struct param_t
    {
        Eigen::DiagonalMatrix<scalar_t, 3> Q{1,1,1};
        Eigen::DiagonalMatrix<scalar_t, 2> R{1,1};
        Eigen::DiagonalMatrix<scalar_t, 3> QN{1,1,1};
    };

    FUNCTION(dynamics, (xdot, 3), (x, 3), (u, 2))
    {
        xdot(0) = u(0) * cos(x(2)) * cos(u(1));
        xdot(1) = u(0) * sin(x(2)) * cos(u(1));
        xdot(2) = u(0) * sin(u(1)) / p.d(0).template cast<T>();
    }

    FUNCTION(lagrange_term, (lagrange, 1), (x, 3), (u, 2))
    {
        Eigen::Matrix<T,3,3> Qm = p.Q.toDenseMatrix().template cast<T>();
        Eigen::Matrix<T,2,2> Rm = p.R.toDenseMatrix().template cast<T>();

        lagrange(0) = x.dot(Qm * x) + u.dot(Rm * u);
    }

    FUNCTION(mayer_term, (mayer, 1), (x, 3))
    {
        Eigen::Matrix<T,3,3> Qm = p.Q.toDenseMatrix().template cast<T>();
        mayer(0) = x.dot(Qm * x);
    }

    template<std::size_t len>
    struct unbounded
    {
        static EIGEN_STRONG_INLINE void eval(const param_t& p, const int iteration, 
                                             Vec<scalar_t, len> lb, Vec<scalar_t, len> ub) noexcept
        {
            lb.array() = -INF;
            ub.array() = INF;
        }
    };



    using variables = Make_Variables(scalar_t, param_t,
        (0, X, (var_t<unbounded<3>, 3 * NUM_NODES, NUM_SEGMENTS>)), 
        (2, uss, (var_t<unbounded<2>, 2, NUM_SEGMENTS>)));




    const typename Approximation::diff_mat_t  m_D     = Approximation::compute_diff_matrix();
    const typename Approximation::nodes_t     m_nodes = Approximation::compute_nodes();
    const typename Approximation::q_weights_t m_quad_weights = Approximation::compute_int_weights();
    time_t time_nodes = time_t::Zero();

    scalar_t t_stop = 1;
    scalar_t t_start = 0;

    struct collocation_
    {
        template<typename T>
        static EIGEN_STRONG_INLINE void eval(const param_t& p, Vec<T, NX> out, cVec<T, 3>& x) noexcept
        {
            Eigen::Matrix<T,3,3> Qm = p.Q.toDenseMatrix().template cast<T>();
            mayer(0) = x.dot(Qm * x);
        }
    };
    using collocation = Jacobian<collocation_, scalar_t, param_t, 1, 3>;


    void equalities(const Eigen::Ref<const X>& var,
                    const Eigen::Ref<const static_parameter_t>& p,
                    Eigen::Ref<nlp_eq_constraints_t> constraint) const noexcept
    {
        state_t<scalar_t> f_res; 
        f_res.setZero();
        const scalar_t t_scale = (t_stop - t_start) / (2 * NUM_SEGMENTS);

        /** @badcode: redo with Reshaped expression later*/
        Eigen::Map<const Eigen::Matrix<scalar_t, NX, NUM_NODES>> lox(var.data(), NX, NUM_NODES);
        Eigen::Matrix<scalar_t, NUM_NODES, NX> DX;

        for(int i = 0; i < NUM_SEGMENTS; ++i)
            DX.template block<POLY_ORDER + 1, NX>(i*POLY_ORDER, 0).noalias() = m_D * lox.template block<NX, POLY_ORDER + 1>(0, i*POLY_ORDER).transpose();

        //DX.transposeInPlace();

        int n = 0;
        int t = 0;
        for(int k = 0; k < VARX_SIZE; k += NX)
        {
            dynamics<scalar_t>(var.template segment<NX>(k), var.template segment<NU>(n + VARX_SIZE),
                               var.template segment<NP>(VARX_SIZE + VARU_SIZE), p, time_nodes(t), f_res);
            constraint. template segment<NX>(k) = DX.transpose().col(t);
            constraint. template segment<NX>(k).noalias() -= t_scale * f_res;
            n += NU;
            ++t;
        }
    }

};










/** QP solvers */
using admm_solver = ADMM<RobotOCP::VAR_SIZE, RobotOCP::NUM_EQ, RobotOCP::scalar_t,
                         RobotOCP::MATRIXFMT, linear_solver_traits<RobotOCP::MATRIXFMT>::default_solver>;

using box_admm_solver = boxADMM<RobotOCP::VAR_SIZE, RobotOCP::NUM_EQ, RobotOCP::scalar_t,
                                RobotOCP::MATRIXFMT, linear_solver_traits<RobotOCP::MATRIXFMT>::default_solver>;

using preconditioner_t = polympc::RuizEquilibration<RobotOCP::scalar_t, RobotOCP::VAR_SIZE, RobotOCP::NUM_EQ, RobotOCP::MATRIXFMT>;


int main(void)
{
    using scalar_t = double;
    using opt_t = Opt_t<scalar_t>;

    opt_t opt;
    std::cout << "opt = \n" << opt.m_D << std::endl;

    // using mpc_t = MPC<RobotOCP, MySolver, box_admm_solver>;
    // mpc_t mpc;
    // mpc.ocp().set_Q_coeff(2.0);
    // mpc.settings().max_iter = 20;
    // mpc.settings().line_search_max_iter = 10;
    // mpc.set_time_limits(0, 2);

    // // problem data
    // mpc_t::static_param p; p << 2.0;          // robot wheel base
    // mpc_t::state_t x0; x0 << 0.5, 0.5, 0.5;   // initial condition
    // mpc_t::control_t lbu; lbu << -1.5, -0.75; // lower bound on control
    // mpc_t::control_t ubu; ubu <<  1.5,  0.75; // upper bound on control

    // mpc.set_static_parameters(p);
    // mpc.control_bounds(lbu, ubu);
    // mpc.initial_conditions(x0);

    // polympc::time_point start = polympc::get_time();
    // mpc.solve();
    // polympc::time_point stop = polympc::get_time();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    // std::cout << "MPC status: " << mpc.info().status.value << "\n";
    // std::cout << "Num iterations: " << mpc.info().iter << "\n";
    // std::cout << "Solve time: " << std::setprecision(9) << static_cast<double>(duration.count()) << "[mc] \n";

    // std::cout << "Solution X: " << mpc.solution_x().transpose() << "\n";
    // std::cout << "Solution U: " << mpc.solution_u().transpose() << "\n";

    // auto segment = mpc.solution_x();
    // mpc_t::scalar_t *lox;
    // lox = mpc.solution_x().data(); //segment.data(); //mpc.solver().primal_solution().template head<10>().data();

    // std::cout << "lox: \n";
    // for(int i = 0; i < 10; ++i)
    //     std::cout << lox[i] << " ";
    // std::cout << std::endl;

    // // warm started iteration
    // x0 << 0.3, 0.4, 0.5;
    // mpc.initial_conditions(x0, x0);

    // start = polympc::get_time();
    // mpc.solve();
    // stop = polympc::get_time();
    // duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    // std::cout << "Solve status: " << mpc.info().status.value << "\n";
    // std::cout << "Num iterations: " << mpc.info().iter << "\n";
    // std::cout << "Solve time: " << std::setprecision(9) << static_cast<double>(duration.count()) << "[mc] \n";

    // std::cout << "Solution X: \n" << mpc.solution_x_reshaped() << "\n";
    // std::cout << "Solution U: \n" << mpc.solution_u_reshaped() << "\n";

    // // sample x solution at collocation points [0, 5, 10]
    // std::cout << "x[0]: " << mpc.solution_x_at(0).transpose() << "\n";
    // std::cout << "x[5]: " << mpc.solution_x_at(5).transpose() << "\n";
    // std::cout << "x[10]: " << mpc.solution_x_at(10).transpose() << "\n";

    // std::cout << " ------------------------------------------------ \n";

    // //sample control at collocation points
    // std::cout << "u[0]: " << mpc.solution_u_at(0).transpose() << "\n";
    // std::cout << "u[1]: " << mpc.solution_u_at(1).transpose() << "\n";

    // std::cout << " ------------------------------------------------ \n";

    // // sample state at time 't'
    // std::cout << "x(0.0): " << mpc.solution_x_at(0.0).transpose() << "\n";
    // std::cout << "x(0.5): " << mpc.solution_x_at(0.5).transpose() << "\n";

    // std::cout << " ------------------------------------------------ \n";

    // //  sample control at time 't'
    // std::cout << "u(0.0): " << mpc.solution_u_at(0.0).transpose() << "\n";
    // std::cout << "u(0.5): " << mpc.solution_u_at(0.5).transpose() << "\n";


    // RobotOCP& ocp = mpc.ocp();
    // mpc_t::nlp_solver_t& solver = mpc.solver();
    // using ocp_t = RobotOCP;

    // ocp_t::nlp_eq_constraints_t constraints;
    // ocp_t::nlp_eq_jacobian_t jacobian;

    // jacobian.resize(ocp_t::NUM_EQ, ocp_t::VAR_SIZE);

    // solver.problem.equalities_linearised(solver.primal_solution(), solver.parameters(), constraints, jacobian);
    // std::cout << "constraints = " << constraints.transpose() << std::endl;
    // std::cout << "jacobian = \n"  << std::setprecision(1) << jacobian << std::endl;

    // return EXIT_SUCCESS;
}
