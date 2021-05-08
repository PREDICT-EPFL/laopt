#include "lampc.hpp"
// #include "ipopt_interface.hpp"

/***********************************************************
    Code generated from Python or from user
 ***********************************************************/

template<typename scalar_t>
struct Opt
{
    struct param_t
    {
        Eigen::Matrix<scalar_t, 2, 1> x0 = {-10, -20};
        const Eigen::Matrix<scalar_t, 2, 2> A = {{-1, -2}, {-3, -4}};
        Eigen::Matrix<scalar_t, 2, 1> B = {-1, -2};

        Eigen::Matrix<scalar_t, 2, 1> q = {1, 2}; // Stage-cost weights
        Eigen::Matrix<scalar_t, 1, 1> r = {3}; // Stage-cost weights
    };
    param_t p;

    struct func1_
    {
        template<typename T>
        static EIGEN_STRONG_INLINE void eval(const param_t& p, Vec<T, 2> out, cVec<T, 1>& var0_, cVec<T, 2>& var1_) noexcept
        {
            Eigen::Matrix<T, 2, 1> tmp1;
            dynamics_::template eval<T>(p, tmp1, p.x0.template cast<T>(), var0_);
            out = var1_ - tmp1;
        }
    };
    using func1 = Jacobian<func1_, scalar_t, param_t, 2, 1, 2>;

    struct dynamics_
    {
        template<typename T>
        static EIGEN_STRONG_INLINE void eval(const param_t& p, Vec<T, 2> out, cVec<T, 2>& x, cVec<T, 1>& u) noexcept
        {
            out = p.A.template cast<T>() * x + p.B.template cast<T>() * u;
            out(0) += x(0)*x(1)*u(0);
            out(1) += x(0)*10*u(0);
        }
    };
    using dynamics = Jacobian<dynamics_, scalar_t, param_t, 2, 2, 1>;

    struct func3_
    {
        template<typename T>
        static EIGEN_STRONG_INLINE void eval(const param_t& p, Vec<T, 2> out, cVec<T, 1>& var0_, cVec<T, 2>& var1_, cVec<T, 2>& var2_) noexcept
        {
            Matrix<T, 2, 1> tmp2;
            dynamics_::template eval<T>(p, tmp2, var1_, var0_);
            out = var2_ - tmp2;
        }
    };
    using func3 = Jacobian<func3_, scalar_t, param_t, 2, 1, 2, 2>;

    struct func2_
    {
        template<typename T>
        static EIGEN_STRONG_INLINE void eval(const param_t& p, Vec<T, 2> out, cVec<T, 1>& var0_, cVec<T, 2>& var1_) noexcept
        {
            Matrix<T, 2, 1> tmp3;
            dynamics_::template eval<T>(p, tmp3, var1_, var0_);
            out = var1_ - tmp3;
        }
    };
    using func2 = Jacobian<func2_, scalar_t, param_t, 2, 1, 2>;

    struct stage_cost_
    {
        template<typename T>
        static EIGEN_STRONG_INLINE void eval(const param_t& p, Vec<T, 1> out, cVec<T, 2>& x, cVec<T, 1>& u) noexcept
        {
            out(0) = x.cwiseProduct(p.q.template cast<T>()).dot(x) + u.cwiseProduct(p.r.template cast<T>()).dot(u);
        }
    };
    using stage_cost = Jacobian<stage_cost_, scalar_t, param_t, 1, 2, 1>;



    // Define variable accessors and ordering
    static constexpr int N = 10;
    using xss = var_t<2, 1>;
    using uss = var_t<1, 1, xss>;
    using x = var_t<2, N, uss>;
    using u = var_t<1, N , x>;
    static constexpr std::size_t num_variables = sum_variable_size<xss, uss, x, u>();

    // Define constraints
    using constraints_t = 
        make_constraints<
            num_variables,
            con_t<func1, 1, iterator<u,0,0>, iterator<x,0,0>>,  // dynamics(x(0), u(0)) == x(1)
            con_t<func3, N-2, iterator<u,1,1>, iterator<x,1,1>, iterator<x,2,1>>, // dynamics(u(i), x(i)) == x(i+1) for i in range(0, N-1)
            con_t<func2, 1, iterator<uss>, iterator<xss>>,  // dynamics(xss, uss) == xss
            con_t<stage_cost, N-2, iterator<x,0,1>, iterator<u,0,1>>  // x(i)'*Q*x(i) + u(i)'*R*u(i) for i in range(0, N-1)
        >;
    static constexpr std::size_t num_constraints = constraints_t::constraint_vector_length();
    constraints_t constraints;

    Opt() : constraints(constraints_t::make_constraints()) {}
};





/***********************************************************
    Implementation
 ***********************************************************/
int main()
{
    using scalar_t = double;
    using Opt = Opt<scalar_t>;
    Opt opt;

    // {
    //     std::cout << "\n\n===Test direct computation of functions===\n";
        
    //     Eigen::Matrix<scalar_t, 2, 1> x = {1, 2};
    //     Eigen::Matrix<scalar_t, 1, 1> u = {3};

    //     std::cout << "stage_cost::eval(opt.p, x, u) = " << Opt::stage_cost::eval(opt.p, x, u).transpose() << std::endl;
    //     std::cout << "\n\n";

    //     std::cout << "jac = stage_cost::jac(opt.p, x, u)" << std::endl;
    //     auto jac = Opt::stage_cost::jac(opt.p, x, u);
    //     std::cout << "jac.val = " << jac.val.transpose() << std::endl;
    //     std::cout << "jac.jacobian = \n" << jac.jacobian << std::endl;
    //     std::cout << "\n\n";

    //     std::cout << "hessian = stage_cost::hessian(opt.p, x, u) = " << Opt::stage_cost::eval(opt.p, x, u).transpose() << std::endl;
    //     auto hessian = Opt::stage_cost::hessian(opt.p, x, u);
    //     std::cout << "hessian.val = " << hessian.val.transpose() << std::endl;
    //     std::cout << "hessian.jacobian = \n" << hessian.jacobian << std::endl;
    //     for(int i=0; i<Opt::stage_cost::num_outputs; i++)
    //     {
    //         std::cout << "hessian.hessian(" << i << ")\n";
    //         std::cout << hessian.hessian[i] << std::endl;
    //     }        
    // }


    {
        std::cout << "============================================================\n";
        std::cout << "===Test computation of constraints and objective function===\n";
        std::cout << "============================================================\n";

        // std::cout << "num_variables = " << opt.num_variables << std::endl;
        // std::cout << "num_constraints = " << opt.num_constraints << std::endl;

        Eigen::Matrix<scalar_t, Opt::num_variables, 1> var;
        for(int i=0; i<Opt::num_variables; i++) var[i] = i;
        Eigen::Matrix<scalar_t, Opt::num_constraints, 1> con;
        for(int i=0; i<Opt::num_constraints; i++) con[i] = 0;

        Eigen::Matrix<scalar_t, Opt::num_constraints, Opt::num_variables> J;
        J.setZero();

        Eigen::SparseMatrix<scalar_t> sJ(Opt::num_constraints, Opt::num_variables);
        opt.constraints.initialize_sparse_jacobian(sJ);

        Eigen::Matrix<scalar_t, Opt::num_variables, Opt::num_variables> H;
        H.setZero();
        Eigen::SparseMatrix<scalar_t> sH(Opt::num_variables, Opt::num_variables);
        opt.constraints.initialize_sparse_hessian(sH);

        std::cout << "Hessian sparsity" << std::endl;
        std::cout << sH << std::endl;

        Eigen::Matrix<scalar_t, Opt::num_constraints, 1> hessian_multiplier;
        hessian_multiplier.array() = 1.0;


        std::cout << "==> Computing value of constraints <==\n";
        opt.constraints(opt.p, var, con);
        std::cout << "con = " << con.transpose() << std::endl;
        std::cout << "\n\n";

        std::cout << "==> Computing dense jacobian <==\n";
        opt.constraints(opt.p, var, con, J);
        std::cout << "con = " << con.transpose() << std::endl;
        std::cout << "J = \n" << J << std::endl;
        std::cout << "\n\n";

        std::cout << "==> Computing dense hessian <==\n";
        opt.constraints(opt.p, var, con, J, H, hessian_multiplier);
        std::cout << "con = " << con.transpose() << std::endl;
        std::cout << "J = \n" << J << std::endl;
        std::cout << "H = \n" << H << std::endl;
        std::cout << "\n\n";

        std::cout << "==> Computing sparse jacobian <==\n";
        std::cout << "con = " << con.transpose() << std::endl;
        std::cout << "sJ = \n" << Eigen::MatrixXd(sJ) << std::endl;
        std::cout << "\n\n";

        std::cout << "==> Computing sparse hessian <==\n";
        opt.constraints(opt.p, var, con, sJ, sH, hessian_multiplier);
        std::cout << "con = " << con.transpose() << std::endl;
        std::cout << "sJ = \n" << Eigen::MatrixXd(sJ) << std::endl;
        std::cout << "sH = \n" << Eigen::MatrixXd(sH) << std::endl;
        std::cout << "\n\n";

    }



    // using myNLP = NLP_Ipopt<double, LOpt<double>>;
    // SmartPtr<myNLP> mynlp = new myNLP();
    // Ipopt::SmartPtr<Ipopt::IpoptApplication> app = new Ipopt::IpoptApplication();

    // app->Options()->SetNumericValue("tol", 1e-7);
    // app->Options()->SetStringValue("mu_strategy", "adaptive");
    // app->Options()->SetStringValue("output_file", "ipopt.out");
    // app->Options()->SetStringValue("hessian_approximation", "limited-memory");

    // ApplicationReturnStatus status;
    // status = app->Initialize();
    // if( status != Solve_Succeeded )
    // {
    //     std::cout << std::endl << std::endl << "*** Error during initialization!" << std::endl;
    //     return (int) status;
    // }

    // // Ask Ipopt to solve the problem
    // mynlp->x0 = {0,1};
    // std::cout << "x0 = " << mynlp->x0 << std::endl;
    // status = app->OptimizeTNLP(mynlp);

    // std::cout << "A = \n" << mynlp->A << std::endl;
    // std::cout << "B = \n" << mynlp->B << std::endl;
    // return (int) status;

}