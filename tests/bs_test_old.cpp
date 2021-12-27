/**
 * Unit test for the construction and computation of LAMPC functions.
 */

#include "bsmatrix.hpp"
#include "lampc_function.hpp"

#include <iomanip>
#include <Eigen/Eigenvalues> 
#include <type_traits>

using namespace lampc;

#include "gtest/gtest.h"

namespace {
using namespace BS;

template<typename Scalar, std::size_t N, typename range>
struct BSTest;

template<typename Scalar, std::size_t N, std::size_t... ind>
struct BSTest<Scalar, N, std::integer_sequence<std::size_t, ind...>>
{
    struct param_t
    {
        Eigen::Matrix<Scalar, 2, 1> x0 {-1, -2};
        const Eigen::Matrix<Scalar, 2, 2> A {{1.0, 0.0}, {0.1, 1.0}};
        Eigen::Matrix<Scalar, 2, 1> B {0.1, 0.005};
        Eigen::Matrix<Scalar, 1, 1> ref {3};

        Eigen::Matrix<Scalar, 2, 1> q {1, 1e3}; // Stage-cost weights
        Eigen::Matrix<Scalar, 1, 1> r {1e-3}; // Stage-cost weights
    };


    FUNCTION(dynamics, Scalar, param_t, (xplus, 2), (x, 2), (u, 1))
    {
        xplus = p.A.template cast<T>() * x + x(0) * p.B.template cast<T>() * u;
    }

    FUNCTION(dynamics_eq, Scalar, param_t, (out, 2), (xplus, 2), (x, 2), (u, 1))
    {
        Eigen::Matrix<T, 2, 1> tmp;
        dynamics::template impl<T>(p, tmp, x, u);
        out = tmp - xplus;
    }

    FUNCTION(test_func, Scalar, param_t, (out, 2), (x, 2), (u, 1))
    {
        out << x(0)*x(1)+x(0)*x(0)*10.0,
            u(0)*x(0)*x(1)+u(1)*x(0)*x(0)*10.0;
    }



    // /** Option 1
    //  * Block storage
    //  */

    // Block 





    // Variable x{2, N};
    // Variable u{1, N-1};

    // VariableSet variables;

    // Constraint 

    // BSTest()
    // {
    //     // Set the order here
    //     for(int i=0; i<N; i++)
    //     {
    //         variables.push(x(i));
    //         variables.push(u(i));            
    //     }
    //     variables.push(x(N-1));

    //     variables.finalize(); // Write offset into each variable
    // }



    template<int i>
    struct x : Variable<2> {static constexpr const char* name="x";};

    template<int i>
    struct u : Variable<1> {static constexpr const char* name="u";};

    using variables = std::tuple<x<ind>..., u<ind>..., x<N>>;

    template<int i>
    struct con_dynamics : BS::FunctionConstraint<dynamics_eq, x<i+1>, x<i>, u<i>> {};

    struct con_test : BS::FunctionConstraint<test_func, x<1>, u<3>> {};

    using constraints = std::tuple<con_dynamics<ind>..., con_test>;

  //   BSTest()
  //   {
  //       mat.finalize();

  //       std::cout << "mat = \n" << Eigen::MatrixX<double>(mat) << std::endl;

  //       // Eigen::Matrix<double, con2::size, z::size> B;
  //       // B.array() = 1;

  //       // Eigen::Matrix<double, con1<1>::size, x<1>::size> B1;
  //       // B1.array() = 2;
        
  // //    time_point start = get_time();
  //       // for(int i=0; i<1000000; i++)
  //       // {
  //       //  // B.array() += i;
  //       //  mat.template set<con2, z>(B);
  //       //  mat.template set<con1<1>, x<1>>(B1);
  //       //  // B.array() -= i;
  //       // }
  //    //    time_point stop = get_time();

  //    //    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  //    //    std::cout << "Time " << std::setprecision(9)
  //    //              << static_cast<double>(duration.count()) / 1000000 << " [microseconds]" << "\n";

  //       // std::cout << "mat = \n" << Eigen::MatrixX<double>(mat) << std::endl;

  //   }
};



TEST(BSTest, FunctionMatrix) {
    using Scalar = double;
    const std::size_t N = 5;

    using Test = BSTest<Scalar, N, std::make_integer_sequence<std::size_t, N>>;
    using param_t = Test::param_t;
    using Matrix = BSMatrix<Scalar, Test::constraints, Test::variables>;
    Matrix mat;
    Eigen::SparseMatrix<Scalar> S;
    mat.initialize(S);

    BS::BSVector<double, Test::variables> v;
    v.array() = 0;

    EXPECT_EQ(v.RowsAtCompileTime, 17);

    auto x0 = v.template get<Test::x<0>>();
    x0 << -1, -1;
    Eigen::Vector<Scalar, 17> sol;
    sol.array() = 0;
    sol.template segment<2>(0) << -1, -1;
    // EXPECT_EQ((v - sol).norm(), 0);

    v.template get<Test::x<1>>().array() = 1;
    v.template get<Test::x<2>>().array() = 2;
    v.template get<Test::x<3>>().array() = 3;
    v.template get<Test::x<4>>().array() = 4;
    v.template get<Test::x<5>>().array() = 5;

    Eigen::Vector<Scalar, 17> sol2;
    sol2.array() = 0;
    sol2.template segment<2>(0) << -1, -1;
    sol2.template segment<2>(2) << 1, 1;
    sol2.template segment<2>(4) << 2, 2;
    sol2.template segment<2>(6) << 3, 3;
    sol2.template segment<2>(8) << 4, 4;
    sol2.template segment<2>(15) << 5, 5;
    // EXPECT_EQ((v - sol2).norm(), 0);

    std::cout << "v    = " << v.transpose() << std::endl;
    std::cout << "sol2 = " << sol2.transpose() << std::endl;

}

template<typename Scalar, typename ConstraintTuple, typename VariableTuple>
struct ConstraintSet;

template<typename Scalar, typename... Constraints, typename... Variables>
struct ConstraintSet<Scalar, std::tuple<Constraints...>, std::tuple<Variables...>> 
{
// private:

    using ConstraintTuple = std::tuple<Constraints...>;
    using VariableTuple = std::tuple<Variables...>;

    // Block variables and matrices
    // These contain offset information for the block-variables
    using bs_variable_t = BS::BSVector<Scalar, VariableTuple>;
    using bs_constraint_t = BS::BSVector<Scalar, ConstraintTuple>;
    using bs_jacobian_t = BSMatrix<Scalar, ConstraintTuple, VariableTuple>;

    bs_jacobian_t jacobian_dat;

public:
    static constexpr int num_variables = sum_template<Variables::size...>();
    static constexpr int num_constraints = sum_template<Constraints::size...>();

    using jacobian_t = typename bs_jacobian_t::type;
    using variable_t = typename bs_variable_t::type;
    using constraint_t = typename bs_constraint_t::type;

    ConstraintSet()
    {
        Eigen::SparseMatrix<Scalar> S;
        jacobian_dat.initialize(S);
    }

    void initialize_jacobian(Eigen::SparseMatrix<Scalar>& S)
    {
        jacobian_dat.initialize(S);
    }

    /*
        Evaluate all constraints
     */
    template<typename param_t>
    EIGEN_STRONG_INLINE void operator()(
        const param_t& param,
        Eigen::Ref<variable_t> var, Eigen::Ref<constraint_t> con)
        const noexcept
    {
        con.array() = 0;
        auto l = {(
            Constraints::template eval<bs_variable_t, bs_constraint_t, Constraints>(param, var, con),
            0)...};
    }

    /*
        Evaluate all constraints and jacobian
     */
    template<typename param_t>
    EIGEN_STRONG_INLINE auto operator()(
        const param_t& param, 
        Eigen::Ref<variable_t> var, Eigen::Ref<constraint_t> con, Eigen::Ref<jacobian_t> jac)
        const noexcept
    {
        con.array() = 0;
        // jac.array() = 0;

        // auto set_jacobian = [](auto ret_jac){
        // };

        // auto l = {(
        //     Constraints::template jacobian<bs_jacobian_t>(param, var, con, jac),
        //     // std::cout << "here\n",
        //     // bs_constraint_t::template get<Constraints>(con) = 
        //     0)...};

        // (void)std::initializer_list<int>{ 
        //     (
        //         std::get<ind>(cons)(param, var, con, jac),
        //         0
        //     )...
        // };
    }
};

TEST(BSTest, ConstraintSet) {
    using Scalar = double;
    const std::size_t N = 20;
 
    using Test = BSTest<Scalar, N, std::make_integer_sequence<std::size_t, N>>;
    using param_t = Test::param_t;

    using CS = ConstraintSet<double, Test::constraints, Test::variables>;
    CS cons;
    param_t param;

    CS::variable_t x;
    CS::constraint_t con;
    CS::jacobian_t jac;

    cons.initialize_jacobian(jac);

    for(int i=0; i<CS::num_variables; i++) x[i] = i;

    cons(param, x, con);
    cons(param, x, con, jac);

    std::cout << "con = " << con.transpose() << std::endl;

    std::cout << "CS::bs_constraint_t::get<Test::con_dynamics<0>>(con) = " << CS::bs_constraint_t::get<Test::con_dynamics<0>>(con).transpose() << std::endl;
    std::cout << "CS::bs_constraint_t::get<Test::con_dynamics<1>>(con) = " << CS::bs_constraint_t::get<Test::con_dynamics<1>>(con).transpose() << std::endl;
    std::cout << "CS::bs_constraint_t::get<Test::con_dynamics<2>>(con) = " << CS::bs_constraint_t::get<Test::con_dynamics<2>>(con).transpose() << std::endl;
    std::cout << "CS::bs_constraint_t::get<Test::con_dynamics<3>>(con) = " << CS::bs_constraint_t::get<Test::con_dynamics<3>>(con).transpose() << std::endl;
    std::cout << "CS::bs_constraint_t::get<Test::con_dynamics<4>>(con) = " << CS::bs_constraint_t::get<Test::con_dynamics<4>>(con).transpose() << std::endl;

    std::cout << "jac = \n" << jac << std::endl;
}



}