#include "laopt/differentiable_functions/identity.hpp"
#include "laopt/differentiable_functions/quadratic_cost.hpp"
#include "laopt/differentiable_functions/replicate.hpp"
#include "gtest/gtest.h"

TEST(DifferentiableFunctionsTest, Identity)
{
    using scalar_t = double;

    Eigen::Vector<scalar_t, 3> x_data; x_data << 1, 2, 3;
    laopt::Variable<scalar_t, 3> x(x_data.data());
    Eigen::Vector<scalar_t, 3> weight; weight << 2, 3, 4;

    laopt::Identity id;

    Eigen::Vector<scalar_t, 3> f_val = id.function(x);
    EXPECT_EQ(f_val, x_data);

    scalar_t w_val = id.wsum(weight, x);
    EXPECT_EQ(w_val, 20);

    Eigen::Matrix<scalar_t, 3, 3> j_val; j_val.setZero();
    id.jacobian(j_val, 1.0, x);
    Eigen::Matrix<scalar_t, 3, 3> j_exp; j_exp << 1, 0, 0,
                                                  0, 1, 0,
                                                  0, 0, 1;
    EXPECT_EQ(j_val, j_exp);

    Eigen::Vector<scalar_t, 3> g_val; g_val.setZero();
    id.gradient(g_val, weight, x);
    Eigen::Vector<scalar_t, 3> g_exp; g_exp << 2, 3, 4;
    EXPECT_EQ(g_val, g_exp);

    Eigen::Matrix<scalar_t, 3, 3> h_val; h_val.setZero();
    id.hessian(h_val, weight, x);
    Eigen::Matrix<scalar_t, 3, 3> h_exp; h_exp.setZero();
    EXPECT_EQ(h_val, h_exp);
}

TEST(DifferentiableFunctionsTest, QuadraticCost)
{
    using scalar_t = double;

    Eigen::Vector<scalar_t, 3> x_data; x_data << 1, 2, 3;
    laopt::Variable<scalar_t, 3> x(x_data.data());
    Eigen::Vector<scalar_t, 1> weight; weight << 2;

    laopt::QuadraticCost<Eigen::DiagonalMatrix<scalar_t, 3, 3>> qc;
    qc.P.diagonal() << 1, 2, 3;
    qc.q << 5, 6, 7;
    qc.r = 0.5;

    scalar_t f_val = qc.function(x);
    EXPECT_EQ(f_val, 56.5);

    scalar_t w_val = qc.wsum(weight, x);
    EXPECT_EQ(w_val, 113);

    Eigen::Matrix<scalar_t, 1, 3> j_val; j_val.setZero();
    qc.jacobian(j_val, 1.0, x);
    Eigen::Matrix<scalar_t, 1, 3> j_exp; j_exp << 6, 10, 16;
    EXPECT_EQ(j_val, j_exp);

    Eigen::Vector<scalar_t, 3> g_val; g_val.setZero();
    qc.gradient(g_val, weight, x);
    Eigen::Vector<scalar_t, 3> g_exp; g_exp << 12, 20, 32;
    EXPECT_EQ(g_val, g_exp);

    Eigen::Matrix<scalar_t, 3, 3> h_val; h_val.setZero();
    qc.hessian(h_val, weight, x);
    Eigen::Matrix<scalar_t, 3, 3> h_exp; h_exp.setZero(); h_exp.diagonal() << 2, 4, 6;
    EXPECT_EQ(h_val, h_exp);
}

TEST(DifferentiableFunctionsTest, Replicate)
{
    using scalar_t = double;

    Eigen::Vector<scalar_t, 3> x_data; x_data << 1, 2, 3;
    laopt::Variable<scalar_t, 3> x(x_data.data());
    Eigen::Vector<scalar_t, 9> weight; weight << 2, 3, 4, 5, 6, 7, 8, 9, 10;

    laopt::Replicate<3> rep;

    Eigen::Vector<scalar_t, 9> f_val = rep.function(x);
    Eigen::Vector<scalar_t, 9> f_exp; f_exp << x, x, x;
    EXPECT_EQ(f_val, f_exp);

    scalar_t w_val = rep.wsum(weight, x);
    EXPECT_EQ(w_val, 114);

    Eigen::Matrix<scalar_t, 9, 3> j_val; j_val.setZero();
    rep.jacobian(j_val, 1.0, x);
    Eigen::Matrix<scalar_t, 9, 3> j_exp; j_exp << 1, 0, 0,
                                                  0, 1, 0,
                                                  0, 0, 1,
                                                  1, 0, 0,
                                                  0, 1, 0,
                                                  0, 0, 1,
                                                  1, 0, 0,
                                                  0, 1, 0,
                                                  0, 0, 1;
    EXPECT_EQ(j_val, j_exp);

    Eigen::Vector<scalar_t, 3> g_val; g_val.setZero();
    rep.gradient(g_val, weight, x);
    Eigen::Vector<scalar_t, 3> g_exp; g_exp << 15, 18, 21;
    EXPECT_EQ(g_val, g_exp);

    Eigen::Matrix<scalar_t, 9, 3> h_val; h_val.setZero();
    rep.hessian(h_val, weight, x);
    Eigen::Matrix<scalar_t, 9, 3> h_exp; h_exp.setZero();
    EXPECT_EQ(h_val, h_exp);
}
