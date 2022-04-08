/**
 * Unit test for the construction and computation of LAMPC problems.
 */

#include <iostream>
#include "lampc.hpp"

#include "test_utils.hpp"
#include "gtest/gtest.h"

namespace {

/**
 * Simple general RK4 integrator
 */
template<typename scalar_t, typename diff_t=scalar_t, typename O, typename X, typename... Parameters>
EIGEN_STRONG_INLINE auto rk4(O ode, const scalar_t _h, X& x, Parameters&... params) noexcept
{
  diff_t h = static_cast<diff_t>(_h);
  auto k1 = ode(x,                               params...);
  auto k2 = ode(x+h/static_cast<diff_t>(2.0)*k1, params...);
  auto k3 = ode(x+h/static_cast<diff_t>(2.0)*k2, params...);
  auto k4 = ode(x+h*k3,                          params...);
  return x + h/static_cast<diff_t>(6.0) * (k1 + static_cast<diff_t>(2.0)*k2 + static_cast<diff_t>(2.0)*k3 + k4);
}


/**
 * Computes the jacobian of the dynamics for a simple multiple-shooting transcription
 */
template<typename scalar_t>
struct MultipleShooter
{
  Eigen::Matrix<scalar_t, 2, 2> A{{1,2},{3,4}};
  Eigen::Matrix<scalar_t, 2, 1> B{{5},{6}};

    template<typename diff_t=scalar_t>
    EIGEN_STRONG_INLINE Vector<diff_t, 2> 
        sys(const Ref<const Matrix<diff_t, 2, 1>>& x, const Ref<const Vector<diff_t, 1>>& u) noexcept
    {
      return ((((x(0)) * (A.template cast<diff_t>()).array()).matrix()) * (x)) + ((B.template cast<diff_t>()) * (u));
    }

    template<typename diff_t=scalar_t>
    EIGEN_STRONG_INLINE Vector<diff_t, 2> dsys(
            const Ref<const Vector<diff_t, 2>>& xp, 
            const Ref<const Vector<diff_t, 2>>& x, 
            const Ref<const Vector<diff_t, 1>>& u) noexcept
    {
        auto self=this;
        return xp - rk4<scalar_t, diff_t>([self](auto&... args){return self->template sys<diff_t>(args...);}, 0.1, x, u);
    }

  make_differentiable(dsys, 2, 2,2,1);

  lampc::Problem<scalar_t> prob;
  lampc::Variable<scalar_t>& x;
  lampc::Variable<scalar_t>& u;

  Eigen::VectorX<scalar_t> var; // The problem variable

  template<typename D, typename T>
    void shoot(T& tape)
    {
        for(int i=0; i<x().cols()-1; i++)
            std::tie(std::ignore, tape(-1,{x[i],u[i],x[i+1]})) = dsys(D(),x(i+1),u(i),x(i));
    };

    SparseMatrix<scalar_t>& shoot()
    {
        shoot<lampc::Jacobian>(mat);
        return S;
    }

    lampc::BSMatrix<scalar_t> mat;
    SparseMatrix<scalar_t> S;

  MultipleShooter(int N)   : x(prob.variable(2,N)), u(prob.variable(1,N)), var(prob.size)
  {
        prob.set_variable(var);

        // Create the tape to record the copy sequence
        lampc::BSMatrixTape<scalar_t> tape;
        shoot<lampc::Jacobian>(tape);
        tape.finalize_structure();
        shoot<lampc::Jacobian>(tape);

        // We now replace the tape with a block-sparse matrix
        // which loads the copy-sequence from the tape (or from file)
        mat.initialize_from_tape(tape);
        mat.initialize_matrix(S);
  }
};

TEST(ProblemTest, ShootJacobian) {
    MultipleShooter<double> t(10);

    // Test sparsity structure
    {
        auto ground = triplet_to_sparse<double>(18,29,{{0,0,1},{1,0,1},{0,1,1},{1,1,1},{0,2,1},{1,2,1},{2,2,1},{3,2,1},{0,3,1},{1,3,1},{2,3,1},{3,3,1},{2,4,1},{3,4,1},{4,4,1},{5,4,1},{2,5,1},{3,5,1},{4,5,1},{5,5,1},{4,6,1},{5,6,1},{6,6,1},{7,6,1},{4,7,1},{5,7,1},{6,7,1},{7,7,1},{6,8,1},{7,8,1},{8,8,1},{9,8,1},{6,9,1},{7,9,1},{8,9,1},{9,9,1},{8,10,1},{9,10,1},{10,10,1},{11,10,1},{8,11,1},{9,11,1},{10,11,1},{11,11,1},{10,12,1},{11,12,1},{12,12,1},{13,12,1},{10,13,1},{11,13,1},{12,13,1},{13,13,1},{12,14,1},{13,14,1},{14,14,1},{15,14,1},{12,15,1},{13,15,1},{14,15,1},{15,15,1},{14,16,1},{15,16,1},{16,16,1},{17,16,1},{14,17,1},{15,17,1},{16,17,1},{17,17,1},{16,18,1},{17,18,1},{16,19,1},{17,19,1},{0,20,1},{1,20,1},{2,21,1},{3,21,1},{4,22,1},{5,22,1},{6,23,1},{7,23,1},{8,24,1},{9,24,1},{10,25,1},{11,25,1},{12,26,1},{13,26,1},{14,27,1},{15,27,1},{16,28,1},{17,28,1}});
        EXPECT_TRUE(ground.isApprox(t.S, 0));
    }

    // Compute jacobian
    {
        auto ground = triplet_to_sparse<double>(18,29,{{0,0,1},{1,0,0},{0,1,0},{1,1,1},{0,2,0},{1,2,-1},{2,2,1},{3,2,0},{0,3,-0.5},{1,3,-0.6},{2,3,0},{3,3,1},{2,4,0},{3,4,-1},{4,4,1},{5,4,0},{2,5,-0.5},{3,5,-0.6},{4,5,0},{5,5,1},{4,6,0},{5,6,-1},{6,6,1},{7,6,0},{4,7,-0.5},{5,7,-0.6},{6,7,0},{7,7,1},{6,8,0},{7,8,-1},{8,8,1},{9,8,0},{6,9,-0.5},{7,9,-0.6},{8,9,0},{9,9,1},{8,10,0},{9,10,-1},{10,10,1},{11,10,0},{8,11,-0.5},{9,11,-0.6},{10,11,0},{11,11,1},{10,12,0},{11,12,-1},{12,12,1},{13,12,0},{10,13,-0.5},{11,13,-0.6},{12,13,0},{13,13,1},{12,14,0},{13,14,-1},{14,14,1},{15,14,0},{12,15,-0.5},{13,15,-0.6},{14,15,0},{15,15,1},{14,16,0},{15,16,-1},{16,16,1},{17,16,0},{14,17,-0.5},{15,17,-0.6},{16,17,0},{17,17,1},{16,18,0},{17,18,-1},{16,19,-0.5},{17,19,-0.6},{0,20,-1},{1,20,0},{2,21,-1},{3,21,0},{4,22,-1},{5,22,0},{6,23,-1},{7,23,0},{8,24,-1},{9,24,0},{10,25,-1},{11,25,0},{12,26,-1},{13,26,0},{14,27,-1},{15,27,0},{16,28,-1},{17,28,0}});
        EXPECT_TRUE(ground.isApprox(t.shoot(), 0));
    }

}

}