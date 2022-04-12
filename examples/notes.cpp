/**
 * Unit test for the construction and computation of LAMPC problems.
 */

#include <iostream>
#include "lampc.hpp"

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

}
int main()
{
    MultipleShooter<double> t(10);

    std::cout << "sparsity = \n" << t.S << std::endl;
    std::cout << "shoot = \n" << t.shoot() << std::endl;


}
