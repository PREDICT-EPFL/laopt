#include <pybind11/embed.h>
#include <pybind11/eigen.h>

#include <Eigen/Dense>
using namespace Eigen;

template<typename scalar_t>
Matrix<scalar_t, 2, 1> sys(const Ref<const Matrix<scalar_t, 2, 1>>& x, const Ref<const Matrix<scalar_t, 1, 1>>& u)
{
  scalar_t p = (3.14159265359) * (3);
  Matrix<scalar_t, 2, 2> A = ((Matrix<scalar_t, 2, 2>({{2,2},{3,4}})).array() * (p)).matrix();
  Matrix<scalar_t, 2, 1> B = A(all, 1);
  Matrix<scalar_t, 2, 1> q{{5},{6}};
  q = (B) * (u);
  Matrix<scalar_t, 2, 1> t = ((((sin((A).array()).matrix()) * (x)) + ((B) * (u))).array() + (3)).matrix();
  return t;
}

PYBIND11_MODULE(example, m) {
    m.def("sys", &sys<double>, "None");
}