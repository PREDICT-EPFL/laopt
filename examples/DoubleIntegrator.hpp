#include <Eigen/Core>
#include <Eigen/StdVector>
using namespace Eigen;

namespace DoubleIntegrator
{
    using namespace Eigen;

    const size_t nx = 2;
    const size_t nu = 1;

    template<typename T>
    using StateType = Matrix<T, nx, 1>;

    template<typename T>
    using InputType = Matrix<T, nu, 1>;

    template <typename T>
    auto dynamics(StateType<T> xp,
                  StateType<T> x,
                  InputType<T> u)
    {
        StateType<T> eq;
        eq << xp[0] - (-sin(x[1]) + x[1]*x[0]), xp[1] - cos(x[0])*u[0];
        return eq;
    }
}; // namespace DoubleIntegrator
