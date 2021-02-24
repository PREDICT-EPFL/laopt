// Eigen includes
#include <Eigen/Core>
#include <Eigen/StdVector>
using namespace Eigen;


template<typename Scalar>
class DoubleIntegrator
{
    Matrix<Scalar, 2, 1> dynamics(Matrix<Scalar, 2, 1> xp, Matrix<Scalar, 2, 1> x, Matrix<Scalar, 1, 1> u)
    {
        Matrix<Scalar, 2, 1> out;
        out = xp + x - u;
        return out;
    }
}


struct NLP_traits
{
    using int num_var = 4;
    using int num_eq = 4;
    using int num_ineq = 7;    
};



// The NLP would have the form

template <typename Scalar,
          typename NLP_traits,
          typename eq_offsets // tuple(pair(offset, size), pair(offset, size), ...)
          >
class DenseNLP // Dense Jacobians
{
    using num_var = NLP_traits::num_var;
    using num_eq = NLP_traits::num_eq;
    using num_ineq = NLP_traits::num_ineq;

    Eigen::Matrix<Scalar, num_var, 1> primal;

    Eigen::Matrix<Scalar, num_eq, num_var> J_eq;
    Eigen::Matrix<Scalar, num_eq, 1> g_eq;

    Eigen::Matrix<Scalar, num_ineq, num_var> J_ineq;
    Eigen::Matrix<Scalar, num_ineq, 1> g_ineq;

    Eigen::Matrix<dual, num_var, 1> primal_d; // Dual variables

    // static eq_offsets

    // Generated...
    void eval()
    {
        DoubleIntegrator<dual>::dynamics(col(X, i+1), col(X, i), col(U, i));

        for (int colIndex = 0; colIndex <= length(ceq.varIndices); colIndex++)
        {
            for (int row = offset<i>; row <= offset<i> + size<i>; row++)
            {
            }
        }
    }

    // This is the generated file...
    void update_jacobians()
    {

    }
}


const int N = 5;
auto X = nlp.Variable<2, N>; // tuple(pair(offset, size), pair(offset, size), ...)
auto U = nlp.Variable<1, N>; 

#define col(Var, i) primal.segment(get<i>(Var)::First, get<i>(Var)::Second)

template<typename... Args>
auto WRT(Args...&& args)
{
    return std::forward_as_tuple<Args...>(args...);
}

for (int i=0; i<N-1; i++)
{
    nlp.add_eq(DoubleIntegrator<dual>::dynamics, WRT(col(X, i+1), col(X, i), col(U, i)));
}
