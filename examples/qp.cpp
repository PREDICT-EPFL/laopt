/**
 * Use polyMPC and LACompiler to solve a simple QP
 */
// #include "lampc.hpp"

// #define SEG(len,offset) template segment<len>(offset)

// #include "qp_functions.hpp"
// #include "qp.compiled.hpp"

#include "qp_lib.hpp"

int main()
{
    QP::variable_t x;
    x.array() = 0;

    QP::param_t param;
    QP::equalities::out_t eq;
    QP::equalities::jacobian_t J;

    solve_qp(param, x, eq, J);

    std::cout << "eq = " << eq.transpose() << std::endl;

    param.x0[0] = 100;
    x.array() = 0;
    solve_qp(param, x, eq, J);
    std::cout << "eq = " << eq.transpose() << std::endl;

    std::cout << "J = \n" << Eigen::MatrixX<double>(J) << std::endl;

// typedef Eigen::Triplet<double> T;
// std::array<T,3> tripletList = {T{1,1,4},{2,2,5},{3,3,6}};
// // tripletList[0] = T(1,1,4);
// // tripletList[1] = T(2,2,5);
// // tripletList[2] = T(3,3,6);

// // tripletList.reserve(10);
// // tripletList.push_back(T(1,1,4));
// // tripletList.push_back(T(2,2,5));
// // tripletList.push_back(T(3,3,6));

// Eigen::SparseMatrix<double> mat(10,10);
// mat.setFromTriplets(tripletList.begin(), tripletList.end());

// std::cout << "mat = \n" << Eigen::MatrixX<double>(mat) << std::endl;

    // eq.eval_jacobian(param, x, con_eq);
    // std::cout << "con_eq = " << con_eq.transpose() << std::endl;
    // std::cout << "eq.jacobian = \n" << Eigen::MatrixX<double>(eq.jacobian) << std::endl;

    return 1;
}
