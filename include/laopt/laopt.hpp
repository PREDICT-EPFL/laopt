#ifndef LAOPT_LAOPT_HPP
#define LAOPT_LAOPT_HPP

#include "laopt/bs_matrix/bs_matrix_sparsity.hpp"
#include "laopt/bs_matrix/bs_matrix_tape.hpp"
#include "laopt/bs_matrix/bs_matrix.hpp"
#include "laopt/bs_matrix/bs_matrix_dense.hpp"

#include "laopt/autodiff/differentiable.hpp"
#include "laopt/autodiff/differentiable_functor.hpp"

#include "laopt/variable_map.hpp"

#include "laopt/expressions/expr_base.hpp"
#include "laopt/expressions/scalar_expr.hpp"
#include "laopt/expressions/add_expr.hpp"
#include "laopt/expressions/sub_expr.hpp"
#include "laopt/expressions/function_capture.hpp"

#include "laopt/expressions/constraint_expr.hpp"
#include "laopt/expressions/ineq_constraint_expr.hpp"
#include "laopt/expressions/eq_constraint_expr.hpp"
#include "laopt/expressions/bounded_expr.hpp"

#include "laopt/problem.hpp"

#endif // LAOPT_LAOPT_HPP