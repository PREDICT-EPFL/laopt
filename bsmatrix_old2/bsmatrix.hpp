#ifndef __BSMATRIX_HPP
#define __BSMATRIX_HPP

#include "bsmatrixcore.hpp"

namespace BS
{

template<int size_, typename... Variables>
struct DenseConstraint : Constraint<size_, Block<Variables>...> 
{

};

template<typename Function, typename... Variables>
struct FunctionConstraint : Constraint<Function::num_outputs, Block<Variables>...> 
{
	/** Evaluate the constraint
	 */
	template<typename bs_variable_t, typename bs_constraint_t, typename This, typename param_t>
	static EIGEN_STRONG_INLINE void eval(
		const param_t& param,
		Eigen::Ref<typename bs_variable_t::type> x,
		Eigen::Ref<typename bs_constraint_t::type> con)
		noexcept
	{
		bs_constraint_t::template get<This>(con) = Function::eval(param, bs_variable_t::template get<Variables>(x)...);
  }

	/** Evaluate the jacobian
	 */
	template<typename bs_jacobian_t, typename param_t>
	static EIGEN_STRONG_INLINE void jacobian(
		const param_t& param,
		Eigen::Ref<typename bs_jacobian_t::bs_variable_t::type> x,
		Eigen::Ref<typename bs_jacobian_t::bs_constraint_t::type> con,
		Eigen::Ref<bs_jacobian_t> J)
		noexcept
	{
		// typename Function::jacobian_return_t ret = Function::jac(param, bs_vector_type::template get<Variables>(x)...);
  //       bs_vector_type::template get<This>(con);

  }


	// /** Update the constraint.
	//  * 
	//  * Recompute the Function, and update the constraint with its jacobian
	//  */
	// template<typename Data, typename Scalar>
	// inline void update(Eigen::SparseMatrix<Scalar>& S, Data&& data)
	// {
	// 	auto ret = Function::jac(data.param, 
	// 		data.var.template segment<Variables::size>(
	// 		std::get<BS::get_index<Variables, Variables...>()>
	// 		(this->m_blocks).column)...);

	// 	int column = 0;
	// 	auto l = {(
	// 		this->template set<Variables, Scalar>(S, 
	// 			// ret.jacobian.block(0, column, Function::num_outputs, Variables::size)),
	// 			ret.jacobian.template block<Function::num_outputs, Variables::size>(
	// 				0, column)),
	// 		column += Variables::size,
	// 		0)...};
	// }
};

};
#endif 