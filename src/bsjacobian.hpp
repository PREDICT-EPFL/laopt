#ifndef __BSJacobian_HPP
#define __BSJacobian_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "bsmatrix.hpp"

namespace lampc
{

/**
 * A collection of BSMatrices representing the evaluation and jacobian of a function
 */
template<typename scalar_t>
struct BSJacobianTape
{
	BSMatrixTape<scalar_t> value;
	BSMatrixTape<scalar_t> jacobian;

	template<typename Arg1, typename Arg2>
	void operator=(std::pair<Arg1, Arg2> tup)
	{
		std::tie(value, jacobian) = tup;
	}

	template<typename Arg1>
	void operator=(Arg1 tup)
	{
		value = tup;
	}

	BSJacobianTape<scalar_t>& operator()(std::initializer_list<Segment> rows,
								 					 std::initializer_list<Segment> cols)
	{
		value(rows, 0);
		jacobian(rows, cols);
		return *this;
	}

	BSJacobianTape<scalar_t>& operator()(std::initializer_list<Segment> rows, int col)
	{
		value(rows, 0);
		jacobian(rows, col);
		return *this;
	}
	BSJacobianTape<scalar_t>& operator()(int row, std::initializer_list<Segment> cols)
	{
		value(row, 0);
		jacobian(row, cols);
		return *this;
	}

	BSJacobianTape<scalar_t>& operator()(int row, int col)
	{
		value(row, 0);
		jacobian(row, col);
		return *this;
	}

	/**
	 * Called when all copy operations have been completed once, which fixes the sparsity structure.
	 * 
	 * If rows and cols aren't specified, then they will be taken as large enough to contain all the
	 * blocks copied in.
	 */
	void finalize_structure(int rows=-1, int cols=-1)
	{
		value.finalize_structure(rows, 1);
		jacobian.finalize_structure(rows, cols);
	}

	// /**
	//  * Produce the information required to execute the recorded copies
	//  */
	// void generate_copy_data(std::vector<Segment>& segments, 
	// 												std::vector<CopyInfo>& copies, 
	// 												Eigen::SparseMatrix<scalar_t>& sparsity_structure)
	// {
	// 	int offset = 0;
	// 	for(int i=0; i<copy_lengths.size(); i++)
	// 	{
	// 		copies.push_back({.segment_index=offset, .num_segments_to_copy=copy_lengths[i]});
	// 		offset += copy_lengths[i];
	// 	}
	// 	segments = copy_sequence;
	// 	sparsity_structure = S.template cast<scalar_t>();
	// }

	// int rows() {return S.rows();}
	// int cols() {return S.cols();}

	// std::tuple<Eigen::SparseMatrix<int>, Eigen::SparseMatrix<int>, Eigen::SparseMatrix<int>> 
	// get_sparsity_structure()
	// {
	// 	Eigen::SparseMatrix<int> ret = S;
	// 	for(int i=0; i<ret.nonZeros(); i++) ret.valuePtr()[i] = 1;
	// 	return ret;
	// }

};

template<typename scalar_t>
class BSJacobian
{
	BSMatrix<scalar_t> value;
	BSMatrix<scalar_t> jacobian;

public:

	BSJacobian() {}; // Must call initialize_from_tape before using
	void initialize_from_tape(BSJacobianTape<scalar_t> tape)
	{
		value.initialize_from_tape(tape.value);
		jacobian.initialize_from_tape(tape.jacobian);
	}

	// Initialize from a tape instance
	BSJacobian(BSJacobianTape<scalar_t> tape) 
	{
		initialize_from_tape(tape);
	}

	/**
	 * Initialize J and H to the right sparsity structures and set them
	 * as the target. Use the dense vector value as the target.
	 */
	void initialize_jacobian(Eigen::SparseMatrix<scalar_t>& jac)
	{
		jacobian.initialize_matrix(jac);
	}

	void set_target_value(Eigen::SparseMatrix<scalar_t>& S)       {value.set_target(S);}
	void set_target_value(Eigen::Ref<Eigen::MatrixX<scalar_t>> S) {value.set_target(S);}

	void set_target_jacobian(Eigen::SparseMatrix<scalar_t>& S)       {jacobian.set_target(S);}
	void set_target_jacobian(Eigen::Ref<Eigen::MatrixX<scalar_t>> S) {jacobian.set_target(S);}

	/**
	 * Copy the given block into the target matrix
	 */
	template<typename Arg1, typename Arg2>
	void operator=(std::pair<Arg1, Arg2> tup)
	{
		std::tie(value, jacobian) = tup;
	}

	template<typename Arg1>
	void operator=(Arg1 tup)
	{
		value = tup;
	}

	// These all compile out. Not used in deployment.
	inline BSJacobian<scalar_t>& operator()(std::initializer_list<Segment> rows, std::initializer_list<Segment> cols)	{return *this;}
	inline BSJacobian<scalar_t>& operator()(std::initializer_list<Segment> rows, int col)	{return *this;}
	inline BSJacobian<scalar_t>& operator()(int row, std::initializer_list<Segment> cols)	{return *this;}
	inline BSJacobian<scalar_t>& operator()(int row, int col)	{return *this;}
	void finalize_structure(int rows=-1, int cols=-1)	{}	
};

};

#endif