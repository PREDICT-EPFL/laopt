/** Static memory implementation of a variable block-matrix type.
 */

#ifndef __BSMATRIXCORE_HPP
#define __BSMATRIXCORE_HPP

#include <iostream>
#include <iomanip>
#include <tuple>
#include <array>
#include <numeric>

#include "Eigen/Dense"
#include "Eigen/Sparse"

#include "lampc_utility.hpp"


/** TODO
 * 
 * Sparse Blocks
 * Diagonal Blocks
 * BSMatrix Blocks
 * 
 * Dense matrices
 * 
 * var sets and constraint sets
 */



typedef std::chrono::time_point<std::chrono::system_clock> time_point;
static time_point get_time()
{
    /** OS dependent */
#ifdef __APPLE__
    return std::chrono::system_clock::now();
#elif defined _WIN32 || defined _WIN64 || defined _MSC_VER
    return std::chrono::system_clock::now();
#else
    return std::chrono::high_resolution_clock::now();
#endif
}


namespace BS
{
	namespace detail
	{
	    template<typename T, typename F, int... Is>
	    void
	    apply(T&& t, F f, std::integer_sequence<int, Is...>)
	    {
	        auto l = { (f(std::get<Is>(t)), 0)... };
	    }

		template<typename Tuple, class T, class BinaryOperation, int... Is>
		constexpr
		T accumulate(Tuple& tup, T& init, 
					 BinaryOperation op, 
					 std::integer_sequence<int, Is...>)
		{
	        auto l = { (op(std::get<Is>(tup), init), 0)... };
	        return init;
		}

		/** Return the sum of the integers
		 */
		template<int... i>
		constexpr int sum_int_template()
		{
			int acc = 0;
	        auto l = { (acc += i, 0)... };
	        return acc;
		}

		/** Concantenate two parameter packs
		 */
	    template<class...> struct pack {};

		template<class T = pack<>, class...> 
		struct concat { using type = T; };

		template<class... T1, class... T2, class... Ts>
		struct concat<pack<T1...>, pack<T2...>, Ts...> 
		    : concat<pack<T1..., T2...>, Ts...> {};

		template<class... Ts> 
		using concat_t = typename concat<Ts...>::type;

	} // namespace detail

	/** Call function on each element of the tuple
	 */
	template<typename... Ts, typename F>
	void apply(std::tuple<Ts...>& t, F f)
	{
	    detail::apply(t, f, std::make_integer_sequence<int, sizeof...(Ts)>());
	}

	/** Tuple version of std::accumulate
	 * 
	 * op must be templated and be type-deducable to have the signature
	 * 
	 * T& = op(Ts&, T&)
	 * 
	 */
	template<typename... Ts, class T, class BinaryOperation>
	constexpr
	T accumulate(std::tuple<Ts...>& tup, T init, BinaryOperation op)
	{
		T init_ = init;
		return detail::accumulate(tup, init_, op, std::make_integer_sequence<int, sizeof...(Ts)>());
	}

	/** Return true if What is in Args
	 */
	template<typename What, typename ... Args>
	constexpr bool is_member()
	{
		bool ret = false;
		auto l = {(ret = ret || std::is_same<What, Args>::value, 0)...};
		return ret;
	} 

	/** Return the index of What in Args, or -1
	 */
	template<typename What, typename ... Args>
	constexpr int get_index()
	{
		int ind = -1;
		int i = 0;
		auto l = {(
			ind = std::is_same<What, Args>::value ? i : ind,
			i++,
			0)...};
		return ind;
	}


	/** Interleave two parameter packs
	 */
	template<class... Us>
	struct interleave {
	     template<class... Vs>
	     using with = detail::concat_t<detail::pack<Us, Vs>...>;
	};	




template<int size_>
struct Variable
{
	static constexpr int size = size_;

	// using type = Eigen::Vector<Scalar, size>;
	// using ref = Eigen::Ref<Eigen::Vector<Scalar, size>>;
};


/** A block dense vector indexible by Variables
 */
template<typename Scalar, typename VariableTuple>
struct BSVector;

template<typename Scalar, typename... Variables>
struct BSVector<Scalar, std::tuple<Variables...>> 
	: Eigen::Vector<Scalar, detail::sum_int_template<Variables::size...>()>
{
	using type = Eigen::Vector<Scalar, detail::sum_int_template<Variables::size...>()>;

private:

	/** Return the offset to variable var
	 */
	template<typename var>
	static constexpr inline int offset()
	{
		int acc = 0;
		bool found = false;
        auto l = { (
        	found = found || std::is_same<var, Variables>::value,
        	acc += found ? 0 : Variables::size,
        	0)... };
        return acc;
	}

public:
	/** Return the segment for variable var
	 */
	template<typename var>
	inline Eigen::Ref<Eigen::Vector<Scalar, var::size>> get()
	{
		return this->template segment<var::size>(offset<var>());
	};

	/** Return the segment for variable var from the given vector
	 */
	template<typename var>
	static inline Eigen::Ref<Eigen::Vector<Scalar, var::size>>
		get(Eigen::Ref<Eigen::Vector<Scalar, type::RowsAtCompileTime>> in)
	{
		return in.template segment<var::size>(offset<var>());
	};

};


template<typename var_>
struct Block
{
	// Top left row and column of this block
	// Set in finalize()
	int row, column; 

	using var = var_;
	static constexpr int size = var_::size;

	// Offsets into valuePtr where each column starts
	std::array<int, size> offsets;

	// Has to be set by derived block before finalize is called,
	// or this block will be assumed to be dense
	std::array<int, size> column_length;

	int nonZeros()
	{
		return std::accumulate(column_length.begin(), column_length.end(), 0);
	}

	/** Set the innerIndexPtr for this column assuming that
	 * the non-zeros are contiguous and at the top of the block.
	 * 
	 * Returns the number of nonzeros in the column.
	 */
	template<typename Index>
	int setColumnSparsity_Dense(int column, Index* innerIndexPtr)
	{
		for(int i=0; i<column_length[column]; i++)
			innerIndexPtr[i] = row + i;
		return column_length[column];
	}

	void setDense(int rows)
	{
		for(int i=0; i<size; i++)
			column_length[i] = rows;
	}

	/** Return the number of nonzeros of this block 
	 * in sub-column i
	 */
	inline int getColumnNonZeros(int i)
	{
		assert(i < size && "Requested column larger than size");
		return column_length[i];
	}

	/** Sets the offsets vector. Only called from BSMatrix
	 */
	template<typename Scalar>
	void setOffsets(Eigen::SparseMatrix<Scalar>& S)
	{
		for(int i=0; i<var::size; i++)
			offsets[i] = S.coeff(row, column + i);
	}


	/** Set the block from the dense matrix B
	 * 
	 * We're assuming here that this block is dense
	 */
	template<typename Scalar>
	inline void set(Eigen::SparseMatrix<Scalar>& S, 
		const Eigen::Ref<const Eigen::MatrixX<Scalar>> B)
	{
		for(int i=0; i<var::size; i++)
		{
			Eigen::Map<Eigen::VectorX<Scalar>> target(S.valuePtr() + offsets[i], column_length[i]);
			target = B.col(i);
		}
	}
};


template<int size_, typename... Blocks>
struct Constraint
{
	static const int size = size_;

	// Top row of this constraint.
	// Set in finalize
	int row;

	using BlockTuple = std::tuple<Blocks...>;
	BlockTuple m_blocks;

	Constraint()
	{
		// Default all blocks to dense
		BS::apply(m_blocks, [](auto& e){e.setDense(size);});
	}

	int nonZeros()
	{
		return BS::accumulate(m_blocks, 0, [](auto& e, int& acc){acc += e.nonZeros(); return acc;});
	}

	/** Apply function to block var, 
	 * or no-op if var is not in constraint
	 */
	template<typename var, typename F, typename T,
			int var_ind = BS::get_index<var, typename Blocks::var...>(),
			std::enable_if_t<var_ind != -1, bool> = true>
	inline auto call(F f, T return_default)
	{
		return f(std::get< var_ind >(m_blocks));
	}

	template<typename var, typename F, typename T,
			int var_ind = BS::get_index<var, typename Blocks::var...>(),
			std::enable_if_t<var_ind == -1, bool> = true>
	inline T call(F f, T return_default)
	{
		return return_default;
	} // Should compile away entirely

	template<typename var, typename Scalar,
			 int var_ind = BS::get_index<var, typename Blocks::var...>()>
	inline void set(Eigen::SparseMatrix<Scalar>& S, 
		const Eigen::Ref<const Eigen::MatrixX<Scalar>> B)
	{
		std::get<var_ind>(m_blocks).set(S, B);
	}

	// * Update the constraint.
	//  * 
	//  * This is just a holder - it's meant to be overwritten in derived 
	//  * constraint classes.
	//  * 
	//  * Takes some user data, which is used to update the constraint.
	 
	// template<typename Data, typename Scalar>
	// inline void update(Eigen::SparseMatrix<Scalar>& S, Data&& data)
	// {}
};

template<typename Scalar, typename ConstraintTuple, typename VariableTuple>
struct BSMatrix;

template<typename Scalar, typename... Constraints, typename... Variables>
struct BSMatrix<Scalar, std::tuple<Constraints...>, std::tuple<Variables...>> 
{
	using bs_variable_t = BSVector<Scalar, std::tuple<Variables...>>;
	using bs_constraint_t = BSVector<Scalar, std::tuple<Constraints...>>;

	using type = Eigen::SparseMatrix<Scalar>; // Raw type

	using ConstraintTuple = std::tuple<Constraints...>;
	ConstraintTuple m_constraints;

	static constexpr int RowsAtCompileTime = sum_template<Constraints::size...>();
	static constexpr int ColsAtCompileTime = sum_template<Variables::size...>();

	BSMatrix()
	{}

	template<typename con, typename var>
	inline void set(const Eigen::Ref<const Eigen::MatrixX<Scalar>> B)
	{
		// Eigen::SparseMatrix<Scalar> S;
		std::get<con>(m_constraints).template set<var>(*this, B);
	}

private:
	/** Sets the sparsity pattern for the i'th column of the
	 * variable var, and returns the number of non-zeros in
	 * this column
	 */
	template<typename var>
	int setColumnSparsity(Eigen::SparseMatrix<Scalar>& S, int column, int& offset)
	{
		int nnz_block = 0;
		int nnz = 0;
		auto innerIndexPtr = S.innerIndexPtr();
        auto l = { (
        	nnz_block =
        		std::get<Constraints>(m_constraints).template call<var>(
	        		[&](auto& blk){
						return blk.setColumnSparsity_Dense(column, innerIndexPtr + offset);
	        		}, 0),
        	offset += nnz_block,
        	nnz += nnz_block,
        	0)... 
    	};

        return nnz;
	}

public:
	/** Apply function to all blocks
	 * 
	 * Calling format
	 * f(auto& block)
	 */
	template<typename F>
	void applyBlockwise(F f)
	{
		BS::apply(m_constraints, 
			[&f](auto& con)
			{BS::apply(con.m_blocks, f);});
	}

	/** Apply function to each constraint
	 * 
	 * Calling format
	 * f(auto& constraint, Eigen::SparseMatrx<Scalar>& S)
	 * where S refers to this matrix
	 */
	template<typename F>
	void applyConstraintWise(F f)
	{
		BS::apply(m_constraints, f);
	}

	// /** Call update on each constraint
	//  */
	// template<typename Data>
	// inline void update(Data&& data)
	// {
	// 	BS::apply(m_constraints, [&](auto& con){con.update(*this, data);});
	// }


	void initialize(Eigen::SparseMatrix<Scalar>& S)
	{
		S.resize(RowsAtCompileTime, ColsAtCompileTime);

		// Compute non-zeros of all blocks
		S.reserve(BS::accumulate(m_constraints, 0, 
			[](auto& x, int& acc){acc += x.nonZeros(); return acc;}));

		// Set the location of each block and constraints
		// Iterate over the constraints
		BS::accumulate(m_constraints, 0,
			[](auto& constraint, int& row){
			constraint.row = row;

			// Iterate over the variables
			int column = 0;
			auto l = {(
				// Set the row and column of the blocks
				constraint.template call<Variables>([row, column](auto& blk)
					{
						blk.row = row;
						blk.column = column;
						return 0;
					}, 0),
				column += Variables::size,
				0)...};

			row += constraint.size;
			return row;
		});

		// Compute outerIndexPtr by computing the number of non-zeros in each column
		//
		// We define a new lambda for each variable
		// The lambda iterates through the columns of that varible
		// and computes the number of non-zeros.
		int column = 0;
		int offset = 0; // innerIndexPtr offset
        auto l = { (
        	column += [&column, &offset, &S, this](){
				int nnz = 0;
				for(int i=0; i<Variables::size; i++)
				{
					nnz = setColumnSparsity<Variables>(S, i, offset);
					S.outerIndexPtr()[column + i + 1] = 
						S.outerIndexPtr()[column + i] + nnz;
				}
				return Variables::size;
        	}(), // Create and call lambda
	    	0)...};

        ////
        // Set the offset vectors in each block

        // We first record the offsets into the valuePtr
		for(int i=0; i<S.nonZeros(); i++)
			S.valuePtr()[i] = i;

		// Now grab the location of each block
		this->applyBlockwise([&](auto& blk){blk.setOffsets(S);});
	}
};

};

#endif // __BSMATRIXCORE_HPP