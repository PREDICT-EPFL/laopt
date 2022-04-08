#ifndef __PROBLEM_HPP
#define __PROBLEM_HPP

#include <numeric>

namespace lampc
{

template<typename scalar_t>
struct Variable;

template<typename scalar_t>
struct Problem
{
	Variable<scalar_t>& variable(int n, int m=1)
	{
		auto v = std::make_shared<Variable<scalar_t>>(n,m,size);
		variables.push_back(v);
		size = compute_size(); // Update the total size of the problem variable

		// Return with move symantics is required, 
		// since we need the reference stored in variables to remain valid
		return *(v.get());
	}

	int size = 0;

	// /**
	//  * Called once the problem is defined. Defines the optimization variable.
	//  */
	// void register_variables(std::initializer_list<std::reference_wrapper<Variable<scalar_t>>> vars)
	// {
	// 	variables = vars;

	// 	int offset = 0;
	// 	for(auto v: variables)
	// 	{
	// 		v.get().initialize(offset);
	// 		offset += v.get().num_elements();
	// 	}

	// 	size = compute_size(); // Once all the variables are registered, the size doesn't change
	// }

	void set_variable(Eigen::Ref<Eigen::VectorX<scalar_t>> var)
	{
		for(auto v: variables)
			v->set_var(var);
	}

private:
	// std::vector<std::reference_wrapper<Variable<scalar_t>>> variables;
	std::vector<std::shared_ptr<Variable<scalar_t>>> variables;

	int compute_size() // Total number of optimization variables
	{
		return std::accumulate(variables.begin(), variables.end(), 0, 
							[](int total, std::shared_ptr<Variable<scalar_t>> var) {return total + var->num_elements();});
	}
};


/**
 * Represents a variable, which is an offset into
 * a global decision variable.
 */
template<typename scalar_t>
struct Variable
{
	Variable(int n, int m, int offset) 
		: n(n), m(m), offset(offset), 
			src(NULL,n,m) // Variable is invalid until set_var is called
		{}

	inline int num_elements() { return n*m; } // Total number of elements in the variable (n*m)
	inline int size() { return n; } // Size of a single variable
	inline std::pair<int,int> shape() { return make_pair(n,m); } // Shape of the variable matrix

	/** 
	 * Return the matrix
	 */
	inline Eigen::Map<Eigen::MatrixX<scalar_t>> operator()()
	{
		return src;
	}

	/** 
	 * Return a reference to the index'th variable in the matrix
	 */
	inline Eigen::Ref<Eigen::VectorX<scalar_t>> operator()(int index)
	{
		return src(Eigen::all,index);
	}

	/**
	 * Return a pair of {offset, length}
	 */
	inline Segment operator[](int index)
	{
		return Segment{.index = offset + index * n, .length = n}	;
	}


friend Problem<scalar_t>;
protected:
	inline void set_var(Eigen::Ref<Eigen::VectorX<scalar_t>> var)
	{
	  new (&src) Eigen::Map<Eigen::MatrixX<scalar_t>>(var.data() + offset, n, m);
	}

private:

	// Offset into the global variable
	int offset;
	int n,m;

	// Reference to start of this variable in the global variable
	Eigen::Map<Eigen::MatrixX<scalar_t>> src;
};

};

#endif // __PROBLEM_HPP