/**
 * Compile an LA Problem into data structures that can be incorporated into a solver.
 * 
 */

#ifndef __LACOMPILER_HPP
#define __LACOMPILER_HPP

#include "lampc.hpp"

// Forward declarations
struct variable_t;
struct variableset_t;
struct callable_t;
struct function_t;
struct weightedsum_t;
struct LACompiler;
struct callable_info;
struct call_t;

using variable_p = std::shared_ptr<variable_t>;
using variableset_p = std::shared_ptr<variableset_t>;
using callable_p = std::shared_ptr<callable_t>;
using call_p = std::shared_ptr<call_t>;
using function_p = std::shared_ptr<function_t>;
using weightedsum_p = std::shared_ptr<weightedsum_t>;

/**
 * A function is a list of callables evaluated at a set of arguments
 * 
 * f = [f1(x,y); f2(x,z); f3(z,x); ...]
 * 
 */
class function_t
{
public:
	function_t(std::string _name, LACompiler &compiler) : name(_name), compiler(compiler) {};

	// Add a call to the function
	void call(callable_p callable, std::vector<variable_p> args);

	int num_outputs();
	Eigen::SparseMatrix<int> jacobianStructure();
	Eigen::SparseMatrix<int> hessianStructure();

	friend std::ostream &operator<<(std::ostream &os, function_p const &f);
	friend LACompiler;

protected:
	// List of functions with the arguments
	std::vector<call_p> calls;
	std::string name;

	LACompiler &compiler;

	// Generate code to evaluate the function
	std::string generate();

	std::string generate_eval();
	std::string generate_jacobian();
};

/**
 * Represents a weighted sum of calls sum <w_i, f_i(x)>
 */
class weightedsum_t : public function_t
{
public:
	weightedsum_t(std::string name, LACompiler &compiler) : function_t(name, compiler) {};

	friend LACompiler;

protected:
	std::string generate();

	std::string generate_eval();
	std::string generate_gradient();
	std::string generate_hessian();
};

struct LACompiler
{
	// Name of the class, scalar type and parameter structure used in the generated code
	std::string className; 
	std::string param_t;
	std::string scalar_t;

	std::vector<variable_p> variables;
	std::vector<variableset_p> variablesets;
	std::vector<function_p> functions;
	std::vector<weightedsum_p> weightedsums;
	std::vector<callable_p> callables;

	LACompiler(std::string className, std::string param_t, std::string scalar_t) 
		:	className(className), param_t(param_t), scalar_t(scalar_t)
		{};

	// Generate code for the problem
	std::string generate(); 

	// Create variables
	variable_p variable(std::string name, int len); 
	std::vector<variable_p> variable(std::string name, int len, int number);

	// Create callable and function
	callable_p callable(callable_info info);
	function_p function(std::string name);
	weightedsum_p weighted_sum(std::string name);

	// Get information about the problem
	int num_variables();

private:
	variable_p variable_impl(std::string name, int len);
};

// Human-readable display of problem components
std::ostream &operator<<(std::ostream &os, variable_p const &var);
std::ostream &operator<<(std::ostream &os, std::vector<variable_p> const &vars);
std::ostream &operator<<(std::ostream &os, callable_p const &var);
std::ostream &operator<<(std::ostream &os, std::vector<callable_p> const &callables);


// Information about a runtime callable
struct callable_info
{
	std::string signature;
	std::string name;
	int num_args;
	int num_outputs;
	std::vector<int> input_sizes;

	Eigen::SparseMatrix<int> jacobianStructure;
	std::vector<Eigen::SparseMatrix<int>> hessianStructure;

	// Pull out the required info from the template type F for the callable
	template<typename F>
	callable_info(F)
	{
		signature = std::string(type_name<F>());
		name = std::string(F::name);
		num_args = int(F::num_input_vars);
		num_outputs = int(F::num_outputs);

		jacobianStructure = F::jacobianStructure();

		for(int i=0; i<F::num_outputs; i++)
			hessianStructure.push_back(F::hessianStructure(i));

		input_sizes = F::get_input_sizes();
	}
};

#endif