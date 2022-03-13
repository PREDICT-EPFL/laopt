#ifndef _LA_OPTIMIZATION_COMPILER__HPP
#define _LA_OPTIMIZATION_COMPILER__HPP

#include "la_compiler.hpp"
#include <limits>
#include <regex>
#include <iterator>

#define FMT_HEADER_ONLY
#include "fmt/format.h"
#include "IndentStream.hpp"
#include "colormod.hpp"

struct constraint_t : call_t
{
	Eigen::VectorX<double> lb, ub;
	static int count; // Number of constraints created so far
	std::string name;

	// When displaying the name, use a minimum width
	int name_display_width = 0;
	int bnd_display_width = 0;
	int call_display_width = 0;

	constraint_t( const call_t &c ) : call_t(c),
	lb(c.num_outputs()), ub(c.num_outputs())
	{
		name = fmt::format("con_{}", count++); // Default name
		lb.array() = -std::numeric_limits<double>::infinity();
		ub.array() = std::numeric_limits<double>::infinity();
	}

	bool is_equality() const { return (lb - ub).norm() == 0; }

	// Human readable lb 
	std::string lb_str() const { return bnd_str(lb); }
	std::string ub_str() const { return bnd_str(ub); }

	int bnd_str_len() { return std::max({bnd_str(lb).length(), bnd_str(ub).length()}); }

protected:
	std::string bnd_str(const Eigen::Ref<const Eigen::VectorX<double>> &bnd) const
	{
		std::stringstream os; 
		os << "[" << bnd.transpose() << "]";

		// Replace "inf" with "∞"
		std::string str = std::regex_replace(os.str(), std::regex(R"(inf)"), "∞");

		return str;
	}
};
int constraint_t::count = 0; // Number of constraints created so far


/**
 * Add comparison operators to define constraints
 */
inline constraint_t operator<= (const call_t& call, const Eigen::VectorX<double> val)
{
	constraint_t constraint(call);
	constraint.ub.array() = constraint.ub.array().min(val.array());
	return constraint;
}
inline constraint_t operator>= (const call_t& call, const Eigen::VectorX<double> val)
{
	constraint_t constraint(call);
	constraint.lb.array() = constraint.lb.array().max(val.array());
	return constraint;
}

inline constraint_t operator<= (const constraint_t& call, const Eigen::VectorX<double> val)
{
	constraint_t constraint(call);
	constraint.ub.array() = constraint.ub.array().min(val.array());
	return constraint;
}
inline constraint_t operator>= (const constraint_t& call, const Eigen::VectorX<double> val)
{
	constraint_t constraint(call);
	constraint.lb.array() = constraint.lb.array().max(val.array());
	return constraint;
}


inline constraint_t operator<= (const Eigen::VectorX<double> val, const call_t& call)
	{ return call >= val; }
inline constraint_t operator>= (const Eigen::VectorX<double> val, const call_t& call)
	{ return call <= val; }
inline constraint_t operator<= (const Eigen::VectorX<double> val, const constraint_t& call)
	{ return call >= val; }
inline constraint_t operator>= (const Eigen::VectorX<double> val, const constraint_t& call)
	{ return call <= val; }


/**
 * Broadcast scalar comparisons to vector forms
 */
inline constraint_t operator>= (const call_t& call, const double val)
	{ return call >= Eigen::VectorX<double>::Constant(call.num_outputs(), val); }
inline constraint_t operator>= (const double val, const call_t& call)
	{ return Eigen::VectorX<double>::Constant(call.num_outputs(), val) >= call; }
inline constraint_t operator<= (const call_t& call, const double val)
	{ return call <= Eigen::VectorX<double>::Constant(call.num_outputs(), val); }
inline constraint_t operator<= (const double val, const call_t& call)
	{ return Eigen::VectorX<double>::Constant(call.num_outputs(), val) <= call; }

inline constraint_t operator>= (const constraint_t& call, const double val)
	{ return call >= Eigen::VectorX<double>::Constant(call.num_outputs(), val); }
inline constraint_t operator>= (const double val, const constraint_t& call)
	{ return Eigen::VectorX<double>::Constant(call.num_outputs(), val) >= call; }
inline constraint_t operator<= (const constraint_t& call, const double val)
	{ return call <= Eigen::VectorX<double>::Constant(call.num_outputs(), val); }
inline constraint_t operator<= (const double val, const constraint_t& call)
	{ return Eigen::VectorX<double>::Constant(call.num_outputs(), val) <= call; }

/** 
 * Equalities
 */
inline constraint_t operator== (const call_t& call, const double val)
	{ return val <= call <= val; }
inline constraint_t operator== (const double val, const call_t& call)
	{ return val <= call <= val; }
inline constraint_t operator== (const call_t& call, const Eigen::VectorX<double> val)
	{ return val <= call <= val; }
inline constraint_t operator== (const Eigen::VectorX<double> val, const call_t& call)
	{ return val <= call <= val; }


/**
 * An optimization problem
 */
class LAOptimizationProblem
{
public:
	LAOptimizationProblem(std::string name, 
		   				  std::string param_t="param_t",
		   				  std::string scalar_t="double") :
		compiler(name, param_t, scalar_t),
		variables(compiler.variables)
		{
			constraints = compiler.function("constraints");
		}

	variable_p variable(std::string name, int len) 
		{return compiler.variable(name,len);}
	std::vector<variable_p> variable(std::string name, int len, int number)
		{return compiler.variable(name,len,number);}

	callable_t& callable(callable_info info) 
		{return *(compiler.callable(info));}

	std::vector<variable_p> &variables;
	function_p constraints;	

	// Cast the list of calls to constraints
	std::vector<constraint_t*> get_constraints()
	{
		std::vector<constraint_t*> cons;
		for(auto& con: constraints->calls)
			cons.push_back(static_cast<constraint_t*>(con.get()));
		return cons;
	}

    friend LAOptimizationProblem& operator<<(LAOptimizationProblem& prob, const constraint_t& con);

    std::string generate()
    {
    	/** Generate the upper and lower bounds */
    	for(auto& con : constraints->calls)
    	{
			auto p = *static_cast<constraint_t*>(con.get());
			std::cout << p.name << std::endl;
    	}

    	return compiler.generate();
    }

	// Set the name of the last constraint on the stack
	void set_name_of_previous_constraint(std::string name)
	{
		auto p = static_cast<constraint_t*>(constraints->calls.back().get());
		p->name = name;
	}

	friend std::ostream &operator<<(std::ostream &os, LAOptimizationProblem &prob);

protected:
	LACompiler compiler;

	struct id_factory
	{
		LAOptimizationProblem *prob;
		id_factory(LAOptimizationProblem* prob) : prob(prob) {}

		call_t operator()(variable_p var)
		{
			auto callablex = prob->compiler.callable(callable_info(var));
			return callablex->operator()(var);
		}
	};

public:
	// Create an identity constraint factory
	id_factory id() { return id_factory(this); };

};

/**
 * Adding constraints to the problem
 */
LAOptimizationProblem& operator<<(LAOptimizationProblem& prob, const constraint_t& con)
{
	prob.constraints->add_call(std::make_shared<constraint_t>(con));
	return prob;
}

struct ConstraintNameBuffer{
    std::string name;
    ConstraintNameBuffer(std::string name) : name(name){}
};

LAOptimizationProblem& operator<<(LAOptimizationProblem& prob, ConstraintNameBuffer m)
{
	prob.set_name_of_previous_constraint(m.name);
	return prob;
}

inline std::string validate_name(std::string name)
{
	std::string str = std::regex_replace(name, std::regex(R"( )"), "_");
	return str;
}

inline ConstraintNameBuffer setname(std::string name)
{
	return ConstraintNameBuffer(validate_name(name));
}

inline ConstraintNameBuffer setname(std::string name, int ind)
{
	return ConstraintNameBuffer(fmt::format("{}_{}", validate_name(name), ind));
}


/**
 * Human readable display of the problem
 */

std::ostream &operator<<(std::ostream &os, constraint_t const &constraint)
{
	if(constraint.is_equality())
		os << fmt::format("{:>{}} | {:>{}}    {:{}} == {:<{}}", 
				constraint.name, constraint.name_display_width,
				"", constraint.bnd_display_width,
				constraint.str(), constraint.call_display_width,
				constraint.ub_str(), constraint.bnd_display_width);
	else
		os << fmt::format("{:>{}} | {:>{}} <= {:{}} <= {:<{}}", 
				constraint.name, constraint.name_display_width,
				constraint.lb_str(), constraint.bnd_display_width,
				constraint.str(), constraint.call_display_width,
				constraint.ub_str(), constraint.bnd_display_width);

	return os;
}

/**
 * Call the function func on each element of vec and choose the max based on the outpu.
 * Return the func(max_element)
 */
template<typename T, typename F>
int get_max(std::vector<T*> vec, F func)
{
	auto t = std::max_element(vec.begin(), vec.end(),
			[func](T* c1, T* c2) 
			{return func(c1) < func(c2);});
	return func(*t);
}


std::ostream &operator<<(std::ostream &os, LAOptimizationProblem &prob)
{
	IndentStream o;
	o << "=== === === === === === === === === === ===\n";

    Color::Modifier bold(Color::FG_RED);
    Color::Modifier def(Color::FG_DEFAULT);

	o << bold << "Optimization problem " << prob.compiler.className << def << std::endl;
	o << std::endl;

	o << bold << "Variables:\n" << def << IndentStream::indent;
	o << prob.compiler.variablesets << "\n\n";
	o << IndentStream::outdent;

	o << bold << "Constraints:\n" << def << IndentStream::indent;

	// Compute the max width of all names
	auto cons = prob.get_constraints();
	int bnd_len = get_max(cons, [](constraint_t* con){return con->bnd_str_len();});
	int name_len = get_max(cons, [](constraint_t* con){return con->name.length();});
	int call_len = get_max(cons, [](constraint_t* con){return con->str().length();});

	for(auto& con: cons)
	{
		con->name_display_width = name_len;
		con->bnd_display_width = bnd_len;
		con->call_display_width = call_len;
		o << *con << "\n";
	}
	o << IndentStream::outdent;

	o << "=== === === === === === === === === === ===\n";

	os << o.str();
	return os;
}


#endif