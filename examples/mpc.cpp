#include <iostream>
#include <typeinfo>
#include <vector>
#include <tuple>
#include <functional>
#include <string>
#include <array>
#include <chrono>
// #include <array_traits>
#include <utility>
using namespace std;

// Eigen includes
#include <Eigen/Core>
#include <Eigen/StdVector>
using namespace Eigen;

// autodiff include
#include <autodiff/forward.hpp>
#include <autodiff/forward/eigen.hpp>
using namespace autodiff;

// boost hana
#include <boost/hana.hpp>
namespace hana = boost::hana;
using namespace hana::literals;

// Good ref: https://www.fluentcpp.com/2019/03/08/stl-algorithms-on-tuples/

namespace util
{
    /// Compile time foreach for tuples
    template <typename Tuple, typename Callable>
    void forEach(Tuple &&tuple, Callable &&callable)
    {
        std::apply(
            [&callable](auto &&... args) { (callable(std::forward<decltype(args)>(args)), ...); },
            std::forward<Tuple>(tuple));
    }
}; // namespace util

template <size_t _size>
struct Variable {
    typedef Matrix<dual, _size, 1> Vector;
    enum { NeedsToAlign = (sizeof(Vector)%16)==0 };
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(NeedsToAlign)

    static const size_t size = _size;

    const size_t offset;
    // const std::string name;
    const Vector x;
    // Variable(const std::string name)
    //     : offset(-1), name(name)
    // {
    //     x.Zero();
    // };

    // Variable(const std::string name, const int num) 
    //     : name(name + to_string(num)), offset(-1)
    // {
    //     x.Zero();
    // };

    std::string to_string()
    {
        std::stringstream ss;
        // ss << name << " = [" << x.transpose() << "]" << " offset = " << offset;
        ss << "[" << x.transpose() << "]" << " offset = " << offset;
        return ss.str();
    }

    friend std::ostream& operator<<( std::ostream& o, const Variable& var ) 
    {
        o << var.to_string();
        return o;
    }

    Vector& get_x()   {return x;}

    void set_x(Ref<VectorXd> val) {
        x << val.segment(offset, size); // Copy and convert to dual variable
    };
};

// Create a std::array, calling a non-default constructor on each one, passing an index number
// http://www.trivialorwrong.com/2015/12/08/meta-sequences-in-c++.html
template<typename Ctor, size_t... S>
std::array<std::result_of_t<Ctor(size_t)>, sizeof...(S)> MakeArray(Ctor&& ctor, std::index_sequence<S...>)
{
   return std::array<std::result_of_t<Ctor(size_t)>, sizeof...(S)> {ctor(S)...};
}

template<size_t N, typename Ctor>
std::array<std::result_of_t<Ctor(size_t)>, N> MakeArray(Ctor&& ctor)
{
   return MakeArray(std::forward<Ctor>(ctor), std::make_index_sequence<N>());
}

// Create a static array of Variables
template<size_t n, size_t N> 
auto VariableArray()//std::string name)
{ 
    // return MakeArray<N>([name](size_t index){return Variable<n>(name + to_string(index));});
    return MakeArray<N>([](size_t index) { return Variable<n>(); });//name + to_string(index)); });
}

// template <size_t n, size_t N>
// struct VariableArray : public std::array<Variable<n>, N>
// {
//     static constexpr size_t size = n * N;

//     return MakeArray<N>([name](size_t index) 
//     { return Variable<n>(name + to_string(index)); });

//     // template <typename... As,
//     //           typename std::enable_if<sizeof...(As) == N>::type * = nullptr>
//     // constexpr foo(As &&... as)
//     //     : std::array<Variable<n>, N>{{std::forward<As>(as)...}}
//     // {
//     // }
// };

// template <size_t n, size_t N>
// struct foo : public std::array<Variable<n>, N>
// {
//     static constexpr size_t size = n * N;
//     template <typename... As,
//               typename std::enable_if<sizeof...(As) == N>::type * = nullptr>
//     constexpr foo(As &&... as)
//         : std::array<Variable<n>, N>{{std::forward<As>(as)...}}
//     {
//     }
// };




namespace DoubleIntegrator {
    const size_t nx = 2;
    const size_t nu = 1;

    template<typename T>
    using StateType = Matrix<T, nx, 1>;

    template<typename T>
    using InputType = Matrix<T, nu, 1>;

    template<typename T>
    auto dynamics(const StateType<T>& xp, 
                  const StateType<T>& x, 
                  const InputType<T>& u)
    {
        StateType<T> eq;
        eq << xp[0] - (-sin(x[1]) + x[1]*x[0]), xp[1] - cos(x[0])*u[0];
        return eq;
    }
};


template<size_t size_f,      // Size of the return type of Function
         size_t size_x,      // Sum of the sizes of the Args
         typename Function,  // Function that takes a set of dual eigen vectors, and returns a dual eigen vector
         typename Args,      // Tuple of references to the dual vectors in the Variables
         typename Vars       // Tuple of reference to the Variables
         >
struct Constraint_Impl
{
    Function &f;
    Args args;
    Vars vars;

    using Jacobian_t = Matrix<double, size_f, size_x>;
    using Value_t = Matrix<double, size_f, 1>;

    Jacobian_t J; // Jacobian
    Value_t val;    // Function value

    std::string name;

    Constraint_Impl(std::string name, Function &f, Args args, Vars vars)
        : f(f), args(args), vars(vars), name(name)
          {};

    // Compute local version of jacobian and function value
    void update()
    {
        Matrix<dual, size_f, 1> result;
        J = jacobian(f, args, args, result);
        for (size_t i=0; i<size_f; i++)
            val[i] = result[i].val;
    }

    // Copies out the blocks df/dxi in J[:, xi.offset + [0:xi.size]]
    void fill_jacobian(Ref<Matrix<double, size_f, Dynamic>> bigJ)
    {
        assert(J.rows() == bigJ.rows());
        size_t col = 0;
        util::forEach(vars, [&](auto var){
            bigJ.block(0, var.offset, bigJ.rows(), var.size) 
                = J.block(0, col, J.rows(), var.size);
            col += var.size;
        });

        // {(std::cout << vars)...;};
        // (void)l;
        // forEach()
        // // assert(col_index + J.cols() <= size_x);
        // cout << "col_index = " << col_index << " J.cols() = " << J.cols() << endl;
        // J = this->J.block(0, col_index, size_f, col_index + J.cols());
    }

    // Return the constraint function
    auto eval()
    {
        return std::apply(f, args);
    }

    // Return the jacobian of the constraint function wrt local vars
    auto eval_jacobian()
    {
        return jacobian(f, args, args);
    }

    std::string to_string()
    {
        std::stringstream ss;

        ss << name << "(";
        string sep = "";
        util::forEach(vars, [&](auto var) {
            ss << sep << var.name;
            sep = ", ";
        });
        ss << ")";

        return ss.str();
    }

    friend std::ostream& operator<<( std::ostream& o, Constraint_Impl& con ) 
    {
        o << con.to_string() << endl;

        // auto indent = std::string(4, ' ');
        // o << indent << "Value = [" << con.val.transpose() << "]\n";
        // o << indent << "Jacobian = \n";

        // IOFormat OctaveFmt(StreamPrecision, 0, ", ", ";\n", indent, "", "[", "]");        
        // cout << con.J.format(OctaveFmt) << endl;

        return o;
    }
};

// Helper function to create a constraint
template<typename Function, typename... ArgTypes>
auto Constraint(std::string name, Function& f, ArgTypes&... args)
{
    // Compute the size of the returned vector, and the number of vars
    using result_type = decltype(std::declval<Function>()(std::declval<Matrix<double,ArgTypes::size,1>>()...));
    static constexpr size_t size_f = result_type::RowsAtCompileTime;
    static constexpr size_t size_x = (ArgTypes::size + ...);

    // cout << "size_f = " << size_f << endl;
    // cout << "size_x = " << size_x << endl;

    // Build a tuple of the dual vectors inside the variables passed in
    auto arg_tuple = std::forward_as_tuple((args.x) ...);
    auto var_tuple = std::forward_as_tuple(args ...);

    return Constraint_Impl<size_f, size_x, Function, decltype(arg_tuple), decltype(var_tuple)>
        (name, f, arg_tuple, var_tuple);
}

// Create a static array of Constraints
template<size_t N, typename Function>
auto ConstraintArray(Function constraint_generator)
{ 
    return MakeArray<N>([&](size_t index){return constraint_generator(index);});
}

// TODO: Compute the sizes of everything at compiletime...
template <typename Scalar, size_t num_vars, size_t num_ineq, size_t num_eq>
struct NLP
{

    Matrix<Scalar, num_ineq, num_vars> Ji; // Jacobian of inequality constraints
    Matrix<Scalar, num_ineq, 1> gi;        // Current value of the inequality constraints

    Matrix<Scalar, num_ineq, num_vars> Je; // Jacobian of equality constraints
    Matrix<Scalar, num_ineq, 1> ge;        // Current value of the equality constraints

    Matrix<Scalar, num_vars, 1> x; // Primal optimization variable

    // Current number of registered constraints (rows of jacobian, not constraint functions)
    size_t num_equality_constraints = 0;
    size_t num_inequality_constraints = 0;

    constexpr NLP()
    {
        // We need to zero these here, since only the
        // non-zero elements are updated during the optimization
        Ji.setZero();
        gi.setZero();
        Je.setZero();
        ge.setZero();

        x.setZero();
    }

    // TODO (1): Convert these vectors to fixed-sized arrays of std::function
    // TODO (2): Convert these to tuples of different lambda functions and generate
    //           the call sequences at compile-time
    // Notes on the speed penalty of std::function
    //  https://stackoverflow.com/questions/14306497/performance-of-stdfunction-compared-to-raw-function-pointer-and-void-this

    // Functions to compute the jacobians and functions
    using ConstraintFunction = std::function<void(void)>;
    std::vector<ConstraintFunction> eval_ineq_jacobians;
    std::vector<ConstraintFunction> eval_eq_jacobians;

    // Functions to print out data about all the variables and constraints
    using PrintFunction = std::function<std::string(void)>;
    std::vector<PrintFunction> print_vars;
    std::vector<PrintFunction> print_equalities;
    std::vector<PrintFunction> print_inequalities;

    // Function to update all of the dual variables to the current value of the primal
    using VarSetFunction = std::function<void(void)>;
    std::vector<VarSetFunction> varset_funcs;

    // Copy the current value of the primal to the dual values of all the vars
    // Evaluate all functions and their jacobians
    void update()
    {
        for (auto &func : varset_funcs) 
            func();
        for (auto &func : eval_eq_jacobians)
            func();
    }

    int offset = 0; // Used to compute offsets of variables into the primal vector
    template <size_t n>
    void addVariable(Variable<n> &var)
    {
        var.offset = offset;
        offset += var.size;

        print_vars.push_back([&var]() { return var.to_string(); });
        varset_funcs.push_back([&var, this]() { var.set_x(this->x); });
    }

    template <size_t n, size_t N>
    void addVariable(std::array<Variable<n>, N> &vars)
    {
        for (auto &var : vars) addVariable(var);
    }

    template <size_t size_f, size_t size_x, typename F, typename A, typename V>
    void addEquality(Constraint_Impl<size_f, size_x, F, A, V> &con)
    {
        print_equalities.push_back([&con]() { return con.to_string(); });

        size_t start_row = num_equality_constraints;
        eval_eq_jacobians.push_back([&con, this, start_row]() {
            // Compute jacobians and values
            con.update();

            // Copy then here
            con.fill_jacobian(
                this->Je.block(start_row, 0, size_f, this->Je.cols())
            );
        });
        num_equality_constraints += size_f;
    }

    template <typename C, size_t N>
    void addEquality(std::array<C, N> &constraints)
    {
        for (auto &con : constraints)
            addEquality(con);
    }

    friend ostream &operator<<(ostream &os, const NLP &nlp)
    {
        os << "-------------------------------\nNLP" << endl;

        os << "Primal var = " << nlp.x.transpose() << endl;

        os << "Variables" << endl;
        for (auto &f : nlp.print_vars)
            os << std::string(4, ' ') << f() << endl;

        if (nlp.print_equalities.size() > 0)
        {
            os << "Equality constraints" << endl;
            for (auto &f : nlp.print_equalities)
                os << std::string(4, ' ') << f() << endl;
        }

        return os;
    }
};

// Returns the total number of variables in the array
template <size_t N>
constexpr size_t count_vars(std::array<int, N> &X)
{
    return N;
}

template <size_t n, size_t N>
constexpr size_t count_vars(std::array<Variable<n>, N> &X)
{
    return n * N;
}

template <size_t n>
constexpr size_t count_vars(Variable<n> &x)
{
    return n;
}

template <size_t n>
class Test
{
    Matrix<double, 3, 4> M;
};

template <size_t n, size_t N>
constexpr size_t count_vars(std::array<Test<n>, N> &x)
{
    return n * N;
}

template <size_t n>
constexpr size_t count_vars(Test<n> &x)
{
    return n;
}

template <typename... ArgTypes>
constexpr size_t count_vars(ArgTypes... args)
{
    // using result_type = decltype(std::declval<Function>()(std::declval<Matrix<double, ArgTypes::size, 1>>()...));
    return (count_vars(args) + ...);
    // return 5;
}


template<size_t num_vars>
struct NLP2
{

    NLP2()
    {

    };
};



// Xvar = make_tuple(2, N)
// Uvar = make_tuple(1, N)
// x0 = 4

// Vars = make_tuple(make_tuple(Xvar), make_tuple(Uvar), make_tuple(4, 1))
// constexpr num_vars = count_vars(Vars)

// tie(X, U, x0) = make_vars(Vars) // Not constexpr

const size_t N = 5; // Horizon

// constexpr std::array<int, 5> Y = {1,2,3,4,5};

// auto X = VariableArray<2, N>();
// auto U = VariableArray<1, N - 1>();
// auto x0 = Variable<5>();

// constexpr std::array<Test<2>, 5> Z;

// using namespace DoubleIntegrator;
// auto q = Constraint("test1", dynamics<dual>, X[1], X[0], U[0]);

// auto dynamic_constraints = ConstraintArray<N - 1>(
//     [](int i) {
//         return Constraint(string("dynamics_") + to_string(i),
//                           dynamics<dual>, X[i + 1], X[i], U[i]);
//     });


// auto Vars = make_tuple(X, U, x0);

// constexpr size_t num_vars = N * 2 + (N - 1) * 1;
// const size_t num_ineq = 3 * N;
// const size_t num_eq = 2 * N;
// NLP<double, num_vars, num_ineq, num_eq> nlp;

// constexpr size_t num_vars = count_vars(Z[0]);

// constexpr size_t num_vars = count_vars(x0, X); //count_vars(X, x0, U);

// NLP2<num_vars> nlp2();

int main()
{

    // cout << "num_vars = " << num_vars << endl;

    // // auto dynamic_constraints = ConstraintArray<N - 1>(
    // //     [&X, &U](int i) {
    // //         return Constraint(string("dynamics_") + to_string(i),
    // //                           dynamics<dual>, X[i + 1], X[i], U[i]);
    // //     });
    // nlp.addVariable(X);
    // nlp.addVariable(U);

    // // TODO: Pull these sizes out at compile-time from the var-list, and more to a helper
    // // constexpr size_t num_vars = NUM_VARS(X, U, x0);

    // // Controls the order of the variables

    // nlp.addEquality(dynamic_constraints);

    // cout << nlp << endl;

    // nlp.x << 1,2,3,4,5,6,7,8,9,10,11,12,13,14;

    // nlp.update();

    // cout << "nlp.J = " << endl << nlp.Je << endl;

    return 0;
}

// q.update();
// cout << "Jacobian = " << endl << q.J << endl;
// cout << "val = " << endl << q.val << endl;

// X[0].x << 1, 2;
// U[0].x << 3;
// q.update();
// cout << "Jacobian = " << endl << q.J << endl;
// cout << "val = " << endl << q.val << endl;
