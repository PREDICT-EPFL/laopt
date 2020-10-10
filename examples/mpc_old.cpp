// C++ includes
#include <iostream>
#include <typeinfo>  //for 'typeid' to work  
#include <vector>
#include <tuple>
#include <functional>
#include <string>
#include <array>
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

using dual2nd = HigherOrderDual<2>;

namespace util {
/// Compile time foreach for tuples
template <typename Tuple, typename Callable>
void forEach(Tuple&& tuple, Callable&& callable)
{
    std::apply(
        [&callable](auto&&... args) { (callable(std::forward<decltype(args)>(args)), ...); },
        std::forward<Tuple>(tuple)
    );
}
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


template <size_t _size>
struct Variable {
    typedef Matrix<dual, _size, 1> Vector;
    enum { NeedsToAlign = (sizeof(Vector)%16)==0 };
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(NeedsToAlign)

    static const size_t size = _size;

    size_t offset;
    const std::string name;
    const Vector x;
    Variable(const std::string name) 
        : name(name), offset(-1)
    {
        x.Zero();
    };

    Variable(const std::string name, const int num) 
        : name(name + to_string(num)), offset(-1)
    {
        x.Zero();
    };

    void print()
    {
        cout << name << " = [" << x.transpose() << "]" << " offset = " << offset;
    }

    friend std::ostream& operator<<( std::ostream& o, const Variable& var ) 
    {
        o << var.name << " = [" << var.x.transpose() << "]" << " offset = " << var.offset;
        return o;
    }

    Vector& get_x()
    {return x;}

    void set_x(VectorXd& val) {
        x << val.segment(offset, size); // Copy and convert to dual variable
    };
};

// Create a static array of Variables
template<size_t n, size_t N> 
auto VariableArray(std::string name)
{ 
    return MakeArray<N>([](size_t index){return Variable<n>("x" + to_string(index));});
}

// TODO: Compute the sizes of everything at compiletime...
template<typename SCALAR, size_t num_vars, size_t num_ineq, size_t num_eq>
struct NLP {

    Matrix<SCALAR, num_ineq, num_vars> Ji; // Jacobian of inequality constraints
    Matrix<SCALAR, num_ineq, 1> gi;        // Current value of the inequality constraints

    Matrix<SCALAR, num_ineq, num_vars> Je; // Jacobian of equality constraints
    Matrix<SCALAR, num_ineq, 1> ge;        // Current value of the equality constraints

    Matrix<SCALAR, num_vars, 1> x; // Primal optimization variable


    constexpr NLP()
    {
        // We need to zero these here, since only the 
        // non-zero elements are updated during the optimization
        Ji.setZero();
        gi.setZero();
        Je.setZero();
        ge.setZero();
    }

    // Functions to compute the jacobians and functions
    std::vector<std::function<void()>> eval_ineq_jacobians;
    std::vector<std::function<void()>> eval_eq_jacobians;

    // Computes the jacobian and function value
    // - args is a tuple of Variables [x1 x2 x3 ...]
    // - Compute the blocks of the jacobian J1 = df/dx1, J2 = df/dx2, ...
    // template<typename Functor, typename VarTypes...>
    // void add_equality(Functor f, Args&& args)


    // // Computes the jacobian and function value
    // // - args is a tuple of vars [x1 x2 x3 ...]
    // // - Compute the blocks of the jacobian J1 = df/dx1, J2 = df/dx2, ...
    // template<typename Functor, typename Args>
    // void add_equality(Functor f, Args&& args)
    // {
    //     // Rows of the Jacobian that we're computing
    //     auto F = std::apply(f, args);
    //     const int row_min = num_eq_constraints;
    //     const int num_rows = F.size();
    //     num_eq_constraints += num_rows;

    //     // Compute the Jacobian wrt each variable
    //     util::forEach(args, [&](auto&& var) {
    //         // Compute the columns of the Jacobian for this block
    //         // Find the reference to var, so we know where it is in the jacobian
    //         auto it = std::find_if(vars.begin(), vars.end(), 
    //             [&var](const VarInfo& e) {
    //                 return &(std::get<0>(e)) == &var;
    //                 });
    //         if (it == vars.end()) {
    //             cerr << "ERROR - INVALID VARIABLE USED" << endl;
    //             // TODO: Throw an error properly here!
    //             exit(-1);    
    //         }
    //         int col_min = std::get<1>(*it);
    //         int num_cols = std::get<0>(*it).size();

    //         // Create a lambda that fills in the right section of J
    //         Jacobian_Generators.push_back(
    //             [row_min, col_min, num_rows, num_cols, &f, &var, &args] 
    //             (MatrixXd& J) 
    //             {J.block(row_min, col_min, num_rows, num_cols) = jacobian(f, wrt(var), args);}
    //         );

    //         // TODO: Record the function value too!
    //     });
    // }

    // Functions to print out data about all the variables
    using PrintFunction = std::function<void(void)>;
    std::vector<PrintFunction> print_vars;


    int offset = 0; // Used to compute offsets of variables into the primal vector
    template <size_t n>
    void addVariable(Variable<n>& var)
    {
        var.offset = offset;
        offset += var.size;

        PrintFunction f = [&var]() {var.print();}; // Assumes var is returned by move
        print_vars.push_back(f);
    }

    template <size_t n, size_t N>
    void addVariable(std::array<Variable<n>, N>& vars)
    {
        for (auto& var: vars) addVariable(var);
    }

    friend ostream& operator<<(ostream& os, const NLP& nlp)
    {
        os << "-------------------------------\nNLP" << endl;

        os << "Variables" << endl;
        for( auto &f : nlp.print_vars)
        {
            f();
            os << endl;
        }
        return os;
    }


    // TODO:
    // add_linear_constraint - Doesn't re-compute the jacobian when the system changes
    // add_bounds (or specify when adding the variable?)
};



// template <typename T>
namespace DoubleIntegrator {
    template<typename T>
    using StateType = Matrix<T, 2, 1>;

    template<typename T>
    using InputType = Matrix<T, 1, 1>;

    template<typename T>
    // auto dynamics(const Ref<const StateType<T>>& xp, 
    //                      const Ref<const StateType<T>>& x, 
    //                      const Ref<const InputType<T>>& u)
    auto dynamics(const StateType<T>& xp, 
                  const StateType<T>& x, 
                  const InputType<T>& u)
    {
        StateType<T> eq;
        eq << xp[0] - (-sin(x[1]) + x[1]*x[0]), xp[1] - cos(x[0])*u[0];
        return eq;
    }

    // StateType dynamics() {} // Fake signature to tell the return type

    // auto inequalities(const StateType& x, const InputType& u)
    // {
    //     // ||x||_inf <= 1
    //     Matrix<T, Dynamic, 1> ineq(6);
    //     ineq[0] = x[0] - 1;
    //     ineq[1] = x[1] - 1;
    //     ineq[2] = -1 - x[0];
    //     ineq[3] = -1 - x[1];

    //     // ||u|| <= 5
    //     ineq[4] = u[0] - 5;
    //     ineq[5] = -5 - u[0];
    //     return ineq;
    // }
};

// void foo(int i, int j) {};

// template<typename Function, typename Wrt, typename Args, typename Result>
// auto jacobian(const Function& f, Wrt&& wrt, Args&& args, Result& F) -> Eigen::MatrixXd

template <typename F>
struct lazy_val 
{
        F m_computation;

        lazy_val(F computation)
            : m_computation(computation)
        {
        } 

        // decltype(m_computation()) operator() const
        // {
        //     return m_computation();
        // }
};

template <typename F>
inline lazy_val<F> make_lazy_val(F&& computation)
{
    return lazy_val<F>(std::forward<F>(computation));
}

// template<typename Function, typename... Ts>
// struct Con2
// {
//     Con2(Function f, std::tuple<Ts...> const& theArgs)
//     {
//         auto eigenArgs = make_tuple(            
//         )
//     }
// };

template<typename Function, typename... ArgTypes>
struct Constraint
{
    // using Function = (R(C::*)(ArgTypes...);

    // This mess "calls" the Function with a set of default arguments, and gets the return type
    // It doesn't really call the function, since this is done at compile type
    using result_type = decltype(std::declval<Function>()(std::declval<Matrix<double,ArgTypes::size,1>>()...));

    Function _f;
    static constexpr size_t nf = result_type::RowsAtCompileTime;
    static constexpr size_t nx = (ArgTypes::size + ...);

    Matrix<dual, nf, nx> J;  // Jacobian
    Matrix<dual, nf, 1> F;  // Function value

    // std::function<void()> update;
    std::function<result_type()> eval;

    // std::function<void (ArgTypes...)> f;
    // std::tuple myArgs;

    Constraint(Function f, ArgTypes&... args)
        : _f(f)
    {
        eval = [&] (){return f(args.x ...);};
        cout << "Calling f = " << f(args.x ...).transpose() << endl;
        // eval = [&](){return f(args.x ...);};

        // myArgs = std::make_tuple(args.x ...);

        // auto A = std::make_tuple(args.x ...);
        // cout << "f(args...) = " << f(args.x ...) << endl;
        // J = jacobian(f, wrt((args.x)...), at((args.x)...));
        // cout << "J = \n" << J << endl;
        // update = [this, f, args...]()
        // {
        //     this->J = jacobian(f, wrt((args.x)...), at((args.x)...));
        // };
    };

    void update()
    {

    }

    // Updates the constraint value and computes the Jacobian
    // void operator() ()
    // {
    //     J = jacobian(sys.dynamics, 
    //              wrt(nlp.x.segment(xp.startIndex, xp.size), 
    //                  nlp.x.segment(x.startIndex, x.size), 
    //                  nlp.x.segment(u.startIndex, u.size)),
    //              at(nlp.x.segment(xp.startIndex, xp.size), 
    //                 nlp.x.segment(x.startIndex, x.size), 
    //                 nlp.x.segment(u.startIndex, u.size)), )
    // }
};


// https://stackoverflow.com/questions/16868129/how-to-store-variadic-template-arguments
namespace helper
{
    template <std::size_t... Ts>
    struct index {};

    template <std::size_t N, std::size_t... Ts>
    struct gen_seq : gen_seq<N - 1, N - 1, Ts...> {};

    template <std::size_t... Ts>
    struct gen_seq<0, Ts...> : index<Ts...> {};
}

template <typename... Ts>
class Action_impl
{
    private:
        std::function<void (Ts...)> f;
        std::tuple<Ts...> args;
    public:
        template <typename F>
        Action_impl(F&& func, Ts&&... args)
            : f(std::forward<F>(func)),
              args(std::make_tuple(std::forward<Ts>(args)...))
        {}

        template <typename... Args, std::size_t... Is>
        void func(std::tuple<Args...>& tup, helper::index<Is...>)
        {
            f(std::get<Is>(tup)...);
        }

        template <typename... Args>
        void func(std::tuple<Args...>& tup)
        {
            func(tup, helper::gen_seq<sizeof...(Args)>{});
        }

        void act()
        {
            func(args);
        }
};

template <typename F, typename... Args>
Action_impl<Args...> make_action(F&& f, Args&&... args)
{
    return Action_impl<Args...>(std::forward<F>(f), std::forward<Args>(args)...);
}


template<typename Function, typename... Args>
struct Con2
{
    // Args args;
    // Function f;
    // std::function<void (Args...)> f;
    // std::tuple<Args...> args;

    Con2(Function&& f, Args&... args) 
    // :
        // f(std::forward<Function>(f)),
        // args(std::make_tuple(std::forward<Args>(args)...))
        {
        }

    
};

// Constraint<F, std::tuple<Matrix<double, Args::size, 1>...>>
// template <typename F, typename... Args>
// auto make_constraint(F&& f, Args&&... args)
// {
//     auto args_t = std::forward_as_tuple(args.x ...);
//     return Con2(f, args_t);
// }




// template <typename Functor>
// struct Test {
//     Test(Functor f) {};
// };

int main()
{
    {
        const int N = 5; // Horizon

        auto X = VariableArray<2, N>("x");
        auto U = VariableArray<1, N-1>("u");

        // DoubleIntegrator<dual> sys;

        constexpr size_t num_vars = N*2 + (N-1)*1;
        const size_t num_ineq = 10;
        const size_t num_eq = 10;
        NLP<double, num_vars, num_ineq, num_eq> nlp;
        nlp.addVariable(X);
        nlp.addVariable(U);

        // Should be possible to make NLP static by creating NLP with
        // NLP<X, U> nlp
        // ... and then using a variadic template to count the size of all the vars
        // ... but we still wouldn't know the sizes of all the constraints
        // We'd have to do 
        //  C = ConstraintArray<num_funcs, N>(...)
        // and then create NLP with
        //  NLP<VARS<X, U>, EQ_CONSTRAINTS<C1, C2>, INEQ_CONSTRAINTS<xxx>> nlp;
        // ... would be a templated disaster though...

        // Create a function object with the signature
        // ceq(const VectorXd& x, MatrixXd& J, VectorXd& val)
        // J = jacobian(f(x))
        // val = f(x)
        // auto ceq = Constraint(sys.dynamics, X[1], X[0], U[0]);

        cout << nlp << endl;

        using namespace DoubleIntegrator;
        Constraint c(dynamics<dual>, X[1], X[0], U[0]);

        cout << "Constraint" << endl;
        cout << "c.nf = " << c.nf << endl;
        cout << "c.nx = " << c.nx << endl;

        // cout << "X[0].x = " << X[0].x.transpose() << endl;

        // cout << "c.eval() = " << c.eval().transpose() << endl;

        auto args = std::make_tuple(X[1].x, X[0].x, U[0].x);

        // auto con = make_constraint(dynamics<dual>, X[1], X[0], U[0]);
        Con2 q(dynamics<dual>, X[1], X[0], U[0]);

        // cout << "con.f(con.args)" << std::apply(con.f, con.args) << endl;

        // auto add = make_action([] (int a, int b) { std::cout << a+b; }, 2, 3);
        // VectorXd x(3); 
        // x << 1,2,3;
        // auto add = make_action([] (int& x) {cout << x << endl;}, 4);

        // cout << "Calling delayed function\n";
        // add.act();
        // cout << "Call complete\n";


        // cout << "c.eval() = " << c.eval().transpose() << endl;

        // auto l = lazy_val(4 * X[0].x);

        // cout << "Lazy = " << l.m_computation() << endl;
        // lazy

        // c.update();

        // nlp.add_equality(c);

        // cout << "sys.dynamics = \n" << 
        // sys.dynamics(nlp.x.segment(xp.startIndex, xp.size), 
        //              nlp.x.segment(x.startIndex, x.size), 
        //              nlp.x.segment(u.startIndex, u.size))
        //              << endl;

        // cout << "Jacobian of sys.dynamics = \n" << 
        // jacobian(sys.dynamics, 
        //          wrt(nlp.x.segment(xp.startIndex, xp.size), 
        //              nlp.x.segment(x.startIndex, x.size), 
        //              nlp.x.segment(u.startIndex, u.size)),
        //          at(nlp.x.segment(xp.startIndex, xp.size), 
        //             nlp.x.segment(x.startIndex, x.size), 
        //             nlp.x.segment(u.startIndex, u.size)))
        //      << endl;


    };

    // {
    //     Matrix<dual, 2, 1> x;
    //     x << 1, 2;
    //     cout << "testFunc(x) = " << testFunc<dual>(x).transpose() << endl;

    //     VectorXdual y(6);
    //     // Ref<Matrix<dual, 3, 1>> y_fixed(y.segment(2, 3));

    //     y << 1,2,3,4,5,6;
    //     // y_fixed << 10,11,12;

    //     cout << "y = " << y.transpose() << endl;
    //     // cout << "y_fixed = " << y_fixed.transpose() << endl;

    //     cout << "testFunc(y.segment(2,3)) = " << testFunc<dual>(y.segment(2,2)).transpose() << endl;

    //     // VectorXd F(2);
    //     // decltype(testFunc<dual>(x)) F;
    //     VectorXdual F(2);
    //     cout << "Jacobian = \n" << jacobian(testFunc<dual>, wrt(y.segment(2,2)), at(y.segment(2,2)), F) << endl;
    //     cout << "Function value = " << F.transpose() << endl;





    //     // VectorXd A_dyn(2);
    //     // Ref<Matrix<double, 2, 1>> A_fixed(A_dyn);

    //     // A_dyn << 1,2;

    //     // cout << "A_dyn = \n" << A_dyn << endl << "A_fixed = \n" << A_fixed << endl;

    //     // y << 1,2,3;
    //     // cout << "testFunc(y) = " << testFunc(y_fixed).transpose() << endl;
    // };

    // {
    //     ConstraintManager m;
    //     m.add(Constraint<void, float>(fx));
    //     m.add(Constraint<void, int>(fy));
    //     // m.execute("print");
    //     // m.execute("print1", 1);
    // };

    // {
    //     Matrix<dual, 2, 1> x(2);
    //     Matrix<dual, 2, 1> y(2);
    //     x << 1.2, 3.4;
    //     y << 3.5, 6.7;

    //     testStruct<dual> s;
    //     cout << s.testFunc(x, y) << endl;

    //     // auto f = std::function(testStruct::testFunc);

    //     cout << "J = \n" << jacobian(testStruct<dual>::testFunc, wrt(x), at(x, y)) << endl;
    // };

    // {
    //     using Sys = DoubleIntegrator<dual>;
    //     Sys sys;
    //     Sys::StateType x;
    //     Sys::StateType xp;
    //     Sys::InputType u;

    //     xp << 2.4, 5.3;
    //     x << 1.2, 3.4;
    //     u << 5.6;

    //     cout << "dynamics = " << sys.dynamics(xp, x, u) << endl;
    //     cout << " J = \n" << jacobian(sys.dynamics, wrt(x), at(xp, x, u)) << endl;
    // };

    // {
    //     DoubleIntegrator<dual> sys;

    //     NLP nlp;
    //     const int N = 5; // Horizon
    //     auto x1 = nlp.getVar(2);
    //     auto u1 = nlp.getVar(1);
    //     auto x2 = nlp.getVar(2);
    //     auto u2 = nlp.getVar(1);
    //     auto x3 = nlp.getVar(2);
    //     auto u3 = nlp.getVar(1);
    //     auto x4 = nlp.getVar(2);
    //     auto u4 = nlp.getVar(1);
    //     auto x5 = nlp.getVar(2);

    //     nlp.add_equality(sys.dynamics, wrt(x2, x1, u1));
    //     nlp.add_equality(sys.dynamics, wrt(x3, x2, u2));
    //     nlp.add_equality(sys.dynamics, wrt(x4, x3, u3));
    //     nlp.add_equality(sys.dynamics, wrt(x5, x4, u4));
    //     nlp.compile();

    //     cout << "num_ineq_constraints = " << nlp.num_ineq_constraints << endl;
    //     cout << "num_eq_constraints = " << nlp.num_eq_constraints << endl;
    //     cout << "num_variables = " << nlp.num_variables << endl;

    //     cout << "Ji = " << endl << nlp.Ji << endl;

    //     for(const auto &f : nlp.Jacobian_Generators) {
    //         f(nlp.Ji);
    //     }

    //     x1 << NLP::VarType::Random(2,1);
    //     x2 << NLP::VarType::Random(2,1);
    //     x3 << NLP::VarType::Random(2,1);
    //     x4 << NLP::VarType::Random(2,1);
    //     x5 << NLP::VarType::Random(2,1);

    //     u1 << NLP::VarType::Random(1,1);
    //     u2 << NLP::VarType::Random(1,1);
    //     u3 << NLP::VarType::Random(1,1);
    //     u4 << NLP::VarType::Random(1,1);

    //     for(const auto &f : nlp.Jacobian_Generators) {
    //         f(nlp.Ji);
    //     }

    //     cout << "Ji = " << endl << nlp.Ji << endl;

    //     // x << 1.2, 3.4;
    //     // u << 3.4;
    //     // y << 1.5, 3.4, 12.3;

    //     // for(const auto &f : nlp.Jacobian_Generators) {
    //     //     f(nlp.Ji);
    //     // }

    //     // cout << "Ji = " << endl << nlp.Ji << endl;
    // };

    return 0;
}



/*
The flow

NLP nlp;

// Records order and size of each variable
x = nlp.getVariable<3>("x"); 
u = nlp.getVariable<2>("u");

// Define functions that are our constraints
template <typename T>
Matrix<T, 1, 1> f(Matrix<T, 3, 1>& x, Matrix<T, 2, 1>& u)
{
    Matrix<T, 1, 1> R;
    R << x[0] * u[1];
    return R;
}

// Stores the location data for the Jf/dx and Jf/du
// - evaluate f once with zero args to determine size
F = nlp.addConstraint(f, x, u)

// - allocate single primal variable
// - grab pointers to all sub-variables
// - bind functions to evaluate jacobians of all constriants, etc
//   - build a tuple and forward args
//   - the bound functions will all have the same format, so can have them be members of the constraints
// - create a list of bound functions to call to update all jacobians and copy them in to the master Jacobians
nlp.compile();

// The online flow:
x << 3.4, 2.3; // Set individual vars. Overloaded to set the sub-vector of the primal
nlp.x = Vector; // Copy into the primal vector

// Compute gradients, etc
// Just runs through the evaluation list
nlp.evaluate_gradients();

// Can get the value and jacobian of any constraint at the current value of x
// - This will re-compute the value without buffering, or copying into big-J
F.val()
F.jacobian()

 */





// // To store a function call with static templates...
// template <typename Callable>
// struct func_t
// {
//     Callable f; // Grabs the function call type
//     decltype(f()) val; // This is the return-type of the function
// };


// // Suppose I've got a function with overloaded dual types
// template <typename T>
// Vector<T, 5> f(Vector<T, 2> x, Vector<T, 1> u)
// {
//     Vector<T, 5> r;
//     ... do r stuff;
//     return r;
// }

// // Want to create a Jacobian call for f and store it in a container
// // The call to be stored is
// template <typename Callable>
// auto get_jacobian(Callable f, Vector<dual, 2> &x, Vector<dual, 1> &u, Vector<double, Callable::RowsAtCompileTime> &F)
// {
//     return [] () jacobian(f, wrt(x, u), args(x, u), F); // For now this is doing a dynamic allocation...
// }

// // // Takes in a list of Variable types, and returns a function that 
// // template <typename Tuple, typename Callable>
// // void variable_to_x(Tuple&& tuple, Callable&& callable)
// // {
// //     std::apply(
// //         [&callable](auto&&... args) { (callable(std::forward<decltype(args)>(args)), ...); },
// //         std::forward<Tuple>(tuple)
// //     );
// // }


// // template <typename Callable>
// class JFunction
// {
//     Callable J; // Function object


//     public:
//     JFunction(Callable J, Variable x...) :
//         f(f)
//         {};
// }

// // TODO: Re-write all the helper functions hessian, jacobian, etc to use static allocation



// template<size>
// Variable
// {
//     Vector<dual, size> x;
//     name...
//     offset...

//     void set_x(VectorXd& val) {
//         x << val.segment(offset, size);
//     }
// }

// We store the functions in a vector



    // Constraints have the form
    // lb <= g(x) <= ub
    // We want to compute the linearization around the point x0
    // lb <= Jg(x0) * (x - x0) + g(x0) <= ub
    // lb - g(x0) + Jg(x0) * x0 <= Jg(x0) * x <= ub - g(x0) + Jg(x0) * x0
