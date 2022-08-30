#include <Eigen/Core>
#include <iostream>
#include <iomanip>

    template <typename T>
    auto type_name() noexcept {
#ifdef __clang__
        std::string name = __PRETTY_FUNCTION__;
        std::string prefix = "auto laopt::type_name() [T = ";
        std::string suffix = "]";
#elif defined(__GNUC__)
        std::string name = __PRETTY_FUNCTION__;
        std::string prefix = "auto laopt::type_name() [with T = ";
        std::string suffix = "]";
#elif defined(_MSC_VER)
        std::string name = __FUNCSIG__;
        std::string prefix = "auto __cdecl laopt::type_name<";
        std::string suffix = ">(void) noexcept";
#else
        std::string name = "Error: unsupported compiler";
        std::string prefix = "";
        std::string suffix = "";
#endif
        return name.substr(prefix.length(), name.length() - prefix.length() - suffix.length());
    }


// Forward declare all specializations
template<typename Derived>
int get_row_index(Derived&& M, int row);
template<typename Derived, typename RowIndices, typename ColIndices>
int get_row_index(Eigen::IndexedView<Derived,RowIndices,ColIndices>& M, int row);
template<typename Derived>
int get_row_index(Eigen::VectorBlock<Derived>& M, int row);
template<typename XprType, int BlockRows, int BlockCols, bool InnerPanel> 
int get_row_index(Eigen::Block<XprType,BlockRows,BlockCols,InnerPanel>& M, int row);




template<typename Derived>
int get_row_index(Derived&& M, int row)
{
    std::cout << "[Default " << row << "] ";
    // std::cout << "type(M) = " << type_name<decltype(M)>() << std::endl;

    return row;
}

template<typename Derived, typename RowIndices, typename ColIndices>
int get_row_index(Eigen::IndexedView<Derived,RowIndices,ColIndices>& M, int row)
{
    int rowval = M.rowIndices()[row];
    auto nest = M.nestedExpression();

    std::cout << "[IndexedView " << rowval << "] ";
    // std::cout << "type(nest) = " << type_name<decltype(nest)>() << std::endl;

    return get_row_index(nest, rowval);
}

template<typename Derived>
int get_row_index(Eigen::VectorBlock<Derived>& M, int row)
{
    int rowval = row + M.startRow();
    auto nest = M.nestedExpression();

    std::cout << "[VectorBlock " << rowval << "] ";
    // std::cout << "type(nest) = " << type_name<decltype(nest)>() << std::endl;

    return get_row_index(nest, rowval);
}

template<typename XprType, int BlockRows, int BlockCols, bool InnerPanel> 
int get_row_index(Eigen::Block<XprType,BlockRows,BlockCols,InnerPanel>& M, int row)
{
    int rowval = row + M.startRow();
    auto nest = M.nestedExpression();

    std::cout << "[Block " << rowval << "] ";
    // std::cout << "type(nest) = " << type_name<decltype(nest)>() << std::endl;

    return get_row_index(nest, rowval);
}

template<typename T>
void print_indices(std::string name, const T& expr)
{
    std::cout << name << std::endl;
    auto q = expr;
    // std::cout << "type(expr) = " << type_name<T>() << std::endl;
    for(int i=0; i<q.rows(); i++)
        std::cout << get_row_index(q, i) << ", " << std::endl;
    std::cout << "\n";
}

int main()
{

    // Declare our master variable
    Eigen::Map<Eigen::Vector<double, 30>> master_variable(NULL);

    // Our variables are just indices into the master
    constexpr int N = 10;
    auto x = master_variable(Eigen::seqN(4, Eigen::fix<N>));

    // We can't refer to the memory of x because there isn't any allocated yet, 
    // but we can get the index information

    print_indices("x", x);
    print_indices("x({1,3,1})", x({1,3,1}));
    print_indices("x({1,3,1})({2,2,1})", x({1,3,1})({2,2,1}));
    print_indices("x({1,3,1})({2,2,1})(Eigen::lastN(2))", x({1,3,1})({2,2,1})(Eigen::lastN(2)));

    return 0;
}
