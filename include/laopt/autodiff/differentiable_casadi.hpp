#ifndef LAOPT_DIFFERENTIABLE_CASADI_HPP
#define LAOPT_DIFFERENTIABLE_CASADI_HPP

#include <Eigen/Dense>
#include "laopt/autodiff/autodiff_scalar.hpp"
#include "laopt/autodiff/eigen_casadi_support.hpp"
#include "laopt/autodiff/differentiable_options.hpp"

namespace laopt
{

/**
 * This is a faster implementation of casadi::FunctionBuffer using less run-time checks.
 */
class FastCasadiFunctionBuffer {
    casadi::Function f;
    std::vector<double> w;
    std::vector<casadi_int> iw;
    std::vector<const double*> arg;
    // arg_buffer is to buffer non-continuous arguments
    std::vector<std::vector<double>> arg_buffer;
    std::vector<double*> res;
    int mem;
public:
    FastCasadiFunctionBuffer(const casadi::Function& f) : f(f)
    {
        size_t n_args = f.sz_arg();

        w.resize(f.sz_w());
        iw.resize(f.sz_iw());
        arg.resize(n_args);
        arg_buffer.resize(n_args);
        res.resize(f.sz_res());
        mem = (int) f.checkout();

        for (size_t i = 0; i < n_args; i++) {
            arg_buffer[i].resize(f.nnz_in((casadi_int) i));
        }
    }

    ~FastCasadiFunctionBuffer()
    {
        f.release(mem);
    }

    template<typename T>
    void set_arg_buffer(casadi_int i, const Eigen::MatrixBase<T>& arg)
    {
        constexpr size_t n = Eigen::MatrixBase<T>::RowsAtCompileTime;
        Eigen::Map<Eigen::Vector<double, n>> map(arg_buffer[i].data(), n);
        map = arg;
        set_arg(i, arg_buffer[i].data(), n);
    }

    void set_arg(casadi_int i, const double* a, casadi_int size)
    {
        assert(size >= f.nnz_in(i) && "Buffer is not large enough.");
        arg[i] = a;
    }

    void set_res(casadi_int i, double* a, casadi_int size)
    {
        assert(size >= f.nnz_out(i) && "Buffer is not large enough.");
        res[i] = a;
    }

    void eval()
    {
        f(casadi::get_ptr(arg), casadi::get_ptr(res), casadi::get_ptr(iw), casadi::get_ptr(w), mem);
    }
};

template<typename Derived, int Options>
class DifferentiableImplCasadi
{
protected:
    std::size_t jacobian_tag_id_counter = 0;
    std::size_t hessian_tag_id_counter = 0;

    std::vector<std::shared_ptr<casadi::Function>> casadi_jacobian_sparse;
    std::vector<std::shared_ptr<casadi::Function>> casadi_hessian_sparse;
    std::vector<std::shared_ptr<FastCasadiFunctionBuffer>> casadi_jacobian_buffer;
    std::vector<std::shared_ptr<FastCasadiFunctionBuffer>> casadi_hessian_buffer;

    template<typename Tag, typename OutJacobian, typename AScalar, typename... Args, int Opts = Options>
    EIGEN_STRONG_INLINE typename std::enable_if<(has_tag_override<Tag>::type::value && has_tag_casadi<Tag>::value) ||
                                                (!has_tag_override<Tag>::type::value && (Opts & CASADI_JACOBIAN) != 0)>::type
    jacobian_impl_autodiff_eval(
        const Tag& tag, // Function to call
        OutJacobian& out_jacobian, // Outputs
        const AScalar& alpha, // Scaling factor
        const Args&... args) noexcept // Function arguments
    {
        using Info = typename Derived::template FuncInfo<Tag, Args...>;

        build_casadi_jacobian(tag, args...);
        auto& buffer = casadi_jacobian_buffer[get_jacobian_tag_id<Tag>()];

        int offset = 0;
        (void) std::initializer_list<int>{
            (
                offset = set_casadi_buffer(offset, *buffer, args),
                0
            )...
        };

        Eigen::Matrix<double, Info::n_outputs, Info::n_inputs> res;
        buffer->set_res(0, res.data(), Info::n_outputs * Info::n_inputs);
        buffer->eval();

        out_jacobian += alpha * res;
    }

    template<typename Tag, typename SparsityNullMat, typename AScalar, typename... Args, int Opts = Options>
    EIGEN_STRONG_INLINE typename std::enable_if<(has_tag_override<Tag>::type::value && has_tag_casadi<Tag>::value) ||
                                                (!has_tag_override<Tag>::type::value && (Opts & CASADI_JACOBIAN) != 0)>::type
    jacobian_impl_autodiff_sparsity(
        const Tag& tag, // Function to call
        BSSliceSparsity<BSMatrixSparsity, SparsityNullMat>& out_jacobian, // Outputs
        const AScalar& alpha, // Scaling factor
        const Args&... args) noexcept // Function arguments
    {
        using Info = typename Derived::template FuncInfo<Tag, Args...>;

        build_casadi_jacobian(tag, args...);

        const casadi::Sparsity& sparsity = casadi_jacobian_sparse[get_jacobian_tag_id<Tag>()]->sparsity_out(0);
        for (int col = 0; col < Info::n_inputs; col++) {
            for (int i = sparsity.colind()[col]; i < sparsity.colind()[col + 1]; i++) {
                int row = sparsity.row()[i];
                out_jacobian(row, col) = 1;
            }
        }
    }

    template<typename Tag, typename OutHessian, typename Weight, typename... Args, int Opts = Options>
    EIGEN_STRONG_INLINE typename std::enable_if<(has_tag_override<Tag>::type::value && has_tag_casadi<Tag>::value) ||
                                                (!has_tag_override<Tag>::type::value && (Opts & CASADI_HESSIAN) != 0)>::type
    hessian_impl_autodiff_eval(
        const Tag& tag,
        OutHessian& out_hessian,
        const Eigen::MatrixBase<Weight>& weight,
        const Args&... args) noexcept
    {
        using Info = typename Derived::template FuncInfo<Tag, Args...>;

        build_casadi_hessian(tag, weight, args...);
        auto& buffer = casadi_hessian_buffer[get_hessian_tag_id<Tag>()];

        int offset = set_casadi_buffer(0, *buffer, weight);
        (void) std::initializer_list<int>{
            (
                offset = set_casadi_buffer(offset, *buffer, args),
                0
            )...
        };

        Eigen::Matrix<double, Info::n_inputs, Info::n_inputs> res;
        buffer->set_res(0, res.data(), Info::n_inputs * Info::n_inputs);
        buffer->eval();

        out_hessian += res;
    }

    template<typename Tag, typename SparsityNullMat, typename Weight, typename... Args, int Opts = Options>
    EIGEN_STRONG_INLINE typename std::enable_if<(has_tag_override<Tag>::type::value && has_tag_casadi<Tag>::value) ||
                                                (!has_tag_override<Tag>::type::value && (Opts & CASADI_HESSIAN) != 0)>::type
    hessian_impl_autodiff_sparsity(
        const Tag& tag,
        BSSliceSparsity<BSMatrixSparsity, SparsityNullMat>& out_hessian,
        const Eigen::MatrixBase<Weight>& weight,
        const Args&... args) noexcept
    {
        using Info = typename Derived::template FuncInfo<Tag, Args...>;

        build_casadi_hessian(tag, weight, args...);

        const casadi::Sparsity& sparsity = casadi_hessian_sparse[get_hessian_tag_id<Tag>()]->sparsity_out(0);
        for (int col = 0; col < Info::n_inputs; col++) {
            for (int i = sparsity.colind()[col]; i < sparsity.colind()[col + 1]; i++) {
                int row = sparsity.row()[i];
                out_hessian(row, col) = 1;
            }
        }
    }

    template<typename Tag>
    std::size_t get_jacobian_tag_id()
    {
        static std::size_t id = jacobian_tag_id_counter++;
        return id;
    }

    template<typename Tag>
    std::size_t get_hessian_tag_id()
    {
        static std::size_t id = hessian_tag_id_counter++;
        return id;
    }

    template<typename Tag, typename Tuple, size_t... I>
    auto call_function_from_tuple(const Tag& tag, Tuple& t, std::index_sequence<I...>)
    {
        return static_cast<Derived*>(this)->function(tag, std::get<I>(t)...);
    }

    template<typename Tag, typename Tuple>
    auto call_function_from_tuple(const Tag& tag, Tuple& t)
    {
        static constexpr auto size = std::tuple_size<Tuple>::value;
        return call_function_from_tuple(tag, t, std::make_index_sequence<size>{});
    }

    template<typename Tag, typename... Args>
    EIGEN_STRONG_INLINE void
    build_casadi_jacobian(const Tag& tag, const Args&... args) noexcept
    {
        using Info = typename Derived::template FuncInfo<Tag, Args...>;

        assert(get_jacobian_tag_id<Tag>() <= casadi_jacobian_sparse.size());
        if (get_jacobian_tag_id<Tag>() == casadi_jacobian_sparse.size())
        {
            std::vector<casadi::SX> casadi_args;
            std::vector<casadi::SX> casadi_vars;
            // we have to go into tuple land here to ensure make_casadi_sx is executed in order which is enforced by the brace initialization list
            std::tuple<decltype(make_casadi_sx(casadi_args, casadi_vars, args))...> func_args{make_casadi_sx(casadi_args, casadi_vars, args)...};
            Eigen::Vector<casadi::SX, Info::n_outputs> out = to_matrix_type(call_function_from_tuple(tag, func_args));
            casadi::SX casadi_out = casadi::SX::vertcat({out.data(), out.data() + Info::n_outputs});

            casadi_jacobian_sparse.emplace_back(std::shared_ptr<casadi::Function>(new casadi::Function(
                "jacobian_sparse",
                casadi_args,
                {casadi::SX::jacobian(casadi_out, casadi::SX::vertcat(casadi_vars))})));
            casadi::Dict jit_options = {{"flags", "-O3"}};
            casadi::Function jacobian_dense(
                "jacobian_dense",
                casadi_args,
                {casadi::SX::densify(casadi::SX::jacobian(casadi_out, casadi::SX::vertcat(casadi_vars)))},
                {{"jit", !(Options & CASADI_NO_JIT)}, {"compiler", "shell"}, {"jit_options", jit_options}});
            casadi_jacobian_buffer.emplace_back(std::make_unique<FastCasadiFunctionBuffer>(jacobian_dense));
        }
    }

    template<typename Tag, typename Weight, typename... Args>
    EIGEN_STRONG_INLINE void
    build_casadi_hessian(const Tag& tag, const Eigen::MatrixBase<Weight>& weight, const Args&... args) noexcept
    {
        using Info = typename Derived::template FuncInfo<Tag, Args...>;

        assert(get_hessian_tag_id<Tag>() <= casadi_hessian_sparse.size());
        if (get_hessian_tag_id<Tag>() == casadi_hessian_sparse.size())
        {
            std::vector<casadi::SX> casadi_args;
            std::vector<casadi::SX> casadi_vars;
            Eigen::Vector<casadi::SX, Info::n_outputs> casadi_weight = make_casadi_sx(casadi_args, casadi_vars, weight);
            // we have to go into tuple land here to ensure make_casadi_sx is executed in order which is enforced by the brace initialization list
            std::tuple<decltype(make_casadi_sx(casadi_args, casadi_vars, args))...> func_args{make_casadi_sx(casadi_args, casadi_vars, args)...};
            Eigen::Vector<casadi::SX, Info::n_outputs> out = to_matrix_type(call_function_from_tuple(tag, func_args));
            casadi::SX out_weight = casadi_weight.dot(out);

            casadi_hessian_sparse.emplace_back(std::shared_ptr<casadi::Function>(new casadi::Function(
                "hessian_sparse",
                casadi_args,
                {casadi::SX::hessian(out_weight, casadi::SX::vertcat(casadi_vars))})));
            casadi::Dict jit_options = {{"flags", "-O3"}};
            casadi::Function hessian_dense(
                "hessian_dense",
                casadi_args,
                {casadi::SX::densify(casadi::SX::hessian(out_weight, casadi::SX::vertcat(casadi_vars)))},
                {{"jit", !(Options & CASADI_NO_JIT)}, {"compiler", "shell"}, {"jit_options", jit_options}});
            casadi_hessian_buffer.emplace_back(std::make_unique<FastCasadiFunctionBuffer>(hessian_dense));
        }
    }

    template<typename T>
    EIGEN_STRONG_INLINE typename std::enable_if<is_variable<T>::value, Eigen::Vector<casadi::SX, Eigen::MatrixBase<T>::RowsAtCompileTime>>::type
    make_casadi_sx(std::vector<casadi::SX>& casadi_args, std::vector<casadi::SX>& casadi_vars, const Eigen::MatrixBase<T>& x) noexcept
    {
        constexpr size_t rows = Eigen::MatrixBase<T>::RowsAtCompileTime;
        constexpr size_t cols = Eigen::MatrixBase<T>::ColsAtCompileTime;
        static_assert(rows >= 0 && cols == 1, "vector based parameters can not be dynamic");
        Eigen::Vector<casadi::SX, rows> y;
        for (size_t i = 0; i < rows; i++) {
            y[i] = casadi::SX::sym("arg" + std::to_string(i), 1);
        }
        casadi_args.push_back(casadi::SX::vertcat({y.data(), y.data() + rows}));
        casadi_vars.push_back(casadi::SX::vertcat({y.data(), y.data() + rows}));
        return y;
    }

    template<typename T>
    EIGEN_STRONG_INLINE typename std::enable_if<!is_variable<T>::value, Eigen::Vector<casadi::SX, Eigen::MatrixBase<T>::RowsAtCompileTime>>::type
    make_casadi_sx(std::vector<casadi::SX>& casadi_args, std::vector<casadi::SX>& casadi_vars, const Eigen::MatrixBase<T>& x) noexcept
    {
        constexpr size_t rows = Eigen::MatrixBase<T>::RowsAtCompileTime;
        constexpr size_t cols = Eigen::MatrixBase<T>::ColsAtCompileTime;
        static_assert(rows >= 0 && cols == 1, "vector based parameters can not be dynamic");
        Eigen::Vector<casadi::SX, rows> y;
        for (size_t i = 0; i < rows; i++) {
            y[i] = casadi::SX::sym("arg" + std::to_string(i), 1);
        }
        casadi_args.push_back(casadi::SX::vertcat({y.data(), y.data() + rows}));
        return y;
    }

    template<typename T>
    EIGEN_STRONG_INLINE typename std::enable_if<!is_variable<T>::value && !std::is_base_of<Eigen::MatrixBase<T>, T>::value, casadi::SX>::type
    make_casadi_sx(std::vector<casadi::SX>& casadi_args, std::vector<casadi::SX>& casadi_vars, const T& x) noexcept
    {
        casadi::SX sym = casadi::SX::sym("arg", 1);
        casadi_args.push_back(sym);
        return sym;
    }

    template<typename T>
    EIGEN_STRONG_INLINE typename std::enable_if<(Eigen::MatrixBase<T>::Flags & Eigen::DirectAccessBit) != 0, int>::type
    set_casadi_buffer(int offset, FastCasadiFunctionBuffer& buffer, const Eigen::MatrixBase<T>& x) noexcept
    {
        constexpr size_t n = Eigen::MatrixBase<T>::RowsAtCompileTime;
        buffer.set_arg(offset, x.derived().data(), n);
        return offset + 1;
    }

    template<typename T>
    EIGEN_STRONG_INLINE typename std::enable_if<(Eigen::MatrixBase<T>::Flags & Eigen::DirectAccessBit) == 0, int>::type
    set_casadi_buffer(int offset, FastCasadiFunctionBuffer& buffer, const Eigen::MatrixBase<T>& x) noexcept
    {
        // x is non-continuous in memory
        buffer.set_arg_buffer(offset, x.derived());
        return offset + 1;
    }

    template<typename T>
    EIGEN_STRONG_INLINE typename std::enable_if<!std::is_base_of<Eigen::MatrixBase<T>, T>::value, int>::type
    set_casadi_buffer(int offset, FastCasadiFunctionBuffer& buffer, const T& x) noexcept
    {
        buffer.set_arg(offset, &x, 1);
        return offset + 1;
    }
};

} // namespace laopt

#endif // LAOPT_DIFFERENTIABLE_CASADI_HPP