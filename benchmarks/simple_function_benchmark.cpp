#include <benchmark/benchmark.h>

#include "laopt/laopt.hpp"
#include "casadi/casadi.hpp"

#include "simple_function_benchmark_casadi_codegen.h"

using namespace casadi;

struct LAOptTestFunction : public laopt::Differentiable<LAOptTestFunction, true>
{
    const Eigen::Matrix<double, 2, 2> A{{1, 2}, {3, 4}};
    const Eigen::Matrix<double, 2, 1> B{{5}, {6}};

    template<typename X, typename U, typename Scalar = typename Eigen::MatrixBase<X>::Scalar>
    EIGEN_STRONG_INLINE Eigen::Vector<Scalar, 2>
    function_impl(const Eigen::MatrixBase<X>& x, const Eigen::MatrixBase<U>& u) noexcept
    {
        return x(0) * (A * x + B * u);
    }
};

static void BM_LAOPT_FUNCTION(benchmark::State& state)
{
    LAOptTestFunction test_function;
    Eigen::Vector<double, 2> x{1, 1};
    Eigen::Vector<double, 1> u{1};
    Eigen::Vector<double, 2> value;

    for (auto _: state)
    {
        value = test_function.function(x, u);
        benchmark::DoNotOptimize(value);
    }
}

static void BM_LAOPT_JACOBIAN(benchmark::State& state)
{
    LAOptTestFunction test_function;
    Eigen::Vector<double, 2> x{1, 1};
    Eigen::Vector<double, 1> u{1};
    Eigen::Vector<double, 2> value;
    Eigen::Matrix<double, 2, 3> jacobian;

    for (auto _: state)
    {
        test_function.jacobian(value, jacobian, x, u);
        benchmark::DoNotOptimize(value);
        benchmark::DoNotOptimize(jacobian);
    }
}

static void BM_LAOPT_WSUM(benchmark::State& state)
{
    LAOptTestFunction test_function;
    Eigen::Vector<double, 2> x{1, 1};
    Eigen::Vector<double, 1> u{1};
    Eigen::Vector<double, 2> weight{1, 1};
    double value;

    for (auto _: state)
    {
        value = test_function.wsum(weight, x, u);
        benchmark::DoNotOptimize(value);
    }
}

static void BM_LAOPT_GRADIENT(benchmark::State& state)
{
    LAOptTestFunction test_function;
    Eigen::Vector<double, 2> x{1, 1};
    Eigen::Vector<double, 1> u{1};
    Eigen::Vector<double, 2> weight{1, 1};
    double value;
    Eigen::Vector<double, 3> gradient;

    for (auto _: state)
    {
        value = test_function.gradient(gradient, weight, x, u);
        benchmark::DoNotOptimize(value);
        benchmark::DoNotOptimize(gradient);
    }
}

static void BM_LAOPT_HESSIAN(benchmark::State& state)
{
    LAOptTestFunction test_function;
    Eigen::Vector<double, 2> x{1, 1};
    Eigen::Vector<double, 1> u{1};
    Eigen::Vector<double, 2> weight{1, 1};
    double value;
    Eigen::Vector<double, 3> gradient;
    Eigen::Matrix<double, 3, 3> hessian;

    for (auto _: state)
    {
        value = test_function.hessian(gradient, hessian, weight, x, u);
        benchmark::DoNotOptimize(value);
        benchmark::DoNotOptimize(gradient);
        benchmark::DoNotOptimize(hessian);
    }
}

struct CasadiSXTestFunction
{
    SX x = SX::sym("x", 2);
    SX u = SX::sym("u", 1);
    SX w = SX::sym("w", 2);

    DM A = DM::vertcat({DM::horzcat({1, 2}), DM::horzcat({3, 4})});
    DM B = DM::vertcat({DM::horzcat({5}), DM::horzcat({6})});

    SX out = x(0) * (SX::mtimes(A, x) + SX::mtimes(B, u));
    SX out_wsum = SX::dot(w, out);

    Function function = Function("function", {x, u}, {out});
    Function jacobian = Function("jacobian", {x, u}, {SX::jacobian(out, SX::vertcat({x, u}))});
    Function wsum = Function("wsum", {w, x, u}, {out_wsum});
    Function gradient = Function("gradient", {w, x, u}, {SX::gradient(out_wsum, SX::vertcat({x, u}))});
    Function hessian = Function("hessian", {w, x, u}, {SX::hessian(out_wsum, SX::vertcat({x, u}))});

    void generate_code()
    {
        Dict opts;
        opts["cpp"] = true;
        opts["with_header"] = true;
        CodeGenerator generator("simple_function_benchmark_casadi_codegen", opts);

        generator.add(function);
        generator.add(jacobian);
        generator.add(wsum);
        generator.add(gradient);
        generator.add(hessian);

        generator.generate();
    }
};

static void BM_CASADI_SX_FUNCTION(benchmark::State& state)
{
    CasadiSXTestFunction test_function;
    DM x = DM::vertcat({1, 1});
    DM u = DM::vertcat({1});
    DM value;

    for (auto _: state)
    {
        value = test_function.function(SXVector({x, u}))[0];
        benchmark::DoNotOptimize(value);
    }
}

static void BM_CASADI_SX_JACOBIAN(benchmark::State& state)
{
    CasadiSXTestFunction test_function;
    DM x = DM::vertcat({1, 1});
    DM u = DM::vertcat({1});
    DM jacobian;

    for (auto _: state)
    {
        jacobian = test_function.jacobian(SXVector({x, u}))[0];
        benchmark::DoNotOptimize(jacobian);
    }
}

static void BM_CASADI_SX_WSUM(benchmark::State& state)
{
    CasadiSXTestFunction test_function;
    DM x = DM::vertcat({1, 1});
    DM u = DM::vertcat({1});
    DM weight = DM::vertcat({1, 1});
    DM wsum;

    for (auto _: state)
    {
        wsum = test_function.wsum(SXVector({weight, x, u}))[0];
        benchmark::DoNotOptimize(wsum);
    }
}

static void BM_CASADI_SX_GRADIENT(benchmark::State& state)
{
    CasadiSXTestFunction test_function;
    DM x = DM::vertcat({1, 1});
    DM u = DM::vertcat({1});
    DM weight = DM::vertcat({1, 1});
    DM gradient;

    for (auto _: state)
    {
        gradient = test_function.gradient(SXVector({weight, x, u}))[0];
        benchmark::DoNotOptimize(gradient);
    }
}

static void BM_CASADI_SX_HESSIAN(benchmark::State& state)
{
    CasadiSXTestFunction test_function;
    DM x = DM::vertcat({1, 1});
    DM u = DM::vertcat({1});
    DM weight = DM::vertcat({1, 1});
    DM hessian;

    for (auto _: state)
    {
        hessian = test_function.hessian(SXVector({weight, x, u}))[0];
        benchmark::DoNotOptimize(hessian);
    }
}

static casadi_int get_nnz_from_sparsity(const casadi_int* sparsity)
{
    casadi_int n_rows = sparsity[0];
    casadi_int n_cols = sparsity[1];
    if (sparsity[2] == 1)
    {
        // dense matrix
        return n_rows * n_cols;
    }

    // sparse matrix
    return sparsity[2 + n_cols];
}

static casadi_real** allocate_sparse(casadi_int sz, const casadi_int* (*func_sparsity_in)(casadi_int i))
{
    casadi_real** arg = new casadi_real*[sz];
    for (casadi_int i = 0; i < sz; i++)
    {
        arg[i] = new casadi_real[get_nnz_from_sparsity(func_sparsity_in(i))];
    }
    return arg;
}

static void BM_CASADI_CODEGEN_FUNCTION(benchmark::State& state)
{
    casadi_int sz_arg, sz_res, sz_iw, sz_w;
    function_work(&sz_arg, &sz_res, &sz_iw, &sz_w);

    casadi_real** arg = allocate_sparse(sz_arg, &function_sparsity_in);
    arg[0][0] = 1;
    arg[0][1] = 1;
    arg[1][0] = 1;

    casadi_real** res = allocate_sparse(sz_res, &function_sparsity_out);

    for (auto _: state)
    {
        function((const casadi_real**) arg, res, nullptr, nullptr, 0);
        benchmark::DoNotOptimize(res);
    }
}

static void BM_CASADI_CODEGEN_JACOBIAN(benchmark::State& state)
{
    casadi_int sz_arg, sz_res, sz_iw, sz_w;
    jacobian_work(&sz_arg, &sz_res, &sz_iw, &sz_w);

    casadi_real** arg = allocate_sparse(sz_arg, &jacobian_sparsity_in);
    arg[0][0] = 1;
    arg[0][1] = 1;
    arg[1][0] = 1;

    casadi_real** res = allocate_sparse(sz_res, &jacobian_sparsity_out);

    for (auto _: state)
    {
        jacobian((const casadi_real**) arg, res, nullptr, nullptr, 0);
        benchmark::DoNotOptimize(res);
    }
}

static void BM_CASADI_CODEGEN_WSUM(benchmark::State& state)
{
    casadi_int sz_arg, sz_res, sz_iw, sz_w;
    wsum_work(&sz_arg, &sz_res, &sz_iw, &sz_w);

    casadi_real** arg = allocate_sparse(sz_arg, &wsum_sparsity_in);
    arg[0][0] = 1;
    arg[0][1] = 1;
    arg[1][0] = 1;
    arg[1][1] = 1;
    arg[2][0] = 1;

    casadi_real** res = allocate_sparse(sz_res, &wsum_sparsity_out);

    for (auto _: state)
    {
        wsum((const casadi_real**) arg, res, nullptr, nullptr, 0);
        benchmark::DoNotOptimize(res);
    }
}

static void BM_CASADI_CODEGEN_GRADIENT(benchmark::State& state)
{
    casadi_int sz_arg, sz_res, sz_iw, sz_w;
    gradient_work(&sz_arg, &sz_res, &sz_iw, &sz_w);

    casadi_real** arg = allocate_sparse(sz_arg, &gradient_sparsity_in);
    arg[0][0] = 1;
    arg[0][1] = 1;
    arg[1][0] = 1;
    arg[1][1] = 1;
    arg[2][0] = 1;

    casadi_real** res = allocate_sparse(sz_res, &gradient_sparsity_out);

    for (auto _: state)
    {
        gradient((const casadi_real**) arg, res, nullptr, nullptr, 0);
        benchmark::DoNotOptimize(res);
    }
}

static void BM_CASADI_CODEGEN_HESSIAN(benchmark::State& state)
{
    casadi_int sz_arg, sz_res, sz_iw, sz_w;
    hessian_work(&sz_arg, &sz_res, &sz_iw, &sz_w);

    casadi_real** arg = allocate_sparse(sz_arg, &hessian_sparsity_in);
    arg[0][0] = 1;
    arg[0][1] = 1;
    arg[1][0] = 1;
    arg[1][1] = 1;
    arg[2][0] = 1;

    casadi_real** res = allocate_sparse(sz_res, &hessian_sparsity_out);

    for (auto _: state)
    {
        hessian((const casadi_real**) arg, res, nullptr, nullptr, 0);
        benchmark::DoNotOptimize(res);
    }
}

BENCHMARK(BM_LAOPT_FUNCTION);
BENCHMARK(BM_LAOPT_JACOBIAN);
BENCHMARK(BM_LAOPT_WSUM);
BENCHMARK(BM_LAOPT_GRADIENT);
BENCHMARK(BM_LAOPT_HESSIAN);

BENCHMARK(BM_CASADI_SX_FUNCTION);
BENCHMARK(BM_CASADI_SX_JACOBIAN);
BENCHMARK(BM_CASADI_SX_WSUM);
BENCHMARK(BM_CASADI_SX_GRADIENT);
BENCHMARK(BM_CASADI_SX_HESSIAN);

BENCHMARK(BM_CASADI_CODEGEN_FUNCTION);
BENCHMARK(BM_CASADI_CODEGEN_JACOBIAN);
BENCHMARK(BM_CASADI_CODEGEN_WSUM);
BENCHMARK(BM_CASADI_CODEGEN_GRADIENT);
BENCHMARK(BM_CASADI_CODEGEN_HESSIAN);

BENCHMARK_MAIN();

//int main()
//{
//    CasadiSXTestFunction test_function;
//    test_function.generate_code();
//
//    return 0;
//}
