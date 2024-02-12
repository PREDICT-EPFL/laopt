#include <benchmark/benchmark.h>

#include "laopt/laopt.hpp"
#include "casadi/casadi.hpp"

#include "chain_mass_benchmark_casadi_codegen.h"

using namespace casadi;

// example taken from
// https://github.com/acados/acados/blob/master/examples/acados_python/chain_mass/export_chain_mass_model.py

template<int n_mass = 10>
struct LAOptChainMass : public laopt::Differentiable<LAOptChainMass<n_mass>>
{
    const Eigen::Vector<double, 3> x0{0, 0, 0}; // fix mass (at wall)
    const double m = 0.033; // mass of the balls
    const double D = 1.0; // spring constant
    const double L = 0.033; // rest length of spring

    static constexpr int M = n_mass - 2; // number of intermediate masses

    static constexpr int NX = (2 * M + 1) * 3;
    static constexpr int NU = 3;

    template<typename X, typename U, typename Scalar = typename Eigen::MatrixBase<X>::Scalar>
    EIGEN_STRONG_INLINE Eigen::Vector<Scalar, NX>
    function_impl(const Eigen::MatrixBase<X>& x, const Eigen::MatrixBase<U>& u) noexcept
    {
        // x = (x_pos((M+1)*3), x_vel(M*3))

        Eigen::Vector<Scalar, 3 * M> f; // force on intermediate masses
        f.setZero();

        for (Eigen::Index i = 0; i < M; i++) {
            f(3 * i + 2) = -9.81;
        }

        for (Eigen::Index i = 0; i < M + 1; i++) {
            Eigen::Vector<Scalar, 3> dist;
            if (i == 0) {
                dist = x(Eigen::seq(i*3, (i+1)*3-1)) - x0;
            } else {
                dist = x(Eigen::seq(i*3, (i+1)*3-1)) - x(Eigen::seq((i-1)*3, i*3-1));
            }
            Scalar scale = D / m * (1 - L / dist.norm());
            Eigen::Vector<Scalar, 3> F = scale * dist;

            // mass on the right
            if (i < M) {
                f(Eigen::seq(i*3, (i+1)*3-1)) -= F;
            }

            // mass on the left
            if (i > 0) {
                f(Eigen::seq((i-1)*3, i*3-1)) += F;
            }
        }

        Eigen::Vector<Scalar, NX> x_dot;
        x_dot << x(Eigen::lastN(M*3)), u, f;
        return x_dot;
    }
};

template<int n_mass>
static void BM_LAOPT_FUNCTION(benchmark::State& state)
{
    LAOptChainMass<n_mass> chain_mass;
    laopt::common_functions::RK4<LAOptChainMass<n_mass>, double> chain_mass_d(chain_mass, 0.2);

    Eigen::Vector<double, LAOptChainMass<n_mass>::NX> x = Eigen::Vector<double, LAOptChainMass<n_mass>::NX>::Random();
    Eigen::Vector<double, LAOptChainMass<n_mass>::NU> u = Eigen::Vector<double, LAOptChainMass<n_mass>::NU>::Random();
    Eigen::Vector<double, LAOptChainMass<n_mass>::NX> value;

    for (auto _: state)
    {
        value = chain_mass_d.function(x, u);
        benchmark::DoNotOptimize(value);
    }
}

template<int n_mass>
static void BM_LAOPT_JACOBIAN(benchmark::State& state)
{
    LAOptChainMass<n_mass> chain_mass;
    laopt::common_functions::RK4<LAOptChainMass<n_mass>, double> chain_mass_d(chain_mass, 0.2);

    laopt::IndexedVector<Eigen::Vector<double, LAOptChainMass<n_mass>::NX>> x;
    x = Eigen::Vector<double, LAOptChainMass<n_mass>::NX>::Random();
    laopt::IndexedVector<Eigen::Vector<double, LAOptChainMass<n_mass>::NU>> u;
    u = Eigen::Vector<double, LAOptChainMass<n_mass>::NU>::Random();
    Eigen::Matrix<double, LAOptChainMass<n_mass>::NX, LAOptChainMass<n_mass>::NX + LAOptChainMass<n_mass>::NU> jacobian;

    for (auto _: state)
    {
        chain_mass_d.jacobian(jacobian, x, u);
        benchmark::DoNotOptimize(jacobian);
    }
}

template<int n_mass>
static void BM_LAOPT_WSUM(benchmark::State& state)
{
    LAOptChainMass<n_mass> chain_mass;
    laopt::common_functions::RK4<LAOptChainMass<n_mass>, double> chain_mass_d(chain_mass, 0.2);

    laopt::IndexedVector<Eigen::Vector<double, LAOptChainMass<n_mass>::NX>> x;
    x = Eigen::Vector<double, LAOptChainMass<n_mass>::NX>::Random();
    laopt::IndexedVector<Eigen::Vector<double, LAOptChainMass<n_mass>::NU>> u;
    u = Eigen::Vector<double, LAOptChainMass<n_mass>::NU>::Random();
    Eigen::Vector<double, LAOptChainMass<n_mass>::NX> weight = Eigen::Vector<double, LAOptChainMass<n_mass>::NX>::Random();
    double value;

    for (auto _: state)
    {
        value = chain_mass_d.wsum(weight, x, u);
        benchmark::DoNotOptimize(value);
    }
}

template<int n_mass>
static void BM_LAOPT_GRADIENT(benchmark::State& state)
{
    LAOptChainMass<n_mass> chain_mass;
    laopt::common_functions::RK4<LAOptChainMass<n_mass>, double> chain_mass_d(chain_mass, 0.2);

    laopt::IndexedVector<Eigen::Vector<double, LAOptChainMass<n_mass>::NX>> x;
    x = Eigen::Vector<double, LAOptChainMass<n_mass>::NX>::Random();
    laopt::IndexedVector<Eigen::Vector<double, LAOptChainMass<n_mass>::NU>> u;
    u = Eigen::Vector<double, LAOptChainMass<n_mass>::NU>::Random();
    Eigen::Vector<double, LAOptChainMass<n_mass>::NX> weight = Eigen::Vector<double, LAOptChainMass<n_mass>::NX>::Random();
    Eigen::Vector<double, LAOptChainMass<n_mass>::NX + LAOptChainMass<n_mass>::NU> gradient;

    for (auto _: state)
    {
        chain_mass_d.gradient(gradient, weight, x, u);
        benchmark::DoNotOptimize(gradient);
    }
}

template<int n_mass>
static void BM_LAOPT_HESSIAN(benchmark::State& state)
{
    LAOptChainMass<n_mass> chain_mass;
    laopt::common_functions::RK4<LAOptChainMass<n_mass>, double> chain_mass_d(chain_mass, 0.2);

    laopt::IndexedVector<Eigen::Vector<double, LAOptChainMass<n_mass>::NX>> x;
    x = Eigen::Vector<double, LAOptChainMass<n_mass>::NX>::Random();
    laopt::IndexedVector<Eigen::Vector<double, LAOptChainMass<n_mass>::NU>> u;
    u = Eigen::Vector<double, LAOptChainMass<n_mass>::NU>::Random();
    Eigen::Vector<double, LAOptChainMass<n_mass>::NX> weight = Eigen::Vector<double, LAOptChainMass<n_mass>::NX>::Random();
    Eigen::Matrix<double, LAOptChainMass<n_mass>::NX + LAOptChainMass<n_mass>::NU, LAOptChainMass<n_mass>::NX + LAOptChainMass<n_mass>::NU> hessian;

    for (auto _: state)
    {
        chain_mass_d.hessian(hessian, weight, x, u);
        benchmark::DoNotOptimize(hessian);
    }
}

template<int n_mass = 10>
struct CasadiSXChainMass
{
    DM x0 = DM::vertcat({0, 0, 0}); // fix mass (at wall)
    const double m = 0.033; // mass of the balls
    const double D = 1.0; // spring constant
    const double L = 0.033; // rest length of spring

    static constexpr int M = n_mass - 2; // number of intermediate masses

    static constexpr int NX = (2 * M + 1) * 3;
    static constexpr int NU = 3;

    Function function;
    Function jacobian;
    Function wsum;
    Function gradient;
    Function hessian;

    CasadiSXChainMass()
    {
        SX x_pos = SX::sym("x_pos", (M + 1) * 3); // position of fix mass eliminated
        SX x_vel = SX::sym("x_vel", M * 3);

        SX x = SX::vertcat({x_pos, x_vel});
        SX u = SX::sym("u", NU);

        SX f = SX::zeros(3 * M);

        for (Eigen::Index i = 0; i < M; i++) {
            f(3 * i + 2) = -9.81;
        }

        for (int i = 0; i < M + 1; i++) {
            SX dist;
            if (i == 0) {
                dist = x_pos(Slice(i*3, (i+1)*3));
                dist -= x0;
            } else {
                dist = x_pos(Slice(i*3, (i+1)*3)) - x_pos(Slice((i-1)*3, i*3));
            }
            SX scale = D / m * (1 - L / SX::norm_2(dist));
            SX F = scale * dist;

            // mass on the right
            if (i < M) {
                f(Slice(i*3, (i+1)*3)) -= F;
            }

            // mass on the left
            if (i > 0) {
                f(Slice((i-1)*3, i*3)) += F;
            }
        }

        SX x_dot = SX::vertcat({x_vel, u, f});
        Function f_x_dot = Function("function", {x, u}, {x_dot});

        double h = 0.2;
        SX k1 = f_x_dot(SXVector({x,          u}))[0];
        SX k2 = f_x_dot(SXVector({x+h*0.5*k1, u}))[0];
        SX k3 = f_x_dot(SXVector({x+h*0.5*k2, u}))[0];
        SX k4 = f_x_dot(SXVector({x+h*k3,     u}))[0];

        SX x_next = x + h/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4);

        SX w = SX::sym("w", NX);
        SX x_next_wsum = SX::dot(w, x_next);

        function = Function("function", {x, u}, {x_next});
        jacobian = Function("jacobian", {x, u}, {SX::jacobian(x_next, SX::vertcat({x, u}))});
        wsum = Function("wsum", {w, x, u}, {x_next_wsum});
        gradient = Function("gradient", {w, x, u}, {SX::gradient(x_next_wsum, SX::vertcat({x, u}))});
        hessian = Function("hessian", {w, x, u}, {SX::hessian(x_next_wsum, SX::vertcat({x, u}))});
    }

    void generate_code()
    {
        Dict opts;
        opts["cpp"] = true;
        opts["with_header"] = true;
        CodeGenerator generator("chain_mass_benchmark_casadi_codegen", opts);

        generator.add(function);
        generator.add(jacobian);
        generator.add(wsum);
        generator.add(gradient);
        generator.add(hessian);

        generator.generate();
    }
};

template<int n_mass>
static void BM_CASADI_SX_FUNCTION(benchmark::State& state)
{
    CasadiSXChainMass<n_mass> chain_mass;
    DM x = DM::rand(CasadiSXChainMass<n_mass>::NX);
    DM u = DM::rand(CasadiSXChainMass<n_mass>::NU);
    DM value;

    for (auto _: state)
    {
        value = chain_mass.function(DMVector({x, u}))[0];
        benchmark::DoNotOptimize(value);
    }
}

template<int n_mass>
static void BM_CASADI_SX_JACOBIAN(benchmark::State& state)
{
    CasadiSXChainMass<n_mass> chain_mass;
    DM x = DM::rand(CasadiSXChainMass<n_mass>::NX);
    DM u = DM::rand(CasadiSXChainMass<n_mass>::NU);
    DM jacobian;

    for (auto _: state)
    {
        jacobian = chain_mass.jacobian(DMVector({x, u}))[0];
        benchmark::DoNotOptimize(jacobian);
    }
}

template<int n_mass>
static void BM_CASADI_SX_WSUM(benchmark::State& state)
{
    CasadiSXChainMass<n_mass> chain_mass;
    DM x = DM::rand(CasadiSXChainMass<n_mass>::NX);
    DM u = DM::rand(CasadiSXChainMass<n_mass>::NU);
    DM weight = DM::rand(CasadiSXChainMass<n_mass>::NX);
    DM wsum;

    for (auto _: state)
    {
        wsum = chain_mass.wsum(DMVector({weight, x, u}))[0];
        benchmark::DoNotOptimize(wsum);
    }
}

template<int n_mass>
static void BM_CASADI_SX_GRADIENT(benchmark::State& state)
{
    CasadiSXChainMass<n_mass> chain_mass;
    DM x = DM::rand(CasadiSXChainMass<n_mass>::NX);
    DM u = DM::rand(CasadiSXChainMass<n_mass>::NU);
    DM weight = DM::rand(CasadiSXChainMass<n_mass>::NX);
    DM gradient;

    for (auto _: state)
    {
        gradient = chain_mass.gradient(DMVector({weight, x, u}))[0];
        benchmark::DoNotOptimize(gradient);
    }
}

template<int n_mass>
static void BM_CASADI_SX_HESSIAN(benchmark::State& state)
{
    CasadiSXChainMass<n_mass> chain_mass;
    DM x = DM::rand(CasadiSXChainMass<n_mass>::NX);
    DM u = DM::rand(CasadiSXChainMass<n_mass>::NU);
    DM weight = DM::rand(CasadiSXChainMass<n_mass>::NX);
    DM hessian;

    for (auto _: state)
    {
        hessian = chain_mass.hessian(DMVector({weight, x, u}))[0];
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
    casadi_real** res = allocate_sparse(sz_res, &hessian_sparsity_out);

    for (auto _: state)
    {
        hessian((const casadi_real**) arg, res, nullptr, nullptr, 0);
        benchmark::DoNotOptimize(res);
    }
}


BENCHMARK_TEMPLATE(BM_LAOPT_FUNCTION, 5);
BENCHMARK_TEMPLATE(BM_LAOPT_JACOBIAN, 5);
BENCHMARK_TEMPLATE(BM_LAOPT_WSUM, 5);
BENCHMARK_TEMPLATE(BM_LAOPT_GRADIENT, 5);
BENCHMARK_TEMPLATE(BM_LAOPT_HESSIAN, 5);

BENCHMARK_TEMPLATE(BM_CASADI_SX_FUNCTION, 5);
BENCHMARK_TEMPLATE(BM_CASADI_SX_JACOBIAN, 5);
BENCHMARK_TEMPLATE(BM_CASADI_SX_WSUM, 5);
BENCHMARK_TEMPLATE(BM_CASADI_SX_GRADIENT, 5);
BENCHMARK_TEMPLATE(BM_CASADI_SX_HESSIAN, 5);

BENCHMARK(BM_CASADI_CODEGEN_FUNCTION);
BENCHMARK(BM_CASADI_CODEGEN_JACOBIAN);
BENCHMARK(BM_CASADI_CODEGEN_WSUM);
BENCHMARK(BM_CASADI_CODEGEN_GRADIENT);
BENCHMARK(BM_CASADI_CODEGEN_HESSIAN);

BENCHMARK_MAIN();

//int main()
//{
//    CasadiSXChainMass<5> chain_mass;
//    chain_mass.generate_code();
//
//    return 0;
//}
