#ifndef SRC_MODELINGBASEEIGEN_HPP
#define SRC_MODELINGBASEEIGEN_HPP

#include <iostream>
#include <vector>

#include <Eigen/Core>
#include <unsupported/Eigen/AutoDiff>

namespace flight_model {
namespace eigen_model {

std::vector<double> toVec(const Eigen::MatrixXd &eigenXd)
{
    return std::vector<double>(eigenXd.data(), eigenXd.size() + eigenXd.data());
}

template<typename Scalar_, int nx_, int nu_ = 0, int nd_ = 0, int ny_ = 0>
class DynamicsBase
{
public:
    static const int nx = nx_;
    static const int nu = nu_;
    static const int nd = nd_;
    static const int ny = ny_;

    /* Templated types */
    template<typename scalar_t> using state_t = Eigen::Matrix<scalar_t, nx, 1>;
    template<typename scalar_t> using control_t = Eigen::Matrix<scalar_t, nu, 1>;
    template<typename scalar_t> using dynamic_params_t = Eigen::Matrix<scalar_t, nd, 1>;
    template<typename scalar_t> using output_t = Eigen::Matrix<scalar_t, ny, 1>;
    template<typename T> using constref_t = const Eigen::Ref<const T>;
    template<typename T> using ref_t = Eigen::Ref<T>;

    /* Scalar types */
    using Scalar = Scalar_;
    using State = state_t<Scalar>;
    using Control = control_t<Scalar>;
    using DynamicParams = dynamic_params_t<Scalar>;
    using Output = output_t<Scalar>;
    using SystemMat = Eigen::Matrix<Scalar, nx, nx>;
    using ControlMat = Eigen::Matrix<Scalar, nx, nu>;

    /* AD (nx+nu) types */
    using Derivates = Eigen::Matrix<Scalar, nx + nu, 1>;
    using AdScalar = Eigen::AutoDiffScalar<Derivates>;
    using AdState = state_t<AdScalar>;
    using AdControl = control_t<AdScalar>;
    using AdDynParams = dynamic_params_t<AdScalar>;
    using AdOutput = output_t<AdScalar>;

    DynamicsBase()
    {
        x_trim_ubound.setZero();
        x_trim_lbound.setZero();
        u_trim_ubound.setZero();
        u_trim_lbound.setZero();
        u_physical_ubound.setZero();
        u_physical_lbound.setZero();
        x_default.setZero();
        x_trim.setZero();
        u_trim.setZero();
    }

    State get_default_initial_state() const { return x_default; }
    std::vector<Scalar> get_default_initial_state_vec() const { return toVec(x_default); }
    State get_trim_state() { return x_trim; }
    Control get_trim_control() { return u_trim; }

//    template<typename scalar_t>
//    void dynamics(Eigen::Ref<state_t<scalar_t>> state_dot,
//                  const Eigen::Ref<const state_t<scalar_t>> &state,
//                  const Eigen::Ref<const control_t<scalar_t>> &control,
//                  const Eigen::Ref<const dynamic_params_t<scalar_t>> &dyn_params,
//                  Eigen::Ref<output_t<scalar_t>> output) const
//    {
//        std::cout << "DynamicsBase::dynamics(): Hide me!\n";
//    }
    virtual void dynamics(ref_t<state_t<Scalar>> state_dot,
                          constref_t<state_t<Scalar>> &state,
                          constref_t<control_t<Scalar>> &control,
                          constref_t<dynamic_params_t<Scalar>> &dyn_params,
                          ref_t<output_t<Scalar>> output) const
    {
        std::cout << "DynamicsBase::dynamics<Scalar>(): Override me!\n";
    }

    virtual void dynamics(ref_t<state_t<AdScalar>> state_dot,
                          constref_t<state_t<AdScalar>> &state,
                          constref_t<control_t<AdScalar>> &control,
                          constref_t<dynamic_params_t<Scalar>> &dyn_params,
                          ref_t<output_t<AdScalar>> output) const
    {
        std::cout << "DynamicsBase::dynamics<AdScalar>(): Override me!\n";
    }

    void linearize_at(Eigen::Matrix<Scalar, nx, nx> &A, Eigen::Matrix<Scalar, nx, nu> &B,
                      const State &state, const Control &control, const DynamicParams &dynParams = DynamicParams())
    {
        // Set linearization point and seed derivatives
        const int NXU = nx + nu;
        AdState x0 = state;
        AdControl u0 = control;
        int derivative_idx = 0;
        for (int ix = 0; ix < nx; ++ix)
            x0(ix).derivatives() = Eigen::Matrix<Scalar, NXU, 1>::Unit(NXU, derivative_idx++);
        for (int iu = 0; iu < nu; ++iu)
            u0(iu).derivatives() = Eigen::Matrix<Scalar, NXU, 1>::Unit(NXU, derivative_idx++);

        // Propagate through the dynamics function
        AdState x_dot;
        DynamicParams d = dynParams;
        AdOutput y; // Dummy
        dynamics(x_dot, x0, u0, d, y);

        // Obtain Jacobian and segment it in df_dx (=A) and df_du (=B)
        Eigen::Matrix<Scalar, nx, NXU> df_dxu;
        for (int ix = 0; ix < nx; ++ix)
            df_dxu.row(ix) = x_dot(ix).derivatives();

        A = df_dxu.template block<nx, nx>(0, 0);
        B = df_dxu.template block<nx, nu>(0, nx);
        //std::cout << "A (=df_dx): \n" << A << "\n";
        //std::cout << "B (=df_du): \n" << B << "\n";
    }


    void trim(const DynamicParams &dyn_params, const Eigen::Matrix<Scalar, nx + nu, 1> &xu0,
              const Control &LBU = {}, const Control &UBU = {},
              const State &LBX = {}, const State &UBX = {}) {}

    State x_trim_ubound;
    State x_trim_lbound;
    Control u_trim_ubound;
    Control u_trim_lbound;
    Control u_physical_ubound;
    Control u_physical_lbound;
protected:
    State x_default;
    State x_trim;
    Control u_trim;
};

} //namespace eigen_model
} //namespace flight_model

#endif //SRC_MODELINGBASEEIGEN_HPP
