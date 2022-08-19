#ifndef KITE_HPP
#define KITE_HPP

#include "casadi/casadi.hpp"
#include <Eigen/Core>
#include <unsupported/Eigen/src/AutoDiff/AutoDiffScalar.h>
#include "laopt/eigen_autodiff_fix.hpp"
#include "yaml-cpp/yaml.h"

namespace kite_math {
/** quaternion arithmetics */
template<typename scalar_t, typename Derived>
Eigen::Quaternion<scalar_t> T1quat(const Derived &rotAng)
{
    Eigen::Quaternion<scalar_t> q;
    q = Eigen::AngleAxis<scalar_t>(-rotAng, Eigen::Matrix<scalar_t, 3, 1>(1, 0, 0));
    return q;
}
template<typename scalar_t, typename Derived>
Eigen::Quaternion<scalar_t> T2quat(const Derived &rotAng)
{
    Eigen::Quaternion<scalar_t> q;
    q = Eigen::AngleAxis<scalar_t>(-rotAng, Eigen::Matrix<scalar_t, 3, 1>(0, 1, 0));
    return q;
}
template<typename scalar_t, typename Derived>
Eigen::Quaternion<scalar_t> T3quat(const Derived &rotAng)
{
    Eigen::Quaternion<scalar_t> q;
    q = Eigen::AngleAxis<scalar_t>(-rotAng, Eigen::Matrix<scalar_t, 3, 1>(0, 0, 1));
    return q;
}

namespace casadi_quat {
/** quaternion arithmetics */
template<typename sym_t>
sym_t T1quat(const sym_t &rotAng) { return sym_t::vertcat({cos(-rotAng / 2.0), sin(-rotAng / 2.0), 0, 0}); }
template<typename sym_t>
sym_t T2quat(const sym_t &rotAng) { return sym_t::vertcat({cos(-rotAng / 2.0), 0, sin(-rotAng / 2.0), 0}); }
template<typename sym_t>
sym_t T3quat(const sym_t &rotAng) { return sym_t::vertcat({cos(-rotAng / 2.0), 0, 0, sin(-rotAng / 2.0)}); }

template<typename sym_t>
sym_t quat_multiply(const sym_t &q1, const sym_t &q2)
{
    sym_t s1 = q1(0);
    sym_t v1 = q1(casadi::Slice(1, 4), 0);

    sym_t s2 = q2(0);
    sym_t v2 = q2(casadi::Slice(1, 4), 0);

    sym_t s = (s1 * s2) - sym_t::dot(v1, v2);
    sym_t v = sym_t::cross(v1, v2) + (s1 * v2) + (s2 * v1);

    return sym_t::vertcat({s, v});
}

template<typename sym_t>
sym_t quat_inverse(const sym_t &q)
{
    std::vector<sym_t> tmp{q(0), -q(1), -q(2), -q(3)};
    return sym_t::vertcat(tmp);
}

template<typename sym_t>
sym_t quat_transform(const sym_t &q_ba, const sym_t &a_vect)
{
    sym_t tmp = quat_multiply(q_ba, quat_multiply(sym_t::vertcat({0, a_vect}), quat_inverse(q_ba)));
    return tmp(casadi::Slice(1, 4), 0);
}
}
}
namespace kite_model {

template<int new_rows, int new_cols, typename DerivedType, int DerivedRows, int DerivedCols>
Eigen::Matrix<DerivedType, new_rows, new_cols>
reshape(const Eigen::Matrix<DerivedType, DerivedRows, DerivedCols> &mat_in)
{
    return Eigen::Map<const Eigen::Matrix<DerivedType, new_rows, new_cols>>(mat_in.data(), new_rows, new_cols);
}

template<typename DerivedType, int DerivedRows, int DerivedCols>
void toEigen(const casadi::DM &casadi_matrix, Eigen::Matrix<DerivedType, DerivedRows, DerivedCols> &eigen_matrix)
{
    eigen_matrix = Eigen::Matrix<DerivedType, DerivedRows, DerivedCols>::Map(
            casadi::DM::densify(casadi_matrix).nonzeros().data(), casadi_matrix.size1(), casadi_matrix.size2());
}
Eigen::MatrixXd toEigenX(const casadi::DM &casadi_matrix)
{
    const long rows = casadi_matrix.size1();
    const long cols = casadi_matrix.size2();
    Eigen::MatrixXd eigen_matrix(rows, cols);
    eigen_matrix.setZero();
    std::memcpy(eigen_matrix.data(), casadi_matrix.ptr(), sizeof(double) * rows * cols);
    return eigen_matrix;
}
std::vector<double> toVec(const casadi::DM &casadi_matrix)
{
    return casadi_matrix.nonzeros();
}
std::vector<double> toVec(const Eigen::MatrixXd &eigenXd)
{
    return std::vector<double>(eigenXd.data(), eigenXd.size() + eigenXd.data());
}
casadi::DM toDM(const Eigen::MatrixXd &eigenXd)
{
    const casadi::DM dm_vec{toVec(eigenXd)};
    return reshape(dm_vec, eigenXd.rows(), eigenXd.cols());
}

enum StateRepresentation
{
    Undefined,
    AttQuat, AttEuler, LongitudinalEulerAoa, LongitudinalUW, LongitudinalFlightPath
};
double get_value(const YAML::Node &node, const std::string &name)
{
    bool property_found{false};
    double property_value = 0.0;
    for (YAML::const_iterator it = node.begin(); it != node.end(); ++it)
    {
        std::string key = it->first.as<std::string>();
        switch (it->second.Type())
        {
            case YAML::NodeType::Scalar :
                if (key == name)
                {
                    property_value = it->second.as<double>();
                    property_found = true;
                    return property_value;
                }
                break;
            case YAML::NodeType::Map :
                /** iterate over the map */
                for (auto iter = it->second.begin(); iter != it->second.end(); ++iter)
                {
                    if ((iter->first).as<std::string>() == name)
                    {
                        property_found = true;
                        property_value = iter->second.as<double>();
                        return property_value;
                    }
                }
                break;
            case YAML::NodeType::Null :
                std::cout << "real lox \n";
                break;
            default :
                std::cout << "lox \n";
                break;
        }
    }
    casadi_assert(property_found, "get_value(YAML::Node, string): Property '%s' not found", name.c_str());
    return property_value;
}

constexpr bool FIX_V0 = false; // In all models: Normalize rates with V0 instead of Va
constexpr double FIXED_V0 = 12;

namespace eigen_model {
using namespace kite_math;

using Scalar = double;

namespace static_parameters {
/* Static parameters are set once at initialization.
 * Add one enum for each static parameter. */
enum StaticParameterNames
{
    rho,

    static_params_max,
    n_static_params = static_params_max + 1
};

struct StaticParamsBuilder
{
public:
    std::vector<Scalar> vec()
    {
        std::vector<Scalar> vec(std::begin(param_array), std::end(param_array));
        return vec;
    }
    Scalar &operator[](StaticParameterNames param_name)
    {
        return param_array[param_name];
    }
private:
    std::array<Scalar, static_params_max> param_array{};
};
}

struct KiteParams
{
    KiteParams()
    {
        //debug_set_PvwYR_params();
    }
    virtual void load_params_from_yaml(const std::string &yaml_filepath)
    {
//        std::cout << "Loading kite params from: " << yaml_filepath << "\n";
        const YAML::Node config = YAML::LoadFile(yaml_filepath);

        b = get_value(config, "b");
        c = get_value(config, "c");
        AR = get_value(config, "AR");
        S = get_value(config, "S");

        mass = get_value(config, "mass");
        Ixx = get_value(config, "Ixx");
        Iyy = get_value(config, "Iyy");
        Izz = get_value(config, "Izz");
        Ixz = get_value(config, "Ixz");

        e_oswald = get_value(config, "e_oswald");
        CD0 = get_value(config, "CD0");

        CL0 = get_value(config, "CL0");
        CLa = get_value(config, "CLa");
        Cm0 = get_value(config, "Cm0");
        Cma = get_value(config, "Cma");

        CYb = get_value(config, "CYb");
        Cl0 = get_value(config, "Cl0");
        Clb = get_value(config, "Clb");
        Cn0 = get_value(config, "Cn0");
        Cnb = get_value(config, "Cnb");

        CLq = get_value(config, "CLq");
        Cmq = get_value(config, "Cmq");

        CYp = get_value(config, "CYp");
        Clp = get_value(config, "Clp");
        Cnp = get_value(config, "Cnp");

        CYr = get_value(config, "CYr");
        Clr = get_value(config, "Clr");
        Cnr = get_value(config, "Cnr");

        CLde = get_value(config, "CLde");
        Cmde = get_value(config, "Cmde");

        Clda = get_value(config, "Clda");
        Cnda = get_value(config, "Cnda");

        CYdr = get_value(config, "CYdr");
        Cldr = get_value(config, "Cldr");
        Cndr = get_value(config, "Cndr");

        TC_thr = get_value(config, "TC_thr");
        TC_dE = get_value(config, "TC_dE");
        TC_dR = get_value(config, "TC_dR");
        TC_dA = get_value(config, "TC_dA");
    }

    Scalar dummy{0};
    Scalar g{9.806};
    Scalar rho{1.1589};

    Scalar b{};
    Scalar c{};
    Scalar AR{};
    Scalar S{};

    Scalar mass{};
    Scalar Ixx{};
    Scalar Iyy{};
    Scalar Izz{};
    Scalar Ixz{};

    Scalar e_oswald{};
    Scalar CD0{};

    Scalar CL0{};
    Scalar CLa{};
    Scalar Cm0{};
    Scalar Cma{};

    Scalar CYb{};
    Scalar Cl0{};
    Scalar Clb{};
    Scalar Cn0{};
    Scalar Cnb{};

    Scalar CLq{};
    Scalar Cmq{};

    Scalar CYp{};
    Scalar Clp{};
    Scalar Cnp{};

    Scalar CYr{};
    Scalar Clr{};
    Scalar Cnr{};

    Scalar CLde{};
    Scalar Cmde{};

    Scalar Clda{};
    Scalar Cnda{};

    Scalar CYdr{};
    Scalar Cldr{};
    Scalar Cndr{};

    Scalar TC_thr{};
    Scalar TC_dE{};
    Scalar TC_dR{};
    Scalar TC_dA{};

    // Thrust Scalareters
    Scalar b_thrust_ang{0};

private:
    /* Apply hardcoded Pvw-YR parameters (for debugging) */
    void debug_set_PvwYR_params()
    {
        b = 1.8;
        c = 0.18523;
        AR = 10.016;
        S = 0.32347;

        mass = 1.3474;
        Ixx = 0.0832;
        Iyy = 0.0667;
        Izz = 0.1173;
        Ixz = -0.00215;

        e_oswald = 0.9;
        CD0 = 0.035221;

        CL0 = 0.85305;
        CLa = 5.6602;
        Cm0 = -0.097313;
        Cma = -1.1554;

        CYb = -0.4624;
        Cl0 = 0;
        Clb = -0.08264;
        Cn0 = 0;
        Cnb = 0.070033;

        CLq = 0;
        Cmq = -30.2134;

        CYp = -1.5297e-06;
        Clp = -0.36389;
        Cnp = -0.032347;

        CYr = 1.0802;
        Clr = 0.18488;
        Cnr = -0.067061;

        CLde = 0;
        Cmde = -1.2639;

        Clda = -0.16626;
        Cnda = -8.2532e-08;

        CYdr = 0.39882;
        Cldr = 0.022692;
        Cndr = -0.058704;

        TC_thr = 0.15689;
        TC_dE = 0.02;
        TC_dR = 0.02;
        TC_dA = 0.06;
    }
};

template<typename scalar_t>
scalar_t getThrust(const scalar_t &throttle, const scalar_t &Va)
{
    /* Thrust manifold coefficients */
    const double a = 17.6582;
    const double b = -0.6554;
    const double ts = -0.0266;

    /* Max. RPM(Va) */
    const double RPM_max0 = 6300.0;
    const double RPM_max_per_Va = 900.0 / 22.0;
    scalar_t RPM_max = RPM_max0 + RPM_max_per_Va * Va;
    scalar_t RPM = throttle * RPM_max;
    scalar_t RPM_scaled = RPM / 8000.0;

    const double thrust_offset = 0.0124;
    scalar_t thrust_solius_12x6 = a * (RPM_scaled - ts) * (RPM_scaled - ts) + b * RPM_scaled * Va - thrust_offset;
    scalar_t thrust = thrust_solius_12x6 * 6.3739 / 10.82;
    return thrust;

//    return throttle * throttle * 6.3739; // Solver gives always zero thrust.
//    return throttle * 6.3739; // Solver uses thrust.
}

/** Longitudinal 4-state fixed-wing model -------------------------------------------------------------------------- **/
template<typename Scalar_, int nx_, int nu_ = 0, int np_ = 0, int ny_ = 0>
class DynamicsBase
{
public:
    static const int nx = nx_;
    static const int nu = nu_;
    static const int np = np_;
    static const int ny = ny_;

    /* Templated types */
    template<typename scalar_t> using state_t = Eigen::Matrix<scalar_t, nx, 1>;
    template<typename scalar_t> using control_t = Eigen::Matrix<scalar_t, nu, 1>;
    template<typename scalar_t> using param_t = Eigen::Matrix<scalar_t, np, 1>;
    template<typename scalar_t> using output_t = Eigen::Matrix<scalar_t, ny, 1>;
    template<typename T> using constref_t = const Eigen::Ref<const T>;
    template<typename T> using ref_t = Eigen::Ref<T>;

    /* Scalar types */
    using Scalar = Scalar_;
    using State = state_t<Scalar>;
    using Control = control_t<Scalar>;
    using Param = param_t<Scalar>;
    using Output = output_t<Scalar>;
    using SystemMat = Eigen::Matrix<Scalar, nx, nx>;
    using ControlMat = Eigen::Matrix<Scalar, nx, nu>;

    /* AD (nx+nu) types */
    using Derivates = Eigen::Matrix<Scalar, nx + nu, 1>;
    using AdScalar = Eigen::AutoDiffScalar<Derivates>;
    using AdState = state_t<AdScalar>;
    using AdControl = control_t<AdScalar>;
    using AdParam = param_t<AdScalar>;
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

    virtual void set_state_representation(StateRepresentation stateRep) { state_representation = stateRep; }
    StateRepresentation get_state_representation() const { return state_representation; }
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
    virtual void dynamics(Eigen::Ref<state_t<Scalar>> state_dot,
                          const Eigen::Ref<const state_t<Scalar>> &state,
                          const Eigen::Ref<const control_t<Scalar>> &control,
                          const Eigen::Ref<const param_t<Scalar>> &param,
                          Eigen::Ref<output_t<Scalar>> output) const
    {
        std::cout << "DynamicsBase::dynamics<Scalar>(): Override me!\n";
    }

    virtual void dynamics(Eigen::Ref<state_t<AdScalar>> state_dot,
                          const Eigen::Ref<const state_t<AdScalar>> &state,
                          const Eigen::Ref<const control_t<AdScalar>> &control,
                          const Eigen::Ref<const param_t<AdScalar>> &param,
                          Eigen::Ref<output_t<AdScalar>> output) const
    {
        std::cout << "DynamicsBase::dynamics<AdScalar>(): Override me!\n";
    }

    void linearize_at(Eigen::Matrix<Scalar, nx, nx> &A, Eigen::Matrix<Scalar, nx, nu> &B,
                      const State &state, const Control &control, const Param &param = Param())
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
        Param p = param;
        AdOutput y; // Dummy
        dynamics(x_dot, x0, u0, p, y);

        // Obtain Jacobian and segment it in df_dx (=A) and df_du (=B)
        Eigen::Matrix<Scalar, nx, NXU> df_dxu;
        for (int ix = 0; ix < nx; ++ix)
            df_dxu.row(ix) = x_dot(ix).derivatives();

        A = df_dxu.template block<nx, nx>(0, 0);
        B = df_dxu.template block<nx, nu>(0, nx);
        //std::cout << "A (=df_dx): \n" << A << "\n";
        //std::cout << "B (=df_du): \n" << B << "\n";
    }


    void trim(const Param &param, const Eigen::Matrix<Scalar, nx + nu, 1> &xu0,
              const Control &LBU = {}, const Control &UBU = {},
              const State &LBX = {}, const State &UBX = {}) {}

    State x_trim_ubound;
    State x_trim_lbound;
    Control u_trim_ubound;
    Control u_trim_lbound;
    Control u_physical_ubound;
    Control u_physical_lbound;
protected:
    StateRepresentation state_representation{Undefined};
    State x_default;
    State x_trim;
    Control u_trim;
};

/** Longitudinal 4-state fixed-wing model -------------------------------------------------------------------------- **/
class LonKiteDynamics : public KiteParams,
                        public DynamicsBase<Scalar, 4, 2, 0, 3>
{
public:
    explicit LonKiteDynamics(StateRepresentation stateRep = LongitudinalEulerAoa)
    {
        state_representation = stateRep;
        init();
    }
    void set_state_representation(StateRepresentation stateRep) override
    {
        state_representation = stateRep;
        init();
    }
    void set_static_params(const std::vector<Scalar> &static_params)
    {
        /* Map static parameters from vector to KiteParams params (base class members) */
        rho = static_params[static_parameters::rho];
    }

    template<typename scalar_t>
    void dynamics(ref_t <state_t<scalar_t>> state_dot,
                  constref_t <state_t<scalar_t>> &state,
                  constref_t <control_t<scalar_t>> &control,
                  constref_t <param_t<scalar_t>> &param,
                  ref_t <output_t<scalar_t>> output) const
    {
        using vector3_t = Eigen::Matrix<scalar_t, 3, 1>;
        using vector2_t = Eigen::Matrix<scalar_t, 2, 1>;

        /** State variables **/
        scalar_t Va, alpha, wy, pitch, gamma, vx, vz;
        switch (state_representation)
        {
            case LongitudinalFlightPath:
            {
                Va = state(0);
                alpha = state(1);
                wy = state(2);
                gamma = state(3);
                break;
            }
            case LongitudinalEulerAoa:
            {
                Va = state(0);
                alpha = state(1);
                wy = state(2);
                pitch = state(3);
                break;
            }
            case LongitudinalUW:
            default:
            {
                vx = state(0);
                vz = state(1);
                wy = state(2);
                pitch = state(3);
                break;
            }
        }

        /** Control variables **/
        scalar_t dE = control(0);    // Elevator deflection [positive causing negative pitch movement (nose down)] [rad]
        scalar_t dF = control(1);    // Throttle [-]

        /** ============================================================================================================ **/
        /** Start of model **/
        if (state_representation == LongitudinalUW)
        {
            Va = vector2_t(vx, vz).norm();
            alpha = atan2(vz, vx);
            gamma = pitch - alpha;
        }

        /** ---------------------------------------------------------- **/
        /** Thrust Forces and Moments in body frame **/
        /** ---------------------------------------------------------- **/
        scalar_t thrust = getThrust(dF, Va);
        vector3_t b_F_thrust(thrust * cos(b_thrust_ang),
                             static_cast<scalar_t>(0),
                             thrust * sin(b_thrust_ang));

        vector3_t b_r_thrust(static_cast<scalar_t>(0.25),
                             static_cast<scalar_t>(0),
                             static_cast<scalar_t>(0));

        vector3_t b_M_thrust = b_r_thrust.template cross(b_F_thrust);
        scalar_t M_thrust = b_M_thrust(1);

        /** ---------------------------------------------------------- **/
        /** Aerodynamic Forces and Moments in aerodynamic (wind) frame **/
        /** ---------------------------------------------------------- **/
        scalar_t V0;
        if (FIX_V0) V0 = FIXED_V0;
        else V0 = Va;

        scalar_t dyn_press = 0.5 * rho * Va * Va;

        /** Forces in x, y, z directions: -Drag, Side force, -Lift **/
        scalar_t CL = CL0 + CLa * alpha + CLq * c / (2.0 * V0) * wy + CLde * dE;
        scalar_t CD = CD0 + CL * CL / (M_PI * e_oswald * AR);

        scalar_t LIFT = dyn_press * S * CL;
        scalar_t DRAG = dyn_press * S * CD;

        /** Moments about x, y, z axes: L, M, N **/
        scalar_t M = dyn_press * S * c * (Cm0 + Cma * alpha + c / (2.0 * V0) * Cmq * wy + Cmde * dE);

        /** Dynamics **/
        scalar_t Va_dot, alpha_dot, wy_dot, pitch_dot, gamma_dot, vx_dot, vz_dot;

        switch (state_representation)
        {
            case LongitudinalFlightPath:
            {
                Va_dot = -DRAG / mass + cos(alpha + b_thrust_ang) * thrust / mass - g * sin(gamma);
                gamma_dot = (LIFT / mass + sin(alpha + b_thrust_ang) * thrust / mass - g * cos(gamma)) / Va;
                alpha_dot = wy - gamma_dot;
                wy_dot = (M + M_thrust) / Iyy;

                state_dot << Va_dot, alpha_dot, wy_dot, gamma_dot;
                output << vx, vz, pitch;
                break;
            }
            case LongitudinalEulerAoa:
            {
                Va_dot = -DRAG / mass + cos(alpha + b_thrust_ang) * thrust / mass - g * sin(pitch - alpha);
                alpha_dot =
                        (-LIFT / mass - sin(alpha + b_thrust_ang) * thrust / mass + g * cos(pitch - alpha) + wy * Va) /
                        Va;
                wy_dot = (M + M_thrust) / Iyy;
                pitch_dot = wy;

                state_dot << Va_dot, alpha_dot, wy_dot, pitch_dot;
                output << vx, vz, gamma;
                break;
            }
            case LongitudinalUW:
            default:
            {
                scalar_t X = -cos(alpha) * DRAG + sin(alpha) * LIFT;
                scalar_t Z = -cos(alpha) * LIFT - sin(alpha) * DRAG;
                vx_dot = (X + b_F_thrust(0)) / mass - g * sin(pitch) - wy * vz;
                vz_dot = (Z + b_F_thrust(2)) / mass + g * cos(pitch) + wy * vx;
                wy_dot = (M + M_thrust) / Iyy;
                pitch_dot = wy;

                state_dot << vx_dot, vz_dot, wy_dot, pitch_dot;
                output << Va, alpha, gamma; //, state_dot; // Add dynamics to output for debugging
                break;
            }
        }
        /** End of model **/
        /** ============================================================================================================ **/
    }

    void dynamics(ref_t <State> state_dot,
                  constref_t <State> &state,
                  constref_t <Control> &control,
                  constref_t <Param> &param,
                  ref_t <Output> output) const override
    {
        dynamics<Scalar>(state_dot, state, control, param, output);
    }
    void dynamics(ref_t <AdState> state_dot,
                  constref_t <AdState> &state,
                  constref_t <AdControl> &control,
                  constref_t <AdParam> &param,
                  ref_t <AdOutput> output) const override
    {
        dynamics<AdScalar>(state_dot, state, control, param, output);
    }

private:
    void init()
    {
        // Default state
        x_default << 12, -0.05, -0.05, -0.05; // {Va/vx} {aoa/vz} {wy}, {pitch/gamma})

        // Physical control bounds
        u_physical_ubound << 21.0 * M_PI / 180.0, 1;
        u_physical_lbound << -21.0 * M_PI / 180.0, 0;

        // Trim bounds - state
        // Va
        x_trim_ubound(0) = 15;
        x_trim_lbound(0) = 10;
        // alpha
        x_trim_ubound(1) = 20 * M_PI / 180;
        x_trim_lbound(1) = -20 * M_PI / 180;
        // wy
        x_trim_ubound(2) = 5 * M_PI / 180;
        x_trim_lbound(2) = -5 * M_PI / 180;
        // pitch/gamma
        x_trim_ubound(3) = 20 * M_PI / 180;
        x_trim_lbound(3) = -20 * M_PI / 180;

        if (state_representation == LongitudinalUW)
        {
            // vz
            x_trim_ubound(1) = 5;
            x_trim_lbound(1) = -5;
        }

        // Trim bounds - controls
        // dE
        u_trim_ubound(0) = 30 * M_PI / 180;
        u_trim_lbound(0) = -30 * M_PI / 180;
        // dF
        u_trim_ubound(0) = 0;
        u_trim_lbound(0) = 0;
    }
};

/** Longitudinal 6-state fixed-wing model (4 + 2D airplane position) ----------------------------------------------- **/
class LonKiteDynamicsWithPos : public KiteParams,
                               public DynamicsBase<Scalar, LonKiteDynamics::nx + 2, LonKiteDynamics::nu,
                                       LonKiteDynamics::np, LonKiteDynamics::ny>
{
public:
    LonKiteDynamicsWithPos(const StateRepresentation &stateRep = LongitudinalEulerAoa) :
            lonKiteDynamics(stateRep)
    {
        state_representation = stateRep;
        x_default << lonKiteDynamics.get_default_initial_state(), 0, -100;

        // Trim bounds
        x_trim_ubound << lonKiteDynamics.x_trim_ubound, 0, 0;
        x_trim_lbound << lonKiteDynamics.x_trim_lbound, 0, 0;
        u_trim_ubound = lonKiteDynamics.u_trim_ubound;
        u_trim_lbound = lonKiteDynamics.u_trim_lbound;
    }

    void set_state_representation(StateRepresentation stateRep) override
    {
        lonKiteDynamics.set_state_representation(stateRep);
        state_representation = stateRep;
    }

    void load_params_from_yaml(const std::string &yaml_filepath) override
    {
        lonKiteDynamics.load_params_from_yaml(yaml_filepath);
        KiteParams::load_params_from_yaml(yaml_filepath);
    }
    void set_static_params(const std::vector<Scalar> &static_params)
    {
        lonKiteDynamics.set_static_params(static_params);

        /* Map static parameters from vector to KiteParams params (base class members) */
        //rho = static_params[static_parameters::rho];
        // This class has no static_params
    }

    template<typename scalar_t>
    void dynamics(ref_t <state_t<scalar_t>> state_dot,
                  constref_t <state_t<scalar_t>> &state,
                  constref_t <control_t<scalar_t>> &control,
                  constref_t <param_t<scalar_t>> &param,
                  ref_t <output_t<scalar_t>> output) const
    {
        /** State, Control, Parameter parsing **/
        LonKiteDynamics::state_t<scalar_t> unaug_state = state.template segment<LonKiteDynamics::nx>(0);

        /** ============================================================================================================ **/
        /** Start of model **/
        scalar_t vHor, vDown;

        switch (state_representation)
        {
            case LongitudinalFlightPath:
            {
                scalar_t Va = state(0);
                scalar_t gamma = state(3);
                vHor = Va * cos(gamma);
                vDown = -Va * sin(gamma);
                break;
            }
            case LongitudinalEulerAoa:
            {
                scalar_t Va = state(0);
                scalar_t gamma = state(3) - state(1);
                vHor = Va * cos(gamma);
                vDown = -Va * sin(gamma);
                break;
            }
            case LongitudinalUW:
            default:
            {
                scalar_t vx = state(0);
                scalar_t vz = state(1);
                scalar_t pitch = state(3);
                vHor = vx * cos(pitch) + vz * sin(pitch);
                vDown = -vx * sin(pitch) + vz * cos(pitch);
                break;
            }
        }

        Eigen::Matrix<scalar_t, 2, 1> aug_state_dot;
        aug_state_dot << vHor, vDown;

        /** End of model **/
        /** ============================================================================================================ **/

        /** Evaluate un-augmented dynamics, append augmented dynamics **/
        LonKiteDynamics::state_t<scalar_t> unaug_state_dot;
        lonKiteDynamics.dynamics<scalar_t>(unaug_state_dot, unaug_state, control, param, output);

        state_dot << unaug_state_dot, aug_state_dot;
    }
    void dynamics(ref_t <State> state_dot,
                  constref_t <State> &state,
                  constref_t <Control> &control,
                  constref_t <Param> &param,
                  ref_t <Output> output) const override
    {
        dynamics<Scalar>(state_dot, state, control, param, output);
    }
    void dynamics(ref_t <AdState> state_dot,
                  constref_t <AdState> &state,
                  constref_t <AdControl> &control,
                  constref_t <AdParam> &param,
                  ref_t <AdOutput> output) const override
    {
        dynamics<AdScalar>(state_dot, state, control, param, output);
    }

private:
    LonKiteDynamics lonKiteDynamics;
};

/** Full 13-state kite model --------------------------------------------------------------------------------------- **/
class KiteDynamics : public KiteParams,
                     public DynamicsBase<Scalar, 13, 4, 3, 10>
{
public:
    KiteDynamics()
    {
        // Default state
        x_default << 12, 0, 0, // v(vx vy vz)
                0, 0, 0,  // w(wx wy wz)
                0, 0, -100, // r(N E D)
                1, 0, 0, 0; // q_nb(qw qx qy qz)

        //                Throttle, elev,            rud,                 aileron
        u_physical_ubound << 1, 21.0 * M_PI / 180.0, 26.5 * M_PI / 180.0, 20.5 * M_PI / 180.0;
        u_physical_lbound << 0, -21.0 * M_PI / 180.0, -26.5 * M_PI / 180.0, -20.5 * M_PI / 180.0;
    }

    void set_static_params(const std::vector<Scalar> &static_params)
    {
        /* Map static parameters from vector to KiteParams params (base class members) */
        rho = static_params[static_parameters::rho];
    }

    /* Other static params */
    Eigen::Vector<Scalar, 3> n_vW = {0,0,0};

    template<typename scalar_t>
    void dynamics(Eigen::Ref<state_t < scalar_t>>
    state_dot,
    const Eigen::Ref<const state_t <scalar_t>> &state,
    const Eigen::Ref<const control_t <scalar_t>> &control,
    const Eigen::Ref<const param_t <scalar_t>> &param,
            Eigen::Ref<output_t < scalar_t>>
    output) const
    {
        using quat_t = Eigen::Quaternion<scalar_t>;
        using vector4_t = Eigen::Matrix<scalar_t, 4, 1>;
        using vector3_t = Eigen::Matrix<scalar_t, 3, 1>;

        /* Aircraft Inertia Matrix */
        Eigen::Matrix<scalar_t, 3, 3> J;
        J.setZero();
        J.diagonal() << Ixx, Iyy, Izz;
        J(0, 2) = Ixz;
        J(2, 0) = Ixz;

        /** State, Control, Parameter parsing **/
//        scalar_t vx = state(0);
//        scalar_t vy = state(1);
//        scalar_t vz = state(2);
//        vector3_t v(vx, vy, vz);
        vector3_t v(state.template segment<3>(0));

//        scalar_t wx = state(3);
//        scalar_t wy = state(4);
//        scalar_t wz = state(5);
//        vector3_t w(wx, wy, wz);
        vector3_t w(state.template segment<3>(3));

        scalar_t qw = state(9);
        scalar_t qx = state(10);
        scalar_t qy = state(11);
        scalar_t qz = state(12);
        quat_t q(qw, qx, qy, qz); // = q_nb // Quaternion initialization, not just segment!
        quat_t q_bn = q.inverse();

        scalar_t dF = control(0);
        scalar_t dE = control(1);
        scalar_t dR = control(2);
        scalar_t dA = control(3);

        /* Parameters */
//        scalar_t vW_N = dyn_params(0);
//        scalar_t vW_E = dyn_params(1);
//        scalar_t vW_D = dyn_params(2);
//        vector3_t n_vW(vW_N, vW_E, vW_D);
//        vector3_t n_vW(dyn_params.template segment<3>(0));

        /** ============================================================================================================ **/
        /** Start of model **/
        vector3_t b_vW = q_bn * n_vW;
        vector3_t b_va = v - b_vW;

        /* Aerodynamic variables (Airspeed, angle of attack, side slip angle) */
        scalar_t Va = b_va.norm();
        scalar_t alpha = (b_va(0) > 0.0) ? atan(b_va(2) / b_va(0)) : M_PI / 2.0;
        scalar_t beta = (Va > 0.0) ? asin(b_va(1) / Va) : 0.0;
        //std::cout << "alpha / " <<  b_va(2) / b_va(0) << ", beta / " << b_va(1) / Va << "\n";

        /* Measured airspeed component (pitot tube orientation dependent) */
        vector3_t r_sens(0.11, 0.22, -0.05);
        vector3_t b_va_meas = b_va + r_sens.cross(w);
        /* At fast body yawrates, the body rotation (thus pitot tube is faster than the CoG) is clearly negligible
         * under the effect of the pitot measurement direction being rotated out of the airflow (sideslip) */

        /* Pitot tube is oriented about 5 degrees above body x axes */
        quat_t q_sens_b;
        q_sens_b = Eigen::AngleAxis<scalar_t>(-5.0 * M_PI / 180.0, vector3_t(0, 1, 0));
        vector3_t sens_va = q_sens_b * b_va_meas;
        scalar_t Va_pitot = sens_va(0);

        /** ---------------------------------------------------------- **/
        /** Thrust Forces and Moments in body frame **/
        /** ---------------------------------------------------------- **/
        scalar_t b_thrust_ang = 0;
        scalar_t thrust = getThrust(dF, Va);

        vector3_t b_F_thrust(cos(b_thrust_ang), 0, sin(b_thrust_ang));
        b_F_thrust *= thrust;

        vector3_t b_r_thrust(0.25, 0, 0);
        vector3_t b_M_thrust = b_r_thrust.cross(b_F_thrust);

        /** ---------------------------------------------------------- **/
        /** Aerodynamic Forces and Moments in aerodynamic (wind) frame **/
        /** ---------------------------------------------------------- **/
        Eigen::Quaternion<scalar_t> q_ba = T2quat<scalar_t, scalar_t>(alpha) * T3quat<scalar_t, scalar_t>(-beta);

        scalar_t V0 = (FIX_V0) ? FIXED_V0 : Va;
        //if (V0 < 0.1) V0 = 0.1;

        scalar_t dyn_press = 0.5 * rho * Va * Va;
        scalar_t CL = CL0 + CLa * alpha + CLq * c / (2.0 * V0) * w(1) + CLde * dE;
        scalar_t CD = CD0 + CL * CL / (M_PI * e_oswald * AR);

        /** Forces in x, y, z directions: -Drag, Side force, -Lift **/
        scalar_t LIFT = dyn_press * S * CL;
        scalar_t DRAG = dyn_press * S * CD;
        scalar_t SF = dyn_press * S * (CYb * beta + b / (2.0 * V0) * (CYp * w(0) + CYr * w(2)) + CYdr * dR);

        vector3_t a_F_aero(-DRAG, SF, -LIFT);

        /** Moments about x, y, z axes: L, M, N **/
        scalar_t L = dyn_press * S * b *
                     (Cl0 + Clb * beta + b / (2.0 * V0) * (Clp * w(0) + Clr * w(2)) + Clda * dA + Cldr * dR);

        scalar_t M = dyn_press * S * c *
                     (Cm0 + Cma * alpha + c / (2.0 * V0) * Cmq * w(1) + Cmde * dE);

        scalar_t N = dyn_press * S * b *
                     (Cn0 + Cnb * beta + b / (2.0 * V0) * (Cnp * w(0) + Cnr * w(2)) + Cnda * dA + Cndr * dR);

        vector3_t b_M_aero(L, M, N);

        /** Aerodynamic Forces and Moments in body frame **/
        vector3_t b_F_aero = q_ba * a_F_aero;

        /** ---------------------------------------- **/
        /** Gravitation, Tether (body frame) **/
        /** ---------------------------------------- **/
        /** Gravitational acceleration **/
        vector3_t b_g = q_bn * vector3_t(0, 0, g);

        /** Tether force and moment **/
        vector3_t b_F_tether(0, 0, 0);

        vector3_t tether_outlet_position(0, 0, 0);
        vector3_t b_M_tether = tether_outlet_position.cross(b_F_tether);

        /** ----------------------------- **/
        /** Motion equations (body frame) **/
        /** ----------------------------- **/
        /** Linear motion equation **/
        vector3_t spec_nongrav_force = (b_F_aero + b_F_thrust + b_F_tether) / mass;
        vector3_t v_dot = spec_nongrav_force + b_g - w.cross(v);

        /** Angular motion equation **/
        vector3_t w_dot = J.inverse() * (b_M_aero + b_M_thrust + b_M_tether - w.cross(J * w));

        /** ------------------------------------ **/
        /** Kinematic Equations (geodetic frame) **/
        /** ------------------------------------ **/
        /** Translation: Aircraft position derivative **/
        vector3_t pos_dot = q * v;                            // q = q_nb, v (body frame)

        /** Rotation: Aircraft attitude derivative **/
        /* Quaternion representation */
        double lambda = -5;
        quat_t tmp = q * quat_t(0, w(0), w(1), w(2));
        vector4_t q_dot = 0.5 * vector4_t(tmp.w(), tmp.x(), tmp.y(), tmp.z()) // q = q_nb, w = omega (body frame)
                          + 0.5 * lambda * vector4_t(q.w(), q.x(), q.y(), q.z()) * (q.dot(q) - 1); // Quaternion norm
        // stabilization term, as in Gros: 'Baumgarte Stabilisation over the SO(3) Rotation Group for Control',
        // improved: lambda negative and SX::dot(q, q) instead of lambda positive and 1/SX::dot(q, q).

        /** End of model **/
        /** ============================================================================================================ **/

        state_dot << v_dot, w_dot, pos_dot, q_dot;

        /** Additional output mappings **/
        vector3_t spec_tether_force = b_F_tether / mass;

        output << Va_pitot, Va, alpha, beta, spec_nongrav_force, spec_tether_force;
    }
private:
};

/** Full 17-state kite model (13 + 4 actuator states) -------------------------------------------------------------- **/
class AugKiteDynamics : public KiteParams,
                        public DynamicsBase<Scalar, KiteDynamics::nx + 4, KiteDynamics::nu,
                                KiteDynamics::np, KiteDynamics::ny>
{
public:
    AugKiteDynamics()
    {
        x_default << kiteDynamics.get_default_initial_state(), 0, 0, 0, 0;

        u_physical_ubound = kiteDynamics.u_physical_ubound;
        u_physical_lbound = kiteDynamics.u_physical_lbound;
    }

    void load_params_from_yaml(const std::string &yaml_filepath) override
    {
        kiteDynamics.load_params_from_yaml(yaml_filepath);
        KiteParams::load_params_from_yaml(yaml_filepath); // Both necessary!
    }
    void set_static_params(const std::vector<Scalar> &static_params)
    {
        kiteDynamics.set_static_params(static_params);

        /* Map static parameters from vector to KiteParams params (base class members) */
        //rho = static_params[static_parameters::rho];
        // This class has no static_params
    }

    template<typename scalar_t>
    void dynamics(Eigen::Ref<state_t < scalar_t>>
    state_dot,
    const Eigen::Ref<const state_t <scalar_t>> &state,
    const Eigen::Ref<const control_t <scalar_t>> &control,
    const Eigen::Ref<const param_t <scalar_t>> &param,
            Eigen::Ref<output_t < scalar_t>>
    output) const
    {
        /** State, Control, Parameter parsing **/
        KiteDynamics::state_t<scalar_t> unaug_state = state.template segment<KiteDynamics::nx>(0);

        constexpr int nx_aug = nx - KiteDynamics::nx;
        Eigen::Matrix<scalar_t, nx_aug, 1> aug_state = state.template segment<nx_aug>(KiteDynamics::nx);

        scalar_t dF = aug_state(0);
        scalar_t dE = aug_state(1);
        scalar_t dR = aug_state(2);
        scalar_t dA = aug_state(3);

        scalar_t dF_cmd = control(0);
        scalar_t dE_cmd = control(1);
        scalar_t dR_cmd = control(2);
        scalar_t dA_cmd = control(3);

        /** ============================================================================================================ **/
        /** Start of model **/
        /* Actuator dynamics */
        scalar_t dF_dot = (dF_cmd - dF) / TC_thr;
        scalar_t dE_dot = (dE_cmd - dE) / TC_dE;
        scalar_t dR_dot = (dR_cmd - dR) / TC_dR;
        scalar_t dA_dot = (dA_cmd - dA) / TC_dA;

        Eigen::Matrix<scalar_t, nx_aug, 1> actuator_pos_dot;
        actuator_pos_dot << dF_dot, dE_dot, dR_dot, dA_dot;

        /** End of model **/
        /** ============================================================================================================ **/

        /** Evaluate un-augmented dynamics, append augmented dynamics **/
        KiteDynamics::state_t<scalar_t> unaug_state_dot;
        kiteDynamics.template dynamics<scalar_t>(unaug_state_dot, unaug_state, aug_state, param, output);

        state_dot << unaug_state_dot, actuator_pos_dot;
    }

private:
    KiteDynamics kiteDynamics;
};

}

}

#endif // KITE_HPP
