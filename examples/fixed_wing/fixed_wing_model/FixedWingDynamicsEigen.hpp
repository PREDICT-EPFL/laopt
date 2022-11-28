#ifndef SRC_FIXEDWINGDYNAMICSEIGEN_HPP
#define SRC_FIXEDWINGDYNAMICSEIGEN_HPP

#include <Eigen/Geometry>

#include "flight_model_utils.hpp"
#include "DynamicsBaseEigen.hpp"
#include "ParamsBaseEigen.hpp"
#include "QuaternionMathEigen.hpp"

namespace flight_model {
namespace eigen_model {
namespace fixed_wing {

using Scalar = double;

struct BasicFixedWingParams : public ParamsBase<Scalar>
{
    Param dummy{0};
    Param g{9.806};
    Param rho{1.225};

    Param b{};
    Param c{};
    Param AR{};
    Param S{};

    Param mass{};
    Param Ixx{};
    Param Iyy{};
    Param Izz{};
    Param Ixz{};

    Param turb_ref_length{};

    void debug_set_PvwYR_params()
    {
        b = 1.8;
        c = 0.18523;
        AR = 10.016;
        S = 0.32347;

        mass = 1.3474;
    }
    void load_params_from_yaml(const std::string &yaml_filepath)
    {
        const YAML::Node yamlNode = YAML::LoadFile(yaml_filepath);
        b = get_value(yamlNode, "b");
        c = get_value(yamlNode, "c");
        AR = get_value(yamlNode, "AR");
        S = get_value(yamlNode, "S");

        mass = get_value(yamlNode, "mass");
        Ixx = get_value(yamlNode, "Ixx");
        Iyy = get_value(yamlNode, "Iyy");
        Izz = get_value(yamlNode, "Izz");
        Ixz = get_value(yamlNode, "Ixz");

        turb_ref_length = get_value(yamlNode, "turb_ref_length");
    }
};
struct LonFixedWingAeroParams : public ParamsBase<Scalar>
{
    Param e_oswald{};
    Param CD0{};

    Param CL0{};
    Param CLa{};
    Param Cm0{};
    Param Cma{};

    Param CLq{};
    Param Cmq{};

    Param CLde{};
    Param Cmde{};

    Param TC_thr{};
    Param TC_dE{};

    // Thrust parameters
    Param b_thrust_ang{0};

    /* Apply hardcoded Pvw-YR parameters (for debugging) */
    void debug_set_PvwYR_params()
    {
        e_oswald = 0.9;
        CD0 = 0.035221;

        CL0 = 0.85305;
        CLa = 5.6602;
        Cm0 = -0.097313;
        Cma = -1.1554;

        CLq = 0;
        Cmq = -30.2134;

        CLde = 0;
        Cmde = -1.2639;

        TC_thr = 0.15689;
        TC_dE = 0.02;
    }
    void load_params_from_yaml(const std::string &yaml_filepath)
    {
        const YAML::Node yamlNode = YAML::LoadFile(yaml_filepath);

        e_oswald = get_value(yamlNode, "e_oswald");
        CD0 = get_value(yamlNode, "CD0");

        CL0 = get_value(yamlNode, "CL0");
        CLa = get_value(yamlNode, "CLa");
        Cm0 = get_value(yamlNode, "Cm0");
        Cma = get_value(yamlNode, "Cma");

        CLq = get_value(yamlNode, "CLq");
        Cmq = get_value(yamlNode, "Cmq");

        CLde = get_value(yamlNode, "CLde");
        Cmde = get_value(yamlNode, "Cmde");

        TC_thr = get_value(yamlNode, "TC_thr");
        TC_dE = get_value(yamlNode, "TC_dE");
    }
};
struct LatFixedWingAeroParams : public ParamsBase<Scalar>
{
    Param CYb{};
    Param Cl0{};
    Param Clb{};
    Param Cn0{};
    Param Cnb{};

    Param CYp{};
    Param Clp{};
    Param Cnp{};

    Param CYr{};
    Param Clr{};
    Param Cnr{};

    Param Clda{};
    Param Cnda{};

    Param CYdr{};
    Param Cldr{};
    Param Cndr{};

    Param TC_dR{};
    Param TC_dA{};

    /* Apply hardcoded Pvw-YR parameters (for debugging) */
    void debug_set_PvwYR_params()
    {
        CYb = -0.4624;
        Cl0 = 0;
        Clb = -0.08264;
        Cn0 = 0;
        Cnb = 0.070033;

        CYp = -1.5297e-06;
        Clp = -0.36389;
        Cnp = -0.032347;

        CYr = 1.0802;
        Clr = 0.18488;
        Cnr = -0.067061;

        Clda = -0.16626;
        Cnda = -8.2532e-08;

        CYdr = 0.39882;
        Cldr = 0.022692;
        Cndr = -0.058704;

        TC_dR = 0.02;
        TC_dA = 0.06;
    }
    void load_params_from_yaml(const std::string &yaml_filepath)
    {
        const YAML::Node yamlNode = YAML::LoadFile(yaml_filepath);

        CYb = get_value(yamlNode, "CYb");
        Cl0 = get_value(yamlNode, "Cl0");
        Clb = get_value(yamlNode, "Clb");
        Cn0 = get_value(yamlNode, "Cn0");
        Cnb = get_value(yamlNode, "Cnb");

        CYp = get_value(yamlNode, "CYp");
        Clp = get_value(yamlNode, "Clp");
        Cnp = get_value(yamlNode, "Cnp");

        CYr = get_value(yamlNode, "CYr");
        Clr = get_value(yamlNode, "Clr");
        Cnr = get_value(yamlNode, "Cnr");

        Clda = get_value(yamlNode, "Clda");
        Cnda = get_value(yamlNode, "Cnda");

        CYdr = get_value(yamlNode, "CYdr");
        Cldr = get_value(yamlNode, "Cldr");
        Cndr = get_value(yamlNode, "Cndr");

        TC_dR = get_value(yamlNode, "TC_dR");
        TC_dA = get_value(yamlNode, "TC_dA");
    }
};

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

constexpr bool FIX_V0 = false; // In all models: Normalize rates with V0 instead of Va
constexpr double FIXED_V0 = 12;

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
struct LonFixedWingParams : public BasicFixedWingParams, LonFixedWingAeroParams
{
    void load_params_from_yaml(const std::string &yaml_filepath)
    {
        BasicFixedWingParams::load_params_from_yaml(yaml_filepath);
        LonFixedWingAeroParams::load_params_from_yaml(yaml_filepath);
    }
};
class LonFixedWingDynamics : public LonFixedWingParams,
                             public DynamicsBase<Scalar, 4, 2, 0, 3>
{
public:
    explicit LonFixedWingDynamics(flight_model::StateRepresentation stateRep = LongitudinalEulerAoa)
    {
        state_representation = stateRep;
        init();
    }
    StateRepresentation get_state_representation() const { return state_representation; }
    void set_state_representation(StateRepresentation stateRep)
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
                  constref_t <DynamicParams> &dyn_params,
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
                  constref_t <DynamicParams> &dyn_params,
                  ref_t <Output> output) const override
    {
        dynamics<Scalar>(state_dot, state, control, dyn_params, output);
    }
    void dynamics(ref_t <AdState> state_dot,
                  constref_t <AdState> &state,
                  constref_t <AdControl> &control,
                  constref_t <DynamicParams> &dyn_params,
                  ref_t <AdOutput> output) const override
    {
        dynamics<AdScalar>(state_dot, state, control, dyn_params, output);
    }

protected:
    StateRepresentation state_representation{flight_model::StateRepresentation::Undefined};
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
class LonFixedWingDynamicsWithPos : public LonFixedWingParams,
                                    public DynamicsBase<Scalar,
                                            LonFixedWingDynamics::nx + 2, LonFixedWingDynamics::nu,
                                            LonFixedWingDynamics::nd, LonFixedWingDynamics::ny>
{
public:
    LonFixedWingDynamicsWithPos(const StateRepresentation &stateRep = LongitudinalEulerAoa) :
            lonFixedWingDynamics(stateRep)
    {
        state_representation = stateRep;
        x_default << lonFixedWingDynamics.get_default_initial_state(), 0, -100;

        // Trim bounds
        x_trim_ubound << lonFixedWingDynamics.x_trim_ubound, 0, 0;
        x_trim_lbound << lonFixedWingDynamics.x_trim_lbound, 0, 0;
        u_trim_ubound = lonFixedWingDynamics.u_trim_ubound;
        u_trim_lbound = lonFixedWingDynamics.u_trim_lbound;
    }

    void set_state_representation(StateRepresentation stateRep)
    {
        lonFixedWingDynamics.set_state_representation(stateRep);
        state_representation = stateRep;
    }

    void load_params_from_yaml(const std::string &yaml_filepath)
    {
        LonFixedWingParams::load_params_from_yaml(yaml_filepath);
        // Need to update params in member instance, too
        lonFixedWingDynamics.load_params_from_yaml(yaml_filepath);
    }
    void set_static_params(const std::vector<Scalar> &static_params)
    {
        lonFixedWingDynamics.set_static_params(static_params);

        /* Map static parameters from vector to KiteParams params (base class members) */
        //rho = static_params[static_parameters::rho];
        // This class has no static_params
    }

    template<typename scalar_t>
    void dynamics(ref_t <state_t<scalar_t>> state_dot,
                  constref_t <state_t<scalar_t>> &state,
                  constref_t <control_t<scalar_t>> &control,
                  constref_t <DynamicParams> &dyn_params,
                  ref_t <output_t<scalar_t>> output) const
    {
        /** State, Control, Parameter parsing **/
        LonFixedWingDynamics::state_t<scalar_t> unaug_state = state.template segment<LonFixedWingDynamics::nx>(0);

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
        LonFixedWingDynamics::state_t<scalar_t> unaug_state_dot;
        lonFixedWingDynamics.dynamics<scalar_t>(unaug_state_dot, unaug_state, control, dyn_params, output);

        state_dot << unaug_state_dot, aug_state_dot;
    }
    void dynamics(ref_t <State> state_dot,
                  constref_t <State> &state,
                  constref_t <Control> &control,
                  constref_t <DynamicParams> &dyn_params,
                  ref_t <Output> output) const override
    {
        dynamics<Scalar>(state_dot, state, control, dyn_params, output);
    }
    void dynamics(ref_t <AdState> state_dot,
                  constref_t <AdState> &state,
                  constref_t <AdControl> &control,
                  constref_t <DynamicParams> &dyn_params,
                  ref_t <AdOutput> output) const override
    {
        dynamics<AdScalar>(state_dot, state, control, dyn_params, output);
    }

protected:
    StateRepresentation state_representation{flight_model::StateRepresentation::Undefined};
private:
    LonFixedWingDynamics lonFixedWingDynamics;
};

/** Full 13-state kite model --------------------------------------------------------------------------------------- **/
struct FixedWingParams : public BasicFixedWingParams, LonFixedWingAeroParams, LatFixedWingAeroParams
{
    void load_params_from_yaml(const std::string &yaml_filepath)
    {
        BasicFixedWingParams::load_params_from_yaml(yaml_filepath);
        LonFixedWingAeroParams::load_params_from_yaml(yaml_filepath);
        LatFixedWingAeroParams::load_params_from_yaml(yaml_filepath);
    }
};
class FixedWingDynamics : public FixedWingParams,
                          public DynamicsBase<Scalar, 13, 4, 3, 10>
{
public:
    FixedWingDynamics()
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

    template<typename scalar_t>
    void dynamics(ref_t <state_t<scalar_t>> state_dot,
                  constref_t <state_t<scalar_t>> &state,
                  constref_t <control_t<scalar_t>> &control,
                  constref_t <DynamicParams> &dyn_params,
                  ref_t <output_t<scalar_t>> output) const
    {
        using quat_t = Eigen::Quaternion<scalar_t>;
        using vec4_t = Eigen::Vector<scalar_t, 4>;
        using vec3_t = Eigen::Vector<scalar_t, 3>;
        using Quat = Eigen::Quaternion<Scalar>;
        using Vec4 = Eigen::Vector<Scalar, 4>;
        using Vec3 = Eigen::Vector<Scalar, 3>;

        /* Aircraft Inertia Matrix */
        Eigen::Matrix<Scalar, 3, 3> J;
        J.setZero();
        J.diagonal() << Ixx, Iyy, Izz;
        J(0, 2) = Ixz;
        J(2, 0) = Ixz;

        /** State, Control, Parameter parsing **/
//        scalar_t vx = state(0);
//        scalar_t vy = state(1);
//        scalar_t vz = state(2);
//        vec3_t v(vx, vy, vz);
        vec3_t v(state.template segment<3>(0));

//        scalar_t wx = state(3);
//        scalar_t wy = state(4);
//        scalar_t wz = state(5);
//        vec3_t w(wx, wy, wz);
        vec3_t w(state.template segment<3>(3));

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
//        vec3_t n_vW(vW_N, vW_E, vW_D);
        vec3_t n_vW(dyn_params.template segment<3>(0));

        /** ============================================================================================================ **/
        /** Start of model **/
        vec3_t b_vW = q_bn * n_vW;
        vec3_t b_va = v - b_vW;

        /* Aerodynamic variables (Airspeed, angle of attack, side slip angle) */
        scalar_t Va = b_va.norm();
        scalar_t alpha = (b_va(0) > 0.0) ? static_cast<scalar_t>(atan2(b_va(2), b_va(0))) : static_cast<scalar_t>(M_PI / 2.0);
        scalar_t beta = (Va > 0.0) ? static_cast<scalar_t>(asin(b_va(1) / Va)) : static_cast<scalar_t>(0.0);
        //std::cout << "alpha / " <<  b_va(2) / b_va(0) << ", beta / " << b_va(1) / Va << "\n";

        /* Measured airspeed component (pitot tube orientation dependent) */
        Vec3 r_sens(0.11, 0.22, -0.05); //TODO: This cannot be const
        vec3_t b_va_meas = b_va + r_sens.cross(w);
        /* At fast body yawrates, the body rotation (thus pitot tube is faster than the CoG) is clearly negligible
         * under the effect of the pitot measurement direction being rotated out of the airflow (sideslip) */

        /* Pitot tube is oriented about 5 degrees above body x axes */
        Quat q_sens_b;
        q_sens_b = Eigen::AngleAxis<Scalar>(-5.0 * M_PI / 180.0, Vec3(0, 1, 0));
        vec3_t sens_va = static_cast<quat_t>(q_sens_b) * b_va_meas;
        scalar_t Va_pitot = sens_va(0);

        /** ---------------------------------------------------------- **/
        /** Thrust Forces and Moments in body frame **/
        /** ---------------------------------------------------------- **/
        const Scalar b_thrust_ang = 0;
        scalar_t thrust = getThrust(dF, Va);

        const Vec3 b_eF_thrust(cos(b_thrust_ang), 0, sin(b_thrust_ang));
        vec3_t b_F_thrust = thrust * b_eF_thrust;

        const Vec3 b_r_thrust(0.25, 0, 0);
        vec3_t b_M_thrust = b_r_thrust.cross(b_F_thrust);

        /** ---------------------------------------------------------- **/
        /** Aerodynamic Forces and Moments in aerodynamic (wind) frame **/
        /** ---------------------------------------------------------- **/
        Eigen::Quaternion<scalar_t> q_ba = quatmath::T2quat<scalar_t>(alpha) * quatmath::T3quat<scalar_t>(-beta);

        scalar_t V0 = (FIX_V0) ? static_cast<scalar_t>(FIXED_V0) : Va;
        //if (V0 < 0.1) V0 = 0.1;

        scalar_t dyn_press = 0.5 * rho * Va * Va;

        vec3_t a_F_aero, b_M_aero;
        if (use_simple_aerodynamics)
        {
            scalar_t CL = CL0 + CLa * alpha;
            scalar_t CD = CD0 + CL * CL / (M_PI * e_oswald * AR);

            /** Forces in x, y, z directions: -Drag, Side force, -Lift **/
            scalar_t LIFT = dyn_press * S * CL;
            scalar_t DRAG = dyn_press * S * CD;
            scalar_t SF = dyn_press * S * CYb * beta;
            a_F_aero << -DRAG, SF, -LIFT;

            /** Moments about x, y, z axes: L, M, N **/
            scalar_t L = dyn_press * S * b * (Clb * beta + Clda * dA);
            scalar_t M = dyn_press * S * c * (Cm0 + Cma * alpha + Cmde * dE);
            scalar_t N = dyn_press * S * b * (Cnb * beta + Cndr * dR);
            b_M_aero << L, M, N;
        }
        else
        {
            scalar_t CL = CL0 + CLa * alpha + CLq * c / (2.0 * V0) * w(1) + CLde * dE;
            scalar_t CD = CD0 + CL * CL / (M_PI * e_oswald * AR);

            /** Forces in x, y, z directions: -Drag, Side force, -Lift **/
            scalar_t LIFT = dyn_press * S * CL;
            scalar_t DRAG = dyn_press * S * CD;
            scalar_t SF = dyn_press * S * (CYb * beta + b / (2.0 * V0) * (CYp * w(0) + CYr * w(2)) + CYdr * dR);

            a_F_aero << -DRAG, SF, -LIFT;

            /** Moments about x, y, z axes: L, M, N **/
            scalar_t L = dyn_press * S * b *
                         (Cl0 + Clb * beta + b / (2.0 * V0) * (Clp * w(0) + Clr * w(2)) + Clda * dA + Cldr * dR);

            scalar_t M = dyn_press * S * c *
                         (Cm0 + Cma * alpha + c / (2.0 * V0) * Cmq * w(1) + Cmde * dE);

            scalar_t N = dyn_press * S * b *
                         (Cn0 + Cnb * beta + b / (2.0 * V0) * (Cnp * w(0) + Cnr * w(2)) + Cnda * dA + Cndr * dR);

            b_M_aero << L, M, N;
        }

        /** Aerodynamic Forces and Moments in body frame **/
        vec3_t b_F_aero = q_ba * a_F_aero;

        /** ---------------------------------------- **/
        /** Gravitation, Tether (body frame) **/
        /** ---------------------------------------- **/
        /** Gravitational acceleration **/
        vec3_t b_g = q_bn * Vec3(0, 0, g);

        /** Tether force and moment **/
        Vec3 b_F_tether(0, 0, 0);

        const Vec3 tether_outlet_position(0, 0, 0);
        vec3_t b_M_tether = tether_outlet_position.cross(b_F_tether);

        /** ----------------------------- **/
        /** Motion equations (body frame) **/
        /** ----------------------------- **/
        /** Linear motion equation **/
        vec3_t spec_nongrav_force = (b_F_aero + b_F_thrust + b_F_tether) / mass;
        vec3_t v_dot = spec_nongrav_force + b_g - w.cross(v);

        /** Angular motion equation **/
        vec3_t w_dot = J.inverse() * (b_M_aero + b_M_thrust + b_M_tether - w.cross(J * w));

        /** ------------------------------------ **/
        /** Kinematic Equations (geodetic frame) **/
        /** ------------------------------------ **/
        /** Translation: Aircraft position derivative **/
        vec3_t pos_dot = q * v;                            // q = q_nb, v (body frame)

        /** Rotation: Aircraft attitude derivative **/
        /* Quaternion representation */
        double lambda = -5;
        quat_t tmp = q * quat_t(static_cast<scalar_t>(0), w(0), w(1), w(2));
        vec4_t q_dot = 0.5 * vec4_t(tmp.w(), tmp.x(), tmp.y(), tmp.z()) // q = q_nb, w = omega (body frame)
                          + 0.5 * lambda * vec4_t(q.w(), q.x(), q.y(), q.z()) * (q.dot(q) - 1); // Quaternion norm
        // stabilization term, as in Gros: 'Baumgarte Stabilisation over the SO(3) Rotation Group for Control',
        // improved: lambda negative and SX::dot(q, q) instead of lambda positive and 1/SX::dot(q, q).

        /** End of model **/
        /** ============================================================================================================ **/

        state_dot << v_dot, w_dot, pos_dot, q_dot;

        /** Additional output mappings **/
        vec3_t spec_tether_force = b_F_tether / mass;

        output << Va_pitot, Va, alpha, beta, spec_nongrav_force, spec_tether_force;
    }

    bool use_simple_aerodynamics{false};
private:
};

/** Full 17-state kite model (13 + 4 actuator states) -------------------------------------------------------------- **/
class AugFixedWingDynamics : public FixedWingParams,
                             public DynamicsBase<Scalar, FixedWingDynamics::nx + 4, FixedWingDynamics::nu,
                                     FixedWingDynamics::nd, FixedWingDynamics::ny>
{
public:
    AugFixedWingDynamics()
    {
        x_default << fixedWingDynamics.get_default_initial_state(), 0, 0, 0, 0;

        u_physical_ubound = fixedWingDynamics.u_physical_ubound;
        u_physical_lbound = fixedWingDynamics.u_physical_lbound;
    }

    void load_params_from_yaml(const std::string &yaml_filepath)
    {
        FixedWingParams::load_params_from_yaml(yaml_filepath);
        // Need to update params in member instance, too
        fixedWingDynamics.load_params_from_yaml(yaml_filepath);
    }
    void set_static_params(const std::vector<Scalar> &static_params)
    {
        fixedWingDynamics.set_static_params(static_params);

        /* Map static parameters from vector to KiteParams params (base class members) */
        //rho = static_params[static_parameters::rho];
        // This class has no static_params
    }

    template<typename scalar_t>
    void dynamics(ref_t <state_t<scalar_t>> state_dot,
                  constref_t <state_t<scalar_t>> &state,
                  constref_t <control_t<scalar_t>> &control,
                  constref_t <DynamicParams> &dyn_params,
                  ref_t <output_t<scalar_t>> output) const
    {
        /** State, Control, Parameter parsing **/
        FixedWingDynamics::state_t<scalar_t> unaug_state = state.template segment<FixedWingDynamics::nx>(0);

        constexpr int nx_aug = nx - FixedWingDynamics::nx;
        Eigen::Matrix<scalar_t, nx_aug, 1> aug_state = state.template segment<nx_aug>(FixedWingDynamics::nx);

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
        FixedWingDynamics::state_t<scalar_t> unaug_state_dot;
        fixedWingDynamics.template dynamics<scalar_t>(unaug_state_dot, unaug_state, aug_state, dyn_params, output);

        state_dot << unaug_state_dot, actuator_pos_dot;
    }

protected:
    StateRepresentation state_representation{flight_model::StateRepresentation::Undefined};
private:
    FixedWingDynamics fixedWingDynamics;
};

} //namespace fixed_wing
} //namespace eigen_model
} //namespace flight_model

#endif //SRC_FIXEDWINGDYNAMICSEIGEN_HPP
