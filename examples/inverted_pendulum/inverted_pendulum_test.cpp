#include <iostream>
#include <chrono>

#include "laopt/laopt.hpp"

#include "inverted_pendulum_ocp.hpp"
#include "laopt/tools/multiple_shooting.hpp"
#include "laopt/tools/radau_collocation.hpp"

#include "examples_helper.hpp"

int main()
{
    /** Select variants to build **/
    constexpr bool with_eigen_AD = true;
    constexpr bool with_casadi_AD = false;
    constexpr bool with_multiple_shooting = false;
    constexpr bool with_radau_collocation = true;
    constexpr bool with_ipopt = false;
    constexpr bool with_sqp_piqp = true;

    /* Choose OCP and Transcription */
    using Ocp = InvertedPendulumOcp;

    /* Construct OCP and set OCP-specific properties */
    std::shared_ptr<Ocp> ocp = std::make_shared<Ocp>();

    ocp->t0 = 0;

    ocp->tf_ub = ocp->t0 + 2;
    ocp->tf_lb = ocp->t0 + 1.5;
    ocp->w_tf = 3;

    ocp->angle_ref = 0.0 * M_PI / 180.0;
    // ref_offset, us
    ocp->p_ub << 1, 1;
    ocp->p_lb << 0, -1;

    ocp->u_ub << 3;
    ocp->u_lb << -3;

    ocp->set_x0({M_PI, 0});

    auto print_ocp_opt = [&](auto& transcription)
    {
        Ocp::Param opt_params = transcription->get_p_opt();
        std::cout << "opt_params: ref_offset: " << opt_params(0) << ", us: " << opt_params(1) << "\n\n";
    };

    /* {Autodiff - Transcription - Solver} combinations */
    const int N = 20;

    const int D_poly = 4;
    const int N_segs = 3;

    using namespace laopt_examples;

    if (with_eigen_AD)
    {
        std::cout << "---------------------------\n"
                     "Eigen autodiff (default)\n"
                     "---------------------------\n";

        if (with_multiple_shooting)
        {
            std::cout << "> Multiple shooting\n";
            using MS_Eigen = laopt_tools::MultipleShooting<Ocp, N, laopt::ERK4>;
            if (with_ipopt)
            {
                auto transcription = run_ipopt<MS_Eigen>(ocp);
                print_ocp_opt(transcription);
            }
            if (with_sqp_piqp)
            {
                auto transcription = run_sqp_piqp<MS_Eigen>(ocp);
                print_ocp_opt(transcription);
            }
        }
        if (with_radau_collocation)
        {
            std::cout << "> Radau Collocation\n";
            using Radau_Eigen = laopt_tools::RadauCollocation<Ocp, N_segs, D_poly>;
            if (with_ipopt)
            {
                auto transcription = run_ipopt<Radau_Eigen>(ocp);
                print_ocp_opt(transcription);
            }
            if (with_sqp_piqp)
            {
                auto transcription = run_sqp_piqp<Radau_Eigen>(ocp);
                print_ocp_opt(transcription);
            }
        }
    }

    if (with_casadi_AD)
    {
#ifdef LAOPT_WITH_CASAD
        std::cout << "---------------------------\n"
                     "CasADi autodiff\n"
                     "---------------------------\n";
        if (with_multiple_shooting)
        {
            std::cout << "> Multiple shooting\n";
            using MS_Cas = laopt_tools::MultipleShooting<Ocp, N, laopt::ERK4, laopt::CASADI_ALL | laopt::CASADI_NO_JIT>;
            if (with_ipopt)
            {
                auto transcription = run_ipopt<MS_Cas>(ocp);
                print_ocp_opt(transcription);
            }
            if (with_sqp_piqp)
            {
                auto transcription = run_sqp_piqp<MS_Cas>(ocp);
                print_ocp_opt(transcription);
            }
        }
        if (with_radau_collocation)
        {
            std::cout << "> Radau Collocation shooting\n";
            using Radau_Casadi = laopt_tools::RadauCollocation<Ocp, N_segs, D_poly, laopt::CASADI_ALL | laopt::CASADI_NO_JIT>;
            if (with_ipopt)
            {
                auto transcription = run_ipopt<Radau_Casadi>(ocp);
                print_ocp_opt(transcription);
            }
            if (with_sqp_piqp)
            {
                auto transcription = run_sqp_piqp<Radau_Casadi>(ocp);
                print_ocp_opt(transcription);
            }
        }
#endif
    }

    return 0;
}
