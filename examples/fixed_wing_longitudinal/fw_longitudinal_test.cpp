#include <iostream>
#include <iomanip>

#include "LonOcpEigen.hpp"
#include "../double_integrator_v0/multiple_shooting_transcription.hpp"
#include "laopt/ipopt_wrapper.hpp"

int main()
{
    /* Choose OCP and Transcription */
    using Ocp = lon_ocp::LonFlightOCP;
    const int N = 30;
    using Transcription = MultipleShootingTranscription<Ocp, N>;

    /* Define specific Tape, LAMPC, and IPOPT problem types for the resulting NLP */
    using Tape = laopt::TapeInfo<Transcription>;
    using OptProblem = laopt::Problem<Transcription>;
    using Solver = laopt::IpoptWrapper<OptProblem>;

    /* Construct and setup OCP */
    Ocp ocp;
    ocp.model.set_state_representation(kite_model::LongitudinalFlightPath);
    ocp.model.load_params_from_yaml("eg4_xflr-Pvw-YR.yaml");

    ocp.objectives[lon_ocp::TrackAngle] = true;
//    ocp.objectives[lon_ocp::TrackVa] = true;
    ocp.objectives[lon_ocp::MinimizeControl] = true;

    ocp.tf = 1.5;

    ocp.pitch_ref = -20.0 * M_PI / 180.0;
    ocp.Va_ref = 11.0;

    ocp.mayer_multiplier = 10;
    ocp.W_pitch_err = 10;
    ocp.W_Va_err = 1;
    ocp.R.diagonal() << 1, 0.1;

    ocp.ubu << 0.8 * ocp.model.u_physical_ubound(0), 0.001;
    ocp.lbu << 0.8 * ocp.model.u_physical_lbound(0), 0;

    /* Transcribe OCP */
    Transcription transcription(ocp);
    Tape tape = laopt::generate_tape(transcription, laopt::generate_sparsity(transcription));

    /* Construct LAMPC and IPOPT problems for transcribed OCP using according tape, link decision variables between problems */
    OptProblem optProblem(transcription, tape); // Tape is optional here and could also be generated internally
    Solver solver(optProblem);

    /* Set initial guess for state trajectory */
    for(auto& x: transcription.X) { x << ocp.model.get_default_initial_state(); }

    /* Set initial state and solve the problem (loop later) */
    ocp.x0 << ocp.model.get_default_initial_state();
    solver.solve();

    /* Print out the solution */
    std::cout << std::endl << std::endl << std::endl << std::endl;
    std::cout << std::setprecision(6) << std::defaultfloat;

    Eigen::Matrix<Ocp::scalar_t, Ocp::NX, transcription.X.size()> X;
    Eigen::Matrix<Ocp::scalar_t, Ocp::NU, transcription.U.size()> U;
    int i{0};
    i = 0; for(auto& x: transcription.X) X.col(i++) << x;
    i = 0; for(auto& u: transcription.U) U.col(i++) << u;

    std::cout << "T = \n" << transcription.T.transpose() << std::endl;
    std::cout << "X = \n" << X << std::endl;
    std::cout << "U = \n" << U << std::endl;
    std::cout << "obj = " << optProblem.eval_objective(laopt::Eval(), solver().sol_primal) << std::endl;

    return 0;
}
