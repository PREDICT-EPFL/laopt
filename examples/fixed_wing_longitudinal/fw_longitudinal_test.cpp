#include <iostream>
#include <iomanip>

#include "LonOcpEigen.hpp"
#include "MultipleShooting.hpp"
#include "laopt/ipopt_wrapper.hpp"

int main()
{
    /* Choose OCP and Transcription */
    using Ocp = lon_ocp::LonFlightOCP;

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

    /* Set initial state */
    ocp.set_x0(ocp.model.get_default_initial_state());

    /* Solve with Multiple Shooting transcription */
    {
        const int N = 30;
        using Transcription = transcription::MultipleShooting<Ocp, N>;

        /* Define specific Tape, LAMPC, and IPOPT problem types for the resulting NLP */
        using Tape = laopt::TapeInfo<Transcription>;
        using OptProblem = laopt::Problem<Transcription>;
        using Solver = laopt::IpoptWrapper<OptProblem>;

        /* Construct transcription for OCP, optionally generate/store tape for that combination */
        Transcription transcription(ocp);
        Tape tape = laopt::generate_tape(transcription, laopt::generate_sparsity(transcription));

        /* Construct laOPT and IPOPT problems for transcribed OCP using according tape */
        OptProblem optProblem(transcription, tape); // Tape is optional here and could also be generated internally
        Solver solver(optProblem);

        /* Set initial guess for state trajectory */
        transcription.set_X_guess(ocp.model.get_default_initial_state());
        std::cout << "X_guess = \n" << transcription.get_Xopt() << "\n";

        solver.solve();
        solver.solve();

        /* Print out the solution */
        std::cout << "\n\n";
        std::cout << std::setprecision(6) << std::defaultfloat;

        Transcription::StateTrajectory Xopt = transcription.get_Xopt();
        Transcription::InputTrajectory Uopt = transcription.get_Uopt();

        std::cout << "T = \n" << transcription.get_Topt().transpose() << "\n";
        std::cout << "Xopt = \n" << Xopt << "\n";
        std::cout << "Uopt = \n" << Uopt << "\n";
        const double objective_eval = optProblem.eval_objective(laopt::Eval(), solver.sol_primal);
        std::cout << "obj: " << objective_eval << "\n";
    }

    return 0;
}
