/**
 * Double Integrator example calling IPOpt
 */

#include "double_integrator.hpp"

Problem tape_to_problem(lampc::TapeInfo<OCP>& tape, OCP& ocp)
{
  return Problem(ocp, tape);
}
