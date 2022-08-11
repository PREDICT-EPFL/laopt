#include "double_integrator_tape.hpp"

laopt::TapeInfo<OCP> double_integrator_tape(OCP& ocp)
{
  return laopt::generate_tape(ocp, laopt::generate_sparsity(ocp));
}
