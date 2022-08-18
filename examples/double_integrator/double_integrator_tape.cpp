#include "double_integrator_tape.hpp"

lampc::TapeInfo<OCP> double_integrator_tape(OCP& ocp)
{
  return lampc::generate_tape(ocp, lampc::generate_sparsity(ocp));
}
