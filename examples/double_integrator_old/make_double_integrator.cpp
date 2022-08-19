#include "double_integrator.hpp"

TapeInfo<OCP> create_double_integrator(OCP& ocp)
{
  Problem prob = laopt::generate(ocp);
  return prob;
}
