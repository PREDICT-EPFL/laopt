#include "double_integrator.hpp"

TapeInfo<OCP> create_double_integrator(OCP& ocp)
{
  Problem prob = lampc::generate(ocp);
  return prob;
}
