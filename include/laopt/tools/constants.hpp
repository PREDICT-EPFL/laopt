#ifndef LAOPT_CONSTANTS_HPP
#define LAOPT_CONSTANTS_HPP

namespace laopt_tools
{

enum OCPOptions
{
    FixedEndTime = 0x0,
    FreeEndTime  = 0x1,

    /* Discrete dynamics: MultipleShooting calls ControlProblem::discrete_dynamics_impl()
     * directly instead of integrating ControlProblem::dynamics_impl(). Continuous (this bit
     * unset) is the default. Not combinable with FreeEndTime. */
    DiscreteDynamics = 0x2,
};

} // namespace laopt_tools

#endif //LAOPT_CONSTANTS_HPP
