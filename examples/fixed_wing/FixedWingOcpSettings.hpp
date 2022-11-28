#ifndef SRC_FIXEDWINGOCPSETTINGS_HPP
#define SRC_FIXEDWINGOCPSETTINGS_HPP

#include <array>

namespace fixed_wing_ocp {

enum ControlObjectiveEnum
{
    TrackAngle,
    TrackVa,
    MinimizeControl,

    _CONTROLOBJECTIVES_MAX
};
using ControlObjectives = std::array<bool, _CONTROLOBJECTIVES_MAX>;

constexpr double TIME_HORIZON = 1.0;
constexpr int POLY_ORDER = 4;
constexpr int NUM_SEGMENTS = 2;
constexpr int N_TRAJ_POINTS = NUM_SEGMENTS * POLY_ORDER + 1;

constexpr double SEGM_TIMESPAN = TIME_HORIZON / NUM_SEGMENTS;
}

#endif //SRC_FIXEDWINGOCPSETTINGS_HPP
