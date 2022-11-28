#ifndef SRC_QUATERNIONMATHEIGEN_HPP
#define SRC_QUATERNIONMATHEIGEN_HPP

#include <Eigen/Core>

namespace flight_model {
namespace eigen_model {
namespace quatmath {

using namespace eigen_model;

/** Quaternion transformations **/
template<typename EigenScalar, typename Scalar>
Eigen::Quaternion<EigenScalar> T1quat(const Scalar &rotAng)
{
    Eigen::Quaternion<EigenScalar> q;
    q = Eigen::AngleAxis<EigenScalar>(-rotAng, Eigen::Vector<EigenScalar, 3>(static_cast<EigenScalar>(1),
                                                                             static_cast<EigenScalar>(0),
                                                                             static_cast<EigenScalar>(0)));
    return q;
}
template<typename EigenScalar, typename Scalar>
Eigen::Quaternion<EigenScalar> T2quat(const Scalar &rotAng)
{
    Eigen::Quaternion<EigenScalar> q;
    q = Eigen::AngleAxis<EigenScalar>(-rotAng, Eigen::Vector<EigenScalar, 3>(static_cast<EigenScalar>(0),
                                                                             static_cast<EigenScalar>(1),
                                                                             static_cast<EigenScalar>(0)));
    return q;
}
template<typename EigenScalar, typename Scalar>
Eigen::Quaternion<EigenScalar> T3quat(const Scalar &rotAng)
{
    Eigen::Quaternion<EigenScalar> q;
    q = Eigen::AngleAxis<EigenScalar>(-rotAng, Eigen::Vector<EigenScalar, 3>(static_cast<EigenScalar>(0),
                                                                             static_cast<EigenScalar>(0),
                                                                             static_cast<EigenScalar>(1)));
    return q;
}
/** quaternions to euler angles **/
template<typename Scalar>
Eigen::Matrix<Scalar, 3, 1> quat2eul(const Eigen::Quaternion<Scalar> &q_nb)
{
    const Scalar q1 = q_nb.x();
    const Scalar q2 = q_nb.y();
    const Scalar q3 = q_nb.z();
    const Scalar q4 = q_nb.w();

    /* Relevant components of the Direction Cosine Matrix */
    const Scalar T_bg_23 = 2.0 * (q2 * q3 + q4 * q1);
    const Scalar T_bg_33 = 2.0 * (q4 * q4 + q3 * q3) - 1.0;
    const Scalar T_bg_13 = 2.0 * (q1 * q3 - q4 * q2);
    const Scalar T_bg_12 = 2.0 * (q1 * q2 + q4 * q3);
    const Scalar T_bg_11 = 2.0 * (q4 * q4 + q1 * q1) - 1.0;

    /* Euler angles: Roll, Pitch, Yaw */
    Eigen::Matrix < Scalar, 3, 1 > euler_angles(atan2(T_bg_23, T_bg_33),
                                                -asin(T_bg_13),
                                                atan2(T_bg_12, T_bg_11));
    return euler_angles;
}

} //namespace quatmath
} //namespace eigen_model
} //namespace flight_model

#endif //SRC_QUATERNIONMATHEIGEN_HPP
