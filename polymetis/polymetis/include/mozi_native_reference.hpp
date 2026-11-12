#pragma once
#include <Eigen/Dense>
#include <array>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <string>

namespace mozi {
constexpr double native_max_flange_error_m=.015;
constexpr double native_max_orientation_error_rad=.08726646259971647; // 5 degrees

inline std::array<double,2> native_tracking_error(const std::array<double,16>& actual_ee,
    const std::array<double,16>& desired_ee,const std::array<double,16>& flange_to_ee) {
  Eigen::Map<const Eigen::Matrix4d> actual(actual_ee.data()),desired(desired_ee.data()),tool(flange_to_ee.data());
  if(!actual.allFinite() || !desired.allFinite() || !tool.allFinite())
    throw std::runtime_error("Native tracking guard received nonfinite feedback");
  // Compare the flange, including the lever arm of the configured TCP.
  const Eigen::Vector3d offset=tool.block<3,3>(0,0).transpose()*tool.block<3,1>(0,3);
  const Eigen::Vector3d actual_flange=actual.block<3,1>(0,3)-actual.block<3,3>(0,0)*offset;
  const Eigen::Vector3d desired_flange=desired.block<3,1>(0,3)-desired.block<3,3>(0,0)*offset;
  const double cosine=( (desired.block<3,3>(0,0).transpose()*actual.block<3,3>(0,0)).trace()-1.)*.5;
  return {(actual_flange-desired_flange).norm(),std::acos(std::clamp(cosine,-1.,1.))};
}

inline void check_native_tracking(const std::array<double,16>& actual_ee,
    const std::array<double,16>& desired_ee,const std::array<double,16>& flange_to_ee) {
  const auto error=native_tracking_error(actual_ee,desired_ee,flange_to_ee);
  if(error[0]>native_max_flange_error_m || error[1]>native_max_orientation_error_rad)
    throw std::runtime_error("Native tracking deviation guard: flange "+std::to_string(error[0]*1000.)+
        " mm, orientation "+std::to_string(error[1]*180./3.141592653589793)+
        " deg exceeds 15 mm / 5 deg; experiment aborted");
}

inline bool native_route(unsigned actual,unsigned requested,unsigned policy,bool valid,
                         uint64_t record,uint64_t expected,unsigned finish) {
  if((actual!=1 && actual!=2) || requested!=actual || policy!=actual || !valid ||
      !record || record!=expected || finish>1)
    throw std::runtime_error("Native policy, mode, recording or owner lease lost; experiment aborted");
  return finish==0;
}

inline std::array<double,16> native_reference(const std::array<double,16>& origin,
    const std::array<double,16>& desired,double blend,unsigned phase) {
  Eigen::Map<const Eigen::Matrix4d> initial(origin.data());
  Eigen::Map<const Eigen::Matrix4d> goal(desired.data());
  auto rigid=[](const Eigen::Matrix4d& matrix) {
    return matrix.allFinite() && (matrix.row(3)-Eigen::RowVector4d(0,0,0,1)).norm()<1e-8 &&
        (matrix.block<3,3>(0,0).transpose()*matrix.block<3,3>(0,0)-Eigen::Matrix3d::Identity()).norm()<1e-6 &&
        std::abs(matrix.block<3,3>(0,0).determinant()-1)<1e-6;
  };
  if(!rigid(initial) || !rigid(goal) || !std::isfinite(blend) || blend<0 || blend>1 ||
      phase>3 || (phase==0 && blend!=0) || (phase>=2 && blend!=1))
    throw std::runtime_error("Invalid native reference transform/phase");
  Eigen::Matrix4d target=goal;
  if(phase<2) {
    if((goal.block<3,1>(0,3)-initial.block<3,1>(0,3)).norm()>.03 ||
        Eigen::AngleAxisd(initial.block<3,3>(0,0).transpose()*goal.block<3,3>(0,0)).angle()>.15)
      throw std::runtime_error("Native prelude exceeds 30 mm / 0.15 rad; no catch-up");
    Eigen::Quaterniond q0(initial.block<3,3>(0,0)),q1(goal.block<3,3>(0,0));
    target.block<3,3>(0,0)=q0.slerp(blend,q1).toRotationMatrix();
    target.block<3,1>(0,3)=(1.-blend)*initial.block<3,1>(0,3)+blend*goal.block<3,1>(0,3);
  }
  std::array<double,16> result;
  std::copy(target.data(),target.data()+16,result.begin());
  return result;
}
} // namespace mozi
