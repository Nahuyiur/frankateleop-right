// Copyright (c) Facebook, Inc. and its affiliates.

// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "spdlog/spdlog.h"
#include <array>
#include <assert.h>
#include <chrono>
#include <memory>
#include <thread>
#include "mozi_fault_window.hpp"
#include "mozi_tracking_tape.hpp"

#include "yaml-cpp/yaml.h"

#include "polymetis/utils.h"
#include <grpcpp/grpcpp.h>

#include <franka/exception.h>
#include <franka/model.h>
#include <franka/robot.h>

#define NUM_DOFS 7
#define FRANKA_HZ 1000.0
#define M_PI 3.14159265358979323846
#define RECOVERY_WAIT_SECS 1
#define RECOVERY_MAX_TRIES 3

class FrankaTorqueControlClient {
private:
  std::unique_ptr<mozi::FaultWindow> mozi_capture_;
  std::unique_ptr<mozi::TrackingTape> tracking_tape_;
  unsigned tracking_path_=0; // 0 A, 1 B, 2 C, 3 stopped handoff, 4 latched fault
  uint64_t native_cycle_=0, previous_command_robot_ms_=0, stationary_since_=0;
  uint64_t native_record_id_=0;
  double previous_source_index_=-1;
  std::array<double,16> native_initial_pose_{};
  bool native_initial_set_=false;
  void recordTracking(const franka::RobotState& state, const std::array<double,16>& pose);
  void updateServerCommand(const franka::RobotState &libfranka_robot_state,
                           std::array<double, NUM_DOFS> &torque_out);
  void checkStateLimits(const franka::RobotState &libfranka_robot_state,
                        std::array<double, NUM_DOFS> &torque_out);
  void postprocessTorques(std::array<double, NUM_DOFS> &torque_applied);

  template <std::size_t N>
  void computeSafetyReflex(std::array<double, N> values,
                           std::array<double, N> lower_limit,
                           std::array<double, N> upper_limit, bool invert_lower,
                           std::array<double, N> &safety_torques, double h,
                           double k, const char *item_name);

  // gRPC
  std::unique_ptr<PolymetisControllerServer::Stub> stub_;
  grpc::Status status_;

  TorqueCommand torque_command_;
  RobotState robot_state_;

  // libfranka
  bool mock_franka_;
  bool readonly_mode_;

  std::unique_ptr<franka::Robot> robot_ptr_;
  std::unique_ptr<franka::Model> model_ptr_;
  std::array<double, NUM_DOFS> torque_commanded_, torque_safety_,
      torque_applied_;

  // Torque processing
  bool limit_rate_;
  double lpf_cutoff_freq_;
  std::array<double, NUM_DOFS> torque_applied_prev_;

  std::array<double, 3> cartesian_pos_ulimits_, cartesian_pos_llimits_;
  std::array<double, NUM_DOFS> joint_pos_ulimits_, joint_pos_llimits_,
      joint_vel_limits_, joint_torques_limits_;
  double elbow_vel_limit_;

  // safety controller
  std::unordered_map<std::string, bool> active_constraints_map_;
  bool is_safety_controller_active_;
  double margin_cartesian_pos_, margin_joint_pos_, margin_joint_vel_;
  double k_cartesian_pos_, k_joint_pos_, k_joint_vel_;

public:
  /**
  TODO
  */
  FrankaTorqueControlClient(std::shared_ptr<grpc::Channel> channel,
                            YAML::Node config);
  void run();
};
