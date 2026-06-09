#include "polymetis/clients/franka_hand_client.hpp"

#include "spdlog/spdlog.h"
#include <string>
#include <thread>
#include <time.h>
#include <chrono>
#include <cmath>
#include <cstdint>

#include <grpc/grpc.h>

#include "polymetis.grpc.pb.h"
#include "polymetis/utils.h"

using grpc::ClientContext;

FrankaHandClient::FrankaHandClient(std::shared_ptr<grpc::Channel> channel,
                                   YAML::Node config)
    : stub_(GripperServer::NewStub(channel)) {
  // Connect to gripper
  std::string robot_ip = config["robot_ip"].as<std::string>();
  spdlog::info("Connecting to robot_ip {}", robot_ip);
  gripper_.reset(new franka::Gripper(robot_ip));

  // Initialize gripper
  gripper_->homing();
  is_moving_ = false;

  // Initialize server connection
  franka::GripperState franka_gripper_state = gripper_->readOnce();
  prev_cmd_width_ = franka_gripper_state.width;

  GripperMetadata metadata;
  metadata.set_max_width(franka_gripper_state.max_width);
  metadata.set_hz(GRIPPER_HZ);

  ClientContext context;
  Empty empty;
  stub_->InitRobotClient(&context, metadata, &empty);

  spdlog::info("Connected.", robot_ip);
}

void FrankaHandClient::getGripperState(void) {
  franka::GripperState franka_gripper_state = gripper_->readOnce();

  gripper_state_.set_width(franka_gripper_state.width);
  gripper_state_.set_is_grasped(franka_gripper_state.is_grasped);
  gripper_state_.set_is_moving(is_moving_);
  gripper_state_.set_prev_command_successful(prev_cmd_successful_);

  // gripper_state.time();  // Use current timestamp instead!
  setTimestampToNow(gripper_state_.mutable_timestamp());
}

void FrankaHandClient::applyGripperCommand(GripperCommand gripper_cmd) {
  is_moving_ = true;

  if (gripper_cmd.grasp()) {
    spdlog::info("Grasping at width {} at speed={}", gripper_cmd.width(),
                 gripper_cmd.speed());
    double eps_inner = (gripper_cmd.epsilon_inner() < 0)
                           ? EPSILON_INNER
                           : gripper_cmd.epsilon_inner();
    double eps_outer = (gripper_cmd.epsilon_outer() < 0)
                           ? EPSILON_OUTER
                           : gripper_cmd.epsilon_outer();
    prev_cmd_successful_ =
        gripper_->grasp(gripper_cmd.width(), gripper_cmd.speed(),
                        gripper_cmd.force(), eps_inner, eps_outer);

  } else {
    spdlog::info("Moving to width {} at speed={}", gripper_cmd.width(),
                 gripper_cmd.speed());
    prev_cmd_successful_ =
        gripper_->move(gripper_cmd.width(), gripper_cmd.speed());
  }

  is_moving_ = false;
}

void FrankaHandClient::run(void) {
  const int64_t period_ns = 1000000000LL / GRIPPER_HZ;

  int timestamp_s;
  int timestamp_ns;
  float cmd_width;

  while (true) {
    // Run control step
    getGripperState();

    grpc::ClientContext context;
    status_ = stub_->ControlUpdate(&context, gripper_state_, &gripper_cmd_);

    if (!is_moving_) {
      // Skip if command not updated
      timestamp_s = gripper_cmd_.timestamp().seconds();
      timestamp_ns = gripper_cmd_.timestamp().nanos();
      cmd_width = gripper_cmd_.width();
      bool has_timestamp = timestamp_s || timestamp_ns;
      bool new_cmd = timestamp_s != prev_cmd_timestamp_s_ ||
                     timestamp_ns != prev_cmd_timestamp_ns_;
      bool width_changed = fabs(cmd_width - prev_cmd_width_) > 0.045;
      if (new_cmd && has_timestamp && width_changed) {
        // applyGripperCommand() in separate thread
        GripperCommand cmd_to_apply = gripper_cmd_;
        std::thread th(&FrankaHandClient::applyGripperCommand, this, cmd_to_apply);
        th.detach();
        prev_cmd_timestamp_s_ = timestamp_s;
        prev_cmd_timestamp_ns_ = timestamp_ns;
        prev_cmd_width_ = cmd_width;
      }
    }

    // Spin once
    std::this_thread::sleep_for(std::chrono::nanoseconds(period_ns));
  }
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    spdlog::error("Usage: franka_hand_client /path/to/cfg.yaml");
    return 1;
  }
  YAML::Node config = YAML::LoadFile(argv[1]);

  // Launch client
  std::string control_address = config["control_ip"].as<std::string>() + ":" +
                                config["control_port"].as<std::string>();
  FrankaHandClient franka_hand_client(
      grpc::CreateChannel(control_address, grpc::InsecureChannelCredentials()),
      config);
  franka_hand_client.run();

  // Termination
  spdlog::info("Wait for shutdown; press CTRL+C to close.");

  return 0;
}
