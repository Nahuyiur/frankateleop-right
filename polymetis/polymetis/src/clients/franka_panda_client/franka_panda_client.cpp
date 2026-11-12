// Copyright (c) Facebook, Inc. and its affiliates.

// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "polymetis/clients/franka_panda_client.hpp"

#include "real_time.hpp"
#include "mozi_native_reference.hpp"
#include "spdlog/spdlog.h"
#include "yaml-cpp/yaml.h"
#include <Eigen/Dense>
#include <math.h>
#include <stdexcept>
#include <string>
#include <time.h>
#include <unistd.h>

#include <fstream>
#include <sstream>

#include <grpc/grpc.h>

using grpc::ClientContext;
using grpc::Status;

namespace {
uint64_t mozi_error_mask(const franka::Errors& error) noexcept {
  uint64_t mask=0;
  if(error.joint_position_limits_violation) mask |= (1ULL << 0);
  if(error.cartesian_position_limits_violation) mask |= (1ULL << 1);
  if(error.self_collision_avoidance_violation) mask |= (1ULL << 2);
  if(error.joint_velocity_violation) mask |= (1ULL << 3);
  if(error.cartesian_velocity_violation) mask |= (1ULL << 4);
  if(error.force_control_safety_violation) mask |= (1ULL << 5);
  if(error.joint_reflex) mask |= (1ULL << 6);
  if(error.cartesian_reflex) mask |= (1ULL << 7);
  if(error.max_goal_pose_deviation_violation) mask |= (1ULL << 8);
  if(error.max_path_pose_deviation_violation) mask |= (1ULL << 9);
  if(error.cartesian_velocity_profile_safety_violation) mask |= (1ULL << 10);
  if(error.joint_position_motion_generator_start_pose_invalid) mask |= (1ULL << 11);
  if(error.joint_motion_generator_position_limits_violation) mask |= (1ULL << 12);
  if(error.joint_motion_generator_velocity_limits_violation) mask |= (1ULL << 13);
  if(error.joint_motion_generator_velocity_discontinuity) mask |= (1ULL << 14);
  if(error.joint_motion_generator_acceleration_discontinuity) mask |= (1ULL << 15);
  if(error.cartesian_position_motion_generator_start_pose_invalid) mask |= (1ULL << 16);
  if(error.cartesian_motion_generator_elbow_limit_violation) mask |= (1ULL << 17);
  if(error.cartesian_motion_generator_velocity_limits_violation) mask |= (1ULL << 18);
  if(error.cartesian_motion_generator_velocity_discontinuity) mask |= (1ULL << 19);
  if(error.cartesian_motion_generator_acceleration_discontinuity) mask |= (1ULL << 20);
  if(error.cartesian_motion_generator_elbow_sign_inconsistent) mask |= (1ULL << 21);
  if(error.cartesian_motion_generator_start_elbow_invalid) mask |= (1ULL << 22);
  if(error.cartesian_motion_generator_joint_position_limits_violation) mask |= (1ULL << 23);
  if(error.cartesian_motion_generator_joint_velocity_limits_violation) mask |= (1ULL << 24);
  if(error.cartesian_motion_generator_joint_velocity_discontinuity) mask |= (1ULL << 25);
  if(error.cartesian_motion_generator_joint_acceleration_discontinuity) mask |= (1ULL << 26);
  if(error.cartesian_position_motion_generator_invalid_frame) mask |= (1ULL << 27);
  if(error.force_controller_desired_force_tolerance_violation) mask |= (1ULL << 28);
  if(error.controller_torque_discontinuity) mask |= (1ULL << 29);
  if(error.start_elbow_sign_inconsistent) mask |= (1ULL << 30);
  if(error.communication_constraints_violation) mask |= (1ULL << 31);
  if(error.power_limit_violation) mask |= (1ULL << 32);
  if(error.joint_p2p_insufficient_torque_for_planning) mask |= (1ULL << 33);
  if(error.tau_j_range_violation) mask |= (1ULL << 34);
  if(error.instability_detected) mask |= (1ULL << 35);
  if(error.joint_move_in_wrong_direction) mask |= (1ULL << 36);
  if(error.cartesian_spline_motion_generator_violation) mask |= (1ULL << 37);
  if(error.joint_via_motion_generator_planning_joint_limit_violation) mask |= (1ULL << 38);
  if(error.base_acceleration_initialization_timeout) mask |= (1ULL << 39);
  if(error.base_acceleration_invalid_reading) mask |= (1ULL << 40);
  if(static_cast<bool>(error) && mask==0) mask=(1ULL<<63);
  return mask;
}
mozi::Sample mozi_state_sample(const franka::RobotState& s) noexcept {
  mozi::Sample sample;
  sample.mono=mozi::mono_ns();sample.wall=mozi::wall_ns();
  sample.robot_ms=s.time.toMSec();sample.robot_mode=static_cast<uint32_t>(s.robot_mode);
  sample.current_errors=mozi_error_mask(s.current_errors);
  sample.last_motion_errors=mozi_error_mask(s.last_motion_errors);
  sample.success_rate=s.control_command_success_rate;
  sample.q=s.q;sample.dq=s.dq;sample.q_d=s.q_d;sample.dq_d=s.dq_d;
  sample.tau=s.tau_J;sample.tau_d=s.tau_J_d;sample.tau_ext=s.tau_ext_hat_filtered;
  sample.O_T_EE=s.O_T_EE;sample.O_F_ext_hat_K=s.O_F_ext_hat_K;
  sample.joint_contact=s.joint_contact;sample.joint_collision=s.joint_collision;
  sample.cartesian_contact=s.cartesian_contact;sample.cartesian_collision=s.cartesian_collision;
  sample.valid=1;return sample;
}
} // namespace


FrankaTorqueControlClient::FrankaTorqueControlClient(
    std::shared_ptr<grpc::Channel> channel, YAML::Node config)
    : stub_(PolymetisControllerServer::NewStub(channel)) {
  std::string robot_client_metadata_path =
      config["robot_client_metadata_path"].as<std::string>();

  // Load robot client metadata
  std::ifstream file(robot_client_metadata_path);
  assert(file);
  std::stringstream buffer;
  buffer << file.rdbuf();
  file.close();
  RobotClientMetadata metadata;
  assert(metadata.ParseFromString(buffer.str()));

  // Initialize robot client with metadata
  ClientContext context;
  Empty empty;
  Status status = stub_->InitRobotClient(&context, metadata, &empty);
  assert(status.ok());

  // Allocate diagnostic buffers before connecting/entering the control loop.
  // Output failures and disabled diagnostics never affect motion decisions.
  const char* enabled=std::getenv("MOZI_NATIVE_FAULT_WINDOW");
  if(!enabled || std::string(enabled)!="0") {
    try {
      mozi::Options options;
      const char* directory=std::getenv("MOZI_FAULT_WINDOW_DIR");
      options.directory=std::string(directory?directory:"/home/pnp/frankateleop/logs/fault_windows/native")+"/"+std::to_string(getpid());
      YAML::Emitter yaml;yaml<<config;
      const char* trigger_name=std::getenv("MOZI_FAULT_TRIGGER_NAME");
      options.trigger_name=trigger_name?trigger_name:"mozi_fault_native";
      options.metadata="{\"pid\":"+std::to_string(getpid())+",\"config_yaml\":"+mozi::FaultWindow::quote(yaml.c_str())+
        ",\"urdf_xml\":"+mozi::FaultWindow::quote(metadata.urdf_file().c_str())+
        ",\"error_names\":[\"joint_position_limits_violation\",\"cartesian_position_limits_violation\",\"self_collision_avoidance_violation\",\"joint_velocity_violation\",\"cartesian_velocity_violation\",\"force_control_safety_violation\",\"joint_reflex\",\"cartesian_reflex\",\"max_goal_pose_deviation_violation\",\"max_path_pose_deviation_violation\",\"cartesian_velocity_profile_safety_violation\",\"joint_position_motion_generator_start_pose_invalid\",\"joint_motion_generator_position_limits_violation\",\"joint_motion_generator_velocity_limits_violation\",\"joint_motion_generator_velocity_discontinuity\",\"joint_motion_generator_acceleration_discontinuity\",\"cartesian_position_motion_generator_start_pose_invalid\",\"cartesian_motion_generator_elbow_limit_violation\",\"cartesian_motion_generator_velocity_limits_violation\",\"cartesian_motion_generator_velocity_discontinuity\",\"cartesian_motion_generator_acceleration_discontinuity\",\"cartesian_motion_generator_elbow_sign_inconsistent\",\"cartesian_motion_generator_start_elbow_invalid\",\"cartesian_motion_generator_joint_position_limits_violation\",\"cartesian_motion_generator_joint_velocity_limits_violation\",\"cartesian_motion_generator_joint_velocity_discontinuity\",\"cartesian_motion_generator_joint_acceleration_discontinuity\",\"cartesian_position_motion_generator_invalid_frame\",\"force_controller_desired_force_tolerance_violation\",\"controller_torque_discontinuity\",\"start_elbow_sign_inconsistent\",\"communication_constraints_violation\",\"power_limit_violation\",\"joint_p2p_insufficient_torque_for_planning\",\"tau_j_range_violation\",\"instability_detected\",\"joint_move_in_wrong_direction\",\"cartesian_spline_motion_generator_violation\",\"joint_via_motion_generator_planning_joint_limit_violation\",\"base_acceleration_initialization_timeout\",\"base_acceleration_invalid_reading\"]}";
      mozi_capture_=std::make_unique<mozi::FaultWindow>(std::move(options));
    }catch(const std::exception& ex){spdlog::warn("Native diagnostic capture unavailable: {}",ex.what());}
  }

  tracking_tape_=std::make_unique<mozi::TrackingTape>("/home/pnp/work/rtc-abc-recordings");

  // Connect to robot
  mock_franka_ = config["mock"].as<bool>();
  readonly_mode_ = config["readonly"].as<bool>();
  if (!mock_franka_) {
    spdlog::info("Connecting to Franka Emika...");
    robot_ptr_.reset(new franka::Robot(config["robot_ip"].as<std::string>()));
    model_ptr_.reset(new franka::Model(robot_ptr_->loadModel()));
    spdlog::info("Connected.");
  } else {
    spdlog::info(
        "Launching Franka client in mock mode. No robot is connected.");
  }

  if (readonly_mode_) {
    spdlog::info("Launching Franka client in read only mode. No control will "
                 "be executed.");
  }

  // Set initial state & action
  for (int i = 0; i < NUM_DOFS; i++) {
    torque_commanded_[i] = 0.0;
    torque_safety_[i] = 0.0;
    torque_applied_prev_[i] = 0.0;

    torque_command_.add_joint_torques(0.0);

    robot_state_.add_joint_positions(0.0);
    robot_state_.add_joint_velocities(0.0);
    robot_state_.add_prev_joint_torques_computed(0.0);
    robot_state_.add_prev_joint_torques_computed_safened(0.0);
    robot_state_.add_motor_torques_measured(0.0);
    robot_state_.add_motor_torques_external(0.0);
    robot_state_.add_motor_torques_desired(0.0);
  }

  // Parse yaml
  limit_rate_ = config["limit_rate"].as<bool>();
  lpf_cutoff_freq_ = config["lpf_cutoff_frequency"].as<double>();

  cartesian_pos_ulimits_ =
      config["limits"]["cartesian_pos_upper"].as<std::array<double, 3>>();
  cartesian_pos_llimits_ =
      config["limits"]["cartesian_pos_lower"].as<std::array<double, 3>>();
  joint_pos_ulimits_ =
      config["limits"]["joint_pos_upper"].as<std::array<double, NUM_DOFS>>();
  joint_pos_llimits_ =
      config["limits"]["joint_pos_lower"].as<std::array<double, NUM_DOFS>>();
  joint_vel_limits_ =
      config["limits"]["joint_vel"].as<std::array<double, NUM_DOFS>>();
  elbow_vel_limit_ = config["limits"]["elbow_vel"].as<double>();
  joint_torques_limits_ =
      config["limits"]["joint_torques"].as<std::array<double, NUM_DOFS>>();

  is_safety_controller_active_ =
      config["safety_controller"]["is_active"].as<bool>();

  margin_cartesian_pos_ =
      config["safety_controller"]["margins"]["cartesian_pos"].as<double>();
  margin_joint_pos_ =
      config["safety_controller"]["margins"]["joint_pos"].as<double>();
  margin_joint_vel_ =
      config["safety_controller"]["margins"]["joint_vel"].as<double>();

  k_cartesian_pos_ =
      config["safety_controller"]["stiffness"]["cartesian_pos"].as<double>();
  k_joint_pos_ =
      config["safety_controller"]["stiffness"]["joint_pos"].as<double>();
  k_joint_vel_ =
      config["safety_controller"]["stiffness"]["joint_vel"].as<double>();

  // Set collision behavior
  if (!mock_franka_) {
    robot_ptr_->setCollisionBehavior(
        config["collision_behavior"]["lower_torque"]
            .as<std::array<double, NUM_DOFS>>(),
        config["collision_behavior"]["upper_torque"]
            .as<std::array<double, NUM_DOFS>>(),
        config["collision_behavior"]["lower_force"].as<std::array<double, 6>>(),
        config["collision_behavior"]["upper_force"]
            .as<std::array<double, 6>>());
  }
}

void FrankaTorqueControlClient::run() {
  // Create callback function that relays information between gRPC server and
  // robot
  auto control_callback = [&](const franka::RobotState &libfranka_robot_state,
                              franka::Duration period) -> franka::Torques {
    auto diagnostic=mozi_state_sample(libfranka_robot_state);
    diagnostic.period_us=period.toMSec()*1000ULL;
    diagnostic.rpc_begin=mozi::mono_ns();
    try {
    // Compute torque components
    updateServerCommand(libfranka_robot_state, torque_commanded_);
    diagnostic.rpc_end=mozi::mono_ns();
    diagnostic.state_wall=uint64_t(robot_state_.timestamp().seconds())*1000000000ULL+robot_state_.timestamp().nanos();
    diagnostic.torque_wall=uint64_t(torque_command_.timestamp().seconds())*1000000000ULL+torque_command_.timestamp().nanos();
    checkStateLimits(libfranka_robot_state, torque_safety_);

    // Aggregate & clamp torques
    for (int i = 0; i < NUM_DOFS; i++) {
      torque_applied_[i] = torque_commanded_[i] + torque_safety_[i];
    }
    postprocessTorques(torque_applied_);

    // Record final applied torques
    for (int i = 0; i < NUM_DOFS; i++) {
      robot_state_.set_prev_joint_torques_computed_safened(i,
                                                           torque_applied_[i]);
    }

    diagnostic.policy_torque=torque_commanded_;
    diagnostic.safety_torque=torque_safety_;
    diagnostic.returned_torque=torque_applied_;
    diagnostic.valid|=2;
    if(mozi_capture_) {
      mozi_capture_->record(diagnostic);
      if(diagnostic.rpc_end-diagnostic.rpc_begin>100000000ULL)
        mozi_capture_->event("control_update_slow","",true);
    }
    if(tracking_path_==0)recordTracking(libfranka_robot_state,libfranka_robot_state.O_T_EE_c);
    const auto& command=torque_command_.native_tracking();
    if(tracking_path_==1) {
      if(!mozi::native_route(1,command.path(),torque_command_.tracking_policy_path(),torque_command_.tracking_valid(),
                            command.record_id(),native_record_id_,command.finish()))
        return franka::MotionFinished(franka::Torques(torque_applied_));
    }
    if(tracking_path_==0 && command.path()>0) {
      if(!mozi::native_route(command.path(),command.path(),torque_command_.tracking_policy_path(),torque_command_.tracking_valid(),
                            command.record_id(),command.record_id(),command.finish()))
        return franka::MotionFinished(franka::Torques(torque_applied_));
      bool stationary=true;
      for(double dq:libfranka_robot_state.dq)stationary=stationary && std::abs(dq)<.01;
      if(!stationary)stationary_since_=0;
      else if(!stationary_since_)stationary_since_=libfranka_robot_state.time.toMSec();
      if(stationary_since_ && libfranka_robot_state.time.toMSec()-stationary_since_>=2000 &&
          tracking_tape_->ready()==command.record_id())
        return franka::MotionFinished(franka::Torques(torque_applied_));
    } else stationary_since_=0;
    return torque_applied_;
    } catch(...) {
      diagnostic.rpc_end=mozi::mono_ns();
      diagnostic.state_wall=uint64_t(robot_state_.timestamp().seconds())*1000000000ULL+robot_state_.timestamp().nanos();
      if(mozi_capture_)mozi_capture_->record(diagnostic);
      throw;
    }
  };

  // Run robot
  if (!mock_franka_ && !readonly_mode_) {
    auto pose_callback = [&](const franka::RobotState& state,franka::Duration) -> franka::CartesianPose {
      // B's torque callback has already processed this SAME state. C has no torque callback.
      if(tracking_path_==2) {
        updateServerCommand(state,torque_commanded_);
        checkStateLimits(state,torque_safety_);
        // C cannot apply our external reflex torque: stop at the same guard instead.
        for(double value:torque_safety_)
          if(!std::isfinite(value) || std::abs(value)>1e-9)
            throw std::runtime_error("Native Cartesian reference reached existing software safety margin");
      }
      const auto& command=torque_command_.native_tracking();
      mozi::native_route(tracking_path_,command.path(),torque_command_.tracking_policy_path(),torque_command_.tracking_valid(),
                         command.record_id(),native_record_id_,command.finish());
      if(command.pose_size()!=16)throw std::runtime_error("Native pose requires 16 values");
      if(!native_initial_set_) {
        native_initial_pose_=state.O_T_EE_c;
        native_initial_set_=true;
      }
      std::array<double,16> desired;
      for(int i=0;i<16;++i)desired[i]=command.pose(i);
      auto pose=mozi::native_reference(native_initial_pose_,desired,command.prelude_blend(),command.phase());
      recordTracking(state,pose);
      if(command.finish()==1)return franka::MotionFinished(franka::CartesianPose(pose));
      return franka::CartesianPose(pose);
    };
    for(;;) {
      try {
        if(tracking_path_>=3) {
          // Read-only stopped handoff. A fallback/default policy cannot resume motion.
          // A latched fault requires operator recovery/restart, never automatic A fallback.
          robot_ptr_->read([&](const franka::RobotState& state) {
            updateServerCommand(state,torque_commanded_);
            return !(tracking_path_==3 && torque_command_.tracking_valid() &&
                     torque_command_.tracking_policy_path()==0 &&
                     torque_command_.native_tracking().path()==0);
          });
          tracking_path_=0;
        }
        if(tracking_path_==0) {
          robot_ptr_->control(control_callback,limit_rate_,lpf_cutoff_freq_);
          if(torque_command_.native_tracking().finish()==1){tracking_path_=3;continue;}
          tracking_path_=torque_command_.native_tracking().path();
          native_record_id_=torque_command_.native_tracking().record_id();
          if(tracking_path_!=1 && tracking_path_!=2)
            throw std::runtime_error("Unexpected torque control completion");
          native_initial_set_=false;
          previous_source_index_=-1;
          previous_command_robot_ms_=0;
        } else {
          if(tracking_path_==1)robot_ptr_->control(control_callback,pose_callback,limit_rate_,lpf_cutoff_freq_);
          else {
            const auto& stiffness=torque_command_.native_tracking().stiffness();
            if(stiffness.size()!=6)throw std::runtime_error("Cartesian stiffness requires six values");
            std::array<double,6> values;
            for(int i=0;i<6;++i) {
              values[i]=stiffness[i];
              if(!std::isfinite(values[i]) || values[i]<=0 || values[i]>(i<3?1500.:45.))
                throw std::runtime_error("Cartesian stiffness outside experiment bounds");
            }
            robot_ptr_->setCartesianImpedance(values);
            robot_ptr_->control(pose_callback,franka::ControllerMode::kCartesianImpedance,limit_rate_,lpf_cutoff_freq_);
          }
          tracking_path_=3;
        }
      } catch(const std::exception& ex) {
        if(mozi_capture_) {
          uint64_t mask=0;
          const auto* control=dynamic_cast<const franka::ControlException*>(&ex);
          if(control && !control->log.empty()) {
            auto last=mozi_state_sample(control->log.back().state);
            last.source=1;mask=last.current_errors|last.last_motion_errors;
            mozi_capture_->record(last);
          }
          mozi_capture_->event("control_exception",ex.what(),true,mask);
        }
        spdlog::error("Robot is unable to be controlled: {}",ex.what());
        if(tracking_path_!=0 || torque_command_.native_tracking().record_id()) {
          tracking_path_=4;
          tracking_tape_->select(0);
          // Stay stopped and preserve a latched failure. Do not repeatedly retry a failed read.
          try {
            robot_ptr_->read([&](const franka::RobotState& state) {
              updateServerCommand(state,torque_commanded_);return true;
            });
          } catch(const std::exception& read_error) {
            spdlog::error("Latched native experiment fault: {}",read_error.what());
            for(;;)std::this_thread::sleep_for(std::chrono::seconds(1));
          }
        } else {
          bool recovered=false;
          for(int i=0;i<RECOVERY_MAX_TRIES;++i) {
            usleep(1000000*RECOVERY_WAIT_SECS);
            try{robot_ptr_->automaticErrorRecovery();recovered=true;break;}
            catch(const std::exception& recovery){spdlog::error("Recovery failed: {}",recovery.what());}
          }
          if(!recovered)return;
        }
      }
    }

  } else {
    // Run mocked robot control loops
    franka::RobotState robot_state;
    franka::Duration duration;

    int period = 1.0 / FRANKA_HZ;
    int period_ns = period * 1.0e9;

    struct timespec abs_target_time;
    while (true) {
      clock_gettime(CLOCK_REALTIME, &abs_target_time);
      abs_target_time.tv_nsec += period_ns;

      // Pull data from robot if in readonly mode
      if (readonly_mode_) {
        robot_state = robot_ptr_->readOnce();
      }

      // Perform control loop with dummy variables (robot_state is populated if
      // in readonly mode)
      control_callback(robot_state, duration);

      clock_nanosleep(CLOCK_REALTIME, TIMER_ABSTIME, &abs_target_time, nullptr);
    }
  }
}

void FrankaTorqueControlClient::updateServerCommand(
    /*
     * Send robot states and receive torque command via a request to the
     * controller server.
     */
    const franka::RobotState &libfranka_robot_state,
    std::array<double, NUM_DOFS> &torque_out) {
  // Record robot states
  if (!mock_franka_) {
    bool prev_command_successful = false;

    for (int i = 0; i < NUM_DOFS; i++) {
      robot_state_.set_joint_positions(i, libfranka_robot_state.q[i]);
      robot_state_.set_joint_velocities(i, libfranka_robot_state.dq[i]);
      robot_state_.set_motor_torques_measured(i,
                                              libfranka_robot_state.tau_J[i]);
      robot_state_.set_motor_torques_external(
          i, libfranka_robot_state.tau_ext_hat_filtered[i]);

      // Check if previous command is successful by checking whether
      // constant torque policy for packet drops is applied
      // (If applied, desired torques will be exactly the same as last timestep)
      if (!prev_command_successful &&
          float(libfranka_robot_state.tau_J_d[i]) !=
              robot_state_.motor_torques_desired(i)) {
        prev_command_successful = true;
      }
      robot_state_.set_motor_torques_desired(i,
                                             libfranka_robot_state.tau_J_d[i]);
    }

    robot_state_.set_prev_command_successful(prev_command_successful);

    // Error code: can only set to 0 if no errors and 1 if any errors exist for
    // now
    robot_state_.set_error_code(bool(libfranka_robot_state.current_errors));
  }
  const auto& native=libfranka_robot_state;
  ++native_cycle_;
  robot_state_.set_native_state_valid(!mock_franka_);
  robot_state_.set_prev_command_successful_valid(false);
  robot_state_.set_robot_time_ms(native.time.toMSec());
  robot_state_.set_state_receive_monotonic_ns(mozi::mono_ns());
  robot_state_.set_control_command_success_rate(native.control_command_success_rate);
  robot_state_.set_tracking_control_path(tracking_path_);
  robot_state_.set_native_cycle(native_cycle_);
  robot_state_.set_tracking_log_ready_id(tracking_tape_->ready());
  robot_state_.set_tracking_log_failed(tracking_tape_->failed());
  robot_state_.set_tracking_phase(torque_command_.native_tracking().phase());
  robot_state_.set_tracking_source_index(torque_command_.native_tracking().source_index());
  robot_state_.set_tracking_record_id(torque_command_.native_tracking().record_id());
  robot_state_.set_m_ee(native.m_ee);
  robot_state_.set_m_load(native.m_load);
  robot_state_.mutable_o_t_ee()->Assign(native.O_T_EE.begin(),native.O_T_EE.end());
  robot_state_.mutable_f_t_ee()->Assign(native.F_T_EE.begin(),native.F_T_EE.end());
  robot_state_.mutable_native_q_d()->Assign(native.q_d.begin(),native.q_d.end());
  robot_state_.mutable_native_dq_d()->Assign(native.dq_d.begin(),native.dq_d.end());
  robot_state_.mutable_o_t_ee_d()->Assign(native.O_T_EE_d.begin(),native.O_T_EE_d.end());
  robot_state_.mutable_o_t_ee_c()->Assign(native.O_T_EE_c.begin(),native.O_T_EE_c.end());
  robot_state_.mutable_ee_t_k()->Assign(native.EE_T_K.begin(),native.EE_T_K.end());
  robot_state_.mutable_f_x_cee()->Assign(native.F_x_Cee.begin(),native.F_x_Cee.end());
  robot_state_.mutable_f_x_cload()->Assign(native.F_x_Cload.begin(),native.F_x_Cload.end());
  setTimestampToNow(robot_state_.mutable_timestamp());

  // Retrieve torques
  grpc::ClientContext context;
  long int pre_update_ns = getNanoseconds();
  status_ = stub_->ControlUpdate(&context, robot_state_, &torque_command_);
  long int post_update_ns = getNanoseconds();
  if (!status_.ok()) {
    std::string error_msg = "ControlUpdate rpc failed. ";
    throw std::runtime_error(error_msg + status_.error_message());
  }

  robot_state_.set_prev_controller_latency_ms(
      float(post_update_ns - pre_update_ns) / 1e6);

  auto record_id=torque_command_.native_tracking().record_id();
  tracking_tape_->select(tracking_path_==4?0:record_id);
  if(record_id && tracking_tape_->failed())throw std::runtime_error("Tracking recorder failed; experiment aborted");
  assert(torque_command_.joint_torques_size() == NUM_DOFS);
  for (int i = 0; i < NUM_DOFS; i++) {
    torque_out[i] = torque_command_.joint_torques(i);
    robot_state_.set_prev_joint_torques_computed(
        i, torque_command_.joint_torques(i));
  }
}

void FrankaTorqueControlClient::recordTracking(const franka::RobotState& s,
    const std::array<double,16>& pose) {
  const auto& command=torque_command_.native_tracking();
  if(!command.record_id() || tracking_tape_->ready()!=command.record_id())return;
  mozi::TrackingRow row;
  const uint64_t source_wall=uint64_t(robot_state_.timestamp().seconds())*1000000000ULL+robot_state_.timestamp().nanos();
  row.time={source_wall,robot_state_.state_receive_monotonic_ns(),s.time.toMSec(),native_cycle_,command.record_id(),
            tracking_path_,command.phase(),previous_command_robot_ms_,mozi_error_mask(s.current_errors),
            mozi_error_mask(s.last_motion_errors),static_cast<uint64_t>(s.robot_mode)};
  size_t at=0;
  auto one=[&](double value){row.data[at++]=value;};
  auto array=[&](const auto& values){for(double value:values)one(value);};
  auto trace=[&](const auto& values){
    for(int i=0;i<7;++i)one(values.size()==7 && tracking_path_!=2?values[i]:std::numeric_limits<double>::quiet_NaN());
  };
  one(command.source_index());one(previous_source_index_);one(command.prelude_blend());one(s.control_command_success_rate);
  array(s.q);array(s.dq);array(s.q_d);array(s.dq_d);
  trace(torque_command_.tracking_q_ref());trace(torque_command_.tracking_dq_ref());trace(torque_command_.tracking_feedforward());
  if(tracking_path_==2) {
    for(int i=0;i<21;++i)one(std::numeric_limits<double>::quiet_NaN());
  } else {array(torque_commanded_);array(torque_safety_);array(torque_applied_);}
  array(s.tau_J);array(s.tau_J_d);array(s.tau_ext_hat_filtered);
  array(s.O_T_EE);array(s.F_T_EE);array(s.EE_T_K);array(s.O_T_EE_d);array(s.O_T_EE_c);array(pose);
  for(int i=0;i<6;++i)one(command.stiffness_size()==6?command.stiffness(i):std::numeric_limits<double>::quiet_NaN());
  one(s.m_ee);one(s.m_load);array(s.F_x_Cee);array(s.F_x_Cload);
  one(torque_command_.tracking_policy_path());one(torque_command_.tracking_valid()?1.:0.);
  one(robot_state_.prev_controller_latency_ms());
  one((int64_t(torque_command_.timestamp().seconds())*1000000000LL+torque_command_.timestamp().nanos()-int64_t(source_wall))/1000.);
  assert(at==row.data.size());
  tracking_tape_->record(row);
  previous_source_index_=command.source_index();previous_command_robot_ms_=s.time.toMSec();
}

void FrankaTorqueControlClient::checkStateLimits(
    const franka::RobotState &libfranka_robot_state,
    std::array<double, NUM_DOFS> &torque_out) {
  /*
   * Compute robot state limit violations and apply safety mechanisms.
   */
  std::array<double, 3> ee_pos_buf, force_buf;
  std::array<double, 1> elbow_vel_buf, elbow_lim_buf, dummy;

  // No safety checks in mock mode
  if (mock_franka_) {
    return;
  }

  // Reset reflex torques
  for (int i = 0; i < NUM_DOFS; i++) {
    torque_out[i] = 0.0;
  }
  for (int i = 0; i < 3; i++) {
    force_buf[i] = 0.0;
  }

  // Cartesian position limits
  for (int i = 0; i < 3; i++) {
    ee_pos_buf[i] = libfranka_robot_state.O_T_EE[12 + i];
  }
  computeSafetyReflex(ee_pos_buf, cartesian_pos_llimits_,
                      cartesian_pos_ulimits_, false, force_buf,
                      margin_cartesian_pos_, k_cartesian_pos_, "EE position");

  std::array<double, 6 *NUM_DOFS> jacobian_array = model_ptr_->zeroJacobian(
      franka::Frame::kEndEffector, libfranka_robot_state);
  Eigen::Map<const Eigen::Matrix<double, 6, NUM_DOFS>> jacobian(
      jacobian_array.data());
  Eigen::Map<const Eigen::Vector3d> force_xyz_vec(force_buf.data());

  Eigen::VectorXd force_vec(6);
  force_vec.head(3) << force_xyz_vec;
  force_vec.tail(3) << Eigen::Vector3d::Zero();

  Eigen::VectorXd torque_vec(NUM_DOFS);
  torque_vec << jacobian.transpose() * force_vec;
  Eigen::VectorXd::Map(&torque_out[0], NUM_DOFS) = torque_vec;

  // Joint position limits
  computeSafetyReflex(libfranka_robot_state.q, joint_pos_llimits_,
                      joint_pos_ulimits_, false, torque_out, margin_joint_pos_,
                      k_joint_pos_, "Joint position");

  // Joint velocity limits
  computeSafetyReflex(libfranka_robot_state.dq, joint_vel_limits_,
                      joint_vel_limits_, true, torque_out, margin_joint_vel_,
                      k_joint_vel_, "Joint velocity");

  // Miscellaneous velocity limits
  elbow_vel_buf[0] = libfranka_robot_state.delbow_c[0];
  elbow_lim_buf[0] = elbow_vel_limit_;
  computeSafetyReflex(elbow_vel_buf, elbow_lim_buf, elbow_lim_buf, true, dummy,
                      0.0, 0.0, "Elbow velocity");
}

void FrankaTorqueControlClient::postprocessTorques(
    /*
     * Filter & clamp torque to limits
     */
    std::array<double, NUM_DOFS> &torque_applied) {
  for (int i = 0; i < 7; i++) {
    // Clamp torques
    if (torque_applied[i] > joint_torques_limits_[i]) {
      torque_applied[i] = joint_torques_limits_[i];
    }
    if (torque_applied[i] < -joint_torques_limits_[i]) {
      torque_applied[i] = -joint_torques_limits_[i];
    }
  }
}

template <std::size_t N>
void FrankaTorqueControlClient::computeSafetyReflex(
    std::array<double, N> values, std::array<double, N> lower_limit,
    std::array<double, N> upper_limit, bool invert_lower,
    std::array<double, N> &safety_torques, double margin, double k,
    const char *item_name) {
  /*
   * Apply safety mechanisms for a vector based on input values and limits.
   * Throws an error if limits are violated.
   * Also computes & outputs safety controller torques.
   * (Note: invert_lower flips the sign of the lower limit. Used for velocities
   * and torques.)
   */
  double upper_violation, lower_violation;
  double lower_sign = 1.0;
  bool safety_constraint_triggered = false;

  std::string item_name_str(item_name);
  if (invert_lower) {
    lower_sign = -1.0;
  }

  // Init constraint active map
  if (active_constraints_map_.find(item_name_str) ==
      active_constraints_map_.end()) {
    active_constraints_map_.emplace(std::make_pair(item_name_str, false));
  }

  // Check limits & compute safety controller
  for (int i = 0; i < N; i++) {
    upper_violation = values[i] - upper_limit[i];
    lower_violation = lower_sign * lower_limit[i] - values[i];

    // Check hard limits (use active_constraints_map_ to prevent flooding
    // terminal)
    if (upper_violation > 0 || lower_violation > 0) {
      safety_constraint_triggered = true;

      if (!active_constraints_map_[item_name_str]) {
        active_constraints_map_[item_name_str] = true;

        spdlog::warn("Safety limits exceeded: "
                     "\n\ttype = \"{}\""
                     "\n\tdim = {}"
                     "\n\tlimits = {}, {}"
                     "\n\tvalue = {}",
                     item_name_str, i, lower_sign * lower_limit[i],
                     upper_limit[i], values[i]);

        std::string error_str =
            "Safety limits exceeded in FrankaTorqueControlClient. ";
        if (!readonly_mode_) {
          throw std::runtime_error(error_str + "\n");
        } else {
          spdlog::warn(error_str + "Ignoring issue during readonly mode.");
        }
      }
    }

    // Check soft limits & compute feedback forces (safety controller)
    if (is_safety_controller_active_) {
      if (upper_violation > -margin) {
        safety_torques[i] -= k * (margin + upper_violation);
      } else if (lower_violation > -margin) {
        safety_torques[i] += k * (margin + lower_violation);
      }
    }
  }

  // Reset constraint active map
  if (!safety_constraint_triggered && active_constraints_map_[item_name_str]) {
    active_constraints_map_[item_name_str] = false;
    spdlog::info("Safety limits no longer violated: \"{}\"", item_name_str);
  }
}

void *rt_main(void *cfg_ptr) {
  YAML::Node &config = *(static_cast<YAML::Node *>(cfg_ptr));

  // Launch adapter
  std::string control_address = config["control_ip"].as<std::string>() + ":" +
                                config["control_port"].as<std::string>();
  FrankaTorqueControlClient franka_panda_client(
      grpc::CreateChannel(control_address, grpc::InsecureChannelCredentials()),
      config);
  franka_panda_client.run();

  return NULL;
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    spdlog::error("Usage: franka_panda_client /path/to/cfg.yaml");
    return 1;
  }
  YAML::Node config = YAML::LoadFile(argv[1]);
  void *config_void_ptr = static_cast<void *>(&config);

  // Launch thread
  create_real_time_thread(rt_main, config_void_ptr);

  // Termination
  spdlog::info("Wait for shutdown; press CTRL+C to close.");

  return 0;
}
