// Copyright (c) Facebook, Inc. and its affiliates.

// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "torch_server_ops.hpp"
#include <istream>
#include <streambuf>
#include <torch/jit.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>
#include <cmath>

extern "C" {

/*
A preallocated chunk of memory, required to convert a char array to an istream.
*/
class membuf : public std::basic_streambuf<char> {
public:
  membuf(const char *p, size_t l) {
    this->setg((char *)p, (char *)p, (char *)p + l);
  }

  pos_type seekoff(off_type off, std::ios_base::seekdir dir,
                   std::ios_base::openmode which = std::ios_base::in) override {
    if (dir == std::ios_base::cur)
      gbump(off);
    else if (dir == std::ios_base::end)
      setg(eback(), egptr() + off, egptr());
    else if (dir == std::ios_base::beg)
      setg(eback(), eback() + off, egptr());
    return gptr() - eback();
  }

  pos_type seekpos(pos_type sp, std::ios_base::openmode which) override {
    return seekoff(sp - pos_type(off_type(0)), std::ios_base::beg, which);
  }
};

/**
TODO
*/
class memstream : public std::istream {
public:
  memstream(const char *p, size_t l) : std::istream(&_buffer), _buffer(p, l) {
    rdbuf(&_buffer);
  }

private:
  membuf _buffer;
};

struct TorchTensor {
  torch::Tensor data;
};
struct TorchScriptModule {
  torch::jit::script::Module data;
};
struct TorchInput {
  std::vector<torch::jit::IValue> data;
};

struct StateDict {
  c10::Dict<std::string, torch::Tensor> data;
};

TorchRobotState::TorchRobotState(int num_dofs) {
  num_dofs_ = num_dofs;

  // Create initial state dictionary
  rs_timestamp_ = new TorchTensor{torch::zeros(2).to(torch::kInt32)};
  rs_joint_positions_ = new TorchTensor{torch::zeros(num_dofs)};
  rs_joint_velocities_ = new TorchTensor{torch::zeros(num_dofs)};
  rs_motor_torques_measured_ = new TorchTensor{torch::zeros(num_dofs)};
  rs_motor_torques_external_ = new TorchTensor{torch::zeros(num_dofs)};
  rs_native_q_d_ = new TorchTensor{torch::zeros(num_dofs)};
  rs_native_dq_d_ = new TorchTensor{torch::zeros(num_dofs)};
  rs_control_path_ = new TorchTensor{torch::zeros(1)};
  rs_robot_ms_ = new TorchTensor{torch::zeros(1).to(torch::kFloat64)};

  state_dict_ = new StateDict{c10::Dict<std::string, torch::Tensor>()};

  state_dict_->data.insert("timestamp", rs_timestamp_->data);
  state_dict_->data.insert("joint_positions", rs_joint_positions_->data);
  state_dict_->data.insert("joint_velocities", rs_joint_velocities_->data);
  state_dict_->data.insert("motor_torques_measured",
                           rs_motor_torques_measured_->data);
  state_dict_->data.insert("motor_torques_external",
                           rs_motor_torques_external_->data);
  state_dict_->data.insert("native_q_d", rs_native_q_d_->data);
  state_dict_->data.insert("native_dq_d", rs_native_dq_d_->data);
  state_dict_->data.insert("tracking_control_path", rs_control_path_->data);
  state_dict_->data.insert("native_robot_ms", rs_robot_ms_->data);

  input_ = new TorchInput{std::vector<torch::jit::IValue>()};
  input_->data.push_back(state_dict_->data);
}

TorchRobotState::~TorchRobotState() {
  delete rs_timestamp_;
  delete rs_joint_positions_;
  delete rs_joint_velocities_;
  delete rs_motor_torques_measured_;
  delete rs_motor_torques_external_;
  delete rs_native_q_d_;
  delete rs_native_dq_d_;
  delete rs_control_path_;
  delete rs_robot_ms_;
  delete state_dict_;
  delete input_;
}

void TorchRobotState::update_state(int timestamp_s, int timestamp_ns,
                                   std::vector<float> joint_positions,
                                   std::vector<float> joint_velocities,
                                   std::vector<float> motor_torques_measured,
                                   std::vector<float> motor_torques_external) {
  rs_timestamp_->data[0] = timestamp_s;
  rs_timestamp_->data[1] = timestamp_ns;
  for (int i = 0; i < joint_positions.size(); i++) {
    rs_joint_positions_->data[i] = joint_positions[i];
    rs_joint_velocities_->data[i] = joint_velocities[i];
    rs_motor_torques_measured_->data[i] = motor_torques_measured[i];
    rs_motor_torques_external_->data[i] = motor_torques_external[i];
  }
}

void TorchRobotState::update_native(const std::vector<float>& q_d,
                                    const std::vector<float>& dq_d, int path, double robot_ms) {
  if(q_d.size()!=num_dofs_ || dq_d.size()!=num_dofs_)
    throw std::runtime_error("Invalid native target dimensions");
  for(int i=0;i<num_dofs_;++i) {
    rs_native_q_d_->data[i]=q_d[i];
    rs_native_dq_d_->data[i]=dq_d[i];
  }
  rs_control_path_->data[0]=path;
  rs_robot_ms_->data[0]=robot_ms;
}

TorchScriptedController::TorchScriptedController(
    char *data, size_t size, TorchRobotState &init_robot_state) {
  memstream stream(data, size);
  module_ = new TorchScriptModule{torch::jit::load(stream)};

  param_dict_input_ = new TorchInput{std::vector<torch::jit::IValue>()};
  empty_input_ = new TorchInput{std::vector<torch::jit::IValue>()};

  // Warm up controller (TorchScript models take time to compile during first 2
  // queries)
  this->warmup_controller(WARM_UP_ITERS, init_robot_state);
}

TorchScriptedController::~TorchScriptedController() {
  delete module_;
  delete param_dict_input_;
  delete empty_input_;
}

std::vector<float> TorchScriptedController::forward(TorchRobotState &input) {
  torch::NoGradGuard no_grad;
  // Step controller & generate torque command response
  c10::Dict<torch::jit::IValue, torch::jit::IValue> controller_state_dict =
      module_->data.forward(input.input_->data).toGenericDict();

  torch::jit::IValue key = torch::jit::IValue("joint_torques");
  torch::Tensor desired_torque = controller_state_dict.at(key).toTensor();
  tracking_trace.clear();
  native_command.clear();
  auto trace_key=torch::jit::IValue("tracking_trace");
  if(controller_state_dict.contains(trace_key)) {
    auto trace=controller_state_dict.at(trace_key).toTensor();
    if(trace.numel()!=22)throw std::runtime_error("Invalid tracking trace");
    tracking_trace.reserve(22);
    for(int i=0;i<22;++i)tracking_trace.push_back(trace[i].item<float>());
  }
  auto command_key=torch::jit::IValue("native_tracking_command");
  if(controller_state_dict.contains(command_key)) {
    auto command=controller_state_dict.at(command_key).toTensor();
    if(command.numel()!=28)throw std::runtime_error("Invalid native tracking command");
    native_command.reserve(28);
    for(int i=0;i<28;++i) {
      double value=command[i].item<double>();
      if(!std::isfinite(value))throw std::runtime_error("Nonfinite native tracking command");
      native_command.push_back(value);
    }
    for(int i:{0,1,2,3,27})if(native_command[i]!=std::floor(native_command[i]))
      throw std::runtime_error("Native tracking IDs/phases must be integers");
    if(native_command[0]<0 || native_command[0]>2 || native_command[1]<0 || native_command[1]>=281474976710656. ||
       native_command[2]<0 || native_command[2]>2 || native_command[3]<-1 || native_command[3]>60000 ||
       native_command[27]<0 || native_command[27]>3)
      throw std::runtime_error("Native tracking command outside its bounds");
  }

  std::vector<float> result;
  for (int i = 0; i < input.num_dofs_; i++) {
    result.push_back(desired_torque[i].item<float>());
  }
  return result;
}

void TorchScriptedController::warmup_controller(
    int warmup_iters, TorchRobotState &init_robot_state) {
  // Backup
  auto tmp_module_data = module_->data.deepcopy();

  // update() also has a profiling/compilation cold path. UpdateController calls
  // it while holding the same mutex as the 1 kHz forward loop, so compile it
  // here, before this controller can be selected by the live loop. Use only
  // this module's registered update parameters (not nested module parameters).
  // Copies avoid aliasing the inputs with the in-place update destination.
  std::vector<torch::jit::IValue> warmup_update_input;
  if (module_->data.hasattr("_param_dict") &&
      module_->data.find_method("update")) {
    c10::Dict<std::string, torch::Tensor> parameters;
    for (const auto &item : module_->data.attr("_param_dict").toGenericDict()) {
      parameters.insert(item.key().toStringRef(), item.value().toTensor().clone());
    }
    warmup_update_input.push_back(parameters);
  }

  // Warmup
  for (int i = 0; i < warmup_iters; i++) {
    if (!warmup_update_input.empty()) {
      module_->data.get_method("update")(warmup_update_input);
    }
    this->forward(init_robot_state);
  }

  // Restore targets, counters and all other policy state; only compiled code
  // is shared with the warmed instance. No warmup output reaches the robot.
  module_->data = tmp_module_data;
}

bool TorchScriptedController::is_terminated() {
  return module_->data.get_method("is_terminated")(empty_input_->data).toBool();
}

void TorchScriptedController::reset() {
  module_->data.get_method("reset")(empty_input_->data);
}

bool TorchScriptedController::param_dict_load(char *data, size_t size) {
  memstream model_stream(data, size);

  torch::jit::script::Module param_dict_container;
  try {
    param_dict_container = torch::jit::load(model_stream);
  } catch (const c10::Error &e) {
    std::cerr << "error loading the param container:\n";
    std::cerr << e.msg() << std::endl;
    return false;
  }

  // Create controller update input dict
  param_dict_input_->data.clear();
  param_dict_input_->data.push_back(
      param_dict_container.forward(empty_input_->data));

  return true;
}

void TorchScriptedController::param_dict_update_module() {
  module_->data.get_method("update")(param_dict_input_->data);
}

} /* extern "C" */
