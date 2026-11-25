#include "torch_server_ops.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>
#include <cmath>
using Clock = std::chrono::steady_clock;
double us(Clock::time_point start) {
  return std::chrono::duration<double, std::micro>(Clock::now()-start).count();
}
std::vector<char> read(const char* path) {
  std::ifstream f(path, std::ios::binary);
  if(!f) throw std::runtime_error("fixture missing");
  return {std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>()};
}
int main() {
  auto model=read("policy.pt"), params=read("params.pt");
  std::ifstream qfile("q.txt"); std::vector<float> q(7), zero(7,0.);
  for(auto& value:q) qfile >> value;
  TorchRobotState state(7); state.update_state(1,0,q,zero,zero,zero);
  auto start=Clock::now();
  TorchScriptedController controller(model.data(),model.size(),state);
  std::cout << "{\"construct_us\":" << us(start) << "}" << std::endl;
  for(int i=0;i<12;i++) {
    for(int tick=0;tick<10;tick++) controller.forward(state);
    start=Clock::now();
    if(!controller.param_dict_load(params.data(),params.size())) return 2;
    double load=us(start);
    start=Clock::now(); controller.param_dict_update_module();
    double update=us(start);
    start=Clock::now(); auto torque=controller.forward(state);
    double forward=us(start);
    for(auto value:torque) if(!std::isfinite(value)) return 3;
    std::cout << "{\"iteration\":" << i << ",\"load_us\":" << load
              << ",\"update_under_lock_us\":" << update
              << ",\"first_forward_us\":" << forward << "}" << std::endl;
  }
}
