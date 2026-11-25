#include "torch_server_ops.hpp"
#include <fstream>
#include <iterator>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <cmath>
std::vector<char> bytes(const char* p) {
  std::ifstream f(p,std::ios::binary);
  if(!f)throw std::runtime_error("fixture missing");
  return {std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>()};
}
void expect(const std::vector<float>& values,float expected) {
  if(values.size()!=7)throw std::runtime_error("bad output dimension");
  for(float v:values)if(std::abs(v-expected)>1e-6)throw std::runtime_error("warmup leaked state");
}
int main() {
  auto p=bytes("stateful.pt"), u=bytes("stateful_params.pt");
  TorchRobotState state(7);
  TorchScriptedController policy(p.data(),p.size(),state);
  expect(policy.forward(state),4.); // initial target=3, tick=1, updates=0
  expect(policy.forward(state),5.);
  if(!policy.param_dict_load(u.data(),u.size()))return 2;
  policy.param_dict_update_module();
  expect(policy.forward(state),112.); // new target=9, tick=3, exactly 1 update
  if(policy.is_terminated())throw std::runtime_error("warmup changed termination state");
  std::cout << "warmup preserves target, counter, update count and termination; live update takes effect" << std::endl;
}
