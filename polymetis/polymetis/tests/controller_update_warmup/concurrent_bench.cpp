#include "torch_server_ops.hpp"
#include <chrono>
#include <thread>
#include <mutex>
#include <vector>
#include <fstream>
#include <iterator>
#include <iostream>
#include <pthread.h>
using C=std::chrono::steady_clock;
double micros(C::time_point a,C::time_point b){return std::chrono::duration<double,std::micro>(b-a).count();}
void pin(int cpu){cpu_set_t s;CPU_ZERO(&s);CPU_SET(cpu,&s);pthread_setaffinity_np(pthread_self(),sizeof(s),&s);}
std::vector<char> bytes(const char* p){std::ifstream f(p,std::ios::binary);return {std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>()};}
int main(){
 pin(2);auto p=bytes("policy.pt"),u=bytes("params.pt");std::vector<float> q(7),z(7,0.);std::ifstream f("q.txt");for(auto&v:q)f>>v;
 TorchRobotState state(7);state.update_state(1,0,q,z,z,z);TorchScriptedController policy(p.data(),p.size(),state);
 std::mutex lock;auto origin=C::now();std::vector<std::vector<double>> ticks,updates;
 std::thread realtime([&]{pin(3);auto next=origin;for(int i=0;i<1000;i++){
   next+=std::chrono::milliseconds(1);std::this_thread::sleep_until(next);
   auto a=C::now();lock.lock();auto b=C::now();policy.forward(state);auto c=C::now();lock.unlock();
   ticks.push_back({micros(origin,a),micros(a,b),micros(b,c)});
 }});
 for(int i=0;i<12;i++){
   std::this_thread::sleep_until(origin+std::chrono::milliseconds(100+i*67));
   auto a=C::now();policy.param_dict_load(u.data(),u.size());auto b=C::now();
   lock.lock();auto c=C::now();policy.param_dict_update_module();auto d=C::now();lock.unlock();
   updates.push_back({micros(origin,a),micros(a,b),micros(b,c),micros(c,d)});
 }
 realtime.join();
 for(auto&v:updates)std::cout<<"{\"update_at_us\":"<<v[0]<<",\"load_us\":"<<v[1]<<",\"lock_wait_us\":"<<v[2]<<",\"apply_us\":"<<v[3]<<"}\n";
 for(auto&v:ticks)if(v[1]+v[2]>500.)std::cout<<"{\"slow_forward_at_us\":"<<v[0]<<",\"lock_wait_us\":"<<v[1]<<",\"forward_us\":"<<v[2]<<"}\n";
}
