#include "mozi_fault_window.hpp"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <mutex>
#include <vector>

int main(int argc,char** argv){
  mozi::Options options;options.directory=argc>1?argv[1]:"/tmp/mozi-fault-window-fake-test";
  options.pre_ns=20000000ULL;options.post_ns=10000000ULL;options.max_post_ns=20000000ULL;
  std::atomic<bool> entered{false},released{false};
  std::atomic<size_t> observed{0};
  auto capture=std::make_unique<mozi::FaultWindow>(options,[&](const mozi::Window& window){
    entered.store(true);
    // Disk stall occurs only on the writer. Collector/producer must progress.
    while(!released.load())std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if(observed.load()==0)assert(window.count>0);
    assert(window.triggers>=1);observed.fetch_add(1);
  });
  mozi::Sample sample;sample.valid=1;
  for(int i=0;i<35;++i){sample.mono=mozi::mono_ns();sample.wall=mozi::wall_ns();sample.robot_ms=i;capture->record(sample);std::this_thread::sleep_for(std::chrono::milliseconds(1));}
  capture->event("fake_cartesian_reflex","quote: \" and newline\n",true,1ULL<<7);
  for(int i=0;i<15;++i){sample.mono=mozi::mono_ns();capture->record(sample);std::this_thread::sleep_for(std::chrono::milliseconds(1));}
  auto deadline=mozi::mono_ns()+1000000000ULL;
  while(!entered.load() && mozi::mono_ns()<deadline)std::this_thread::sleep_for(std::chrono::milliseconds(1));
  assert(entered.load());
  auto blocked_start=mozi::mono_ns();
  auto consumed_before=capture->consumed();
  std::vector<uint64_t> times;times.reserve(100000);
  for(int i=0;i<100000;++i){
    sample.mono=mozi::mono_ns();sample.wall=mozi::wall_ns();
    auto before=mozi::mono_ns();capture->record(sample);times.push_back(mozi::mono_ns()-before);
    if(i%1000==0)std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  assert(capture->consumed()>consumed_before+1000);
  while(mozi::mono_ns()-blocked_start<10000000000ULL){
    sample.mono=mozi::mono_ns();sample.wall=mozi::wall_ns();capture->record(sample);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  // Two ready/owned slots are bounded; further events count missed windows.
  for(int j=0;j<4;++j){capture->event("fake_recovery_failed","",true);std::this_thread::sleep_for(std::chrono::milliseconds(25));}
  assert(capture->missed_windows()>0);
  released.store(true);
  std::sort(times.begin(),times.end());
  std::cout<<"sample_bytes="<<sizeof(mozi::Sample)<<" capture_bytes="<<sizeof(mozi::FaultWindow)
           <<" record_p99_ns="<<times[times.size()*99/100]
           <<" record_p999_ns="<<times[times.size()*999/1000]
           <<" consumed_while_writer_blocked="<<capture->consumed()-consumed_before
           <<" dropped="<<capture->dropped()<<" missed_windows="<<capture->missed_windows()<<std::endl;
  capture.reset();assert(observed.load()>0);
  // Real file path: trigger metadata, event escaping, binary ABI and disk errors.
  options.pre_ns=100000000ULL;options.post_ns=2000000ULL;
  {
    auto files=std::make_unique<mozi::FaultWindow>(options);
    sample.mono=mozi::mono_ns();sample.wall=mozi::wall_ns();files->record(sample);
    files->event("fake_power_limit_violation","test\n\"",true,1ULL<<32);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  options.directory="/dev/null/cannot-create-diagnostic-directory";
  {
    auto bad=std::make_unique<mozi::FaultWindow>(options);
    sample.mono=mozi::mono_ns();bad->record(sample);bad->event("fake_disk_failed","",true);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
#ifdef __linux__
  options.directory=std::string(argv[1])+"-ipc";
  options.trigger_name="mozi_fake_trigger_"+std::to_string(getpid());
  std::atomic<bool> ipc_seen{false};
  {
    auto ipc=std::make_unique<mozi::FaultWindow>(options,[&](const mozi::Window& window){ipc_seen.store(window.triggers>0);});
    sample.mono=mozi::mono_ns();sample.wall=mozi::wall_ns();ipc->record(sample);
    int sender=socket(AF_UNIX,SOCK_DGRAM|SOCK_NONBLOCK,0);
    mozi::Event marker;marker.mono=mozi::mono_ns();marker.wall=mozi::wall_ns();marker.trigger=true;
    std::strcpy(marker.kind,"fake_policy_switch");
    mozi::send_policy_trigger(sender,options.trigger_name,marker);::close(sender);
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
  }
  assert(ipc_seen.load());
#endif
  std::cout<<"native fake tests OK (no robot connection)"<<std::endl;
}
