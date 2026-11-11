#pragma once
// Diagnostic only: no RPC, allocation, file IO or waiting in producers.
#include <array>
#include <algorithm>
#include <cstdlib>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <memory>
#include <string>
#include <cstddef>
#include <thread>
#include <unistd.h>
#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/un.h>
#endif

namespace mozi {
inline bool background_thread() noexcept {
#ifdef __linux__
  // std::thread can inherit RR/FIFO from the robot startup thread. Never let a
  // diagnostic consumer or writer retain realtime priority.
  sched_param parameter{};
  if(pthread_setschedparam(pthread_self(),SCHED_OTHER,&parameter)!=0)return false;
  setpriority(PRIO_PROCESS,0,10);
  const long cpus=sysconf(_SC_NPROCESSORS_ONLN);
  if(cpus>2){
    cpu_set_t mask;CPU_ZERO(&mask);
    for(long i=2;i<cpus && i<CPU_SETSIZE;++i)CPU_SET(i,&mask);
    sched_setaffinity(0,sizeof(mask),&mask); // Optional, cgroups can restrict it.
  }
#endif
  return true;
}
inline uint64_t mono_ns() noexcept {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline uint64_t wall_ns() noexcept {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();
}

struct Sample {
  uint64_t sequence=0, mono=0, wall=0, robot_ms=0, period_us=0;
  uint64_t rpc_begin=0, rpc_end=0, state_wall=0, torque_wall=0;
  uint64_t current_errors=0, last_motion_errors=0, dropped=0;
  uint32_t valid=0, source=0, robot_mode=0;
  double success_rate=0;
  std::array<double,7> q{},dq{},q_d{},dq_d{},tau{},tau_d{},tau_ext{};
  std::array<double,7> policy_torque{},safety_torque{},returned_torque{};
  std::array<double,16> O_T_EE{};
  std::array<double,6> O_F_ext_hat_K{};
  std::array<double,7> joint_contact{},joint_collision{};
  std::array<double,6> cartesian_contact{},cartesian_collision{};
};
static_assert(sizeof(Sample)<=1536,"fixed sample budget exceeded");
static_assert(std::atomic<uint64_t>::is_always_lock_free,"requires lock-free counters");

struct Event {
  uint64_t mono=0,wall=0,mask=0,sequence=0;
  bool trigger=false;
  char kind[48]{};
  char message[256]{};
};
#ifdef __linux__
inline sockaddr_un trigger_address(const std::string& name) noexcept {
  sockaddr_un address{};address.sun_family=AF_UNIX;
  // Abstract namespace: no filesystem entry, no stale socket after SIGKILL.
  std::strncpy(address.sun_path+1,name.c_str(),80);return address;
}
inline socklen_t trigger_address_size(const std::string& name) noexcept {
  return offsetof(sockaddr_un,sun_path)+1+std::min(size_t(80),name.size());
}
inline void send_policy_trigger(int socket,const std::string& name,const Event& e) noexcept {
  if(socket<0)return;
  auto address=trigger_address(name);
  sendto(socket,&e,sizeof(e),MSG_DONTWAIT,reinterpret_cast<sockaddr*>(&address),trigger_address_size(name));
}
#endif
template<class T,size_t N> class Queue {
  std::array<T,N> buffer_{};
  alignas(64) std::atomic<uint64_t> write_{0};
  alignas(64) std::atomic<uint64_t> read_{0};
public:
  bool push(const T& value) noexcept {
    auto w=write_.load(std::memory_order_relaxed);
    if(w-read_.load(std::memory_order_acquire)>=N)return false;
    buffer_[w%N]=value;
    write_.store(w+1,std::memory_order_release);
    return true;
  }
  bool pop(T& value) noexcept {
    auto r=read_.load(std::memory_order_relaxed);
    if(r==write_.load(std::memory_order_acquire))return false;
    value=buffer_[r%N];
    read_.store(r+1,std::memory_order_release);
    return true;
  }
};
struct Window {
  // 0 free, 1 collector owns, 2 ready, 3 writer owns.
  std::atomic<int> state{0};
  uint64_t first=0,end=0,first_wall=0,id=0,drop_begin=0,drop_end=0;
  size_t count=0,triggers=0;
  bool truncated=false;
  std::array<Sample,16384> samples{};
};
struct Options {
  uint64_t pre_ns=5000000000ULL,post_ns=3000000000ULL,max_post_ns=10000000000ULL;
  std::string directory,metadata="{}",trigger_name;
};
class FaultWindow {
  Queue<Sample,4096> samples_;
  Queue<Event,256> events_,output_events_;
  std::atomic_flag event_gate_=ATOMIC_FLAG_INIT;
  std::array<Sample,16384> history_{};
  std::array<Window,2> windows_;
  size_t history_write_=0,history_count_=0;
  Window* active_=nullptr;
  uint64_t last_mask_=0,next_id_=0;
  std::atomic<uint64_t> sequence_{0},dropped_{0},event_dropped_{0};
  std::atomic<uint64_t> consumed_{0},completed_{0},missed_windows_{0},write_errors_{0};
  std::atomic<bool> stop_{false},collector_done_{false};
  Options options_;
  std::function<void(const Window&)> test_writer_;
  std::thread collector_,writer_;
  int trigger_socket_=-1;

  void begin(uint64_t when,uint64_t wall) noexcept {
    if(active_ && when<=active_->end){
      ++active_->triggers;
      active_->end=std::min(active_->first+options_.max_post_ns,when+options_.post_ns);
      return;
    }
    if(active_)finish();
    for(auto& w:windows_){
      int free=0;
      if(!w.state.compare_exchange_strong(free,1,std::memory_order_acq_rel))continue;
      active_=&w;w.first=when;w.end=when+options_.post_ns;w.first_wall=wall;
      w.id=++next_id_;w.count=0;w.triggers=1;w.truncated=false;
      w.drop_begin=dropped_.load();
      uint64_t start=when>options_.pre_ns?when-options_.pre_ns:0;
      for(size_t i=0;i<history_count_;++i){
        auto& s=history_[(history_write_+history_.size()-history_count_+i)%history_.size()];
        if(s.mono>=start && s.mono<=w.end)append(s);
      }
      if(w.count)w.drop_begin=w.samples[0].dropped;
      return;
    }
    missed_windows_.fetch_add(1);
  }
  void append(const Sample& s) noexcept {
    if(!active_ || s.mono>active_->end)return;
    if(active_->count<active_->samples.size())active_->samples[active_->count++]=s;
    else active_->truncated=true;
  }
  void finish() noexcept {
    active_->drop_end=dropped_.load();
    active_->state.store(2,std::memory_order_release);
    active_=nullptr;
  }
  void collect() noexcept {
    if(!background_thread()){collector_done_.store(true);return;}
    while(!stop_.load(std::memory_order_relaxed)){
      bool work=false;
      Sample s;
      // Bound a pass so a busy sample producer cannot starve error events.
      for(size_t n=0;n<512 && samples_.pop(s);++n){
        work=true;
        if(s.source==0 && (s.current_errors & ~last_mask_))begin(s.mono,s.wall);
        if(s.source==0)last_mask_=s.current_errors;
        append(s);
        history_[history_write_]=s;
        history_write_=(history_write_+1)%history_.size();
        history_count_=std::min(history_count_+1,history_.size());
        consumed_.fetch_add(1,std::memory_order_relaxed);
      }
      Event e;
      for(size_t n=0;n<256 && events_.pop(e);++n){
        work=true;
        if(e.trigger)begin(e.mono,e.wall);
        if(!output_events_.push(e))event_dropped_.fetch_add(1);
      }
#ifdef __linux__
      if(trigger_socket_>=0){
        for(size_t n=0;n<32;++n){
          Event external;
          auto bytes=recv(trigger_socket_,&external,sizeof(external),MSG_DONTWAIT);
          if(bytes<0)break;
          if(bytes!=sizeof(external))continue;
          external.kind[sizeof(external.kind)-1]=0;
          external.message[sizeof(external.message)-1]=0;
          // Only diagnostics are accepted; no control action is reachable here.
          if(external.mono>mono_ns() || mono_ns()-external.mono>60000000000ULL)continue;
          work=true;external.sequence=sequence_.load(std::memory_order_relaxed);
          if(external.trigger)begin(external.mono,external.wall);
          if(!output_events_.push(external))event_dropped_.fetch_add(1);
        }
      }
#endif
      if(active_ && mono_ns()>active_->end)finish();
      if(!work)std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    // Capture the partial tail honestly during shutdown; never wait for new state.
    Sample s;
    while(samples_.pop(s))append(s);
    Event e;
    while(events_.pop(e)){
      if(e.trigger)begin(e.mono,e.wall);
      if(!output_events_.push(e))event_dropped_.fetch_add(1);
    }
    if(active_){active_->truncated=true;finish();}
    collector_done_.store(true,std::memory_order_release);
  }
  static std::string escaped(const char* input){
    std::string out="\"";
    for(auto p=input;*p;++p){
      unsigned char c=*p;
      if(c=='"'||c=='\\'){out+='\\';out+=c;}
      else if(c>=32)out+=c;
      else {const char* hex="0123456789abcdef";out+="\\u00";out+=hex[c>>4];out+=hex[c&15];}
    }
    return out+'"';
  }
  template<size_t N> static void array(std::ostream& f,const std::array<double,N>& a){
    f<<'[';for(size_t i=0;i<N;++i){if(i)f<<',';f<<a[i];}f<<']';
  }
  void status_file(){
    std::ofstream f(options_.directory+"/status.json.tmp");
    f<<"{\"schema\":1,\"sample_bytes\":"<<sizeof(Sample)<<",\"consumed\":"<<consumed_.load()
     <<",\"dropped_samples\":"<<dropped_.load()<<",\"dropped_events\":"<<event_dropped_.load()
     <<",\"completed\":"<<completed_.load()<<",\"missed_windows\":"<<missed_windows_.load()
     <<",\"write_errors\":"<<write_errors_.load()<<",\"observed_wall_ns\":"<<wall_ns()<<'}';
    f.close();
    std::filesystem::rename(options_.directory+"/status.json.tmp",options_.directory+"/status.json");
  }
  void write_window(const Window& w){
    if(test_writer_){test_writer_(w);return;}
    std::string name=options_.directory+"/incident-"+std::to_string(w.id);
    // Binary is fixed-layout and paired with ABI/schema metadata. Decoder
    // refuses an incompatible sample size. JSON formatting is background only.
    std::ofstream bin(name+".bin",std::ios::binary);
    bin.exceptions(std::ios::failbit|std::ios::badbit);
    bin.write(reinterpret_cast<const char*>(w.samples.data()),w.count*sizeof(Sample));
    bin.close();
    std::ofstream f(name+".json.tmp");
    f.exceptions(std::ios::failbit|std::ios::badbit);
    f<<"{\"schema\":1,\"sample_bytes\":"<<sizeof(Sample)<<",\"sample_count\":"<<w.count
     <<",\"first_monotonic_ns\":"<<w.first<<",\"end_monotonic_ns\":"<<w.end
     <<",\"first_wall_ns\":"<<w.first_wall<<",\"trigger_count\":"<<w.triggers
     <<",\"pre_ns\":"<<options_.pre_ns<<",\"post_ns\":"<<options_.post_ns
     <<",\"truncated\":"<<(w.truncated?"true":"false")
     <<",\"drop_begin\":"<<w.drop_begin<<",\"drop_end\":"<<w.drop_end
     <<",\"available_pre_ns\":"<<(w.count && w.first>w.samples[0].mono?w.first-w.samples[0].mono:0)
     <<",\"metadata\":"<<options_.metadata<<'}';
    f.close();std::filesystem::rename(name+".json.tmp",name+".json");
  }
  void write() noexcept {
    if(!background_thread()){write_errors_.fetch_add(1);return;}
    std::ofstream events;
    bool directory_ready=false;
    uint64_t next_status=0;
    while(true){
      try {
        if(!directory_ready && !test_writer_){
          std::filesystem::create_directories(options_.directory);
          events.open(options_.directory+"/events.jsonl",std::ios::app);
          events.exceptions(std::ios::failbit|std::ios::badbit);
          directory_ready=true;
        }
        Event e;
        while(output_events_.pop(e)){
          if(test_writer_)continue;
          events<<"{\"monotonic_ns\":"<<e.mono<<",\"wall_ns\":"<<e.wall
                <<",\"kind\":"<<escaped(e.kind)<<",\"message\":"<<escaped(e.message)
                <<",\"mask\":"<<e.mask<<",\"sequence\":"<<e.sequence
                <<",\"trigger\":"<<(e.trigger?"true":"false")<<"}\n";
        }
        for(auto& w:windows_){
          int ready=2;
          if(!w.state.compare_exchange_strong(ready,3,std::memory_order_acq_rel))continue;
          try{write_window(w);completed_.fetch_add(1);}
          catch(...){write_errors_.fetch_add(1);}
          w.state.store(0,std::memory_order_release);
        }
        if(!test_writer_ && mono_ns()>next_status){
          events.flush();status_file();next_status=mono_ns()+1000000000ULL;
        }
      }catch(...){
        write_errors_.fetch_add(1);next_status=mono_ns()+1000000000ULL;
        // A permanently unavailable filesystem must not hang shutdown or keep
        // both bounded window slots occupied forever. Loss is counted.
        for(auto& w:windows_){int ready=2;w.state.compare_exchange_strong(ready,0);}
      }
      if(collector_done_.load(std::memory_order_acquire)){
        bool ready=false;for(auto& w:windows_)ready|=w.state.load()==2;
        if(!ready)break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
  }
public:
  static std::string quote(const char* value){return escaped(value);}
  explicit FaultWindow(Options options,std::function<void(const Window&)> writer={})
      :options_(std::move(options)),test_writer_(std::move(writer)){
#ifdef __linux__
    if(!options_.trigger_name.empty()){
      trigger_socket_=socket(AF_UNIX,SOCK_DGRAM|SOCK_NONBLOCK|SOCK_CLOEXEC,0);
      auto address=trigger_address(options_.trigger_name);
      if(trigger_socket_>=0 && bind(trigger_socket_,reinterpret_cast<sockaddr*>(&address),trigger_address_size(options_.trigger_name))!=0){
        close(trigger_socket_);trigger_socket_=-1;
      }
    }
#endif
    try{collector_=std::thread([this]{collect();});writer_=std::thread([this]{write();});}
    catch(...){stop_.store(true);if(collector_.joinable())collector_.join();if(trigger_socket_>=0)::close(trigger_socket_);throw;}
  }
  ~FaultWindow(){stop_.store(true);collector_.join();writer_.join();if(trigger_socket_>=0)::close(trigger_socket_);}
  FaultWindow(const FaultWindow&)=delete;
  bool record(Sample s) noexcept {
    s.sequence=sequence_.fetch_add(1,std::memory_order_relaxed)+1;
    s.dropped=dropped_.load(std::memory_order_relaxed);
    if(!samples_.push(s)){dropped_.fetch_add(1,std::memory_order_relaxed);return false;}
    return true;
  }
  bool event(const char* kind,const char* message="",bool trigger=false,uint64_t mask=0) noexcept {
    Event e;e.mono=mono_ns();e.wall=wall_ns();e.mask=mask;e.trigger=trigger;
    e.sequence=sequence_.load(std::memory_order_relaxed);
    std::strncpy(e.kind,kind,sizeof(e.kind)-1);
    std::strncpy(e.message,message,sizeof(e.message)-1);
    // Independent bounded MPSC event path: contention drops, never waits.
    if(event_gate_.test_and_set(std::memory_order_acquire)){event_dropped_.fetch_add(1);return false;}
    bool ok=events_.push(e);event_gate_.clear(std::memory_order_release);
    if(!ok)event_dropped_.fetch_add(1);return ok;
  }
  uint64_t consumed()const noexcept{return consumed_.load();}
  uint64_t dropped()const noexcept{return dropped_.load();}
  uint64_t missed_windows()const noexcept{return missed_windows_.load();}
};
static_assert(sizeof(FaultWindow)<96ULL*1024*1024,"fixed memory budget exceeded");
} // namespace mozi
