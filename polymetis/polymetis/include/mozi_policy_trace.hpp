#pragma once
#include "mozi_fault_window.hpp"

namespace mozi {
struct PolicyEvent {
  uint64_t mono=0,wall=0,source_wall=0,generation=0,revision=0,log_index=0;
  uint64_t thread_id=0,entry_ns=0,state_ready_ns=0,locked_ns=0,forwarded_ns=0,completed_ns=0;
  uint32_t kind=0;
};
class PolicyTrace {
  Queue<PolicyEvent,1024> queue_;
  std::atomic_flag gate_=ATOMIC_FLAG_INIT;
  std::atomic<uint64_t> dropped_{0};
  std::atomic<bool> stop_{false};
  std::thread thread_;
public:
  explicit PolicyTrace(std::string directory){
    thread_=std::thread([this,directory]{
      if(!background_thread()){dropped_.fetch_add(1);return;}
      const char* tag=std::getenv("MOZI_FAULT_TRIGGER_NAME");
      std::string trigger=tag?tag:"mozi_fault_native";
      int trigger_socket=-1;
#ifdef __linux__
      trigger_socket=socket(AF_UNIX,SOCK_DGRAM|SOCK_NONBLOCK|SOCK_CLOEXEC,0);
#endif
      auto notify=[&](const PolicyEvent& e){
#ifdef __linux__
        if(e.kind==1 || e.kind==4){
          Event marker;marker.mono=e.mono;marker.wall=e.wall;marker.trigger=true;
          std::strncpy(marker.kind,e.kind==1?"policy_loaded":"policy_terminated",sizeof(marker.kind)-1);
          // Formatting and networking stay entirely on this background thread.
          std::string message="generation="+std::to_string(e.generation)+" revision="+std::to_string(e.revision);
          std::strncpy(marker.message,message.c_str(),sizeof(marker.message)-1);
          send_policy_trigger(trigger_socket,trigger,marker);
        }
#endif
      };
      try {
        std::filesystem::create_directories(directory);
        std::ofstream f(directory+"/policy.jsonl");
        f.exceptions(std::ios::failbit|std::ios::badbit);
        while(!stop_.load()){
          PolicyEvent e;
          while(queue_.pop(e)){
            notify(e);
            f<<"{\"monotonic_ns\":"<<e.mono<<",\"wall_ns\":"<<e.wall
             <<",\"state_source_wall_ns\":"<<e.source_wall<<",\"policy_generation\":"<<e.generation
             <<",\"parameter_revision\":"<<e.revision<<",\"log_index\":"<<e.log_index
             <<",\"kind\":"<<e.kind<<",\"thread_id\":"<<e.thread_id
             <<",\"entry_ns\":"<<e.entry_ns<<",\"state_ready_ns\":"<<e.state_ready_ns
             <<",\"locked_ns\":"<<e.locked_ns<<",\"forwarded_ns\":"<<e.forwarded_ns
             <<",\"completed_ns\":"<<e.completed_ns<<",\"dropped\":"<<dropped_.load()<<"}\n";
          }
          f.flush();std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        PolicyEvent e;while(queue_.pop(e)){
          notify(e);
          f<<"{\"monotonic_ns\":"<<e.mono<<",\"wall_ns\":"<<e.wall
           <<",\"state_source_wall_ns\":"<<e.source_wall<<",\"policy_generation\":"<<e.generation
           <<",\"parameter_revision\":"<<e.revision<<",\"log_index\":"<<e.log_index
           <<",\"kind\":"<<e.kind<<",\"thread_id\":"<<e.thread_id
           <<",\"entry_ns\":"<<e.entry_ns<<",\"state_ready_ns\":"<<e.state_ready_ns
           <<",\"locked_ns\":"<<e.locked_ns<<",\"forwarded_ns\":"<<e.forwarded_ns
           <<",\"completed_ns\":"<<e.completed_ns<<",\"dropped\":"<<dropped_.load()<<"}\n";
        }
      }catch(...){dropped_.fetch_add(1);}
      if(trigger_socket>=0)::close(trigger_socket);
    });
  }
  ~PolicyTrace(){stop_.store(true);thread_.join();}
  bool emit(uint32_t kind,uint64_t generation,uint64_t revision,
            uint64_t index,uint64_t source_wall=0)noexcept{
    PolicyEvent e;e.mono=mono_ns();e.wall=wall_ns();e.kind=kind;
    e.thread_id=static_cast<uint64_t>(pthread_self());
    e.generation=generation;e.revision=revision;e.log_index=index;e.source_wall=source_wall;
    if(gate_.test_and_set(std::memory_order_acquire)){dropped_.fetch_add(1);return false;}
    bool ok=queue_.push(e);gate_.clear(std::memory_order_release);
    if(!ok)dropped_.fetch_add(1);return ok;
  }
  bool timing(uint64_t generation,uint64_t revision,uint64_t index,uint64_t source_wall,
              uint64_t entry,uint64_t state_ready,uint64_t locked,
              uint64_t forwarded,uint64_t completed)noexcept{
    PolicyEvent e;e.mono=completed;e.wall=wall_ns();e.kind=20;
    e.thread_id=static_cast<uint64_t>(pthread_self());
    e.generation=generation;e.revision=revision;e.log_index=index;e.source_wall=source_wall;
    e.entry_ns=entry;e.state_ready_ns=state_ready;e.locked_ns=locked;
    e.forwarded_ns=forwarded;e.completed_ns=completed;
    if(gate_.test_and_set(std::memory_order_acquire)){dropped_.fetch_add(1);return false;}
    bool ok=queue_.push(e);gate_.clear(std::memory_order_release);
    if(!ok)dropped_.fetch_add(1);return ok;
  }
};
} // namespace mozi
