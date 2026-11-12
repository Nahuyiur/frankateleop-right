#pragma once
// Experiment recorder: one bounded, allocation-free producer; background disk IO.
#include "mozi_fault_window.hpp"
#include <limits>
#include <iostream>

namespace mozi {
struct TrackingRow {
  std::array<uint64_t,11> time{};
  std::array<double,209> data{};
};
static_assert(sizeof(TrackingRow)==8*(11+209),"Tracking binary layout changed");

class TrackingTape {
  Queue<TrackingRow,8192> queue_;
  std::atomic<uint64_t> wanted_{0},ready_{0};
  std::atomic<bool> failed_{false},stop_{false};
  std::thread worker_;
  std::string root_;

  void write() noexcept {
    if(!background_thread()){failed_=true;return;}
    std::ofstream output;
    uint64_t opened=0;
    try {
      std::filesystem::create_directories(root_);
      while(!stop_.load()) {
        auto wanted=wanted_.load(std::memory_order_acquire);
        if(wanted!=opened) {
          TrackingRow row;
          while(queue_.pop(row)) {
            if(output && row.time[4]==opened)
              output.write(reinterpret_cast<const char*>(&row),sizeof(row));
          }
          if(output.is_open()){output.flush();if(!output)throw std::runtime_error("tracking disk flush failed");output.close();}
          ready_.store(0);
          opened=0;
          if(wanted) {
            if(std::filesystem::space(root_).available < 512ULL*1024*1024)
              throw std::runtime_error("tracking requires 512 MiB free disk");
            auto path=root_+"/"+std::to_string(wanted)+".bin";
            if(std::filesystem::exists(path))throw std::runtime_error("tracking ID already exists");
            output.open(path,std::ios::binary|std::ios::out);
            output.write("MOZITR1\n",8);
            output.flush();
            if(!output)throw std::runtime_error("tracking file open failed");
            opened=wanted;
            ready_.store(opened,std::memory_order_release);
          }
        }
        TrackingRow row;
        for(int i=0;i<512 && queue_.pop(row);++i) {
          if(!output.is_open() || row.time[4]!=opened)
            throw std::runtime_error("tracking row belongs to a different recording");
          output.write(reinterpret_cast<const char*>(&row),sizeof(row));
          if(!output)throw std::runtime_error("tracking write failed");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      }
      TrackingRow row;
      while(queue_.pop(row))if(output.is_open())output.write(reinterpret_cast<const char*>(&row),sizeof(row));
      if(output.is_open()){output.flush();if(!output)failed_=true;}
    }catch(const std::exception& e){failed_=true;ready_=0;std::cerr<<"Tracking recorder failed: "<<e.what()<<std::endl;}
  }
public:
  explicit TrackingTape(std::string root):root_(std::move(root)) {
    worker_=std::thread([this]{write();});
  }
  ~TrackingTape(){stop_=true;if(worker_.joinable())worker_.join();}
  void select(uint64_t id) noexcept {wanted_.store(id,std::memory_order_release);}
  uint64_t ready() const noexcept {return ready_.load(std::memory_order_acquire);}
  bool failed() const noexcept {return failed_.load();}
  void record(const TrackingRow& row) noexcept {
    if(row.time[4] && ready()==row.time[4] && !queue_.push(row))failed_=true;
  }
};
} // namespace mozi
