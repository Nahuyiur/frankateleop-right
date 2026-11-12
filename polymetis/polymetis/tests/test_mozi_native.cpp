// No robot/network connection. Exercises the helpers used by the real callbacks.
#include "mozi_native_reference.hpp"
#include "mozi_tracking_tape.hpp"
#include <cassert>
#include <filesystem>
#include <iostream>

template<class F>void rejects(F fn){bool failed=false;try{fn();}catch(const std::exception&){failed=true;}assert(failed);}
int main() {
  std::array<double,16> identity={1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
  auto desired=identity;desired[12]=.02;
  assert(mozi::native_reference(identity,desired,0,0)==identity);
  assert(std::abs(mozi::native_reference(identity,desired,.5,1)[12]-.01)<1e-12);
  assert(mozi::native_reference(identity,desired,1,2)==desired);
  auto invalid=identity;invalid[12]=.031;
  rejects([&]{mozi::native_reference(identity,invalid,0,0);});
  invalid=identity;invalid[0]=2;
  rejects([&]{mozi::native_reference(identity,invalid,0,0);});
  rejects([&]{mozi::native_reference(identity,desired,.5,2);});
  assert(mozi::native_route(1,1,1,true,123,123,0));
  assert(!mozi::native_route(1,1,1,true,123,123,1)); // intentional cancel, including pending handoff
  rejects([]{mozi::native_route(1,1,1,true,123,123,2);});
  rejects([]{mozi::native_route(1,2,2,true,123,123,0);});
  rejects([]{mozi::native_route(1,1,1,true,124,123,0);});
  rejects([]{mozi::native_route(1,1,0,true,123,123,0);});
  rejects([]{mozi::native_route(1,1,1,false,123,123,0);});
  std::string directory="/tmp/mozi-native-test-"+std::to_string(getpid());
  {
    auto tape=std::make_unique<mozi::TrackingTape>(directory);
    tape->select(123);
    for(int n=0;n<200 && !tape->ready();++n)std::this_thread::sleep_for(std::chrono::milliseconds(5));
    assert(tape->ready()==123 && !tape->failed());
    for(uint64_t n=0;n<1000;++n){mozi::TrackingRow row;row.time[4]=123;row.time[3]=n+1;row.data[0]=n;tape->record(row);}
    tape->select(0);
    for(int n=0;n<200 && tape->ready();++n)std::this_thread::sleep_for(std::chrono::milliseconds(5));
    assert(tape->ready()==0 && !tape->failed());
    assert(std::filesystem::file_size(directory+"/123.bin")==8+1000*sizeof(mozi::TrackingRow));
    // Existing experiment files are never overwritten.
    tape->select(123);
    for(int n=0;n<200 && !tape->failed();++n)std::this_thread::sleep_for(std::chrono::milliseconds(5));
    assert(tape->failed() && tape->ready()==0);
  }
  std::filesystem::remove_all(directory);
  std::cout<<"PASS native prelude transforms, immutable mode/record ownership, cancel/expiry, bounded recorder flush and collision refusal\n";
}
