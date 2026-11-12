// No robot/network connection. Exercises the helpers used by the real callbacks.
#include "mozi_native_reference.hpp"
#include "mozi_tracking_tape.hpp"
#include <cassert>
#include <filesystem>
#include <iostream>
#include <limits>

template<class F>void rejects(F fn){bool failed=false;try{fn();}catch(const std::exception&){failed=true;}assert(failed);}
int main() {
  std::array<double,16> identity={1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
  {
    auto tool=identity;
    Eigen::Map<Eigen::Matrix4d> transform(tool.data());
    transform.block<3,3>(0,0)=Eigen::AngleAxisd(.3,Eigen::Vector3d::UnitY()).toRotationMatrix();
    transform.block<3,1>(0,3)=Eigen::Vector3d(.044,0,.19);
    const auto flange=mozi::flange_stiffness_frame(tool);
    Eigen::Map<const Eigen::Matrix4d> inverse(flange.data());
    assert((transform*inverse-Eigen::Matrix4d::Identity()).norm()<1e-12);
    mozi::check_flange_stiffness(tool,flange);
    rejects([&]{mozi::check_flange_stiffness(tool,identity);});
    auto invalid=tool;invalid[0]=2;
    rejects([&]{mozi::flange_stiffness_frame(invalid);});
    struct State {std::array<double,16> F_T_EE,EE_T_K;};
    State state{tool,identity};auto pending=identity;int delay=0,sets=0,runs=0;
    auto read=[&]{if(delay && !--delay)state.EE_T_K=pending;return state;};
    auto set=[&](const auto& frame){++sets;pending=frame;delay=2;};
    mozi::with_flange_stiffness(read,set,[&]{++runs;mozi::check_flange_stiffness(state.F_T_EE,state.EE_T_K);});
    assert(runs==1 && sets==2 && state.EE_T_K==identity && state.F_T_EE==tool);
    try {
      mozi::with_flange_stiffness(read,set,[&]{throw std::runtime_error("control failed");});
      assert(false);
    } catch(const std::runtime_error& e) {assert(std::string(e.what())=="control failed");}
    assert(state.EE_T_K==identity);
    sets=0;
    auto lost_ack=[&](const auto& frame){state.EE_T_K=frame;if(++sets==1)throw std::runtime_error("lost setK reply");};
    rejects([&]{mozi::with_flange_stiffness(read,lost_ack,[&]{assert(false);});});
    assert(sets==2 && state.EE_T_K==identity);
    sets=0;
    auto restore_failed=[&](const auto& frame){if(++sets==2)throw std::runtime_error("restore rejected");state.EE_T_K=frame;};
    rejects([&]{mozi::with_flange_stiffness(read,restore_failed,[]{});});
    state.EE_T_K=identity;
    rejects([&]{mozi::with_flange_stiffness(read,[](const auto&){},[&]{assert(false);});});
    assert(state.EE_T_K==identity);
  }
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
  mozi::check_native_tracking(identity,identity,identity);
  auto measured=identity;measured[12]=.0149;
  mozi::check_native_tracking(measured,identity,identity);
  measured[12]=.0151;
  rejects([&]{mozi::check_native_tracking(measured,identity,identity);});
  Eigen::Map<Eigen::Matrix4d> matrix(measured.data());
  matrix.setIdentity();matrix.block<3,3>(0,0)=Eigen::AngleAxisd(.08,Eigen::Vector3d::UnitY()).toRotationMatrix();
  mozi::check_native_tracking(measured,identity,identity);
  matrix.block<3,3>(0,0)=Eigen::AngleAxisd(.09,Eigen::Vector3d::UnitY()).toRotationMatrix();
  rejects([&]{mozi::check_native_tracking(measured,identity,identity);});
  matrix.block<3,3>(0,0)=Eigen::AngleAxisd(.086,Eigen::Vector3d::UnitY()).toRotationMatrix();
  auto tcp=identity;tcp[14]=.19;
  // EE positions coincide and orientation is below 5 degrees, but the flange
  // moves beyond 15 mm due to the TCP lever arm. An EE-only guard would miss it.
  rejects([&]{mozi::check_native_tracking(measured,identity,tcp);});
  measured[12]=std::numeric_limits<double>::quiet_NaN();
  rejects([&]{mozi::check_native_tracking(measured,identity,identity);});
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
  std::cout<<"PASS native transforms, same-cycle flange deviation guard, mode ownership, cancel/expiry, recorder flush and collision refusal\n";
}
