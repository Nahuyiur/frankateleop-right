#include "mozi_policy_trace.hpp"
int main(int argc,char**argv){if(argc!=2)return 1;mozi::PolicyTrace t(argv[1]);t.emit(10,1,2,3);return t.timing(1,2,3,4,100,120,140,700200,700220)?0:2;}
