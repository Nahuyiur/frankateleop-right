# Controller update warmup regression

The native server serializes `UpdateController` and each 1 kHz forward tick with
the same mutex. TorchScript profiles/compiles the first two `update()` calls;
warming only `forward()` left a 2.15–2.19 ms second update on the right MoZI host.

The fix exercises `update()` with copies of the policy's own registered values
before activation, then restores the complete pre-warmup module state. It does
not change torque limits, gains, targets, native protection, or the 0.99 task
communication guard.

Run these tests offline with the same Polymetis environment and shared-library
paths as the backend. No test sends a robot command.

1. Run `stateful_fixture.py` to export the synthetic policy and parameters.
2. Compile `test_warmup.cpp` with C++17, `_GLIBCXX_USE_CXX11_ABI=0`, the
   `torch_isolation/include` directory and `-ltorch_server_ops`.
3. Run the resulting executable in the fixture directory. It verifies that
   construction preserves the initial target, forward/update counters and
   termination flag, and that a subsequent update is actually applied.

For host-specific latency comparison, `fixture.py` exports the current
HybridJointImpedanceControl using read-only server metadata and feedback.
`bench.cpp` links the native torch_server_ops and torchscript_pinocchio libraries
and measures parameter loading, the update under the server mutex, and the next
forward independently. Run it against old and new libraries in separate
processes. Latencies are measurements, not a hard-real-time guarantee.

2026-11-25 MoZI1 right offline result: three cold processes per library;
old second update 2151–2187 us, new first/second update 11–17 us. Stateful
regression passed. Hardware acceptance must be reported separately.

## Remaining online latency and passive trace

The first guarded zero-displacement hardware stream after the warmup patch still
hit a 2.236 ms ControlUpdate RPC and 0.98 communication rate before first target
acknowledgement. The patch removes one measured cold path; it is not full FCI
acceptance. No communication or torque threshold was relaxed.

First-parameter trace kinds: 10 RPC handler entered after priority change;
11 binary stream received; 12 parameter container loaded; 13 controller mutex
acquired. Existing kind 2 is parameter application and kind 3 first forward.
All timestamps use the existing bounded queue/background writer.

concurrent_bench.cpp tests concurrent direct forward/load/update. offline_rpc.py
starts a fake robot on isolated loopback port 50151 and sends no hardware commands;
it reads only live metadata on 50051. Its diagnostic trigger name is isolated.
Do not treat non-realtime synthetic latency as hardware acceptance.

Build note: the host's original CMake cache has an obsolete generated protobuf
header. Rebuilding the server must regenerate both C++ and gRPC files from the
current protos/polymetis.proto and consistently recompile server, run_server and
both protobuf translation units; do not mix the stale cached object files.
