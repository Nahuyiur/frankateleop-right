"""Isolated C++ server + generated protobuf + real TorchScript; no FCI client."""
import io
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

BUILD=Path(os.environ['MOZI_TEST_BUILD'])
sys.path.insert(0,str(BUILD))
import grpc
import polymetis_pb2 as pb
import polymetis_pb2_grpc as rpc
import torch
from test_tracking_native import policy


def blob(module):
    out=io.BytesIO();torch.jit.save(module,out);return out.getvalue()


def main():
    for name,number,kind in [('o_t_ee',13,1),('f_t_ee',14,1),('native_state_valid',16,8),
                             ('robot_time_ms',17,4),('state_receive_monotonic_ns',18,4),('prev_command_successful_valid',19,8)]:
        field=pb.RobotState.DESCRIPTOR.fields_by_name[name]
        assert (field.number,field.type)==(number,kind),'Existing native feedback wire contract changed'
    a,b,c=policy(0),policy(1),policy(2)
    with socket.socket() as probe:
        probe.bind(('127.0.0.1',0));port=probe.getsockname()[1]
    log=(BUILD/'fake-server.log').open('w')
    # No -r, no robot client, and a test-only ephemeral loopback port.
    server=subprocess.Popen([str(BUILD/'run_server'),'-s','127.0.0.1','-p',str(port)],stdout=log,stderr=subprocess.STDOUT)
    channel=grpc.insecure_channel('127.0.0.1:'+str(port))
    try:
        grpc.channel_ready_future(channel).result(timeout=10)
        stub=rpc.PolymetisControllerServerStub(channel)
        stub.InitRobotClient(pb.RobotClientMetadata(dof=7,hz=1000,default_controller=blob(a)),timeout=10)
        state=pb.RobotState(joint_positions=[.002]*7,joint_velocities=[.03]*7,
            motor_torques_measured=[0.]*7,motor_torques_external=[0.]*7,native_state_valid=True,
            native_q_d=[.02]*7,native_dq_d=[.1]*7,tracking_control_path=0,robot_time_ms=10000,
            o_t_ee=list(torch.eye(4).flatten().tolist()),f_t_ee=list(torch.eye(4).flatten().tolist()),
            control_command_success_rate=.997,ee_t_k=list(torch.eye(4).flatten().tolist()))
        def step(path):
            wall=time.time_ns();state.timestamp.seconds=wall//10**9;state.timestamp.nanos=wall%10**9
            state.tracking_control_path=path;state.robot_time_ms+=1;state.native_cycle+=1
            return stub.ControlUpdate(state,timeout=3)
        def install(module):
            data=blob(module)
            call=stub.SetController.future(iter([pb.ControllerChunk(torchscript_binary_chunk=data)]),timeout=10)
            while not call.done():
                step(0)
                time.sleep(.002)
            call.result()
        step(0)
        install(b);out=step(1)
        assert out.tracking_valid and out.tracking_policy_path==1 and out.native_tracking.path==1
        assert max(abs(x-.02) for x in out.tracking_q_ref)<1e-7
        assert max(abs(x-.1) for x in out.tracking_dq_ref)<1e-7
        assert max(abs(x-.0075) for x in out.tracking_feedforward)<1e-6
        assert out.native_tracking.record_id==123 and len(out.native_tracking.pose)==16
        read=stub.GetRobotState(pb.Empty(),timeout=3)
        assert list(read.o_t_ee)==list(state.o_t_ee) and read.robot_time_ms==state.robot_time_ms
        assert abs(read.control_command_success_rate-.997)<1e-10
        install(c);out=step(2)
        assert out.native_tracking.path==2 and all(v==0. for v in out.joint_torques)
        # Simulate an unrelated normal controller: no implicit native continuation.
        from torchcontrol.policies import HybridJointImpedanceControl
        from test_tracking_native import Model,BASELINE
        normal=torch.jit.script(HybridJointImpedanceControl(torch.zeros(7),robot_model=Model(),
            **{k:torch.tensor(v) for k,v in BASELINE.items()}))
        install(normal);out=step(2)
        assert not out.tracking_valid and not out.HasField('native_tracking')
        print('PASS protobuf compatibility, same-cycle native q_d/dq_d/ff, B/C routing, foreign-policy detection')
    finally:
        channel.close();server.terminate()
        try:server.wait(timeout=5)
        except subprocess.TimeoutExpired:server.kill();server.wait()
        log.close()


if __name__=='__main__':main()
