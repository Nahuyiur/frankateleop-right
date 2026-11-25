"""Functional log-export regression on an isolated synthetic robot, never hardware."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import grpc
from polymetis_pb2 import Empty, LogInterval, RobotState
from polymetis_pb2_grpc import PolymetisControllerServerStub

root = Path(__file__).resolve().parent
binary = sys.argv[1]
live = PolymetisControllerServerStub(grpc.insecure_channel('127.0.0.1:50051'))
metadata = live.GetRobotClientMetadata(Empty(), timeout=2)
env = dict(os.environ, MOZI_NATIVE_FAULT_WINDOW='0')
with (root/'offline-log-server.log').open('w') as log:
    server = subprocess.Popen(['taskset','-c','4',str(root/binary),'-s','127.0.0.1','-p','50151'],env=env,stdout=log,stderr=log)
    stop = threading.Event()
    cycles = []
    errors = []
    try:
        channel = grpc.insecure_channel('127.0.0.1:50151')
        grpc.channel_ready_future(channel).result(timeout=10)
        stub = PolymetisControllerServerStub(channel)
        stub.InitRobotClient(metadata, timeout=10)
        q = list(map(float,(root/'q.txt').read_text().split()))
        def pump():
            os.sched_setaffinity(0,{6})
            tick = time.monotonic()
            try:
                while not stop.is_set():
                    now = time.time_ns()
                    state = RobotState(joint_positions=q,joint_velocities=[0.]*7,motor_torques_measured=[0.]*7,motor_torques_external=[0.]*7)
                    state.timestamp.seconds=now//10**9;state.timestamp.nanos=now%10**9
                    started=time.perf_counter_ns();stub.ControlUpdate(state,timeout=2)
                    cycles.append((time.perf_counter_ns()-started)/1e6)
                    tick += .001
                    stop.wait(max(0,tick-time.monotonic()))
            except Exception as exc:
                errors.append(str(exc));stop.set()
        worker=threading.Thread(target=pump);worker.start()
        deadline=time.monotonic()+15
        while len(cycles)<6500 and not errors:
            if time.monotonic()>deadline: raise TimeoutError('synthetic fill deadline')
            stop.wait(.01)
        n=len(cycles);started=time.perf_counter_ns()
        rows=list(stub.GetRobotStateLog(LogInterval(start=100,end=6398),timeout=5))
        elapsed=(time.perf_counter_ns()-started)/1e6
        assert len(rows)==6299, len(rows)
        stream=stub.GetRobotStateLog(LogInterval(start=100,end=6398),timeout=5)
        for i,row in enumerate(stream):
            if i==20: stream.cancel();break
        fresh=stub.GetRobotState(Empty(),timeout=2)
        assert fresh.joint_positions and not errors, errors
        stop.set();worker.join(timeout=3)
        result=dict(binary=binary,hardware_commands=0,export_rows=len(rows),export_ms=elapsed,cycles_during_export=len(cycles)-n,max_cycle_ms=max(cycles[n:]),cancellation_and_followup_read='passed',errors=errors,note='SCHED_OTHER synthetic test; not real-time or hardware acceptance')
        (root/(binary+'.offline-log.json')).write_text(json.dumps(result))
        print(json.dumps(result))
    finally:
        stop.set();server.terminate();server.wait(timeout=5)
