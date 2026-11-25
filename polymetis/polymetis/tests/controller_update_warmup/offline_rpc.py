"""Fake robot on an isolated loopback port. Only GetMetadata touches live server."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import grpc
from polymetis_pb2 import Empty, RobotState, ControllerChunk
from polymetis_pb2_grpc import PolymetisControllerServerStub

out=Path(__file__).resolve().parent
live=PolymetisControllerServerStub(grpc.insecure_channel('127.0.0.1:50051'))
metadata=live.GetRobotClientMetadata(Empty(),timeout=2) # read only
env=os.environ.copy();env.update(MOZI_FAULT_WINDOW_DIR=str(out/'offline-rpc-trace'),MOZI_FAULT_TRIGGER_NAME='mozi_fault_offline_fci_20261125')
log=(out/'offline-rpc-server.log').open('w')
process=subprocess.Popen(['taskset','-c','4',str(out/'run_server.trace'),'-s','127.0.0.1','-p','50151'],env=env,stdout=log,stderr=log)
stop=threading.Event();cycles=[];errors=[]
try:
 channel=grpc.insecure_channel('127.0.0.1:50151');grpc.channel_ready_future(channel).result(timeout=10)
 stub=PolymetisControllerServerStub(channel);stub.InitRobotClient(metadata,timeout=10)
 q=list(map(float,(out/'q.txt').read_text().split()))
 def pump():
  os.sched_setaffinity(0,{6});next_tick=time.monotonic()
  try:
   while not stop.is_set():
    now=time.time_ns();s=RobotState(joint_positions=q,joint_velocities=[0.]*7,motor_torques_measured=[0.]*7,motor_torques_external=[0.]*7)
    s.timestamp.seconds=now//10**9;s.timestamp.nanos=now%10**9
    a=time.time_ns();stub.ControlUpdate(s,timeout=2);b=time.time_ns();cycles.append(dict(start_ns=a,end_ns=b,latency_us=(b-a)/1000))
    next_tick+=.001;stop.wait(max(0,next_tick-time.monotonic()))
  except Exception as e:errors.append(str(e));stop.set()
 worker=threading.Thread(target=pump);worker.start()
 def chunks(path):
  data=path.read_bytes()
  return (ControllerChunk(torchscript_binary_chunk=data[i:i+1024]) for i in range(0,len(data),1024))
 stub.SetController(chunks(out/'policy.pt'),timeout=10)
 calls=[]
 for i in range(12):
  a=time.time_ns();stub.UpdateController(chunks(out/'params.pt'),timeout=2);b=time.time_ns();calls.append(dict(i=i,start_ns=a,end_ns=b,latency_us=(b-a)/1000))
  stop.wait(.067)
 stop.set();worker.join(timeout=3)
 result=dict(isolated=True,hardware_commands=0,cycles=cycles,updates=calls,errors=errors,server_pid=process.pid)
 (out/'offline-rpc-result.json').write_text(json.dumps(result));print(json.dumps({k:v for k,v in result.items() if k!='cycles'}));print('worst_cycles',sorted(cycles,key=lambda x:x['latency_us'],reverse=True)[:10])
finally:
 stop.set();process.terminate();process.wait(timeout=5);log.close()
