"""Real TorchScript and existing DIK/RPC paths, with fake hardware only."""
import copy
import importlib.util
import io
from pathlib import Path
import pickle
import sys
import threading
from types import SimpleNamespace

import numpy as np
import torch
import zmq

ROOT=Path(__file__).resolve().parents[1]/'teleop'
def load(name,filename):
    path={'fr3.py':'robots/fr3.py','robot_node.py':'zmq_core/robot_node.py'}.get(filename,filename)
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module

contract=load('teleop.tracking_gains','tracking_gains.py')
policy_module=load('teleop.tracking_policy','tracking_policy.py')
helper=load('teleop.tracking_impedance','tracking_impedance.py')
telemetry=load('teleop.recording_telemetry','recording_telemetry.py')
fr3=load('candidate_fr3','fr3.py')
sys.modules['teleop.robots.fr3']=fr3
static=load('teleop.tracking_static','tracking_static.py')
node=load('candidate_robot_node','robot_node.py')
clock=[1.]
helper.time=SimpleNamespace(monotonic=lambda:clock[0],time_ns=lambda:int(clock[0]*1e9))
static.time=helper.time


class Model(torch.nn.Module):
    def compute_jacobian(self,q:torch.Tensor):return torch.eye(6,7)
    def inverse_dynamics(self,q:torch.Tensor,dq:torch.Tensor,ddq:torch.Tensor):return dq*.1+ddq+1.
    def forward_kinematics(self,q:torch.Tensor):return q[:3],torch.tensor([0.,0.,0.,1.])


class Interface:
    def __init__(self):
        self.metadata=SimpleNamespace(hz=1000,urdf_file='<fake-only/>',**{'default_'+k:list(v) for k,v in contract.BASELINE.items()})
        self.use_grav_comp=True;self.robot_model=Model();self.default_starts=0;self.episode=10;self.active=True;self.updates=[]
        self.grpc_connection=SimpleNamespace(GetEpisodeInterval=lambda *a,**kw:SimpleNamespace(start=self.episode,end=-1 if self.active else 0))
    def get_joint_positions(self):return torch.zeros(7)
    def start_cartesian_impedance(self):self.default_starts+=1;self.episode+=1;self.active=True
    def start_joint_impedance(self):self.start_cartesian_impedance()
    def is_running_policy(self):return self.active
    def terminate_current_policy(self,**kw):self.active=False
    def send_torch_policy(self,policy,blocking):
        assert blocking is False
        buffer=io.BytesIO();torch.jit.save(torch.jit.script(policy),buffer);buffer.seek(0)
        self.script=torch.jit.load(buffer)
        self.episode+=1;self._teleop_goal_episode=self.episode;self.active=True
        self._teleop_accepted_joint_goal=policy.joint_pos_desired.detach().cpu().tolist()
    def update_current_policy(self,params):
        with torch.no_grad():self.script.update(params)
        self.updates.append({k:v.clone() for k,v in params.items()})
        self._teleop_accepted_joint_goal=params['joint_pos_desired'].tolist()
        return len(self.updates)


def reference(count=20):
    return dict(sha256='a'*64,times_ns=[10**18+i*20_000_000 for i in range(count)],
                eef=[[.4+i*.0001,.1,.3,0.,0.,0.,1.] for i in range(count)])


def robot(start=True, mode=contract.DEFAULT_VELOCITY_MODE):
    r=fr3.fr3Robot.__new__(fr3.fr3Robot);r.robot=Interface();r.control_mode='ee';r._ik_mode='dik'
    if start:
        helper.configure(r,contract.BASELINE,reference(),mode);r._start_control_mode('ee')
    return r


def target(r,joints,arrival_dt=.01):
    clock[0]+=arrival_dt
    e=r._tracking_reference['eef'][r._tracking_reference_index]
    return helper.update_target(r,torch.tensor(joints),e[:3],e[3:])


def test_differentiation_timing_limits_and_exact_torque():
    r=robot();target(r,[0.]*7)
    assert r._tracking_velocity_sample['desired_joint_velocity']==[0.]*7
    target(r,[.004]*7)
    sample=r._tracking_velocity_sample
    assert sample['dt_s']==.02 and np.allclose(sample['raw_joint_velocity'],.2)
    assert np.allclose(sample['desired_joint_velocity'],.1)
    q=torch.full((7,),.002);dq=torch.full((7,),.03)
    with torch.no_grad():actual=r.robot.script({'joint_positions':q,'joint_velocities':dq})['joint_torques']
    kp=torch.tensor(contract.BASELINE['Kq'])+torch.tensor(contract.BASELINE['Kx']+[0.])
    kd=torch.tensor(contract.BASELINE['Kqd'])+torch.tensor(contract.BASELINE['Kxd']+[0.])
    assert torch.allclose(actual,kp*(.004-q)+kd*(.1-dq)+.1*dq,atol=2e-5)
    target(r,[.204]*7)
    assert r._tracking_velocity_sample['limited']
    assert np.max(np.abs(r._tracking_velocity_sample['desired_joint_velocity']))<=1.
    assert np.max(np.abs(np.array(r._tracking_velocity_sample['desired_joint_velocity'])-.1))<=.20001
    results=[]
    for arrival in [.005,.027]:
        r=robot();target(r,[0.]*7,arrival);target(r,[.004]*7,arrival)
        results.append(r._tracking_velocity_sample['desired_joint_velocity'])
    assert results[0]==results[1], 'Network arrival jitter must not change the derivative'


def test_watchdog_hold_repeated_targets_and_restart():
    r=robot();target(r,[0.]*7);target(r,[.004]*7)
    state={'joint_positions':torch.full((7,),.004),'joint_velocities':torch.zeros(7)}
    with torch.no_grad():
        trace=[r.robot.script(state)['joint_torques'].clone() for _ in range(52)]
    assert torch.equal(trace[0],trace[30])
    assert bool(torch.all(torch.abs(trace[40])<torch.abs(trace[0])))
    assert bool(torch.equal(trace[50],torch.zeros(7)))
    target(r,[.008]*7,.08)
    assert r._tracking_velocity_sample['resumed_after_stale'] and r._tracking_velocity_sample['desired_joint_velocity']==[0.]*7
    target(r,[.012]*7);r._hold_current_ee_pose()
    assert r._tracking_velocity_sample['hold'] and r._tracking_velocity_sample['desired_joint_velocity']==[0.]*7
    previous_index=r._tracking_reference_index
    target(r,[.016]*7)
    assert r._tracking_reference_index==previous_index+1 and r._tracking_velocity_sample['desired_joint_velocity']==[0.]*7
    # Identical positions still advance the freshness sequence and produce zero derivative.
    target(r,[.016]*7)
    assert r._tracking_velocity_sample['raw_joint_velocity']==[0.]*7
    r.robot.episode+=1
    for fn in [lambda:helper.snapshot(r),lambda:r._dik_seed([0.]*7),lambda:r._restart_cartesian_at_measured_pose()]:
        try:fn()
        except RuntimeError:pass
        else:raise AssertionError('Foreign/default policy accepted during experiment')
    helper.clear(r);r._start_control_mode('ee')
    assert r.robot.default_starts==1 and helper.snapshot(r)['pending'] is None
    assert r._tracking_velocity_sample is None and helper.defaults(r)==contract.BASELINE


def test_reference_validation_and_dik_telemetry_path():
    for patch in [dict(times_ns=[1,1],eef=reference(2)['eef']),dict(sha256='bad'),dict(eef=[[float('nan')]*7]*20)]:
        try:contract.validate_reference(dict(reference(),**patch))
        except ValueError:pass
        else:raise AssertionError('Invalid trajectory accepted')
    r=robot();e=reference()['eef'][0]
    try:helper.update_target(r,torch.zeros(7),[.6,.1,.3],e[3:])
    except RuntimeError:pass
    else:raise AssertionError('Mismatched trajectory target accepted')
    assert not r.robot.updates and r._tracking_reference_index==0
    helper.update_target(r,torch.zeros(7),e[:3],[-x for x in e[3:]])
    assert r._tracking_reference_index==1
    original=fr3.dik_solve
    fr3.dik_solve=lambda fk,jac,seed,p,rot:(np.asarray(seed)+.004,dict(residual_m=0.,residual_rad=0.,sigma_min=.1))
    events=[];r._recording_telemetry=SimpleNamespace(emit=lambda data,stamp:events.append((data,stamp)))
    try:
        clock[0]+=.01;e=reference()['eef'][1]
        r._update_ee_target(torch.tensor(e[:3]),torch.tensor(e[3:]))
        assert events[-1][0]['tracking_velocity']['reference_index']==1
        assert np.allclose(events[-1][0]['tracking_velocity']['desired_joint_velocity'],.1)
        assert r.robot.updates[-1]['target_sequence'].item()==2
    finally:fr3.dik_solve=original
    r._tracking_reference_index=len(reference()['times_ns'])
    try:helper.update_target(r,torch.zeros(7),e[:3],e[3:])
    except RuntimeError:pass
    else:raise AssertionError('Unexpected extra command accepted')


def test_owned_rpc_configuration_and_restore():
    r=robot(False);server=node.ZMQServerRobot(r,host='127.0.0.1',port=0)
    endpoint=server._socket.getsockopt_string(zmq.LAST_ENDPOINT)
    worker=threading.Thread(target=server.serve);worker.start()
    socket=zmq.Context.instance().socket(zmq.REQ);socket.setsockopt(zmq.LINGER,0);socket.connect(endpoint)
    def call(method,**args):
        socket.send(pickle.dumps(dict(method=method,args=args)));assert socket.poll(2000,zmq.POLLIN)
        return pickle.loads(socket.recv())
    try:
        assert call('get_replay_capabilities')['tracking_impedance']==3
        assert call('get_replay_capabilities')['static_sampling']==1
        assert call('get_replay_capabilities')['native_tracking']==1
        for method in ['configure_tracking_static','freeze_tracking_target','get_tracking_static_sample',
                       'configure_tracking_record','configure_tracking_native','get_tracking_native','start_tracking_native','finish_tracking_native']:
            assert 'error' in call(method), 'Static requests must require the replay owner'
        assert 'error' in call('configure_tracking_impedance',gains=contract.BASELINE,reference=reference())
        assert call('reserve_replay',replay_token='owned')['ok']
        assert 'error' in call('configure_tracking_record',record_id=123,replay_token='other')
        assert call('configure_tracking_record',record_id=123,replay_token='owned')['record_id']==123
        assert 'error' in call('configure_tracking_impedance',gains=contract.BASELINE,replay_token='owned')
        result=call('configure_tracking_impedance',gains=contract.BASELINE,reference=reference(),velocity_mode=contract.DEFAULT_VELOCITY_MODE,replay_token='owned')
        assert result['reference']['count']==20 and result['velocity_reference']==contract.VELOCITY_REFERENCE
        assert call('set_replay_mode',mode='ee',replay_token='owned') is None
        assert call('get_tracking_impedance')['applied']['velocity_reference']==contract.VELOCITY_REFERENCE
        assert 'error' in call('configure_tracking_record',record_id=456,replay_token='owned')
        assert 'error' in call('release_replay',replay_token='other')
        assert call('release_replay',replay_token='owned')['ok']
        assert call('get_tracking_impedance')['applied'] is None and r.robot.default_starts==1
    finally:
        socket.close();server.stop();worker.join(3);server._socket.close();server._context.term()
    assert not worker.is_alive()


def test_zero_velocity_matches_original_policy_and_records_formula():
    from torchcontrol.policies import HybridJointImpedanceControl
    r=robot(mode='zero');target(r,[0.]*7);target(r,[.004]*7)
    assert r._tracking_velocity_sample['desired_joint_velocity']==[0.]*7
    p=helper.snapshot(r)['applied'];law=contract.control_law('zero')
    assert p['formula']==law['formula'] and p['control_law_id']==law['id']
    assert p['velocity_formula']=='dq_des = 0' and p['velocity_reference']['mode']=='zero'
    original=HybridJointImpedanceControl(torch.zeros(7),**{k:torch.tensor(v) for k,v in contract.BASELINE.items()},robot_model=Model(),ignore_gravity=True)
    with torch.no_grad():
        original.update(dict(joint_pos_desired=torch.full((7,),.004)))
        for i in range(55):
            state={'joint_positions':torch.full((7,),.002+i*.00001),'joint_velocities':torch.full((7,),.03)}
            assert torch.allclose(r.robot.script(state)['joint_torques'],original(state)['joint_torques'],atol=1e-6)
    helper.clear(r)
    assert r._tracking_velocity_mode is None
    try:helper.configure(r,contract.BASELINE,reference(),'unknown')
    except ValueError:pass
    else:raise AssertionError('Invalid mode accepted')
    assert r._tracking_impedance_override is None


def static_robot(saved=None):
    r=robot(False);r._max_gripper_width=.09
    helper.configure(r,contract.BASELINE,reference(2))
    static.configure(r,saved,.09)
    r._start_control_mode('ee')
    target(r,[0,0,0,-1,0,1,0]);target(r,[.01,0,0,-1,0,1,0])
    r.measured=[.013,0,0,-1,0,1,0]
    r.width=.09
    r.get_feedback_observations=lambda **kw:dict(
        state_timestamp_ns=helper.time.time_ns(),native_state_valid=True,
        joint_positions=r.measured,joint_velocities=[.001]*7,
        ee_pos_quat=r.measured[:3]+[0.,0.,0.,1.],
        native_F_T_EE=[1.,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],
        joint_torques=[1.]*7,joint_torques_desired=[2.]*7,joint_torques_external=[.1]*7,
        robot_time_ms=100,state_receive_monotonic_ns=helper.time.time_ns(),
        control_command_success_rate=1.,gripper_width=[r.width],gripper_width_source='measured')
    return r


def fails(fn,text):
    try:fn()
    except (RuntimeError,ValueError) as e:assert text.lower() in str(e).lower(),str(e)
    else:raise AssertionError('Invalid static state accepted: '+text)


def test_static_freeze_never_relatches_measured_and_samples_same_state():
    r=static_robot();before=len(r.robot.updates)
    from teleop.native_observation import pose
    read=r.get_feedback_observations
    def native_feedback(**kwargs):
        raw=read(**kwargs)
        matrix=np.eye(4);matrix[:3,3]=r.measured[:3]
        raw['ee_pos_quat']=pose(matrix)
        return raw
    r.get_feedback_observations=native_feedback
    frozen=static.freeze(r)
    assert len(r.robot.updates)==before, 'Freeze must not command the measured pose or restart the policy'
    assert abs(frozen['q_ref'][0]-.01)<1e-8 and frozen['q_ref']!=r.measured
    r.measured[0]=.017
    s=static.sample(r)
    assert s['q'][0]==.017 and abs(s['fk_measured'][0]-.017)<1e-7
    assert frozen==r._tracking_static_frozen and abs(frozen['fk_ref'][0]-.01)<1e-8
    assert np.allclose(s['feedforward_Nm'],.0001,atol=1e-7)
    assert len(r.robot.updates)==before
    fails(lambda:static.freeze(r),'already frozen')
    r.width=.08;fails(lambda:static.sample(r),'gripper')
    r.width=.09;r.robot.episode+=1;fails(lambda:static.sample(r),'no longer active')


def test_static_hold_lost_target_and_incomplete_prefix_fail_closed():
    r=static_robot();r._tracking_reference_index=1
    fails(lambda:static.freeze(r),'complete prefix')
    r._tracking_reference_index=2;static.freeze(r)
    fails(lambda:helper.update_target(r,torch.tensor([.02,0,0,-1,0,1,0]),[0,0,0],[0,0,0,1]),'frozen')
    r._hold_current_ee_pose()
    fails(lambda:static.sample(r),'safety hold')
    r=static_robot();r.robot._teleop_accepted_joint_goal=[0.]*7
    fails(lambda:static.freeze(r),'target changed')
    helper.clear(r)
    assert r._tracking_static_config is None and r._tracking_static_frozen is None


def test_saved_static_target_selected_only_at_endpoint_and_jump_guarded():
    saved=[.015,0,0,-1,0,1,0]
    r=robot(False);r._max_gripper_width=.09
    helper.configure(r,contract.BASELINE,reference(2));static.configure(r,saved,.09);r._start_control_mode('ee')
    seed=np.array([.01,0,0,-1,0,1,0]);computed=seed+.001
    assert np.array_equal(static.select_target(r,computed,seed),computed)
    r._tracking_reference_index=1
    assert np.allclose(static.select_target(r,computed,seed),saved)
    fails(lambda:static.select_target(r,computed,seed+.21),'too far')
    fails(lambda:static.configure(r,saved,.09),'precede positioning')
    r=robot(False);r._max_gripper_width=.09;helper.configure(r,contract.BASELINE,reference())
    fails(lambda:static.configure(r,[0]*7,.09),'joint limits')
    fails(lambda:static.configure(r,None,float('nan')),'gripper')


if __name__=='__main__':
    for fn in [test_differentiation_timing_limits_and_exact_torque,test_watchdog_hold_repeated_targets_and_restart,
               test_reference_validation_and_dik_telemetry_path,test_owned_rpc_configuration_and_restore,
               test_zero_velocity_matches_original_policy_and_records_formula,
               test_static_freeze_never_relatches_measured_and_samples_same_state,
               test_static_hold_lost_target_and_incomplete_prefix_fail_closed,
               test_saved_static_target_selected_only_at_endpoint_and_jump_guarded]:fn()
    print('PASS: real TorchScript torque/expiry, trajectory timing, limits, holds, DIK telemetry, reservation and restoration; no hardware connection.')


def test_online_reference_timing_and_hold():
    from collections import deque
    import pytest
    r=robot(False)
    helper.configure(r,contract.BASELINE,None,online_hz=10)
    r._online_references=deque()
    r._online_velocity_log=deque(maxlen=2048)
    r._start_control_mode('ee')
    assert r.robot.script.fade_ticks == 130 and r.robot.script.zero_ticks == 150
    pose=[.41,.12,.3,0.,0.,0.,1.]
    def send(delta, q):
        stamp=round((clock[0]+delta)*1e9)
        helper.queue_online_reference(r, stamp, pose)
        clock[0]+=delta
        helper.update_target(r,torch.full((7,),q),pose[:3],pose[3:])
        return r._tracking_velocity_sample['desired_joint_velocity']
    assert not any(send(.1,0.))
    assert all(v>0 for v in send(.1,.01))
    assert not any(send(.2,.02))
    helper.update_target(r,torch.zeros(7),hold=True)
    assert not any(send(.1,.01))
    helper.queue_online_reference(r,round((clock[0]+.1)*1e9),pose)
    clock[0]+=.1
    with pytest.raises(RuntimeError,match='does not match'):
        helper.update_target(r,torch.zeros(7),[.5,.12,.3],pose[3:])
