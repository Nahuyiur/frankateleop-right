"""Owned B/C experiments in the existing Polymetis client and FCI connection."""
import copy
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import torch

from teleop.tracking_gains import FORMULA, validate_gains

ROOT=Path('/home/pnp/work/rtc-abc-preload')


def observed(robot):
    raw=robot.get_feedback_observations(recording=True)
    if not raw.get('native_state_valid') or not -20_000_000<=time.time_ns()-raw['state_timestamp_ns']<=150_000_000:
        raise RuntimeError('Fresh native feedback required')
    state=raw.get('tracking_native')
    if not isinstance(state,dict):raise RuntimeError('Native C++ feedback upgrade is not active')
    if state['log_failed'] or state['path']==4:raise RuntimeError('Latched native experiment/recorder fault; manual recovery required')
    if state['path']==2:
        tool=np.asarray(raw['native_F_T_EE']).reshape(4,4,order='F')
        frame=np.asarray(state['EE_T_K']).reshape(4,4,order='F')
        if not np.allclose(tool@frame,np.eye(4),atol=1e-6,rtol=0):
            raise RuntimeError('Native C stiffness frame readback is not the flange')
    return raw,state


def active(robot):
    record=getattr(robot,'_tracking_native',None)
    if not record:raise RuntimeError('No owned native experiment')
    from teleop.tracking_impedance import snapshot
    snapshot(robot) # verifies the active policy episode, not just its parameters
    return record


def validate_integral(value):
    """P1 B-only slow integral; None keeps the unchanged B baseline."""
    if value is None:return None
    if not isinstance(value,dict) or set(value)!={'time_s','leak_s','limit_Nm'}:
        raise ValueError('Integral requires time_s, leak_s and limit_Nm')
    t,leak,limit=value['time_s'],value['leak_s'],value['limit_Nm']
    if type(t) not in (int,float) or not .2<=t<=5:raise ValueError('Integral time must be 0.2-5 s')
    if type(leak) not in (int,float) or not 1<=leak<=30 or leak<=t:raise ValueError('Integral leak must be 1-30 s and exceed the integral time')
    if (not isinstance(limit,list) or len(limit)!=7 or
            any(type(v) not in (int,float) or not np.isfinite(v) or not 0<v<=3 for v in limit)):
        raise ValueError('Integral torque limit must be seven values in (0, 3] Nm')
    return dict(time_s=float(t),leak_s=float(leak),limit_Nm=[float(v) for v in limit])


def validate_friction(value):
    """B-only Coulomb feedforward on desired velocity; None keeps it off."""
    if value is None:return None
    if not isinstance(value,dict) or set(value)!={'coulomb_Nm','velocity_rad_s'}:
        raise ValueError('Friction feedforward requires coulomb_Nm and velocity_rad_s')
    c,v0=value['coulomb_Nm'],value['velocity_rad_s']
    if (not isinstance(c,list) or len(c)!=7 or
            any(type(x) not in (int,float) or not np.isfinite(x) or not 0<=x<=1 for x in c)):
        raise ValueError('Coulomb feedforward must be seven values in [0, 1] Nm')
    if type(v0) not in (int,float) or not .005<=v0<=.2:raise ValueError('Friction smoothing velocity must be 0.005-0.2 rad/s')
    return dict(coulomb_Nm=[float(x) for x in c],velocity_rad_s=float(v0))


def configure(robot, artifact, sha256, path, record_id, stiffness, integral=None, friction=None):
    from teleop.tracking_reference import check,check_native_mode
    from teleop.tracking_policy import NativeTrackingHybridJointImpedanceControl
    from teleop.tracking_impedance import snapshot
    if getattr(robot,'_tracking_native',None):raise RuntimeError('Native experiment already configured')
    if path not in ('B','C'):raise ValueError('Native path must be B or C')
    integral=validate_integral(integral)
    if integral and path!='B':raise ValueError('Integral correction is B-only')
    friction=validate_friction(friction)
    if friction and path!='B':raise ValueError('Friction feedforward is B-only')
    if type(record_id) is not int or not 0<record_id<2**48:raise ValueError('Invalid recording ID')
    if not isinstance(artifact,str) or not re.fullmatch(r'[a-f0-9]{24}\.npz',artifact):raise ValueError('Invalid preload artifact name')
    source=ROOT/artifact
    if source.is_symlink() or source.stat().st_size>20_000_000:raise ValueError('Invalid preload file')
    if hashlib.sha256(source.read_bytes()).hexdigest()!=sha256:raise ValueError('Preload checksum mismatch')
    with np.load(source,allow_pickle=False) as stored:
        poses=stored['poses'].copy()
        metadata=json.loads(str(stored['metadata'].item()))
    peaks=check(poses)
    required_client=check_native_mode(path,poses,metadata)
    if hashlib.sha256(poses.astype('<f8').tobytes()).hexdigest()!=metadata['processed_sha256']:
        raise ValueError('Processed reference checksum mismatch')
    raw,state=observed(robot)
    if state['path']!=0:raise RuntimeError('Native handoff must start from path A')
    if not np.allclose(raw['native_F_T_EE'],metadata['F_T_EE'],atol=1e-8,rtol=0):raise RuntimeError('TCP changed after reference preparation')
    if max(abs(v) for v in raw['joint_velocities'])>=.01:raise RuntimeError('Robot must be stationary before native handoff')
    width=raw.get('gripper_width',())
    if raw.get('gripper_width_source')!='measured' or len(width)!=1 or not np.isfinite(width[0]):
        raise RuntimeError('Fixed measured gripper opening required')
    applied=snapshot(robot)['applied']
    if not applied or applied['controller']!='TrackingHybridJointImpedanceControl':raise RuntimeError('Verified A preparation policy required')
    if required_client and applied.get('runtime',{}).get('binaries_sha256',{}).get(
            '/home/pnp/frankateleop/polymetis/polymetis/build/franka_panda_client')!=required_client:
        raise RuntimeError('Guarded C reference requires the verified 15 mm / 5 degree native client')
    gains=validate_gains(applied['gains'])
    if (not isinstance(stiffness,list) or len(stiffness)!=6 or
            any(type(v) not in (int,float) or not np.isfinite(v) or not 0<v<=(1500 if i<3 else 45) for i,v in enumerate(stiffness))):
        raise ValueError('Invalid native Cartesian stiffness')
    interface=robot.robot
    if not interface.use_grav_comp or interface.metadata.hz!=1000:raise RuntimeError('Verified 1 kHz gravity-compensated model required')
    policy=NativeTrackingHybridJointImpedanceControl(joint_pos_current=interface.get_joint_positions(),
        robot_model=interface.robot_model,ignore_gravity=True,poses=torch.from_numpy(poses),path=1 if path=='B' else 2,
        record_id=record_id,stiffness=torch.tensor(stiffness,dtype=torch.float64),
        **(dict(integral_time_s=integral['time_s'],integral_leak_s=integral['leak_s'],
                integral_limit_Nm=torch.tensor(integral['limit_Nm'],dtype=torch.float32)) if integral else {}),
        **(dict(friction_Nm=torch.tensor(friction['coulomb_Nm'],dtype=torch.float32),
                friction_velocity_rad_s=friction['velocity_rad_s']) if friction else {}),
        **{k:torch.tensor(v,dtype=torch.float32) for k,v in gains.items()})
    # A failed/uncertain SetController must retain ownership and fail closed.
    robot._tracking_native=dict(path=path,record_id=record_id,pending=True)
    interface.send_torch_policy(policy,blocking=False)
    record=dict(path=path,record_id=record_id,artifact=artifact,artifact_sha256=sha256,reference=metadata,
        peaks=peaks,policy_episode=int(interface._teleop_goal_episode),gains=gains if path=='B' else None,
        runtime=applied['runtime'],model_sha256=applied['model_sha256'],
        software_sha256=dict(applied['software_sha256'],**{str(p):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__),Path(__file__).with_name('tracking_reference.py'))}),
        cartesian_stiffness=list(stiffness) if path=='C' else None,prelude_s=3.,
        controller='NativeTrackingHybridJointImpedanceControl' if path=='B' else 'Franka internal Cartesian impedance',
        control_law_id=('hybrid-native-qd-integral-v1' if integral else 'hybrid-native-qd-v1') if path=='B' else 'franka-cartesian-impedance-firmware',
        integral=integral,friction=friction,
        friction_formula=('tau_f = Fc * tanh(dq_d / v0) on native desired velocity in B only; tau = tau_policy + tau_f (+ tau_i)')
            if friction else None,
        integral_formula=('tau_i = Kp(q) z; dz/dt = (q_d - q)/Ti - z/T_leak in phases 1-2; z decays with 0.2 s otherwise; '
                          '|tau_i| <= limit by uniform scaling; tau = tau_policy + tau_i') if integral else None,
        formula=FORMULA if path=='B' else 'w ≈ K_C e_x + D_C(v_ref − v); explanatory model, not a verified complete firmware law',
        velocity_formula='dq_ref = native dq_d from the same RobotState' if path=='B' else 'Franka internal motion generator; no independent KD interface',
        initial_native=copy.deepcopy(state),F_T_EE=list(raw['native_F_T_EE']),
        gripper_width_m=float(raw['gripper_width'][0]),configured_ns=time.time_ns())
    if path=='C':
        record['stiffness_frame']=dict(mode='flange',version='C-flange-K-v1',
            before_EE_T_K=list(state['EE_T_K']),
            requested_EE_T_K=np.linalg.inv(np.asarray(raw['native_F_T_EE']).reshape(4,4,order='F')).T.reshape(16).tolist(),
            verification='native readback before motion and on every C cycle; F_T_EE @ EE_T_K = I',
            exit='restore previous EE_T_K on normal completion and control exception')
    robot._tracking_native=record
    robot._tracking_impedance_applied=dict(record,applied_ns=time.time_ns(),hz=1000,reference=metadata)
    return copy.deepcopy(record)


def status(robot, heartbeat=True):
    record=active(robot)
    if heartbeat:
        robot.robot.update_current_policy(dict(lease_wall_ns=torch.tensor([float(time.time_ns()+3_000_000_000)],dtype=torch.float64)))
    raw,state=observed(robot)
    if state['record_id'] not in (0,record['record_id']):raise RuntimeError('Native recording ID changed')
    if not np.allclose(raw['native_F_T_EE'],record['F_T_EE'],atol=1e-8,rtol=0):raise RuntimeError('TCP changed during experiment')
    width=raw.get('gripper_width',())
    if (raw.get('gripper_width_source')!='measured' or len(width)!=1 or not np.isfinite(width[0])
            or abs(width[0]-record['gripper_width_m'])>.001):
        raise RuntimeError('Fixed measured gripper opening changed or became unavailable: source=%s, error=%s, detail=%s' % (
            raw.get('gripper_width_source'),raw.get('gripper_error_code'),raw.get('gripper_observation_error','')))
    from teleop.tracking_static import fk
    return dict(state,source_ns=raw['state_timestamp_ns'],q=list(raw['joint_positions']),
                dq=list(raw['joint_velocities']),eef=[float(v) for v in raw['ee_pos_quat']],
                fk_measured=fk(robot.robot.robot_model,raw['joint_positions']),
                fk_ref=fk(robot.robot.robot_model,state['q_d']),
                torque_measured=list(raw['joint_torques']),torque_desired=list(raw['joint_torques_desired']),
                torque_external=list(raw['joint_torques_external']),robot_time_ms=raw['robot_time_ms'],
                fci_success_rate=raw['control_command_success_rate'],
                gripper_width_m=float(width[0]),F_T_EE=list(raw['native_F_T_EE']))


def start(robot, start_ns):
    record=active(robot);raw,state=observed(robot)
    if state['path']!=({'B':1,'C':2}[record['path']]) or state['log_ready_id']!=record['record_id']:
        raise RuntimeError('Native controller/recorder not ready')
    if 'started_ns' in record:raise RuntimeError('Native reference already started')
    if type(start_ns) is not int or not .2e9<=start_ns-time.time_ns()<=5e9:raise ValueError('Start must be 0.2–5 s in the future')
    robot.robot.update_current_policy(dict(start_wall_ns=torch.tensor([float(start_ns)],dtype=torch.float64),
        lease_wall_ns=torch.tensor([float(time.time_ns()+3_000_000_000)],dtype=torch.float64)))
    record['started_ns']=start_ns
    return dict(start_ns=start_ns,record_id=record['record_id'])


def finish(robot):
    """Stop native motion before restoring a controller anchored at measured q."""
    record=active(robot)
    raw,state=observed(robot)
    # MotionFinished is only valid at a stationary reference endpoint. A peer
    # fault or operator stop during motion must abort and retain ownership.
    if state['phase']!=3 or max(abs(v) for v in state['dq_d'])>=.001:
        robot.robot.update_current_policy(dict(finish=torch.full((1,),2.),
            lease_wall_ns=torch.tensor([float(time.time_ns()+3_000_000_000)],dtype=torch.float64)))
        raise RuntimeError('Native experiment aborted before stationary endpoint; reservation retained for recovery')
    robot.robot.update_current_policy(dict(finish=torch.ones(1),
        lease_wall_ns=torch.tensor([float(time.time_ns()+3_000_000_000)],dtype=torch.float64)))
    deadline=time.monotonic()+2
    while True:
        raw,state=observed(robot)
        if state['path']==3:break
        if time.monotonic()>deadline:raise TimeoutError('Native finish not acknowledged; reservation retained')
        time.sleep(.01)
    from teleop.tracking_policy import TrackingHybridJointImpedanceControl
    from teleop.tracking_impedance import defaults
    interface=robot.robot
    policy=TrackingHybridJointImpedanceControl(joint_pos_current=interface.get_joint_positions(),
        robot_model=interface.robot_model,ignore_gravity=True,record_id=0,
        **{k:torch.tensor(v,dtype=torch.float32) for k,v in defaults(robot).items()})
    interface.send_torch_policy(policy,blocking=False)
    deadline=time.monotonic()+2
    while True:
        raw,state=observed(robot)
        if state['path']==0 and state['log_ready_id']==0:break
        if time.monotonic()>deadline:raise TimeoutError('Explicit A handoff not acknowledged')
        time.sleep(.01)
    robot._tracking_native=None
    return dict(record_id=record['record_id'],restored_path='A',log_closed=True)
