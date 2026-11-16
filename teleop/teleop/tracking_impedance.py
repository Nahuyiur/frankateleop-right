"""Per-reservation impedance, installed only at the normal post-positioning restart."""
import copy
import hashlib
import inspect
from functools import lru_cache
from pathlib import Path
import subprocess
import time

from teleop.tracking_gains import DEFAULT_VELOCITY_MODE, VELOCITY_REFERENCE, control_law, validate_gains, validate_reference


@lru_cache(maxsize=1)
def runtime_provenance():
    """Read once outside the real-time loop; retain source and runtime identities."""
    root=Path(__file__).resolve().parents[2]
    def git(path,*args):
        result=subprocess.run(['git','-C',str(path),*args],capture_output=True,text=True,timeout=2)
        return result.stdout.strip() if result.returncode==0 else None
    build=root/'polymetis/polymetis/build'
    files=[build/'franka_panda_client',build/'run_server',build/'torch_isolation/libtorch_server_ops.so']
    return dict(repository=str(root),commit=git(root,'rev-parse','HEAD'),
        tracked_changes=git(root,'status','--porcelain','--untracked-files=no'),
        libfranka=git('/home/pnp/libfranka','describe','--tags','--always'),
        binaries_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()})


def defaults(robot):
    return {key:list(getattr(robot.robot.metadata, 'default_'+key))
            for key in ('Kq','Kqd','Kx','Kxd')}


def configure(robot, gains, reference, velocity_mode=DEFAULT_VELOCITY_MODE, integral=None, friction=None):
    from teleop.tracking_native import validate_integral, validate_friction
    law = control_law(velocity_mode)
    gains = validate_gains(gains)
    integral = validate_integral(integral)
    friction = validate_friction(friction)
    if (integral or friction) and law['mode'] == 'zero':
        raise ValueError('Integral/friction terms require the trajectory-difference desired velocity')
    reference = validate_reference(reference)
    if robot.control_mode != 'ee':
        raise ValueError('tracking gains require absolute EEF replay')
    if getattr(robot, '_tracking_impedance_override', None) is not None:
        raise RuntimeError('tracking gains already configured for this reservation')
    if getattr(robot,'_ik_mode','polymetis') != 'dik':
        raise RuntimeError('trajectory velocity requires the verified DIK backend')
    robot._tracking_impedance_override = gains
    robot._tracking_terms = dict(integral=integral, friction=friction)
    robot._tracking_reference = reference
    robot._tracking_velocity_mode = law["mode"]
    robot._tracking_impedance_applied = None
    return dict(configured=gains, applied=False,velocity_reference=law["velocity_reference"],control_law_id=law["id"],
                **({'integral':integral,'friction':friction} if integral or friction else {}),
                reference=dict(sha256=reference['sha256'],count=len(reference['times_ns'])))


def clear(robot):
    if getattr(robot,'_tracking_native',None):
        raise RuntimeError('Finish native control before clearing the replay reservation')
    robot._tracking_impedance_override = None
    robot._tracking_terms = None
    robot._tracking_impedance_applied = None
    robot._tracking_reference = None
    robot._tracking_velocity_mode = None
    robot._tracking_velocity_sample = None
    robot._tracking_static_config = None
    robot._tracking_static_frozen = None
    robot._tracking_last_target = None
    robot._tracking_had_hold = False
    robot._tracking_record_id = 0


def start(robot):
    import torch
    from teleop.tracking_policy import TrackingHybridJointImpedanceControl
    interface = robot.robot
    gains = validate_gains(robot._tracking_impedance_override)
    terms = getattr(robot,'_tracking_terms',None) or {}
    integral,friction = terms.get('integral'),terms.get('friction')
    if not interface.use_grav_comp or interface.metadata.hz != 1000:
        raise RuntimeError('tracking formula requires the verified 1 kHz gravity-compensated backend')
    q = interface.get_joint_positions()
    policy = TrackingHybridJointImpedanceControl(joint_pos_current=q,
        robot_model=interface.robot_model, ignore_gravity=True,
        record_id=getattr(robot,'_tracking_record_id',0),
        **(dict(integral_time_s=integral['time_s'],integral_leak_s=integral['leak_s'],
                integral_limit_Nm=torch.tensor(integral['limit_Nm'],dtype=torch.float32)) if integral else {}),
        **(dict(friction_Nm=torch.tensor(friction['coulomb_Nm'],dtype=torch.float32),
                friction_velocity_rad_s=friction['velocity_rad_s']) if friction else {}),
        **{key:torch.tensor(value,dtype=torch.float32) for key,value in gains.items()})
    actual = {key:torch.diag(getattr(policy.joint_pd,key)).detach().cpu().tolist() for key in gains}
    files = [Path(__file__),Path(inspect.getfile(type(robot))),
             Path(inspect.getfile(type(interface))),Path(inspect.getfile(TrackingHybridJointImpedanceControl)),
             Path(inspect.getfile(TrackingHybridJointImpedanceControl.__bases__[0])),
             Path(inspect.getfile(validate_gains)),Path(inspect.getfile(type(policy.joint_pd))),
             Path(__file__).with_name('tracking_static.py')]
    reference=robot._tracking_reference
    law=control_law(robot._tracking_velocity_mode)
    record = dict(gains=actual, formula=law['formula'], control_law_id=law['id'], velocity_formula=law['velocity_formula'], controller='TrackingHybridJointImpedanceControl',
        path='A',record_id=getattr(robot,'_tracking_record_id',0),
        **(dict(integral=integral,friction=friction,
                terms_formula='tau = tau_policy + Kp(q) z + Fc tanh(dq_des/v0); dz/dt = (q_des - q)/Ti - z/T_leak while targets arrive, 0.2 s fade after 1 s without targets; |Kp z| <= limit') if integral or friction else {}),
        hz=1000,velocity_reference=law['velocity_reference'],static_software_feedforward_Nm=[0.]*7,
        runtime=runtime_provenance(),model_sha256=hashlib.sha256(interface.metadata.urdf_file.encode()).hexdigest(),
        reference=dict(sha256=reference['sha256'],count=len(reference['times_ns']),accepted=0),
        source='float32 gains in policy submitted to Polymetis; SetController acknowledged',
        software_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    interface.send_torch_policy(policy, blocking=False)
    record.update(policy_episode=int(interface._teleop_goal_episode), applied_ns=time.time_ns())
    robot._tracking_impedance_applied = record
    robot._tracking_reference_index = 0
    robot._tracking_previous_target = None
    robot._tracking_velocity = torch.zeros(7)
    robot._tracking_velocity_sequence = 0
    robot._tracking_velocity_sample = None
    robot._tracking_last_update_monotonic = None
    robot._tracking_static_frozen = None
    robot._tracking_last_target = None
    robot._tracking_had_hold = False


def update_target(robot, joints, position=None, quaternion=None, *, hold=False):
    """Pair each incoming EEF with the immutable fixed-replay time table, then differentiate DIK targets."""
    import torch
    if getattr(robot,'_tracking_native',None):
        if hold:
            robot._tracking_had_hold=True
            robot.robot.update_current_policy(dict(finish=torch.full((1,),2.)))
        raise RuntimeError('Native experiment rejects DIK target/HOLD updates and is invalid after HOLD')
    joints=torch.as_tensor(joints,dtype=torch.float32).detach().clone()
    if joints.shape!=(7,) or not bool(torch.isfinite(joints).all()):
        raise ValueError('Invalid desired joint positions')
    index=robot._tracking_reference_index
    reference=robot._tracking_reference
    raw=velocity=torch.zeros(7)
    dt=0.
    clipped=False
    now=time.monotonic()
    previous_time=robot._tracking_last_update_monotonic
    resumed=previous_time is not None and now-previous_time>=VELOCITY_REFERENCE['stale_zero_s']
    if not hold and getattr(robot, '_tracking_static_frozen', None) is not None:
        raise RuntimeError('Static reference is frozen; release before sending another target')
    if hold:
        robot._tracking_had_hold = True
    if not hold:
        if index>=len(reference['times_ns']):
            raise RuntimeError('Unexpected extra target after fixed trajectory')
        pose=torch.cat((torch.as_tensor(position),torch.as_tensor(quaternion))).float()
        expected=torch.tensor(reference['eef'][index],dtype=torch.float32)
        quat_error=min(float(torch.linalg.vector_norm(pose[3:]-expected[3:])),float(torch.linalg.vector_norm(pose[3:]+expected[3:])))
        if (pose.shape!=(7,) or not bool(torch.isfinite(pose).all())
                or float(torch.max(torch.abs(pose[:3]-expected[:3])))>1e-6 or quat_error>1e-5):
            raise RuntimeError('EEF target does not match the fixed trajectory time table')
        previous=None if resumed else robot._tracking_previous_target
        if previous is not None and robot._tracking_velocity_mode != 'zero':
            dt=(reference['times_ns'][index]-reference['times_ns'][index-1])/1e9
            raw=(joints-previous)/dt
            limit=VELOCITY_REFERENCE['max_abs_rad_s']
            bounded=torch.clamp(raw,-limit,limit)
            clipped=not bool(torch.equal(raw,bounded))
            alpha=dt/(VELOCITY_REFERENCE['filter_tau_s']+dt)
            old=robot._tracking_velocity
            filtered=old+alpha*(bounded-old)
            step=VELOCITY_REFERENCE['max_accel_rad_s2']*dt
            velocity=old+torch.clamp(filtered-old,-step,step)
    sequence=robot._tracking_velocity_sequence+1
    updated=robot.robot.update_current_policy(dict(joint_pos_desired=joints,joint_vel_desired=velocity,
                                                  target_sequence=torch.tensor([float(sequence)])))
    if updated == -1:
        raise RuntimeError('Desired position/velocity update was not acknowledged')
    robot._tracking_velocity_sequence=sequence
    robot._tracking_last_update_monotonic=now
    robot._tracking_previous_target=None if hold else joints
    robot._tracking_velocity=velocity
    if not hold:
        robot._tracking_reference_index=index+1
        robot._tracking_impedance_applied['reference']['accepted']=index+1
        robot._tracking_last_target = dict(q_ref=joints.tolist(), eef_reference=expected.tolist(),
            reference_source_ns=reference['times_ns'][index], reference_index=index,
            target_sequence=sequence, parameter_update_index=int(updated), acknowledged_ns=time.time_ns())
    robot._tracking_velocity_sample=dict(velocity_mode=robot._tracking_velocity_mode,reference_index=index if not hold else None,
        reference_source_ns=reference['times_ns'][index] if not hold else None,dt_s=dt,
        raw_joint_velocity=raw.tolist(),desired_joint_velocity=velocity.tolist(),limited=clipped,
        hold=hold,resumed_after_stale=resumed,parameter_update_index=int(updated),acknowledged_ns=time.time_ns())
    return updated


def snapshot(robot):
    record = copy.deepcopy(getattr(robot, '_tracking_impedance_applied', None))
    if record is not None:
        from polymetis_pb2 import Empty
        episode = robot.robot.grpc_connection.GetEpisodeInterval(Empty(),timeout=.25)
        if episode.start != record['policy_episode'] or episode.end != -1:
            raise RuntimeError('experiment policy is no longer active')
        record['last_velocity_command']=copy.deepcopy(getattr(robot,'_tracking_velocity_sample',None))
    return dict(defaults=defaults(robot), pending=copy.deepcopy(getattr(robot,'_tracking_impedance_override',None)),
                applied=record, mode=robot.control_mode)
