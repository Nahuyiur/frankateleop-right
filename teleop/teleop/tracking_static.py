"""Experiment-only static diagnostics. Never relatch a measured pose as the reference."""
import copy
import hashlib
import math
import time

import numpy as np
import torch


def vector(value, count, name):
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValueError('Invalid '+name)
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in value):
        raise ValueError('Invalid '+name)
    return [float(v) for v in value]


def configure(robot, joint_target, gripper_width):
    if getattr(robot, '_tracking_impedance_override', None) is None:
        raise RuntimeError('Configure the owned tracking experiment first')
    if getattr(robot, '_tracking_impedance_applied', None) is not None:
        raise RuntimeError('Static configuration must precede positioning')
    if joint_target is not None:
        joint_target = vector(joint_target, 7, 'saved joint target')
        from teleop.robots.fr3 import EE_JOINT_LIMITS_LOWER, EE_JOINT_LIMITS_UPPER, DIK_LIMIT_CLIP_RAD
        q = np.asarray(joint_target)
        if np.any(q < EE_JOINT_LIMITS_LOWER+DIK_LIMIT_CLIP_RAD) or np.any(q > EE_JOINT_LIMITS_UPPER-DIK_LIMIT_CLIP_RAD):
            raise ValueError('Saved static target exceeds joint limits')
    if type(gripper_width) not in (int, float) or not math.isfinite(gripper_width) or not 0 <= gripper_width <= robot._max_gripper_width:
        raise ValueError('Measured fixed gripper width required')
    config = dict(joint_target=joint_target, gripper_width=float(gripper_width))
    robot._tracking_static_config = config
    return copy.deepcopy(config)


def select_target(robot, computed, seed):
    """Only the final recorded EEF command may select a saved reference."""
    config = getattr(robot, '_tracking_static_config', None)
    if not config or config['joint_target'] is None or robot._tracking_reference_index != len(robot._tracking_reference['times_ns'])-1:
        return computed
    from teleop.robots.fr3 import DIK_STEP_MAX_RAD
    target = np.asarray(config['joint_target'], dtype=np.float32).astype(float)
    if np.max(np.abs(target-np.asarray(seed))) > DIK_STEP_MAX_RAD:
        raise RuntimeError('Saved static target is too far from the accepted prefix; no target sent')
    return target


def checked_target(robot):
    from teleop.tracking_impedance import snapshot
    applied = snapshot(robot)['applied']
    if not applied or not getattr(robot, '_tracking_static_config', None):
        raise RuntimeError('Static experiment is not configured/active')
    if getattr(robot, '_tracking_had_hold', False):
        raise RuntimeError('Safety HOLD invalidated the static experiment')
    target = getattr(robot, '_tracking_last_target', None)
    if not target or robot._tracking_reference_index != len(robot._tracking_reference['times_ns']):
        raise RuntimeError('The complete prefix must be acknowledged before static sampling')
    accepted = getattr(robot.robot, '_teleop_accepted_joint_goal', None)
    if not np.array_equal(np.asarray(accepted), np.asarray(target['q_ref'])):
        raise RuntimeError('Accepted policy target changed during static sampling')
    return applied, target


def fk(model, joints):
    with torch.no_grad():
        position, quaternion = model.forward_kinematics(torch.tensor(joints, dtype=torch.float32))
    pose = torch.cat((position, quaternion)).detach().cpu().tolist()
    return vector(pose, 7, 'model flange pose')


def observation(robot):
    value = robot.get_feedback_observations(recording=True)
    stamp = value.get('state_timestamp_ns', 0)
    if not value.get('native_state_valid') or not -20_000_000 <= time.time_ns()-stamp <= 150_000_000:
        raise RuntimeError('Fresh native feedback required for static sampling')
    width = value.get('gripper_width', ())
    if (value.get('gripper_width_source') != 'measured' or len(width) != 1 or not math.isfinite(width[0])
            or abs(width[0]-robot._tracking_static_config['gripper_width']) > .001):
        raise RuntimeError('Fixed gripper width changed or feedback is unavailable')
    return value


def freeze(robot):
    applied, target = checked_target(robot)
    if getattr(robot, '_tracking_static_frozen', None) is not None:
        raise RuntimeError('Static target already frozen')
    raw = observation(robot)
    result = dict(copy.deepcopy(target), fk_ref=fk(robot.robot.robot_model, target['q_ref']),
                  frozen_ns=time.time_ns(), policy_episode=applied['policy_episode'],
                  F_T_EE=list(raw['native_F_T_EE']), frame='flange_in_robot_base',
                  gripper_width_m=float(raw['gripper_width'][0]),
                  model_sha256=hashlib.sha256(robot.robot.metadata.urdf_file.encode()).hexdigest(),
                  velocity_expiry_s=.05, source='last acknowledged policy target; no measured-pose relatch')
    robot._tracking_static_frozen = result
    return copy.deepcopy(result)


def sample(robot):
    applied, target = checked_target(robot)
    frozen = getattr(robot, '_tracking_static_frozen', None)
    if frozen is None or frozen['policy_episode'] != applied['policy_episode'] or frozen['q_ref'] != target['q_ref']:
        raise RuntimeError('Frozen static reference is no longer active')
    raw = observation(robot)
    if not np.allclose(raw['native_F_T_EE'], frozen['F_T_EE'], atol=1e-7, rtol=0):
        raise RuntimeError('TCP configuration changed during static sampling')
    q = vector(list(raw['joint_positions']), 7, 'measured joints')
    dq = vector(list(raw['joint_velocities']), 7, 'measured velocities')
    model = robot.robot.robot_model
    with torch.no_grad():
        qt, dqt, zero = torch.tensor(q,dtype=torch.float32), torch.tensor(dq,dtype=torch.float32), torch.zeros(7)
        feedforward = model.inverse_dynamics(qt, dqt, zero)-model.inverse_dynamics(qt, zero, zero)
    return dict(source_ns=int(raw['state_timestamp_ns']), q=q, dq=dq,
                eef=vector([float(v) for v in raw['ee_pos_quat']], 7, 'native flange pose'), fk_measured=fk(model, q),
                feedforward_Nm=vector(feedforward.detach().cpu().tolist(), 7, 'software feedforward'),
                feedforward_source='same inverse-dynamics formula evaluated on this cached native sample',
                torque_measured=vector(list(raw['joint_torques']), 7, 'measured torque'),
                torque_desired=vector(list(raw['joint_torques_desired']), 7, 'native desired torque'),
                torque_external=vector(list(raw['joint_torques_external']), 7, 'estimated external torque'),
                robot_time_ms=int(raw['robot_time_ms']),
                state_receive_monotonic_ns=int(raw['state_receive_monotonic_ns']),
                fci_success_rate=float(raw['control_command_success_rate']),
                gripper_width_m=float(raw['gripper_width'][0]), policy_episode=applied['policy_episode'],
                target_sequence=target['target_sequence'])
