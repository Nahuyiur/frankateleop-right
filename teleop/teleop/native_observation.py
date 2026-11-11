"""Native feedback adapters and one-slot caches; never compute observation FK.

RPCs and representation conversion run on independent workers. Consumers only
copy immutable snapshots. Original timestamps survive cache reads and outages.
"""
import math
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation


MAX_AGE_NS = 150_000_000


def timestamp_ns(state):
    return int(state.timestamp.seconds) * 1_000_000_000 + int(state.timestamp.nanos)


def transform(values):
    matrix = np.asarray(values, dtype=float).reshape(4, 4, order='F')
    rotation = matrix[:3, :3]
    if (not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-5)
            or abs(np.linalg.det(rotation) - 1) > 2e-5):
        raise ValueError('invalid native rigid transform')
    return matrix


def pose(matrix):
    return tuple(matrix[:3, 3]) + tuple(Rotation.from_matrix(matrix[:3, :3]).as_quat())


def arm_observation(state):
    """All fields originate in the same native robot sample; no device calls."""
    result = dict(state_timestamp_ns=timestamp_ns(state))
    for name, field in (('joint_positions', 'joint_positions'), ('joint_velocities', 'joint_velocities'),
                        ('joint_torques', 'motor_torques_measured'),
                        ('joint_torques_external', 'motor_torques_external'),
                        ('joint_torques_desired', 'motor_torques_desired')):
        values = tuple(float(x) for x in getattr(state, field))
        if len(values) != 7 or not all(math.isfinite(x) for x in values):
            raise ValueError('invalid native ' + field)
        result[name] = values
    if not getattr(state, 'native_state_valid', False):
        result['observation_error'] = 'Native Franka feedback unavailable; update native client/protobuf (no FK fallback)'
        return result
    try:
        ee = transform(state.o_t_ee)
        tool = transform(state.f_t_ee)
        # Existing MoZI command/model endpoint is panda_link8 (the flange).
        # O_T_F = O_T_EE * inverse(F_T_EE): fixed frame conversion, NOT FK.
        inverse = np.eye(4)
        inverse[:3, :3] = tool[:3, :3].T
        inverse[:3, 3] = -tool[:3, :3].T @ tool[:3, 3]
        flange = ee @ inverse
        result['ee_pos_quat'] = pose(flange)
        result['ee_pose_euler'] = tuple(flange[:3, 3]) + tuple(Rotation.from_matrix(flange[:3, :3]).as_euler('xyz'))
        result['eef_source'] = 'franka.O_T_EE * inverse(F_T_EE)'
        result['native_eef'] = pose(ee)
        result['native_O_T_EE'] = tuple(state.o_t_ee)
        result['native_F_T_EE'] = tuple(state.f_t_ee)
        result['robot_time_ms'] = int(state.robot_time_ms)
        result['state_receive_monotonic_ns'] = int(state.state_receive_monotonic_ns)
        rate = float(state.control_command_success_rate)
        if not math.isfinite(rate) or not 0 <= rate <= 1:
            raise ValueError('invalid FCI communication success rate')
        result['control_command_success_rate'] = rate
        result['command_execution_confirmed'] = None
        result['native_state_valid'] = True
    except (ValueError, TypeError) as exc:
        # No fabricated pose or fallback to a stale/desired position.
        for key in ('ee_pos_quat', 'ee_pose_euler'):
            result.pop(key, None)
        result['native_state_valid'] = False
        result['observation_error'] = str(exc)
    return result


def gripper_observation(state, max_width):
    width = float(state.width)
    valid = int(state.error_code) == 0 and math.isfinite(width) and 0 <= width <= max_width
    return dict(state_timestamp_ns=timestamp_ns(state), gripper_timestamp_ns=timestamp_ns(state),
                gripper_width=(width if valid else float('nan'),),
                gripper_width_source='measured' if valid else 'unavailable',
                gripper_error_code=int(state.error_code), gripper_max_width=max_width)


class LatestObservation:
    """Exactly one snapshot; even a stalled RPC never blocks snapshot()."""
    def __init__(self, read, convert, hz=50, name='native-observation'):
        self.read, self.convert, self.period = read, convert, 1.0 / hz
        self.latest = None
        self.error = 'awaiting native feedback'
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name=name)
        self.thread.start()

    def _run(self):
        while not self.stop.is_set():
            started = time.monotonic()
            try:
                value = self.convert(self.read())
                self.latest = value  # atomic reference swap; never mutate published values
                self.error = ''
            except Exception as exc:
                self.error = str(exc)[:240]
            self.stop.wait(max(0, self.period - (time.monotonic() - started)))

    def snapshot(self, now_ns=None):
        value = self.latest
        now = time.time_ns() if now_ns is None else now_ns
        if value is None:
            raise RuntimeError(self.error)
        age = now - value['state_timestamp_ns']
        if not -20_000_000 <= age <= MAX_AGE_NS:
            raise RuntimeError('native feedback stale/clock invalid: age %.1f ms' % (age / 1e6))
        return dict(value)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=.3)
