import importlib.util
import sys
import io
import pickle
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "teleop"
    / "experiments"
    / "gello_eef_teleop.py"
)
SPEC = importlib.util.spec_from_file_location("gello_eef_teleop", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_relative_mapper_is_continuous_at_anchor():
    leader = np.array([0.4, -0.1, 0.3, 0.2, -0.3, 0.4])
    follower = np.array([0.5, 0.2, 0.45, -0.5, 0.1, -0.2])
    leader_position, leader_rotation = MODULE.pose6_to_parts(leader)
    follower_position, follower_rotation = MODULE.pose6_to_parts(follower)
    mapper = MODULE.RelativePoseMapper(
        leader_position, leader_rotation, follower_position, follower_rotation
    )

    mapped = mapper.map(leader)
    position_error, rotation_error = MODULE.pose_error(mapped, follower)

    assert position_error < 1e-12
    assert rotation_error < 1e-12


def test_relative_mapper_rotates_translation_between_anchor_frames():
    leader_rotation = Rotation.identity()
    follower_rotation = Rotation.from_euler("z", np.pi / 2)
    mapper = MODULE.RelativePoseMapper(
        np.zeros(3), leader_rotation, np.zeros(3), follower_rotation
    )
    mapped = mapper.map([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])

    np.testing.assert_allclose(mapped[:3], [0.0, 0.1, 0.0], atol=1e-12)


def test_pose_step_limits_translation_and_rotation():
    current = np.zeros(6)
    target = np.array([0.3, 0.4, 0.0, 0.0, 0.0, 1.0])
    limited = MODULE.limit_pose_step(current, target, 0.01, 0.08)
    translation, rotation = MODULE.pose_error(current, limited)

    assert np.isclose(translation, 0.01)
    assert np.isclose(rotation, 0.08)


def test_wrapped_joint_step_ignores_two_pi_encoder_wrap():
    previous = np.array([np.pi - 0.01, 0.0])
    current = np.array([-np.pi + 0.01, 0.0])

    assert np.isclose(MODULE.wrapped_joint_step(previous, current), 0.02)


def mapper(leader, follower, base, ps=1, rs=1):
    lp, lr = MODULE.pose6_to_parts(leader)
    fp, fr = MODULE.pose6_to_parts(follower)
    return MODULE.RelativePoseMapper(lp, lr, fp, fr, ps, rs, base)


def test_identical_side_mounts_do_not_rotate_translation_by_tip_posture():
    leader = np.array([0.2, 0, 0.4, 0.3, -0.2, 0.8])
    follower = np.array([0.5, 0.2, -0.1, -0.9, 0.5, -0.3])
    current = leader.copy()
    current[:3] += [0.02, -0.03, 0.01]
    mapped = mapper(leader, follower, Rotation.identity()).map(current)
    np.testing.assert_allclose(mapped[:3] - follower[:3], [0.02, -0.03, 0.01], atol=1e-12)


@pytest.mark.parametrize("leader_mount,follower_mount", [
    ([90, 0, 0], [90, 0, 0]), ([0, -90, 0], [0, -90, 0]),
    ([90, 0, 0], [-90, 0, 0]), ([17, -40, 63], [-22, 80, 5]),
])
def test_physical_world_displacement_and_rotation_are_preserved(leader_mount, follower_mount):
    wl = Rotation.from_euler("xyz", leader_mount, degrees=True)
    wf = Rotation.from_euler("xyz", follower_mount, degrees=True)
    base = wf.inv() * wl
    leader = np.array([0.2, -0.1, 0.4, 0.2, -0.4, 0.7])
    follower = np.array([0.5, 0.2, 0.3, -0.8, 0.1, 0.3])
    world_delta = np.array([0.01, 0.02, -0.03])
    world_rot = Rotation.from_rotvec([0.05, -0.08, 0.02])
    current = leader.copy()
    current[:3] += wl.inv().apply(world_delta)
    current[3:] = (wl.inv() * world_rot * wl * Rotation.from_euler("xyz", leader[3:])).as_euler("xyz")
    mapped = mapper(leader, follower, base).map(current)
    np.testing.assert_allclose(wf.apply(mapped[:3] - follower[:3]), world_delta, atol=1e-12)
    actual_rot = wf * Rotation.from_euler("xyz", mapped[3:]) * Rotation.from_euler("xyz", follower[3:]).inv() * wf.inv()
    assert (actual_rot.inv() * world_rot).magnitude() < 1e-12


def test_physical_map_identity_at_anchor_and_repeat_does_not_integrate():
    leader = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    follower = [0.2, 0.3, -0.1, 0.2, -0.3, -0.4]
    mapping = mapper(leader, follower, Rotation.from_euler("x", 90, degrees=True), .5, .5)
    np.testing.assert_allclose(mapping.map(leader), follower, atol=1e-12)
    current = np.array(leader) + [0.02, 0, 0, 0, 0, 0]
    first = mapping.map(current)
    np.testing.assert_allclose(mapping.map(current), first, atol=1e-12)
    assert np.isclose(np.linalg.norm(first[:3] - np.array(follower[:3])), .01)


@pytest.mark.parametrize("extra", [
    ["--execute"], ["--execute", "--calibration-verified"],
    ["--translation-scale", "nan"], ["--rotation-scale", "0"],
    ["--workspace-x", "1", "-1"], ["--base-alignment-rpy-deg", "0", "inf", "0"],
    ["--execute", "--base-alignment-rpy-deg", "0", "0", "0", "--calibration-verified", "--preview-anchor-only"],
])
def test_bad_or_unverified_cli_is_rejected_before_hardware(extra):
    with pytest.raises(SystemExit):
        MODULE.parse_args(["--teleop-port", "/dev/fake"] + extra)


def test_lease_eof_and_timeout_are_latched_stops():
    lease = MODULE.SupervisorLease(io.StringIO("GO\n"))
    lease.thread.join(1)
    with pytest.raises(MODULE.SafetyStop):
        lease.check()
    lease.stopped.clear()
    lease.last_seen = MODULE.time.monotonic() - 2
    with pytest.raises(MODULE.SafetyStop):
        lease.check()


def test_structured_robot_hold_is_not_counted_as_accepted_command():
    class Socket:
        def send(self, data):
            pass
        def recv(self):
            return pickle.dumps({"status": "hold", "reason": "joint_soft_limit"})
    client = MODULE.RobotNodeClient.__new__(MODULE.RobotNodeClient)
    client._socket = Socket()
    with pytest.raises(MODULE.SafetyStop, match="joint_soft_limit"):
        client.request("command_ee_pose")


def test_right_damping_uses_official_right_path_not_left_or_unused_right_goal(monkeypatch):
    captured = []
    monkeypatch.setattr(MODULE, "TeleopAgent", lambda **kwargs: captured.append(kwargs))
    left = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBZIB2R-if00-port0"
    right = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBJTKP2-if00-port0"
    MODULE.make_gello_agent(left, True, "left")
    MODULE.make_gello_agent(right, True, "right")
    assert captured[0]["dynamixel_config"].dynamixel_goal_position == MODULE.LEFT_DAMPING_GOAL
    assert captured[1]["dynamixel_config"].dynamixel_goal_position == (0, 0, 0, -1.571, 0, 1.571, 0)
    assert captured[1]["dynamixel_config"].joint_signs == MODULE.PORT_CONFIG_MAP[right].joint_signs
