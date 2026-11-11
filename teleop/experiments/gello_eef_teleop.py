#!/usr/bin/env python3
"""Drive a Franka Cartesian controller from a GELLO leader.

The GELLO still reports its calibrated joint state.  This adapter applies the
Franka model's forward kinematics and maps leader motion relative to the
startup pose onto the follower's startup end-effector pose.  The relative
anchor makes startup continuous even when leader and follower are not in the
same joint configuration.

The default is monitor-only.  Pass ``--execute`` explicitly to send commands.
"""

from __future__ import annotations

import argparse
import math
import pickle
import signal
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import zmq
from polymetis import RobotInterface
from scipy.spatial.transform import Rotation

from teleop.agents.teleop_agent import (
    DynamixelRobotConfig,
    PORT_CONFIG_MAP,
    TeleopAgent,
)


DAMPING_TORQUE_ENABLE = (0, 0, 0, 0, 0, 0, 0)
DAMPING_KP_P = (30, 60, 0, 30, 0, 0, 0)
DAMPING_KP_I = (0, 0, 0, 0, 0, 0, 0)
DAMPING_KP_D = (250, 100, 80, 60, 30, 10, 5)
LEFT_DAMPING_GOAL = (
    0.19421,
    0.259328,
    0.192638,
    -1.80504,
    1.2055,
    2.84043,
    -0.736754,
)
# Match the actual official right 4_run_env.sh path (robot_type=None), NOT
# the unused fr3_right goal in run_env.py. Do not copy left damping to right.
RIGHT_DAMPING_GOAL = (0.0, 0.0, 0.0, -1.571, 0.0, 1.571, 0.0)


class SupervisorLease:
    """Stdin lease: both arms wait for GO; EOF/STOP/timeout permanently stops."""

    def __init__(self, stream, timeout=1.0):
        self.go = threading.Event()
        self.stopped = threading.Event()
        self.timeout = timeout
        self.last_seen = time.monotonic()
        self.thread = threading.Thread(target=self._read, args=(stream,), daemon=True)
        self.thread.start()

    def _read(self, stream):
        try:
            for line in stream:
                value = line.strip()
                if value not in {"LEASE", "GO"}:
                    break
                self.last_seen = time.monotonic()
                if value == "GO":
                    self.go.set()
        finally:
            self.stopped.set()

    def check(self):
        if self.stopped.is_set() or time.monotonic() - self.last_seen > self.timeout:
            raise SafetyStop("dual-arm supervisor lease lost; restart and re-anchor both arms")

    def wait_for_go(self, stop):
        while not self.go.wait(0.05):
            self.check()
            if stop[0]:
                raise SafetyStop("stopped before dual-arm GO")
        self.check()


class SafetyStop(RuntimeError):
    """Raised when a live command violates an adapter safety envelope."""


class RobotNodeClient:
    """Small timeout-aware client for the existing robot-node protocol."""

    def __init__(self, host: str, port: int, timeout_ms: int) -> None:
        self.endpoint = "tcp://{}:{}".format(host, port)
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(self.endpoint)

    def request(self, method: str, args: Optional[Dict[str, Any]] = None) -> Any:
        try:
            self._socket.send(pickle.dumps({"method": method, "args": args or {}}))
            result = pickle.loads(self._socket.recv())
        except zmq.Again as exc:
            raise TimeoutError("robot node timed out at {}".format(self.endpoint)) from exc
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError("robot node: {}".format(result["error"]))
        if isinstance(result, dict) and result.get("status") == "hold":
            raise SafetyStop("robot node HOLD: {}".format(result))
        return result

    def close(self) -> None:
        self._socket.close(0)
        self._context.term()


def pose6_to_parts(pose: Sequence[float]) -> Tuple[np.ndarray, Rotation]:
    value = np.asarray(pose, dtype=np.float64).reshape(-1)
    if value.shape != (6,) or not np.all(np.isfinite(value)):
        raise ValueError("expected finite [x,y,z,rx,ry,rz], got {}".format(value))
    return value[:3].copy(), Rotation.from_euler("xyz", value[3:])


def parts_to_pose6(position: np.ndarray, rotation: Rotation) -> np.ndarray:
    return np.r_[np.asarray(position, dtype=np.float64), rotation.as_euler("xyz")]


@dataclass
class RelativePoseMapper:
    leader_position: np.ndarray
    leader_rotation: Rotation
    follower_position: np.ndarray
    follower_rotation: Rotation
    translation_scale: float = 1.0
    rotation_scale: float = 1.0
    # R_followerBase_leaderBase. None is the old anchor-local mapping, retained
    # for direct users of this class; the CLI always supplies a physical map.
    base_alignment: Optional[Rotation] = None

    def map(self, leader_pose: Sequence[float]) -> np.ndarray:
        position, rotation = pose6_to_parts(leader_pose)
        base_alignment = self.base_alignment
        if base_alignment is None:
            base_alignment = self.follower_rotation * self.leader_rotation.inv()
        position_delta = base_alignment.apply(
            (position - self.leader_position) * self.translation_scale
        )
        # Spatial rotation delta in follower base, not an Euler-angle difference.
        relative_rotation = base_alignment * rotation * self.leader_rotation.inv() * base_alignment.inv()
        scaled_rotation = Rotation.from_rotvec(
            relative_rotation.as_rotvec() * self.rotation_scale
        )
        return parts_to_pose6(
            self.follower_position + position_delta,
            scaled_rotation * self.follower_rotation,
        )


def pose_error(first: Sequence[float], second: Sequence[float]) -> Tuple[float, float]:
    first_position, first_rotation = pose6_to_parts(first)
    second_position, second_rotation = pose6_to_parts(second)
    position_error = float(np.linalg.norm(second_position - first_position))
    rotation_error = float((first_rotation.inv() * second_rotation).magnitude())
    return position_error, rotation_error


def limit_pose_step(
    current: Sequence[float],
    target: Sequence[float],
    max_translation: float,
    max_rotation: float,
) -> np.ndarray:
    current_position, current_rotation = pose6_to_parts(current)
    target_position, target_rotation = pose6_to_parts(target)

    translation = target_position - current_position
    translation_norm = float(np.linalg.norm(translation))
    if translation_norm > max_translation:
        translation *= max_translation / translation_norm

    rotation_delta = current_rotation.inv() * target_rotation
    rotation_vector = rotation_delta.as_rotvec()
    rotation_norm = float(np.linalg.norm(rotation_vector))
    if rotation_norm > max_rotation:
        rotation_vector *= max_rotation / rotation_norm

    return parts_to_pose6(
        current_position + translation,
        current_rotation * Rotation.from_rotvec(rotation_vector),
    )


def wrapped_joint_step(previous: Sequence[float], current: Sequence[float]) -> float:
    delta = np.asarray(current, dtype=float) - np.asarray(previous, dtype=float)
    delta = (delta + np.pi) % (2.0 * np.pi) - np.pi
    return float(np.max(np.abs(delta)))


def validate_target(args: argparse.Namespace, anchor: np.ndarray, target: np.ndarray) -> None:
    translation, rotation = pose_error(anchor, target)
    if translation > args.max_anchor_translation:
        raise SafetyStop(
            "target is {:.3f} m from anchor (limit {:.3f} m)".format(
                translation, args.max_anchor_translation
            )
        )
    if rotation > args.max_anchor_rotation:
        raise SafetyStop(
            "target is {:.3f} rad from anchor (limit {:.3f} rad)".format(
                rotation, args.max_anchor_rotation
            )
        )
    if getattr(args, "preview_anchor_only", False):
        return
    for label, value, bounds in zip(
        ("x", "y", "z"), target[:3], (args.workspace_x, args.workspace_y, args.workspace_z)
    ):
        if not bounds[0] <= value <= bounds[1]:
            raise SafetyStop(
                "target {}={:.3f} outside [{:.3f}, {:.3f}]".format(
                    label, value, bounds[0], bounds[1]
                )
            )


def make_gello_agent(port: str, damping: bool, side: str = "left") -> TeleopAgent:
    if port not in PORT_CONFIG_MAP:
        raise ValueError("GELLO port is not calibrated in PORT_CONFIG_MAP: {}".format(port))
    if not damping:
        return TeleopAgent(port=port)
    base = PORT_CONFIG_MAP[port]
    config = DynamixelRobotConfig(
        joint_ids=base.joint_ids,
        joint_offsets=base.joint_offsets,
        joint_signs=base.joint_signs,
        gripper_config=base.gripper_config,
        dynamixel_torque_enable=DAMPING_TORQUE_ENABLE,
        dynamixel_kp_p=DAMPING_KP_P,
        dynamixel_kp_i=DAMPING_KP_I,
        dynamixel_kp_d=DAMPING_KP_D,
        dynamixel_goal_position=LEFT_DAMPING_GOAL if side == "left" else RIGHT_DAMPING_GOAL,
    )
    return TeleopAgent(port=port, dynamixel_config=config)


def forward_kinematics(robot: RobotInterface, joints: Sequence[float]) -> np.ndarray:
    joint_array = np.asarray(joints, dtype=np.float64).reshape(-1)
    if joint_array.shape != (7,) or not np.all(np.isfinite(joint_array)):
        raise SafetyStop("invalid GELLO arm joints: {}".format(joint_array))
    position, quaternion = robot.robot_model.forward_kinematics(
        torch.as_tensor(joint_array, dtype=torch.float32)
    )
    return np.r_[
        position.detach().cpu().numpy(),
        Rotation.from_quat(quaternion.detach().cpu().numpy()).as_euler("xyz"),
    ]


def robot_pose(observation: Dict[str, Any]) -> np.ndarray:
    if "ee_pose_euler" not in observation:
        raise RuntimeError(
            "robot node has no ee_pose_euler; restart it with the current fr3.py"
        )
    pose = np.asarray(observation["ee_pose_euler"], dtype=np.float64).reshape(-1)
    if pose.shape != (6,) or not np.all(np.isfinite(pose)):
        raise RuntimeError("invalid robot ee_pose_euler: {}".format(pose))
    return pose


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--node-port", type=int, default=6002)
    parser.add_argument("--robot-port", type=int, default=50052)
    parser.add_argument("--teleop-port", required=True)
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--base-alignment-rpy-deg", type=float, nargs=3,
                        help="R_followerBase_leaderBase, extrinsic xyz degrees; same mounting = 0 0 0")
    parser.add_argument("--calibration-verified", action="store_true")
    parser.add_argument("--supervised", action="store_true", help="wait for stdin GO and require a continuous lease")
    parser.add_argument("--preview-anchor-only", action="store_true", help="preview only: skip uncalibrated absolute workspace box")
    parser.add_argument("--timeout-ms", type=int, default=300)
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true", help="validate CLI and exit without constructing clients/devices")
    parser.add_argument("--duration-sec", type=float, default=0.0)
    parser.add_argument("--arm-countdown-sec", type=float, default=3.0)
    parser.add_argument("--no-leader-damping", action="store_true")
    parser.add_argument("--translation-scale", type=float, default=1.0)
    parser.add_argument("--rotation-scale", type=float, default=1.0)
    parser.add_argument("--max-step-translation", type=float, default=0.010)
    parser.add_argument("--max-step-rotation", type=float, default=0.080)
    parser.add_argument("--max-anchor-translation", type=float, default=0.25)
    parser.add_argument("--max-anchor-rotation", type=float, default=1.20)
    parser.add_argument("--max-leader-joint-step", type=float, default=0.35)
    parser.add_argument("--max-tracking-translation", type=float, default=0.08)
    parser.add_argument("--max-tracking-rotation", type=float, default=0.60)
    parser.add_argument("--workspace-x", type=float, nargs=2, default=(0.20, 0.75))
    parser.add_argument("--workspace-y", type=float, nargs=2, default=(-0.45, 0.45))
    parser.add_argument("--workspace-z", type=float, nargs=2, default=(0.15, 0.80))
    parser.add_argument("--max-gripper-width", type=float, default=0.09)
    parser.add_argument("--gripper-event-delta", type=float, default=0.02)
    parser.add_argument("--gripper-speed", type=float, default=0.05)
    parser.add_argument("--gripper-force", type=float, default=20.0)
    args = parser.parse_args(argv)
    positive = (
        "hz",
        "timeout_ms",
        "max_step_translation",
        "max_step_rotation",
        "max_anchor_translation",
        "max_anchor_rotation",
        "max_leader_joint_step",
        "max_tracking_translation",
        "max_tracking_rotation",
        "max_gripper_width",
        "gripper_event_delta",
        "translation_scale", "rotation_scale", "gripper_speed", "gripper_force",
    )
    for name in positive:
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error("--{} must be positive".format(name.replace("_", "-")))
    for name in ("workspace_x", "workspace_y", "workspace_z"):
        bounds = getattr(args, name)
        if not all(math.isfinite(x) for x in bounds) or bounds[0] >= bounds[1]:
            parser.error("--{} requires finite LOW < HIGH".format(name.replace("_", "-")))
    if args.execute and (not args.calibration_verified or args.base_alignment_rpy_deg is None):
        parser.error("live EEF requires explicit --base-alignment-rpy-deg and --calibration-verified")
    if args.execute and (args.preview_anchor_only or args.check_only):
        parser.error("execute cannot be combined with preview-anchor-only/check-only")
    if args.base_alignment_rpy_deg is None:
        args.base_alignment_rpy_deg = [0.0, 0.0, 0.0]
    if not all(math.isfinite(x) for x in args.base_alignment_rpy_deg):
        parser.error("base alignment must contain finite degrees")
    for name in ("duration_sec", "arm_countdown_sec"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(name + " must be finite and non-negative")
    return args


def run(args: argparse.Namespace) -> None:
    lease = SupervisorLease(sys.stdin) if args.supervised else None
    node = RobotNodeClient(args.host, args.node_port, args.timeout_ms)
    gello: Optional[TeleopAgent] = None
    stop = [False]

    def request_stop(_signum, _frame) -> None:
        stop[0] = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        mode = str(node.request("get_control_mode"))
        if mode != "ee":
            raise RuntimeError(
                "robot node must be launched in EE mode; got {!r}. "
                "Use left_franka/3_launch_eef_node.sh".format(mode)
            )
        observation = node.request("get_observations")
        follower_anchor = robot_pose(observation)

        model_client = RobotInterface(ip_address=args.host, port=args.robot_port)
        gello = make_gello_agent(args.teleop_port, not args.no_leader_damping, args.side)
        leader_state = np.asarray(gello.act(observation), dtype=np.float64)
        if leader_state.shape != (8,):
            raise RuntimeError("expected 8-D GELLO state, got {}".format(leader_state.shape))
        leader_anchor = forward_kinematics(model_client, leader_state[:7])
        leader_position, leader_rotation = pose6_to_parts(leader_anchor)
        follower_position, follower_rotation = pose6_to_parts(follower_anchor)
        mapper = RelativePoseMapper(
            leader_position,
            leader_rotation,
            follower_position,
            follower_rotation,
            args.translation_scale,
            args.rotation_scale,
            Rotation.from_euler("xyz", args.base_alignment_rpy_deg, degrees=True),
        )
        mapped_anchor = mapper.map(leader_anchor)
        anchor_position_error, anchor_rotation_error = pose_error(
            follower_anchor, mapped_anchor
        )
        print("GELLO EEF adapter preflight passed")
        print("  robot node: {}:{} mode={}".format(args.host, args.node_port, mode))
        print("  GELLO: {}".format(args.teleop_port))
        print("  follower anchor: {}".format(np.array2string(follower_anchor, precision=4)))
        print("  leader FK anchor: {}".format(np.array2string(leader_anchor, precision=4)))
        print(
            "  anchor continuity error: {:.6f} m / {:.6f} rad".format(
                anchor_position_error, anchor_rotation_error
            )
        )
        print(
            "  mode: {}".format(
                "EXECUTE" if args.execute and not args.check_only else "MONITOR ONLY"
            )
        )
        if args.check_only:
            return

        if lease is not None:
            print("GELLO_EEF_READY {}".format(args.side), flush=True)
            lease.wait_for_go(stop)

        if args.execute:
            if not model_client.is_running_policy():
                print("Cartesian policy stopped; restarting at the current pose.")
                model_client.start_cartesian_impedance()
            deadline = time.monotonic() + max(0.0, args.arm_countdown_sec)
            while time.monotonic() < deadline:
                if stop[0]:
                    raise SafetyStop("stopped during arming")
                if lease is not None:
                    lease.check()
                remaining = deadline - time.monotonic()
                print("\rArming in {:.1f}s; keep GELLO still...".format(remaining), end="", flush=True)
                time.sleep(min(0.1, remaining))
            if args.arm_countdown_sec > 0:
                print()

            # The operator may have nudged the leader during the countdown.
            # Re-anchor immediately before the first live command so that the
            # commanded Cartesian path still begins at exactly the current
            # follower pose.
            observation = node.request("get_observations")
            follower_anchor = robot_pose(observation)
            leader_state = np.asarray(gello.act(observation), dtype=np.float64)
            leader_anchor = forward_kinematics(model_client, leader_state[:7])
            leader_position, leader_rotation = pose6_to_parts(leader_anchor)
            follower_position, follower_rotation = pose6_to_parts(follower_anchor)
            mapper = RelativePoseMapper(
                leader_position,
                leader_rotation,
                follower_position,
                follower_rotation,
                args.translation_scale,
                args.rotation_scale,
                Rotation.from_euler("xyz", args.base_alignment_rpy_deg, degrees=True),
            )
            print("Live anchors captured; first EEF target equals current robot pose.")

        previous_leader = leader_state[:7].copy()
        previous_command = follower_anchor.copy()
        previous_gripper = float(np.clip(leader_state[-1], 0.0, 1.0))
        started = time.monotonic()
        next_tick = started
        next_report = started
        command_count = 0

        while not stop[0]:
            if lease is not None:
                lease.check()
            now = time.monotonic()
            if args.duration_sec > 0 and now - started >= args.duration_sec:
                break
            observation = node.request("get_observations")
            leader_state = np.asarray(gello.act(observation), dtype=np.float64)
            if leader_state.shape != (8,) or not np.all(np.isfinite(leader_state)):
                raise SafetyStop("invalid GELLO state: {}".format(leader_state))
            leader_step = wrapped_joint_step(previous_leader, leader_state[:7])
            if leader_step > args.max_leader_joint_step:
                raise SafetyStop(
                    "GELLO sample jumped {:.3f} rad (limit {:.3f} rad)".format(
                        leader_step, args.max_leader_joint_step
                    )
                )
            previous_leader = leader_state[:7].copy()

            leader_pose = forward_kinematics(model_client, leader_state[:7])
            raw_target = mapper.map(leader_pose)
            validate_target(args, follower_anchor, raw_target)
            command = limit_pose_step(
                previous_command,
                raw_target,
                args.max_step_translation,
                args.max_step_rotation,
            )

            actual_pose = robot_pose(observation)
            tracking_translation, tracking_rotation = pose_error(actual_pose, previous_command)
            if args.execute and command_count > max(5, int(args.hz)):
                if tracking_translation > args.max_tracking_translation:
                    raise SafetyStop(
                        "EEF tracking error {:.3f} m exceeds {:.3f} m".format(
                            tracking_translation, args.max_tracking_translation
                        )
                    )
                if tracking_rotation > args.max_tracking_rotation:
                    raise SafetyStop(
                        "EEF tracking error {:.3f} rad exceeds {:.3f} rad".format(
                            tracking_rotation, args.max_tracking_rotation
                        )
                    )

            gripper_closedness = float(np.clip(leader_state[-1], 0.0, 1.0))
            update_gripper = (
                abs(gripper_closedness - previous_gripper) >= args.gripper_event_delta
            )
            gripper_width = args.max_gripper_width * (1.0 - gripper_closedness)
            if args.execute:
                if stop[0]:
                    break
                if lease is not None:
                    lease.check()
                node.request(
                    "command_ee_pose",
                    {
                        "pose_6d": command,
                        "gripper_width": gripper_width,
                        "gripper_speed": args.gripper_speed,
                        "gripper_force": args.gripper_force,
                        "update_gripper": update_gripper,
                    },
                )
                command_count += 1
                previous_command = command
                if update_gripper:
                    previous_gripper = gripper_closedness

            if now >= next_report:
                delta_translation, delta_rotation = pose_error(follower_anchor, raw_target)
                print(
                    "t={:6.1f}s target_delta={:.3f}m/{:.3f}rad "
                    "tracking={:.3f}m/{:.3f}rad gripper={:.2f}{}".format(
                        now - started,
                        delta_translation,
                        delta_rotation,
                        tracking_translation,
                        tracking_rotation,
                        gripper_closedness,
                        " command" if update_gripper and args.execute else "",
                    )
                )
                next_report = now + 1.0

            next_tick += 1.0 / args.hz
            sleep_time = next_tick - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_tick = time.monotonic()
        print("Adapter stopped; Cartesian controller is holding the last commanded pose.")
    finally:
        if gello is not None:
            leader_robot = getattr(gello, "_robot", None)
            if leader_robot is not None and hasattr(leader_robot, "close"):
                leader_robot.close()
        node.close()


def main() -> None:
    args = parse_args()
    if args.validate_only:
        print("GELLO EEF arguments validated (no hardware)")
        return
    try:
        run(args)
    except SafetyStop as exc:
        raise SystemExit("SAFETY STOP: {}".format(exc))


if __name__ == "__main__":
    main()
