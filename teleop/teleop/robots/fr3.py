import os
import time
import torch
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict
from typing import Optional
import numpy as np
from teleop.robots.robot import Robot

LOGGER = logging.getLogger("franka_fr3_robot")

MAX_OPEN = 0.09
GRIPPER_SPEED = 0.1
GRIPPER_FORCE = 20.0
GRIPPER_DEBUG_INTERVAL = 0.5
GRIPPER_COMMAND_THRESHOLD = 0.5
EE_JOINT_LIMITS_LOWER = np.array(
    [-2.64, -1.68, -2.80, -2.94, -2.70, 0.64, -2.91], dtype=float
)
EE_JOINT_LIMITS_UPPER = np.array(
    [2.64, 1.68, 2.80, -0.25, 2.70, 4.41, 2.91], dtype=float
)
# At the 50 Hz teleop command rate, the fastest FR3 wrist joint can move about
# 0.10 rad between checks. Keep a larger default buffer than one command frame.
DEFAULT_EE_JOINT_SOFT_MARGIN_RAD = 0.15


class fr3Robot(Robot):
    """A class representing a UR robot."""

    def __init__(
            self, 
            robot_ip: str = "192.168.1.100", 
            franka_port: int=50051, 
            frankahand_port: int = 50053,
            joint_positions_desired: Optional[torch.Tensor] = None,
            control_mode: str = "joint",
            home_on_init: bool = True,
            open_gripper_on_init: bool = True,
            ):
            
        from polymetis import GripperInterface, RobotInterface
        print(f"Connecting to robot at IP: {robot_ip}")

        self.robot = RobotInterface(
            ip_address=robot_ip,
            port=franka_port,
        )
        self.gripper = GripperInterface(
            ip_address=robot_ip,
            port=frankahand_port,
        )
        self._max_gripper_width = self._get_gripper_max_width()
        if joint_positions_desired is None and home_on_init:
            self.robot.go_home()
        elif joint_positions_desired is None:
            print("Skipping robot.go_home() on init.")
        else:
            if joint_positions_desired.shape != (7,):
                raise ValueError(f"Franka requires 7 joints params, current input is: {joint_positions_desired.shape}")
            print("init robot")
            self.joint_positions_desired = joint_positions_desired
            self.robot.move_to_joint_positions(self.joint_positions_desired)
            
        self.control_mode = control_mode
        self._last_gripper_command = 0.0
        self._last_gripper_command_raw = 0.0
        self._last_gripper_target_width = self._max_gripper_width
        self._last_gripper_command_timestamp = time.time()
        self._last_gripper_command_source = "init"
        self._ee_joint_soft_margin = float(
            os.environ.get(
                "FRANKA_EE_JOINT_SOFT_MARGIN_RAD",
                str(DEFAULT_EE_JOINT_SOFT_MARGIN_RAD),
            )
        )
        if not 0.0 < self._ee_joint_soft_margin < 0.5:
            raise ValueError(
                "FRANKA_EE_JOINT_SOFT_MARGIN_RAD must be in (0, 0.5), got "
                f"{self._ee_joint_soft_margin}"
            )
        self._last_ee_hold_log_time = 0.0
        self._last_ee_hold_reason = ""
        # PICO starts in preview mode and may not issue an EEF command for
        # several seconds. Starting Polymetis' Cartesian policy here would
        # leave libfranka without a desired-pose update and trigger its 1 s
        # communication watchdog before the operator ever squeezes. GELLO
        # keeps the historical eager behavior; PICO opts into lazy startup.
        self._lazy_cartesian_start = (
            control_mode == "ee"
            and os.environ.get("FRANKA_LAZY_EE_START", "0").strip().lower()
            in ("1", "true", "yes", "on")
        )
        self._cartesian_policy_started = False
        if not self._lazy_cartesian_start:
            self._start_control_mode(control_mode)
        self._debug_gripper = os.environ.get("TELEOP_DEBUG_GRIPPER", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._last_gripper_debug_time = 0.0
        # The gripper SDK call can take hundreds of milliseconds while the
        # Cartesian command itself is already accepted by Polymetis.  Keep it
        # off the single ZMQ request thread and collapse queued updates to the
        # latest target so it can never build a latency backlog.
        self._gripper_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="fr3-gripper"
        )
        self._gripper_lock = threading.Lock()
        self._pending_gripper_command = None
        self._gripper_worker_running = False
        if open_gripper_on_init:
            self.gripper.goto(
                width=self._max_gripper_width,
                speed=GRIPPER_SPEED,
                force=GRIPPER_FORCE,
            )
            self._remember_gripper_command(0.0, self._max_gripper_width, "init_open")
            time.sleep(1)
        else:
            print("Skipping gripper open on init.")
            try:
                width = float(np.clip(self.gripper.get_state().width, 0.0, self._max_gripper_width))
                closedness = 1.0 - width / self._max_gripper_width
                self._remember_gripper_command(closedness, width, "init_feedback")
            except Exception as exc:
                print(f"Could not initialize cached gripper command from feedback: {exc}")

    def _start_control_mode(self, control_mode: str) -> None:
        if control_mode == "joint":
            self.robot.start_joint_impedance()
        elif control_mode == "ee":
            self.robot.start_cartesian_impedance()
            self._cartesian_policy_started = True
        else:
            raise ValueError(f"Unsupported fr3 control_mode: {control_mode}")

    def _ensure_joint_controller_running(self) -> None:
        if self.control_mode != "joint":
            return
        try:
            if self.robot.is_running_policy():
                return
        except Exception as exc:
            print(f"Unable to check robot controller state: {exc}")

        print("Joint impedance controller is not running; starting it.")
        self.robot.start_joint_impedance()
        deadline = time.time() + 2.0
        while time.time() < deadline:
            try:
                if self.robot.is_running_policy():
                    return
            except Exception:
                pass
            time.sleep(0.02)
        print("Warning: joint impedance controller did not report running yet.")

    def _target_moves_away_from_limits(
            self,
            joints_tensor,
            unsafe_indices,
            lower_distance,
            upper_distance,
            target_pose,
            ) -> bool:
        """Conservatively predict whether a Cartesian target retreats from limits."""
        try:
            from scipy.spatial.transform import Rotation

            target = np.asarray(target_pose, dtype=float).reshape(6)
            current_position, current_quaternion = self.robot.get_ee_pose()
            current_position_np = current_position.detach().cpu().numpy()
            current_quaternion_np = current_quaternion.detach().cpu().numpy()
            current_rotation = Rotation.from_quat(current_quaternion_np)
            target_rotation = Rotation.from_euler("xyz", target[3:])

            # CartesianImpedanceControl uses world-frame position/orientation
            # error with this Jacobian. Match that convention for the local
            # least-squares joint-direction prediction.
            position_error_world = target[:3] - current_position_np
            rotation_error_world = (
                target_rotation * current_rotation.inv()
            ).as_rotvec()
            desired_twist_world = np.r_[position_error_world, rotation_error_world]
            jacobian = (
                self.robot.robot_model.compute_jacobian(joints_tensor)
                .detach()
                .cpu()
                .numpy()
            )
            predicted_delta = np.linalg.pinv(jacobian, rcond=1e-3) @ desired_twist_world
        except Exception as exc:
            self._log_ee_hold(
                {
                    "status": "hold",
                    "reason": "joint_escape_direction_unavailable",
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )
            return False

        minimum_escape_delta = 1e-4
        for index in unsafe_indices:
            index = int(index)
            near_lower = lower_distance[index] <= upper_distance[index]
            delta = float(predicted_delta[index])
            if near_lower and delta <= minimum_escape_delta:
                return False
            if not near_lower and delta >= -minimum_escape_delta:
                return False
        return True

    def _joint_soft_limit_hold(self, target_pose=None):
        """Return a structured HOLD result when an arm joint nears a hard limit."""
        joints_tensor = self.robot.get_joint_positions()
        if hasattr(joints_tensor, "detach"):
            joints = joints_tensor.detach().cpu().numpy().astype(float)
        else:
            joints = np.asarray(joints_tensor, dtype=float)
        joints = joints.reshape(-1)
        if joints.shape != (7,) or not np.all(np.isfinite(joints)):
            raise RuntimeError(f"Invalid Franka joint feedback: {joints}")

        margin = getattr(
            self, "_ee_joint_soft_margin", DEFAULT_EE_JOINT_SOFT_MARGIN_RAD
        )
        lower_distance = joints - EE_JOINT_LIMITS_LOWER
        upper_distance = EE_JOINT_LIMITS_UPPER - joints
        distances = np.minimum(lower_distance, upper_distance)
        unsafe = np.flatnonzero(distances <= margin)
        if unsafe.size == 0:
            return None

        if target_pose is not None and self._target_moves_away_from_limits(
            joints_tensor,
            unsafe,
            lower_distance,
            upper_distance,
            target_pose,
        ):
            return None

        index = int(unsafe[np.argmin(distances[unsafe])])
        side = "lower" if lower_distance[index] <= upper_distance[index] else "upper"
        limit = (
            EE_JOINT_LIMITS_LOWER[index]
            if side == "lower"
            else EE_JOINT_LIMITS_UPPER[index]
        )
        return {
            "status": "hold",
            "reason": "joint_soft_limit",
            "joint_index": index,
            "joint_position": float(joints[index]),
            "limit_side": side,
            "limit": float(limit),
            "margin_rad": float(margin),
        }

    def _log_ee_hold(self, result) -> None:
        reason = str(result.get("reason", "unknown"))
        now = time.monotonic()
        last_time = getattr(self, "_last_ee_hold_log_time", 0.0)
        last_reason = getattr(self, "_last_ee_hold_reason", "")
        if reason == last_reason and now - last_time < 1.0:
            return
        self._last_ee_hold_log_time = now
        self._last_ee_hold_reason = reason
        print(f"Cartesian safety HOLD: {result}")

    def _hold_current_ee_pose(self) -> None:
        """Replace a potentially unsafe target with the measured current pose."""
        position, orientation = self.robot.get_ee_pose()
        update_idx = self.robot.update_desired_ee_pose(
            position=position, orientation=orientation
        )
        if update_idx == -1:
            raise RuntimeError("Franka IK failed while latching the current EE pose")

    def _guard_cartesian_controller(self, target_pose=None):
        """Hold at soft limits and safely recover a stopped Cartesian policy."""
        soft_limit = self._joint_soft_limit_hold(target_pose)
        if soft_limit is not None:
            if self.robot.is_running_policy():
                self._hold_current_ee_pose()
            self._log_ee_hold(soft_limit)
            return soft_limit

        if self.robot.is_running_policy():
            self._cartesian_policy_started = True
            return None

        # The low-level client may stop a custom policy after automatic error
        # recovery. Restart only from a joint-safe state. Do not wait up to two
        # seconds for is_running_policy(): the 1 kHz Franka watchdog can expire
        # during that poll and the ZMQ caller then reports a false timeout.
        self.robot.start_cartesian_impedance()
        self._cartesian_policy_started = True
        return None

    def num_dofs(self) -> int:
        """Get the number of joints of the robot.

        Returns:
            int: The number of joints of the robot.
        """
        return 8

    def get_control_mode(self) -> str:
        return self.control_mode

    def get_joint_state(self) -> np.ndarray:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        robot_joints = self.robot.get_joint_positions()
        gripper_pos = self.gripper.get_state()
        pos = np.append(robot_joints, gripper_pos.width / self._max_gripper_width)
        return pos

    def command_joint_state(
            self,
            joint_state: np.ndarray,
            gripper_speed: float = GRIPPER_SPEED,
            gripper_force: float = GRIPPER_FORCE,
            update_gripper: bool = True,
            ) -> None:
        """Command the leader robot to a given state.

        Args:
            joint_state (np.ndarray): The state to command the leader robot to.
        """
        import torch

        self._ensure_joint_controller_running()
        desired_joints = torch.tensor(joint_state[:-1])
        try:
            self.robot.update_desired_joint_positions(desired_joints)
        except Exception as exc:
            if "no controller running" not in str(exc):
                raise
            print("Joint controller was missing during update; restarting and retrying.")
            self.robot.start_joint_impedance()
            time.sleep(0.1)
            self.robot.update_desired_joint_positions(desired_joints)
        if update_gripper:
            raw_gripper_action = float(joint_state[-1])
            if not np.isfinite(raw_gripper_action):
                raise ValueError(f"Invalid gripper action: {raw_gripper_action}")
            gripper_action = float(np.clip(raw_gripper_action, 0.0, 1.0))
            target_width = float(
                np.clip(
                    self._max_gripper_width * (1.0 - gripper_action),
                    0.0,
                    self._max_gripper_width,
                )
            )
            self._debug_gripper_command(
                raw_gripper_action,
                gripper_action,
                target_width,
                gripper_speed,
                gripper_force,
                phase="before",
            )
            self.gripper.goto(
                width=target_width,
                speed=gripper_speed,
                force=gripper_force,
            )
            self._debug_gripper_command(
                raw_gripper_action,
                gripper_action,
                target_width,
                gripper_speed,
                gripper_force,
                phase="after",
            )
            self._remember_gripper_command(
                gripper_action,
                target_width,
                "command_joint_state",
            )

    def move_to_joint_positions(
            self,
            joint_positions: np.ndarray,
            gripper_width: Optional[float] = None,
            gripper_speed: float = GRIPPER_SPEED,
            gripper_force: float = GRIPPER_FORCE,
            restart_controller: bool = True,
            ) -> None:
        """Move through Polymetis' blocking joint trajectory API."""
        joints = np.asarray(joint_positions, dtype=float).reshape(-1)
        if joints.shape != (7,):
            raise ValueError(f"Expected 7 joint positions, got shape {joints.shape}")
        if not np.all(np.isfinite(joints)):
            raise ValueError(f"Invalid joint positions: {joints}")

        desired_joints = torch.tensor(joints, dtype=torch.float32)
        self.robot.move_to_joint_positions(desired_joints)

        if gripper_width is not None:
            width = float(np.clip(gripper_width, 0.0, self._max_gripper_width))
            closedness = 1.0 - width / self._max_gripper_width
            self.gripper.goto(
                width=width,
                speed=gripper_speed,
                force=gripper_force,
            )
            self._remember_gripper_command(
                closedness,
                width,
                "move_to_joint_positions",
            )

        if restart_controller:
            self._start_control_mode(self.control_mode)

    def command_ee_pose(
            self,
            pose_6d: np.ndarray,
            gripper_width: float,
            gripper_speed: float = 0.05,
            gripper_force: float = 40.0,
            update_gripper: bool = True,
            ) -> Optional[Dict[str, object]]:
        """Command an absolute EE pose [x, y, z, rx, ry, rz] and gripper width in meters."""
        pose = np.asarray(pose_6d, dtype=float).reshape(-1)
        if pose.shape != (6,):
            raise ValueError(f"Expected pose_6d shape (6,), got {pose.shape}")

        hold_result = self._guard_cartesian_controller(pose)
        if hold_result is not None:
            # A Cartesian soft-limit HOLD must not suppress an independent
            # gripper command. The arm can be held while the operator still
            # opens/closes the fingers.
            if update_gripper:
                width = float(np.clip(gripper_width, 0.0, self._max_gripper_width))
                closedness = 1.0 - width / self._max_gripper_width
                self._remember_gripper_command(
                    closedness,
                    width,
                    "command_ee_pose_hold",
                )
                self._queue_gripper_command(width, gripper_speed, gripper_force)
            return hold_result

        from scipy.spatial.transform import Rotation

        position = torch.tensor(pose[:3], dtype=torch.float32)
        quat = Rotation.from_euler("xyz", pose[3:], degrees=False).as_quat()
        orientation = torch.tensor(quat, dtype=torch.float32)
        try:
            update_idx = self.robot.update_desired_ee_pose(
                position=position, orientation=orientation
            )
        except Exception as exc:
            if "no controller running" not in str(exc).lower():
                raise
            # is_running_policy() can report a stale/optimistic state after
            # Polymetis has stopped the policy. Force a restart and retry the
            # exact same target once; the old code restarted/checks but then
            # re-raised without ever retrying the pose update.
            self.robot.start_cartesian_impedance()
            self._cartesian_policy_started = True
            try:
                update_idx = self.robot.update_desired_ee_pose(
                    position=position, orientation=orientation
                )
            except Exception as retry_exc:
                if "no controller running" not in str(retry_exc).lower():
                    raise
                return {
                    "status": "hold",
                    "reason": "cartesian_controller_restarted",
                    "detail": str(retry_exc),
                }
        if update_idx == -1:
            raise RuntimeError(f"Franka IK failed for pose_6d={pose.tolist()}")

        if update_gripper:
            width = float(np.clip(gripper_width, 0.0, self._max_gripper_width))
            closedness = 1.0 - width / self._max_gripper_width
            self._remember_gripper_command(
                closedness,
                width,
                "command_ee_pose",
            )
            self._queue_gripper_command(width, gripper_speed, gripper_force)

    def _queue_gripper_command(
        self, width: float, speed: float, force: float
    ) -> None:
        """Queue only the newest gripper target and return immediately.

        ZMQ command latency must be governed by the Cartesian update, not by a
        blocking Robotiq/DAS-Finger action.  A single worker preserves command
        ordering while replacing pending intermediate targets with the latest
        controller value.
        """
        executor = getattr(self, "_gripper_executor", None)
        if executor is None:
            # Legacy unit-test adapters construct fr3Robot with object.__new__.
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fr3-gripper")
            self._gripper_executor = executor
            self._gripper_lock = threading.Lock()
            self._pending_gripper_command = None
            self._gripper_worker_running = False
        with self._gripper_lock:
            self._pending_gripper_command = (float(width), float(speed), float(force))
            if self._gripper_worker_running:
                return
            self._gripper_worker_running = True
        executor.submit(self._drain_gripper_commands)

    def _drain_gripper_commands(self) -> None:
        while True:
            with self._gripper_lock:
                command = self._pending_gripper_command
                self._pending_gripper_command = None
                if command is None:
                    self._gripper_worker_running = False
                    return
            width, speed, force = command
            try:
                self.gripper.goto(width=width, speed=speed, force=force)
            except Exception as exc:
                LOGGER.warning(
                    "asynchronous gripper command failed width=%.4f force=%.1f: %s",
                    width,
                    force,
                    exc,
                )

    def _debug_gripper_command(
        self,
        raw_gripper_action: float,
        gripper_action: float,
        target_width: float,
        gripper_speed: float,
        gripper_force: float,
        phase: str,
    ) -> None:
        if not self._debug_gripper:
            return
        now = time.time()
        if now - self._last_gripper_debug_time < GRIPPER_DEBUG_INTERVAL:
            return
        self._last_gripper_debug_time = now
        try:
            gripper_state = self.gripper.get_state()
            current_width = float(gripper_state.width)
            current_width_msg = f"{current_width:.4f}"
            state_msg = (
                f"is_moving={getattr(gripper_state, 'is_moving', 'unknown')} "
                f"is_grasped={getattr(gripper_state, 'is_grasped', 'unknown')} "
                "prev_command_successful="
                f"{getattr(gripper_state, 'prev_command_successful', 'unknown')}"
            )
        except Exception as exc:
            current_width_msg = f"unavailable ({type(exc).__name__}: {exc})"
            state_msg = "state=unavailable"
        print(
            "[fr3 gripper] "
            f"phase={phase} "
            f"action_raw={raw_gripper_action:.3f} "
            f"action_clipped={gripper_action:.3f} "
            f"target_width={target_width:.4f} "
            f"current_width={current_width_msg} "
            f"speed={gripper_speed} force={gripper_force} "
            f"{state_msg}"
        )

    def _get_gripper_max_width(self) -> float:
        max_width = MAX_OPEN
        metadata = getattr(self.gripper, "metadata", None)
        if metadata is not None and hasattr(metadata, "max_width"):
            try:
                max_width = float(metadata.max_width)
            except Exception:
                max_width = MAX_OPEN
        else:
            try:
                max_width = float(self.gripper.get_state().max_width)
            except Exception:
                max_width = MAX_OPEN
        if not np.isfinite(max_width) or max_width <= 0:
            max_width = MAX_OPEN
        return max_width

    def get_observations(self, fast: bool = False) -> Dict[str, np.ndarray]:
        if fast:
            # Teleop anchoring/tracking only needs EEF pose and gripper width.
            # Avoid five synchronous gRPC calls on the command server; the
            # full recorder observation contract remains the default below.
            ee_pos, ee_quat = self.robot.get_ee_pose()
            ee_pos = ee_pos.detach().cpu().numpy()
            ee_quat = ee_quat.detach().cpu().numpy()
            from scipy.spatial.transform import Rotation
            ee_euler = Rotation.from_quat(ee_quat).as_euler("xyz", degrees=False)
            return {
                "ee_pos_quat": np.concatenate([ee_pos, ee_quat]),
                "ee_pose_euler": np.concatenate([ee_pos, ee_euler]),
                "gripper_width": np.array(
                    [self._last_gripper_target_width], dtype=np.float32
                ),
            }
        joints = self.get_joint_state()
        joint_velocities = (
            self.robot.get_joint_velocities()
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        ee_pos, ee_quat = self.robot.get_ee_pose()
        ee_pos = ee_pos.detach().cpu().numpy()
        ee_quat = ee_quat.detach().cpu().numpy()
        pos_quat = np.concatenate([ee_pos, ee_quat])

        from scipy.spatial.transform import Rotation

        ee_euler = Rotation.from_quat(ee_quat).as_euler("xyz", degrees=False)
        pos_euler = np.concatenate([ee_pos, ee_euler])
        gripper_pos = np.array([joints[-1]])
        gripper_width = float(joints[-1] * self._max_gripper_width)
        return {
            # Preserve the existing 8-D compatibility contract: 7 arm joints
            # followed by normalized gripper width.  Callers that only need the
            # arm explicitly slice the first seven values.
            "joint_positions": joints,
            "joint_velocities": joint_velocities,
            "ee_pos_quat": pos_quat,
            "ee_pose_euler": pos_euler,
            "gripper_position": gripper_pos,
            "gripper_width": np.array([gripper_width], dtype=np.float32),
            "gripper_closedness": np.array([self._last_gripper_command_raw], dtype=np.float32),
            "gripper_01closedness": np.array([self._last_gripper_command], dtype=np.float32),
            "gripper_closed": np.array([self._last_gripper_command], dtype=np.float32),
            "gripper_command": np.array([self._last_gripper_command], dtype=np.float32),
            "gripper_command_raw": np.array([self._last_gripper_command_raw], dtype=np.float32),
            "gripper_target_width": np.array([self._last_gripper_target_width], dtype=np.float32),
            "gripper_command_timestamp": np.array([self._last_gripper_command_timestamp], dtype=np.float64),
            "gripper_command_source": self._last_gripper_command_source,
        }

    def _remember_gripper_command(
            self,
            closedness: float,
            target_width: float,
            source: str,
            ) -> None:
        closedness = float(np.clip(closedness, 0.0, 1.0))
        max_width = getattr(self, "_max_gripper_width", MAX_OPEN)
        self._last_gripper_command_raw = closedness
        self._last_gripper_command = 1.0 if closedness >= GRIPPER_COMMAND_THRESHOLD else 0.0
        self._last_gripper_target_width = float(np.clip(target_width, 0.0, max_width))
        self._last_gripper_command_timestamp = time.time()
        self._last_gripper_command_source = source


def main():
    robot = fr3Robot()
    current_joints = robot.get_joint_state()
    # move a small delta 0.1 rad
    move_joints = current_joints + 0.05
    # make last joint (gripper) closed
    move_joints[-1] = 0.5
    time.sleep(1)
    m = robot._max_gripper_width
    robot.gripper.goto(1 * m, speed=GRIPPER_SPEED, force=GRIPPER_FORCE)
    time.sleep(1)
    robot.gripper.goto(1.05 * m, speed=GRIPPER_SPEED, force=GRIPPER_FORCE)
    time.sleep(1)
    robot.gripper.goto(1.1 * m, speed=GRIPPER_SPEED, force=GRIPPER_FORCE)
    time.sleep(1)


if __name__ == "__main__":
    main()
