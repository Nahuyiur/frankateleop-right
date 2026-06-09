import os
import time
import torch
from typing import Dict
from typing import Optional
import numpy as np
from teleop.robots.robot import Robot

MAX_OPEN = 0.09
GRIPPER_SPEED = 0.1
GRIPPER_FORCE = 20.0
GRIPPER_DEBUG_INTERVAL = 0.5


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
        self._start_control_mode(control_mode)
        self._debug_gripper = os.environ.get("TELEOP_DEBUG_GRIPPER", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._last_gripper_debug_time = 0.0
        if open_gripper_on_init:
            self.gripper.goto(
                width=self._max_gripper_width,
                speed=GRIPPER_SPEED,
                force=GRIPPER_FORCE,
            )
            time.sleep(1)
        else:
            print("Skipping gripper open on init.")

    def _start_control_mode(self, control_mode: str) -> None:
        if control_mode == "joint":
            self.robot.start_joint_impedance()
        elif control_mode == "ee":
            self.robot.start_cartesian_impedance()
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
        self.gripper.goto(width=target_width, speed=gripper_speed, force=gripper_force)
        self._debug_gripper_command(
            raw_gripper_action,
            gripper_action,
            target_width,
            gripper_speed,
            gripper_force,
            phase="after",
        )

    def command_ee_pose(
        self,
        pose_6d: np.ndarray,
        gripper_width: float,
        gripper_speed: float = 0.05,
        gripper_force: float = 40.0,
        update_gripper: bool = True,
    ) -> None:
        """Command an absolute EE pose [x, y, z, rx, ry, rz] and gripper width in meters."""
        pose = np.asarray(pose_6d, dtype=float).reshape(-1)
        if pose.shape != (6,):
            raise ValueError(f"Expected pose_6d shape (6,), got {pose.shape}")

        from scipy.spatial.transform import Rotation

        position = torch.tensor(pose[:3], dtype=torch.float32)
        quat = Rotation.from_euler("xyz", pose[3:], degrees=False).as_quat()
        orientation = torch.tensor(quat, dtype=torch.float32)
        update_idx = self.robot.update_desired_ee_pose(
            position=position, orientation=orientation
        )
        if update_idx == -1:
            raise RuntimeError(f"Franka IK failed for pose_6d={pose.tolist()}")

        if update_gripper:
            width = float(np.clip(gripper_width, 0.0, self._max_gripper_width))
            self.gripper.goto(width=width, speed=gripper_speed, force=gripper_force)

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
        try:
            max_width = float(self.gripper.get_state().max_width)
        except Exception:
            max_width = MAX_OPEN
        if not np.isfinite(max_width) or max_width <= 0:
            max_width = MAX_OPEN
        return max_width

    def get_observations(self) -> Dict[str, np.ndarray]:
        joints = self.get_joint_state()
        ee_pos, ee_quat = self.robot.get_ee_pose()
        ee_pos = ee_pos.detach().cpu().numpy()
        ee_quat = ee_quat.detach().cpu().numpy()
        pos_quat = np.concatenate([ee_pos, ee_quat])

        from scipy.spatial.transform import Rotation

        ee_euler = Rotation.from_quat(ee_quat).as_euler("xyz", degrees=False)
        pos_euler = np.concatenate([ee_pos, ee_euler])
        gripper_pos = np.array([joints[-1]])
        return {
            "joint_positions": joints,
            "joint_velocities": joints,
            "ee_pos_quat": pos_quat,
            "ee_pose_euler": pos_euler,
            "gripper_position": gripper_pos,
        }


def main():
    robot = fr3Robot()
    current_joints = robot.get_joint_state()
    # move a small delta 0.1 rad
    move_joints = current_joints + 0.05
    # make last joint (gripper) closed
    move_joints[-1] = 0.5
    time.sleep(1)
    m = 0.09
    robot.gripper.goto(1 * m, speed=255, force=255)
    time.sleep(1)
    robot.gripper.goto(1.05 * m, speed=255, force=255)
    time.sleep(1)
    robot.gripper.goto(1.1 * m, speed=255, force=255)
    time.sleep(1)


if __name__ == "__main__":
    main()
