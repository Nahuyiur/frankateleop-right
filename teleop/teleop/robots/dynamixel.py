from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from teleop.robots.robot import Robot


class DynamixelRobot(Robot):
    """A Dynamixel-based GELLO leader with optional joint damping."""

    def __init__(
        self,
        joint_ids: Sequence[int],
        joint_offsets: Optional[Sequence[float]] = None,
        joint_signs: Optional[Sequence[int]] = None,
        real: bool = False,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 57600,
        gripper_config: Optional[Tuple[int, float, float]] = None,
        start_joints: Optional[np.ndarray] = None,
        dynamixel_torque_enable: Optional[Sequence[int]] = None,
        dynamixel_kp_p: Optional[Sequence[int]] = None,
        dynamixel_kp_i: Optional[Sequence[int]] = None,
        dynamixel_kp_d: Optional[Sequence[int]] = None,
        dynamixel_goal_position: Optional[Sequence[float]] = None,
    ):
        from teleop.dynamixel.driver import (
            DynamixelDriver,
            DynamixelDriverProtocol,
            FakeDynamixelDriver,
        )

        print(f"attempting to connect to port: {port}")
        self._num_arm_joints = len(joint_ids)
        self.gripper_open_close: Optional[Tuple[float, float]]
        if gripper_config is not None:
            assert joint_offsets is not None
            assert joint_signs is not None

            # joint_ids.append(gripper_config[0])
            # joint_offsets.append(0.0)
            # joint_signs.append(1)
            joint_ids = tuple(joint_ids) + (gripper_config[0],)
            joint_offsets = tuple(joint_offsets) + (0.0,)
            joint_signs = tuple(joint_signs) + (1,)
            self.gripper_open_close = (
                gripper_config[1] * np.pi / 180,
                gripper_config[2] * np.pi / 180,
            )
        else:
            self.gripper_open_close = None

        self._joint_ids = joint_ids
        self._driver: DynamixelDriverProtocol

        if joint_offsets is None:
            self._joint_offsets = np.zeros(len(joint_ids))
        else:
            self._joint_offsets = np.array(joint_offsets)

        if joint_signs is None:
            self._joint_signs = np.ones(len(joint_ids))
        else:
            self._joint_signs = np.array(joint_signs)

        assert len(self._joint_ids) == len(self._joint_offsets), (
            f"joint_ids: {len(self._joint_ids)}, "
            f"joint_offsets: {len(self._joint_offsets)}"
        )
        assert len(self._joint_ids) == len(self._joint_signs), (
            f"joint_ids: {len(self._joint_ids)}, "
            f"joint_signs: {len(self._joint_signs)}"
        )
        assert np.all(
            np.abs(self._joint_signs) == 1
        ), f"joint_signs: {self._joint_signs}"

        if real:
            self._driver = DynamixelDriver(joint_ids, port=port, baudrate=baudrate)
            if dynamixel_torque_enable is not None:
                self._configure_damping(
                    torque_enable=dynamixel_torque_enable,
                    kp_p=dynamixel_kp_p,
                    kp_i=dynamixel_kp_i,
                    kp_d=dynamixel_kp_d,
                    goal_position_rad=dynamixel_goal_position,
                )
                self._torque_on = any(
                    enabled == 1 for enabled in dynamixel_torque_enable
                )
            else:
                self._driver.set_torque_mode(False)
                self._torque_on = False
        else:
            self._driver = FakeDynamixelDriver(joint_ids)
            self._torque_on = False
        self._closed = False
        self._last_pos = None
        self._alpha = 0.99

        if start_joints is not None:
            # loop through all joints and add +- 2pi to the joint offsets to get the closest to start joints
            new_joint_offsets = []
            current_joints = self.get_joint_state()
            assert current_joints.shape == start_joints.shape
            if gripper_config is not None:
                current_joints = current_joints[:-1]
                start_joints = start_joints[:-1]
            for idx, (c_joint, s_joint, joint_offset) in enumerate(
                zip(current_joints, start_joints, self._joint_offsets)
            ):
                new_joint_offsets.append(
                    np.pi
                    * 2
                    * np.round((-s_joint + c_joint) / (2 * np.pi))
                    * self._joint_signs[idx]
                    + joint_offset
                )
            if gripper_config is not None:
                new_joint_offsets.append(self._joint_offsets[-1])
            self._joint_offsets = np.array(new_joint_offsets)

    def _configure_damping(
        self,
        torque_enable: Sequence[int],
        kp_p: Optional[Sequence[int]] = None,
        kp_i: Optional[Sequence[int]] = None,
        kp_d: Optional[Sequence[int]] = None,
        goal_position_rad: Optional[Sequence[float]] = None,
    ):
        """Map arm damping targets to the raw motor frame.

        Damping inputs describe arm joints only. If the GELLO trigger gripper is
        present, its register fields are padded with zero and its torque remains
        disabled.
        """
        total_count = len(self._joint_ids)
        arm_count = self._num_arm_joints

        def _pad_arm_values(name, values):
            if values is None:
                return None
            if len(values) != arm_count:
                raise ValueError(f"{name} must contain one value per arm joint")
            return list(values) + [0] * (total_count - arm_count)

        padded_torque = _pad_arm_values("torque_enable", torque_enable)
        if any(value not in (0, 1) for value in torque_enable):
            raise ValueError("torque_enable values must be 0 or 1")

        raw_goal_rad = None
        if goal_position_rad is not None:
            if len(goal_position_rad) != arm_count:
                raise ValueError(
                    "goal_position_rad must contain one value per arm joint"
                )
            raw_goal_rad = (
                np.asarray(goal_position_rad) * self._joint_signs[:arm_count]
                + self._joint_offsets[:arm_count]
            ).tolist()
            raw_goal_rad += [0.0] * (total_count - arm_count)

        self._driver.configure_damping(
            torque_enable=padded_torque,
            kp_p=_pad_arm_values("kp_p", kp_p),
            kp_i=_pad_arm_values("kp_i", kp_i),
            kp_d=_pad_arm_values("kp_d", kp_d),
            goal_position_rad=raw_goal_rad,
        )

    def num_dofs(self) -> int:
        return len(self._joint_ids)

    def get_joint_state(self) -> np.ndarray:
        pos = (self._driver.get_joints() - self._joint_offsets) * self._joint_signs
        assert len(pos) == self.num_dofs()

        if self.gripper_open_close is not None:
            # map pos to [0, 1]
            g_pos = (pos[-1] - self.gripper_open_close[0]) / (
                self.gripper_open_close[1] - self.gripper_open_close[0]
            )
            g_pos = min(max(0, g_pos), 1)
            pos[-1] = g_pos

        if self._last_pos is None:
            self._last_pos = pos
        else:
            # exponential smoothing
            pos = self._last_pos * (1 - self._alpha) + pos * self._alpha
            self._last_pos = pos

        return pos

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        self._driver.set_joints((joint_state + self._joint_offsets).tolist())

    def close(self):
        """Disable all leader torque before closing the serial driver."""
        if getattr(self, "_closed", False):
            return
        driver = getattr(self, "_driver", None)
        if driver is None:
            return
        try:
            driver.close()
        finally:
            self._torque_on = False
            self._closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def set_torque_mode(self, mode: bool):
        if mode == self._torque_on:
            return
        self._driver.set_torque_mode(mode)
        self._torque_on = mode

    def get_observations(self) -> Dict[str, np.ndarray]:
        return {"joint_state": self.get_joint_state()}
