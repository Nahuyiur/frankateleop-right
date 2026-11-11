import os
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from teleop.agents.agent import Agent
from teleop.robots.dynamixel import DynamixelRobot


@dataclass
class DynamixelRobotConfig:
    joint_ids: Sequence[int]
    """The joint ids of TELEOP (not including the gripper). Usually (1, 2, 3 ...)."""

    joint_offsets: Sequence[float]
    """The joint offsets of TELEOP. There needs to be a joint offset for each joint_id and should be a multiple of pi/2."""

    joint_signs: Sequence[int]
    """The joint signs of TELEOP. There needs to be a joint sign for each joint_id and should be either 1 or -1.

    This will be different for each arm design. Refernce the examples below for the correct signs for your robot.
    """

    gripper_config: Tuple[int, int, int]
    """The gripper config of TELEOP. This is a tuple of (gripper_joint_id, degrees in open_position, degrees in closed_position)."""

    dynamixel_torque_enable: Optional[Sequence[int]] = None

    dynamixel_kp_p: Optional[Sequence[int]] = None

    dynamixel_kp_i: Optional[Sequence[int]] = None

    dynamixel_kp_d: Optional[Sequence[int]] = None

    dynamixel_goal_position: Optional[Sequence[float]] = None

    def __post_init__(self):
        assert len(self.joint_ids) == len(self.joint_offsets)
        assert len(self.joint_ids) == len(self.joint_signs)
        if self.dynamixel_torque_enable is not None:
            assert len(self.dynamixel_torque_enable) == len(self.joint_ids), (
                f"dynamixel_torque_enable length {len(self.dynamixel_torque_enable)} "
                f"must match arm joints ({len(self.joint_ids)})"
            )

    def make_robot(
        self, port: str = "/dev/ttyUSB0", start_joints: Optional[np.ndarray] = None
    ) -> DynamixelRobot:
        return DynamixelRobot(
            joint_ids=self.joint_ids,
            joint_offsets=list(self.joint_offsets),
            real=True,
            joint_signs=list(self.joint_signs),
            port=port,
            gripper_config=self.gripper_config,
            start_joints=start_joints,
            dynamixel_torque_enable=(
                list(self.dynamixel_torque_enable)
                if self.dynamixel_torque_enable is not None
                else None
            ),
            dynamixel_kp_p=(
                list(self.dynamixel_kp_p)
                if self.dynamixel_kp_p is not None
                else None
            ),
            dynamixel_kp_i=(
                list(self.dynamixel_kp_i)
                if self.dynamixel_kp_i is not None
                else None
            ),
            dynamixel_kp_d=(
                list(self.dynamixel_kp_d)
                if self.dynamixel_kp_d is not None
                else None
            ),
            dynamixel_goal_position=(
                list(self.dynamixel_goal_position)
                if self.dynamixel_goal_position is not None
                else None
            ),
        )


PORT_CONFIG_MAP: Dict[str, DynamixelRobotConfig] = {
# official side-mount right FR3
    "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBJTKP2-if00-port0": DynamixelRobotConfig(
        joint_ids=(1, 2, 3, 4, 5, 6, 7),
        joint_offsets=(
            0 * np.pi / 2,
            2 * np.pi / 2,
            2 * np.pi / 2,
            3 * np.pi / 2,
            1 * np.pi / 2,
            1 * np.pi / 2,
            9 * np.pi / 4,
        ),
        joint_signs=(1, -1, 1, 1, 1, 1, 1),
        gripper_config=(8, 199.11, 158.31),
    ),
    # legacy FR3 adapter
    "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTB2UWSZ-if00-port0": DynamixelRobotConfig(
        joint_ids=(1, 2, 3, 4, 5, 6, 7),
        joint_offsets=(
            0 * np.pi / 2,
            2 * np.pi / 2,
            4 * np.pi / 2,
            3 * np.pi / 2,
            1 * np.pi / 2 ,
            1 * np.pi / 2,
            9 * np.pi / 4 - 0.860072,
        ),
        joint_signs=(1, 1, 1, 1, 1, 1, 1),
        gripper_config=(8, 207.30, 165.50),
    ),
    # official side-mount left FR3
    "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBZIB2R-if00-port0": DynamixelRobotConfig(
        joint_ids=(1, 2, 3, 4, 5, 6, 7),
        joint_offsets=(
            2 * np.pi / 2,
            2 * np.pi / 2,
            2 * np.pi / 2,
            3 * np.pi / 2,
            2 * np.pi / 2,
            1 * np.pi / 2,
            5 * np.pi / 4,
        ),
        joint_signs=(1, -1, 1, 1, 1, 1, 1),
        gripper_config=(8, 203.11, 161.31),
    ),
# current left FR3 adapter
    "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBJKECV-if00-port0": DynamixelRobotConfig(
        joint_ids=(1, 2, 3, 4, 5, 6, 7),
        joint_offsets=(
            4 * np.pi / 4,
            2 * np.pi / 2,
            0 * np.pi / 2,
            3 * np.pi / 2,
            2 * np.pi / 2 ,
            1 * np.pi / 2,
            3 * np.pi / 4 - 0.905,
        ),
        joint_signs=(1, 1, 1, 1, 1, 1, 1),
        gripper_config=(8, 206.34, 164.54),
    ),
# right FR3
    "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAUMOPA-if00-port0": DynamixelRobotConfig(
        joint_ids=(1, 2, 3, 4, 5, 6, 7),
        joint_offsets=(
            5 * np.pi / 4,
            2 * np.pi / 2,
            4 * np.pi / 2,
            3 * np.pi / 2,
            2 * np.pi / 2 ,
            1 * np.pi / 2,
            5 * np.pi / 4,
        ),
        joint_signs=(1, -1, 1, 1, 1, 1, 1),
        gripper_config=(8, 206.34, 164.54),
    ),
    # official UR leader
    "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAAVS1C-if00-port0": DynamixelRobotConfig(
        joint_ids=(1, 2, 3, 4, 5, 6),
        joint_offsets=(
            3 * np.pi / 2,
            2 * np.pi / 2,
            0 * np.pi / 2,
            2 * np.pi / 2,
            4 * np.pi / 2,
            3 * np.pi / 2,
        ),
        joint_signs=(1, -1, 1, 1, 1, -1),
        gripper_config=(7, 208, 166),
    ),
}


class TeleopAgent(Agent):
    def __init__(
        self,
        port: str,
        dynamixel_config: Optional[DynamixelRobotConfig] = None,
        start_joints: Optional[np.ndarray] = None,
    ):
        if dynamixel_config is not None:
            self._robot = dynamixel_config.make_robot(
                port=port, start_joints=start_joints
            )
        else:
            assert os.path.exists(port), port
            assert port in PORT_CONFIG_MAP, f"Port {port} not in config map"

            config = PORT_CONFIG_MAP[port]
            self._robot = config.make_robot(port=port, start_joints=start_joints)

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        return self._robot.get_joint_state()
        dyna_joints = self._robot.get_joint_state()
        # current_q = dyna_joints[:-1]  # last one dim is the gripper
        current_gripper = dyna_joints[-1]  # last one dim is the gripper

        print(current_gripper)
        if current_gripper < 0.2:
            self._robot.set_torque_mode(False)
            return obs["joint_positions"]
        else:
            self._robot.set_torque_mode(False)
            return dyna_joints
