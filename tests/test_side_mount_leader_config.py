import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "teleop"))

from experiments import run_env  # noqa: E402
from teleop.agents.teleop_agent import (  # noqa: E402
    DynamixelRobotConfig,
    PORT_CONFIG_MAP,
)


LEFT_PORT = (
    "/dev/serial/by-id/"
    "usb-FTDI_USB__-__Serial_Converter_FTBZIB2R-if00-port0"
)
RIGHT_PORT = (
    "/dev/serial/by-id/"
    "usb-FTDI_USB__-__Serial_Converter_FTBJTKP2-if00-port0"
)
UR_PORT = (
    "/dev/serial/by-id/"
    "usb-FTDI_USB__-__Serial_Converter_FTAAVS1C-if00-port0"
)


class SideMountLeaderConfigTest(unittest.TestCase):
    def test_official_fr3_port_profiles(self):
        left = PORT_CONFIG_MAP[LEFT_PORT]
        right = PORT_CONFIG_MAP[RIGHT_PORT]

        np.testing.assert_allclose(
            left.joint_offsets,
            [np.pi, np.pi, np.pi, 3 * np.pi / 2, np.pi, np.pi / 2, 5 * np.pi / 4],
        )
        self.assertEqual(tuple(left.joint_signs), (1, -1, 1, 1, 1, 1, 1))
        self.assertEqual(left.gripper_config, (8, 203.11, 161.31))

        np.testing.assert_allclose(
            right.joint_offsets,
            [0, np.pi, np.pi, 3 * np.pi / 2, np.pi / 2, np.pi / 2, 9 * np.pi / 4],
        )
        self.assertEqual(tuple(right.joint_signs), (1, -1, 1, 1, 1, 1, 1))
        self.assertEqual(right.gripper_config, (8, 199.11, 158.31))

    def test_official_ur_profile(self):
        ur = PORT_CONFIG_MAP[UR_PORT]
        self.assertEqual(tuple(ur.joint_ids), (1, 2, 3, 4, 5, 6))
        np.testing.assert_allclose(
            ur.joint_offsets,
            [3 * np.pi / 2, np.pi, 0, np.pi, 2 * np.pi, 3 * np.pi / 2],
        )
        self.assertEqual(tuple(ur.joint_signs), (1, -1, 1, 1, 1, -1))
        self.assertEqual(ur.gripper_config, (7, 208, 166))

    @patch("teleop.agents.teleop_agent.DynamixelRobot")
    def test_damping_fields_reach_driver_contract(self, robot_cls):
        cfg = DynamixelRobotConfig(
            joint_ids=(1, 2),
            joint_offsets=(0.0, 0.0),
            joint_signs=(1, -1),
            gripper_config=(3, 200, 160),
            dynamixel_torque_enable=(1, 1),
            dynamixel_kp_p=(30, 60),
            dynamixel_kp_i=(0, 0),
            dynamixel_kp_d=(250, 100),
            dynamixel_goal_position=(0.1, -0.2),
        )

        cfg.make_robot(port="mock-port")

        kwargs = robot_cls.call_args.kwargs
        self.assertEqual(kwargs["dynamixel_torque_enable"], [1, 1])
        self.assertEqual(kwargs["dynamixel_kp_p"], [30, 60])
        self.assertEqual(kwargs["dynamixel_kp_i"], [0, 0])
        self.assertEqual(kwargs["dynamixel_kp_d"], [250, 100])
        self.assertEqual(kwargs["dynamixel_goal_position"], [0.1, -0.2])

    def test_run_env_defaults_match_official_side_mount_values(self):
        self.assertEqual(run_env.Args().control_port, 6001)
        self.assertTrue(run_env.Args().torque_enable)
        self.assertEqual(
            run_env.DYNAMIXEL_DAMPING_DEFAULTS,
            {
                "torque_enable": [1, 1, 1, 1, 1, 1, 1],
                "kp_p": [30, 60, 0, 30, 0, 0, 0],
                "kp_i": [0, 0, 0, 0, 0, 0, 0],
                "kp_d": [250, 100, 80, 60, 30, 10, 5],
                "goal_position": [0.0, 0.0, 0.0, -1.571, 0.0, 1.571, 0.0],
            },
        )
        self.assertEqual(
            run_env.DYNAMIXEL_GOAL_POSITION_MAP["fr3_left"],
            [0.19421, 0.259328, 0.192638, -1.80504, 1.2055, 2.84043, -0.736754],
        )
        self.assertEqual(
            run_env.DYNAMIXEL_GOAL_POSITION_MAP["fr3_right"],
            [-0.234954, 0.278439, -0.195666, -1.76859, -0.929469, 2.8677, -1.36821],
        )

    def test_check_only_does_not_construct_damping_config(self):
        class FakeEnv:
            def get_obs(self):
                return {"joint_positions": np.zeros(8)}

        args = run_env.Args(
            agent="teleop",
            control_port=6001,
            teleop_port=LEFT_PORT,
            robot_type="fr3_left",
            torque_enable=True,
            check_only=True,
        )
        with patch.object(run_env, "ZMQClientRobot"):
            with patch.object(run_env, "RobotEnv", return_value=FakeEnv()):
                with patch.object(run_env, "TeleopAgent") as agent_cls:
                    agent_cls.return_value.act.return_value = np.zeros(8)
                    run_env.main(args)

        self.assertNotIn("dynamixel_config", agent_cls.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
