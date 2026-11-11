import threading
import unittest
from unittest import mock

import numpy as np

from teleop.dynamixel.driver import (
    ADDR_GOAL_CURRENT,
    ADDR_GOAL_POSITION,
    ADDR_KP_D,
    ADDR_KP_I,
    ADDR_KP_P,
    ADDR_OPERATING_MODE,
    DEFAULT_GOAL_CURRENT_MA,
    DynamixelDriver,
    FakeDynamixelDriver,
    OPERATING_MODE_CURRENT_POSITION,
)
from teleop.robots.dynamixel import DynamixelRobot


class RecordingDriver:
    def __init__(self):
        self.config = None
        self.closed = False

    def configure_damping(self, **kwargs):
        self.config = kwargs

    def close(self):
        self.closed = True


class RecordingPacketHandler:
    def __init__(self, events):
        self.events = events

    def write1ByteTxRx(self, _port, dxl_id, addr, value):
        self.events.append(("torque", dxl_id, addr, value))
        return 0, 0


class DynamixelDampingTest(unittest.TestCase):
    def test_fake_driver_tracks_mode_current_gains_goal_and_torque_mask(self):
        driver = FakeDynamixelDriver((1, 2))
        driver.configure_damping(
            torque_enable=(1, 0),
            kp_p=(30, 0),
            kp_i=(0, 0),
            kp_d=(250, 5),
            goal_position_rad=(0.25, -0.5),
        )

        self.assertEqual(
            driver._config["operating_mode"],
            [OPERATING_MODE_CURRENT_POSITION, OPERATING_MODE_CURRENT_POSITION],
        )
        self.assertEqual(
            driver._config["goal_current"],
            [DEFAULT_GOAL_CURRENT_MA, DEFAULT_GOAL_CURRENT_MA],
        )
        self.assertEqual(driver._config["torque_enable"], [1, 0])
        self.assertEqual(driver._config["kp_p"], [30, 0])
        self.assertEqual(driver._config["kp_i"], [0, 0])
        self.assertEqual(driver._config["kp_d"], [250, 5])
        np.testing.assert_allclose(driver.get_joints(), [0.25, -0.5])
        self.assertTrue(driver.torque_enabled())
        driver.close()
        self.assertFalse(driver.torque_enabled())

    def test_robot_maps_user_goal_to_raw_frame_and_excludes_gripper(self):
        offsets = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        signs = np.array([1, -1, 1, -1, 1, -1, 1])
        goal = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        robot = DynamixelRobot(
            joint_ids=(1, 2, 3, 4, 5, 6, 7),
            joint_offsets=offsets,
            joint_signs=signs,
            gripper_config=(8, 200.0, 160.0),
            real=False,
        )
        recording_driver = RecordingDriver()
        robot._driver = recording_driver

        robot._configure_damping(
            torque_enable=(1, 1, 1, 1, 1, 1, 1),
            kp_p=(30, 60, 0, 30, 0, 0, 0),
            kp_i=(0, 0, 0, 0, 0, 0, 0),
            kp_d=(250, 100, 80, 60, 30, 10, 5),
            goal_position_rad=goal,
        )

        config = recording_driver.config
        self.assertEqual(config["torque_enable"], [1, 1, 1, 1, 1, 1, 1, 0])
        self.assertEqual(config["kp_p"], [30, 60, 0, 30, 0, 0, 0, 0])
        self.assertEqual(config["kp_i"], [0] * 8)
        self.assertEqual(config["kp_d"], [250, 100, 80, 60, 30, 10, 5, 0])
        expected_raw = (goal * signs + offsets).tolist() + [0.0]
        np.testing.assert_allclose(config["goal_position_rad"], expected_raw)

        robot.close()
        self.assertTrue(recording_driver.closed)

    def test_real_driver_configuration_order_uses_official_register_values(self):
        driver = DynamixelDriver.__new__(DynamixelDriver)
        driver._ids = (1, 2)
        driver._lock = threading.Lock()
        driver._portHandler = object()
        driver._torque_enabled = False
        events = []
        driver._packetHandler = RecordingPacketHandler(events)
        driver.set_torque_mode = lambda enabled: events.append(
            ("all_torque", enabled)
        )
        driver._write_values_to_addr = lambda addr, length, values: events.append(
            ("write", addr, length, list(values))
        )

        with mock.patch("teleop.dynamixel.driver.time.sleep"):
            driver.configure_damping(
                torque_enable=(1, 0),
                kp_p=(30, 60),
                kp_i=(0, 0),
                kp_d=(250, 100),
                goal_position_rad=(np.pi, -np.pi / 2),
            )

        self.assertEqual(events[0], ("all_torque", False))
        self.assertEqual(
            events[1],
            ("write", ADDR_OPERATING_MODE, 1, [OPERATING_MODE_CURRENT_POSITION] * 2),
        )
        self.assertEqual(
            events[2],
            ("write", ADDR_GOAL_CURRENT, 2, [DEFAULT_GOAL_CURRENT_MA] * 2),
        )
        self.assertEqual(events[3], ("write", ADDR_KP_P, 2, [30, 60]))
        self.assertEqual(events[4], ("write", ADDR_KP_I, 2, [0, 0]))
        self.assertEqual(events[5], ("write", ADDR_KP_D, 2, [250, 100]))
        self.assertEqual(events[6][-1], 1)
        self.assertEqual(events[7][-1], 0)
        self.assertEqual(
            events[8],
            ("write", ADDR_GOAL_POSITION, 4, [2048, -1024]),
        )

    def test_invalid_arm_length_is_rejected_before_driver_call(self):
        robot = DynamixelRobot(joint_ids=(1, 2), real=False)
        recording_driver = RecordingDriver()
        robot._driver = recording_driver
        with self.assertRaises(ValueError):
            robot._configure_damping(torque_enable=(1,))
        self.assertIsNone(recording_driver.config)
        robot.close()


if __name__ == "__main__":
    unittest.main()
