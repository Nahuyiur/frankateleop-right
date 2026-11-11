import unittest

import numpy as np
import torch

from teleop.robots.fr3 import fr3Robot


class FakeRobotModel:
    def __init__(self, escape_joint=None):
        self.escape_joint = escape_joint

    def compute_jacobian(self, joints):
        jacobian = torch.zeros((6, 7), dtype=torch.float32)
        if self.escape_joint is not None:
            jacobian[0, self.escape_joint] = 1.0
        return jacobian


class FakePolymetisRobot:
    def __init__(self, joints, running=True, escape_joint=None):
        self.joints = torch.tensor(joints, dtype=torch.float32)
        self.running = running
        self.started = 0
        self.updates = []
        self.robot_model = FakeRobotModel(escape_joint)

    def get_joint_positions(self):
        return self.joints

    def is_running_policy(self):
        return self.running

    def start_cartesian_impedance(self):
        self.started += 1
        self.running = True

    def get_ee_pose(self):
        return torch.tensor([0.4, 0.0, 0.3]), torch.tensor([0.0, 0.0, 0.0, 1.0])

    def update_desired_ee_pose(self, *, position, orientation):
        self.updates.append((position.clone(), orientation.clone()))
        return len(self.updates)


def make_fr3(joints, running=True, escape_joint=None):
    robot = object.__new__(fr3Robot)
    robot.robot = FakePolymetisRobot(
        joints, running=running, escape_joint=escape_joint
    )
    robot.control_mode = "ee"
    robot._ee_joint_soft_margin = 0.10
    robot._last_ee_hold_log_time = 0.0
    robot._last_ee_hold_reason = ""
    return robot


class CartesianSafetyTests(unittest.TestCase):
    def test_near_joint_limit_latches_current_pose_and_returns_hold(self):
        robot = make_fr3([0.0, 0.0, 0.0, -1.5, 2.65, 2.0, 0.0])

        result = robot.command_ee_pose(
            np.zeros(6), 0.04, update_gripper=False
        )

        self.assertEqual(result["status"], "hold")
        self.assertEqual(result["reason"], "joint_soft_limit")
        self.assertEqual(result["joint_index"], 4)
        self.assertEqual(len(robot.robot.updates), 1)
        np.testing.assert_allclose(robot.robot.updates[0][0], [0.4, 0.0, 0.3])

    def test_stopped_controller_restarts_but_does_not_apply_stale_target(self):
        robot = make_fr3([0.0, 0.0, 0.0, -1.5, 0.0, 2.0, 0.0], running=False)

        result = robot.command_ee_pose(
            np.array([0.5, 0.0, 0.3, 0.0, 0.0, 0.0]),
            0.04,
            update_gripper=False,
        )

        self.assertEqual(result["reason"], "cartesian_controller_restarted")
        self.assertEqual(robot.robot.started, 1)
        self.assertEqual(robot.robot.updates, [])

    def test_safe_running_controller_accepts_target(self):
        robot = make_fr3([0.0, 0.0, 0.0, -1.5, 0.0, 2.0, 0.0])

        result = robot.command_ee_pose(
            np.array([0.5, 0.0, 0.3, 0.0, 0.0, 0.0]),
            0.04,
            update_gripper=False,
        )

        self.assertIsNone(result)
        self.assertEqual(len(robot.robot.updates), 1)

    def test_near_upper_limit_allows_target_predicted_to_move_away(self):
        robot = make_fr3(
            [0.0, 0.0, 0.0, -1.5, 2.65, 2.0, 0.0], escape_joint=4
        )

        result = robot.command_ee_pose(
            np.array([0.39, 0.0, 0.3, 0.0, 0.0, 0.0]),
            0.04,
            update_gripper=False,
        )

        self.assertIsNone(result)
        self.assertEqual(len(robot.robot.updates), 1)

    def test_near_upper_limit_rejects_target_predicted_to_move_toward_limit(self):
        robot = make_fr3(
            [0.0, 0.0, 0.0, -1.5, 2.65, 2.0, 0.0], escape_joint=4
        )

        result = robot.command_ee_pose(
            np.array([0.41, 0.0, 0.3, 0.0, 0.0, 0.0]),
            0.04,
            update_gripper=False,
        )

        self.assertEqual(result["reason"], "joint_soft_limit")
        self.assertEqual(len(robot.robot.updates), 1)


if __name__ == "__main__":
    unittest.main()
