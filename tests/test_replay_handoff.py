"""No hardware: same-mode ownership changes must not reset a deflected spring."""
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from teleop.robots.fr3 import fr3Robot


class ReplayHandoffTests(unittest.TestCase):
    def make(self):
        r = object.__new__(fr3Robot)
        r.control_mode = 'ee'
        r._cartesian_policy_started = True
        r.robot = SimpleNamespace(is_running_policy=Mock(return_value=True),
                                  terminate_current_policy=Mock())
        r._start_control_mode = Mock()
        # A held spring can be stationary while its accepted target is offset.
        r._cartesian_hold_anchor = Mock(return_value={'ee_hold_joint_goal': np.full(7, .004)})
        self.state = dict(state_timestamp_ns=time.time_ns(), joint_positions=np.zeros(7),
                          joint_velocities=np.zeros(7))
        r._native_arm = SimpleNamespace(snapshot=lambda: self.state)
        return r

    def test_settled_deflected_hold_survives_acquire_and_release(self):
        r = self.make()
        for _ in range(2):
            self.assertTrue(r.set_replay_mode('ee')['preserved_target'])
        r.robot.terminate_current_policy.assert_not_called()
        r._start_control_mode.assert_not_called()

    def test_motion_gets_measured_pose_hold(self):
        r = self.make(); self.state['joint_velocities'][2] = .02
        self.assertFalse(r.set_replay_mode('ee')['preserved_target'])
        r.robot.terminate_current_policy.assert_called_once_with(return_log=False)
        r._start_control_mode.assert_called_once_with('ee')

    def test_large_unfinished_target_is_not_carried_into_new_lease(self):
        r = self.make(); r._cartesian_hold_anchor.return_value['ee_hold_joint_goal'][1] = .06
        self.assertFalse(r.set_replay_mode('ee')['preserved_target'])

    def test_stopped_policy_restarts(self):
        r = self.make(); r.robot.is_running_policy.return_value = False
        self.assertFalse(r.set_replay_mode('ee')['preserved_target'])
        r.robot.terminate_current_policy.assert_not_called()
        r._start_control_mode.assert_called_once_with('ee')

    def test_unverified_policy_episode_is_not_preserved(self):
        r = self.make(); r._cartesian_hold_anchor.return_value = {}
        self.assertFalse(r.set_replay_mode('ee')['preserved_target'])

    def test_joint_mode_transition_restarts(self):
        r = self.make()
        self.assertFalse(r.set_replay_mode('joint')['preserved_target'])
        self.assertEqual(r.control_mode, 'joint')

    def test_custom_experiment_and_forced_recovery_restart(self):
        for key in ('_tracking_impedance_override', '_tracking_impedance_applied', '_tracking_native'):
            r = self.make(); setattr(r, key, {'active': True})
            self.assertFalse(r.set_replay_mode('ee')['preserved_target'])
        r = self.make()
        self.assertFalse(r.set_replay_mode('ee', force_reanchor=True)['preserved_target'])

    def test_stale_or_nonfinite_feedback_fails_without_policy_mutation(self):
        for change in ('stale', 'nonfinite'):
            r = self.make()
            if change == 'stale': self.state['state_timestamp_ns'] -= 1_000_000_000
            else: self.state['joint_positions'][0] = np.nan
            with self.assertRaisesRegex(RuntimeError, 'fresh measured'):
                r.set_replay_mode('ee')
            r.robot.terminate_current_policy.assert_not_called()
            r._start_control_mode.assert_not_called()

    def test_transport_error_is_not_reported_as_preserved(self):
        r = self.make(); r.robot.is_running_policy.side_effect = TimeoutError('no reply')
        with self.assertRaises(TimeoutError): r.set_replay_mode('ee')
        r._start_control_mode.assert_not_called()


if __name__ == '__main__':
    unittest.main()
