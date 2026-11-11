import os
import time
from polymetis._mozi_timing import timed
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
    [-2.90074, -1.83609, -2.90074, -3.07702, -2.87630, 0.43982, -3.05083], dtype=float
)
EE_JOINT_LIMITS_UPPER = np.array(
    [2.90074, 1.83609, 2.90074, -0.11694, 2.87630, 4.62163, 3.05083], dtype=float
)
# FR3 controller (FCI) position limits, the same values as joint_pos_upper/lower in
# polymetis robot_client/*_franka_hardware.yaml. The HOLD fires DEFAULT margin rad
# inside them; the polymetis spring starts outside this line (margin 0.05).
# 2026-10-02: replaced the old table (libfranka recommended limits minus 0.1) per
# user request. Note J4 now stops at -2.977; on 2026-10-01 the right arm hit
# self-collision avoidance at J4 -3.00, so a fold that far may reflex before HOLD.
DEFAULT_EE_JOINT_SOFT_MARGIN_RAD = 0.10
# A restarted Cartesian policy starts at the measured pose. If the incoming
# target is farther than this, it is the stale goal that caused the reflex
# (Kx=750 N/m: a 40 N reflex needs ~5 cm); hold and require a fresh squeeze.
RECOVERY_MAX_TARGET_OFFSET_M = 0.03
RECOVERY_MAX_TARGET_OFFSET_RAD = 0.20
# 2026-10-03 singularity-robust differential IK (FRANKA_IK_MODE=dik, default).
# Polymetis' update_desired_ee_pose re-solves a plain CLIK (damping 1e-6, no
# null-space term) from the *measured* joints every frame. Three failures seen
# on 10-03 follow from that:
#   * near singular poses the joint target moves ~1/sigma per unit EE motion:
#     J6~pi makes J5/J7 counter-rotate ~3x the operator's wrist tremor (the
#     "J7 shake"); the elbow/shoulder case runs J3 into overspeed (17:17);
#   * seeding from the measured joints copies their null-space coordinate into
#     the target, so the joint spring has no stiffness along the null space;
#   * a far target can land on another IK branch (0.8-1.2 rad jumps, 18:11).
# Instead, continue from the policy's own accepted joint goal and take one damped
# least-squares step per command (resolved-rate), with damping that rises as the
# weighted sigma_min falls; use the null space only to push away from joint
# limits. One step, not an iterated solve: repeating DLS on the same target
# converges back to the undamped solution (4 steps already pass ~60% of it), so
# the damping must act as a per-frame rate along the weak direction. Well
# conditioned directions converge within one 20 ms frame anyway.
DIK_ITERATIONS = 1
DIK_ROTATION_WEIGHT_M = 0.2          # 1 rad of orientation error weighs like 0.2 m
DIK_SIGMA_DAMPING = 0.08             # normal poses sit at 0.10-0.13; home 0.10
DIK_LAMBDA_MAX = 0.08
# near_singular is reported only when it costs the operator something: sigma
# is low *and* the damped step leaves the target visibly behind. sigma alone is
# not enough: on 10-03 one operator worked for minutes at sigma ~0.05 (J6 ~3.4).
DIK_SIGMA_NOTICE = 0.04
DIK_SIGMA_NOTICE_CLEAR = 0.05
DIK_NOTICE_LAG_M = 0.010
DIK_NOTICE_LAG_RAD = np.deg2rad(3.0)
DIK_NOTICE_CLEAR_LAG_M = 0.005
DIK_NOTICE_CLEAR_LAG_RAD = np.deg2rad(1.5)
DIK_STEP_MAX_RAD = 0.20              # per command, per joint
# Per-command task-space step (50 Hz: 1 m/s, 5 rad/s). A sudden target jump
# (stale clutch origin, tracking glitch: 9 cm / 50 deg on 10-03 18:52) is walked
# over several frames instead of becoming a spring offset worth > 60 N at once.
DIK_EE_STEP_M = 0.02
DIK_EE_STEP_RAD = 0.10
DIK_LIMIT_BUFFER_RAD = 0.30
DIK_LIMIT_PUSH_RAD = 0.01            # per command, null-space only
DIK_LIMIT_CLIP_RAD = 0.02
DIK_UNREACHABLE_M = 0.10
DIK_UNREACHABLE_RAD = 0.50
DIK_UNREACHABLE_S = 0.5
# A joint target farther than this from the measured joints means the arm is not
# following (blocked); hold instead of winding up the spring. A goal from the same
# live policy is never silently re-seeded from the measurement: that would turn a
# blocked arm into a repeating push-release sawtooth.
IK_JUMP_HOLD_RAD = 0.45


def dik_solve(fk, jacobian, seed, target_position, target_rotation,
              lower=EE_JOINT_LIMITS_LOWER, upper=EE_JOINT_LIMITS_UPPER):
    """Damped least-squares steps from ``seed`` toward a world-frame EE target.

    ``fk(q) -> (position[3], scipy Rotation)``, ``jacobian(q) -> 6x7`` in
    LOCAL_WORLD_ALIGNED (Polymetis' convention). Returns (q, info).
    """
    weight = np.array([1, 1, 1] + [DIK_ROTATION_WEIGHT_M] * 3, dtype=float)
    q = np.asarray(seed, dtype=float).copy()
    sigma_min, weak = np.inf, None
    for _ in range(DIK_ITERATIONS):
        position, rotation = fk(q)
        error = np.r_[target_position - position, (target_rotation * rotation.inv()).as_rotvec()]
        if np.linalg.norm(error[:3]) < 1e-4 and np.linalg.norm(error[3:]) < 1e-3:
            break
        error[:3] *= min(1.0, DIK_EE_STEP_M / max(np.linalg.norm(error[:3]), 1e-12))
        error[3:] *= min(1.0, DIK_EE_STEP_RAD / max(np.linalg.norm(error[3:]), 1e-12))
        J = weight[:, None] * jacobian(q)
        U, S, Vt = np.linalg.svd(J)
        sigma_min, weak = float(S[-1]), Vt[5]
        damping = 0.0
        if sigma_min < DIK_SIGMA_DAMPING:
            damping = DIK_LAMBDA_MAX ** 2 * (1.0 - (sigma_min / DIK_SIGMA_DAMPING) ** 2)
        gains = S / (S ** 2 + damping)
        step = Vt[:6].T @ (gains * (U.T @ (weight * error)))
        # Null space of the damped map; push only joints inside the buffer.
        projector = np.eye(7) - Vt[:6].T @ np.diag(S * gains) @ Vt[:6]
        push = np.clip((lower + DIK_LIMIT_BUFFER_RAD - q) / DIK_LIMIT_BUFFER_RAD, 0, 1) \
            - np.clip((q - (upper - DIK_LIMIT_BUFFER_RAD)) / DIK_LIMIT_BUFFER_RAD, 0, 1)
        q = q + step + projector @ (DIK_LIMIT_PUSH_RAD * push)
    q = np.clip(q, lower + DIK_LIMIT_CLIP_RAD, upper - DIK_LIMIT_CLIP_RAD)
    position, rotation = fk(q)
    residual_m = float(np.linalg.norm(target_position - position))
    residual_rad = float((target_rotation * rotation.inv()).magnitude())
    if weak is None:
        J = weight[:, None] * jacobian(q)
        _, S, Vt = np.linalg.svd(J)
        sigma_min, weak = float(S[-1]), Vt[5]
    wrist_share = float(weak[4] ** 2 + weak[6] ** 2)
    return q, {
        "sigma_min": sigma_min,
        "kind": "wrist" if wrist_share >= 0.5 else "arm",
        "residual_m": residual_m,
        "residual_rad": residual_rad,
    }


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
        if self.robot.metadata.ee_link_name != 'panda_link8':
            raise ValueError('Native flange observation requires the configured panda_link8 endpoint')
        self.gripper = GripperInterface(
            ip_address=os.environ.get('FRANKA_GRIPPER_HOST', robot_ip),
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
        from teleop.recording_telemetry import configured
        self._recording_telemetry = configured('ik_command/' + os.environ.get('MOZI_ARM', 'unknown'))
        self._last_gripper_command = 0.0
        self._last_gripper_command_raw = 0.0
        self._last_gripper_target_width = self._max_gripper_width
        self._last_gripper_command_timestamp = time.time()
        self._last_gripper_command_source = "init"
        self._gripper_command_snapshot = self._gripper_command_fields()
        self._ee_joint_soft_margin = float(
            os.environ.get(
                "FRANKA_EE_JOINT_SOFT_MARGIN_RAD",
                str(DEFAULT_EE_JOINT_SOFT_MARGIN_RAD),
            )
        )
        if not 0.0 <= self._ee_joint_soft_margin < 0.5:
            raise ValueError(
                "FRANKA_EE_JOINT_SOFT_MARGIN_RAD must be in [0, 0.5), got "
                f"{self._ee_joint_soft_margin}"
            )
        self._last_ee_hold_log_time = 0.0
        self._last_ee_hold_reason = ""
        self._ik_mode = os.environ.get("FRANKA_IK_MODE", "dik").strip().lower()
        if self._ik_mode not in ("dik", "polymetis"):
            raise ValueError(f"FRANKA_IK_MODE must be dik or polymetis, got {self._ik_mode}")
        self._reset_ik_guard()
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

        # Dedicated feedback workers: neither device RPC shares the command
        # server thread. Preserve native production timestamps in one-slot caches.
        from polymetis_pb2 import Empty
        from teleop.native_observation import LatestObservation, arm_observation, gripper_observation
        self._fast_observation_requested_until = 0.
        self._native_arm = LatestObservation(
            lambda: self.robot.grpc_connection.GetRobotState(Empty(), timeout=.1),
            arm_observation, hz=50, name="fr3-native-state")
        self._native_gripper = LatestObservation(
            lambda: self.gripper.grpc_connection.GetState(Empty(), timeout=.1),
            lambda state: gripper_observation(state, self._max_gripper_width),
            hz=30, name="fr3-encoder-state")
        self._native_anchor = LatestObservation(self._read_control_anchor, lambda value: value,
                                                hz=30, name="fr3-control-anchor")

    def _start_control_mode(self, control_mode: str) -> None:
        # A joint policy or a failed restart must not look like a live EE policy.
        self._cartesian_policy_started = False
        self._tracking_impedance_applied = None
        self._reset_ik_guard()
        if control_mode == "joint":
            self.robot.start_joint_impedance()
        elif control_mode == "ee":
            if getattr(self, '_tracking_impedance_override', None) is not None:
                from teleop.tracking_impedance import start
                start(self)
            else:
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
        """Return HOLD at the configured FR3 position boundary; allow retreat."""
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
        if reason == "joint_soft_limit":
            LOGGER.warning(
                "Cartesian safety HOLD: reason=%s joint_index=%s joint_position=%.6f limit_side=%s limit=%.6f margin_rad=%.3f",
                reason,
                result.get("joint_index"),
                float(result.get("joint_position", float("nan"))),
                result.get("limit_side"),
                float(result.get("limit", float("nan"))),
                float(result.get("margin_rad", float("nan"))),
            )
        elif result.get("status") == "notice":
            LOGGER.warning("Cartesian IK notice: %s", result)
        else:
            LOGGER.warning("Cartesian safety HOLD: %s", result)

    def _hold_current_ee_pose(self) -> None:
        """Replace a potentially unsafe target with the measured current pose."""
        if getattr(self, "_ik_mode", "polymetis") == "dik":
            # Latch the measured joints directly: an IK of the current pose can
            # itself jump near singular/J5~0 poses.
            self._reset_ik_guard()
            if getattr(self,'_tracking_impedance_applied',None) is not None:
                from teleop.tracking_impedance import update_target
                update_target(self,self.robot.get_joint_positions(),hold=True)
            else:
                self.robot.update_desired_joint_positions(self.robot.get_joint_positions())
            return
        position, orientation = self.robot.get_ee_pose()
        update_idx = self.robot.update_desired_ee_pose(
            position=position, orientation=orientation
        )
        if update_idx == -1:
            raise RuntimeError("Franka IK failed while latching the current EE pose")

    def _reset_ik_guard(self) -> None:
        self._dik_far_since = None
        self._dik_near_singular = False

    def _dik_seed(self, measured):
        """The live policy's accepted joint goal, or the measured joints."""
        goal = getattr(self.robot, "_teleop_accepted_joint_goal", None)
        episode = getattr(self.robot, "_teleop_goal_episode", None)
        tracking = getattr(self, '_tracking_impedance_applied', None)
        if goal is None or episode is None:
            if tracking is not None:
                raise RuntimeError('Tracking policy lost its accepted joint goal')
            return measured
        goal = np.asarray(goal, dtype=float).reshape(-1)
        if goal.shape != (7,) or not np.all(np.isfinite(goal)):
            return measured
        try:
            from polymetis_pb2 import Empty
            interval = self.robot.grpc_connection.GetEpisodeInterval(Empty(), timeout=0.25)
        except Exception:
            if tracking is not None:
                raise
            return measured
        if tracking is not None and (interval.start != tracking['policy_episode'] or interval.end != -1):
            raise RuntimeError('Tracking policy changed; stop this experiment instead of mixing gains')
        if interval.start != episode or interval.end != -1:
            return measured  # Recovery, homing or replay replaced the policy.
        return goal

    def _update_ee_target(self, position, orientation):
        """Return (update_idx, result) for one Cartesian target.

        update_idx == -1 keeps the original "IK failed" semantics. ``result`` is
        a HOLD (target not sent) or a near-singular notice (target sent).
        """
        if getattr(self, "_ik_mode", "polymetis") != "dik":
            submitted_ns = time.time_ns()
            update_idx = self.robot.update_desired_ee_pose(
                position=position, orientation=orientation
            )
            goal = getattr(self.robot, "_teleop_accepted_joint_goal", None)
            if update_idx != -1 and goal is not None:
                self._record_ik_target(goal, position, orientation, 'polymetis', submitted_ns)
            return update_idx, None
        from scipy.spatial.transform import Rotation

        model = self.robot.robot_model

        def fk(q):
            p, quat = model.forward_kinematics(torch.tensor(q, dtype=torch.float32))
            return (p.detach().cpu().numpy().astype(float),
                    Rotation.from_quat(quat.detach().cpu().numpy().astype(float)))

        def jacobian(q):
            return model.compute_jacobian(torch.tensor(q, dtype=torch.float32)).detach().cpu().numpy().astype(float)

        measured = self.robot.get_joint_positions().detach().cpu().numpy().astype(float).reshape(-1)
        if measured.shape != (7,) or not np.all(np.isfinite(measured)):
            raise RuntimeError(f"Invalid Franka joint feedback: {measured}")
        seed = self._dik_seed(measured)
        target_position = np.asarray(position, dtype=float).reshape(3)
        target_rotation = Rotation.from_quat(np.asarray(orientation, dtype=float).reshape(4))
        q, info = dik_solve(fk, jacobian, seed, target_position, target_rotation)
        q = seed + np.clip(q - seed, -DIK_STEP_MAX_RAD, DIK_STEP_MAX_RAD)
        if not np.all(np.isfinite(q)):
            return -1, None
        offset = q - measured
        worst = int(np.argmax(np.abs(offset)))
        if abs(offset[worst]) > IK_JUMP_HOLD_RAD:
            self._hold_current_ee_pose()
            return None, {
                "status": "hold",
                "reason": "ik_joint_jump",
                "joint_index": worst,
                "delta_rad": round(float(offset[worst]), 3),
                "limit_rad": IK_JUMP_HOLD_RAD,
            }
        now = time.monotonic()
        if info["residual_m"] > DIK_UNREACHABLE_M or info["residual_rad"] > DIK_UNREACHABLE_RAD:
            if self._dik_far_since is None:
                self._dik_far_since = now
            elif now - self._dik_far_since >= DIK_UNREACHABLE_S:
                self._hold_current_ee_pose()
                return -1, None
        else:
            self._dik_far_since = None
        joint_target = torch.tensor(q, dtype=torch.float32)
        submitted_ns = time.time_ns()
        if getattr(self,'_tracking_impedance_applied',None) is not None:
            from teleop.tracking_impedance import update_target
            update_idx = update_target(self,joint_target,position,orientation)
        else:
            update_idx = self.robot.update_desired_joint_positions(joint_target)
        if update_idx != -1:
            self._record_ik_target(joint_target, position, orientation, 'dik', submitted_ns)
        sigma = info["sigma_min"]
        if self._dik_near_singular:
            self._dik_near_singular = sigma < DIK_SIGMA_NOTICE_CLEAR and (
                info["residual_m"] > DIK_NOTICE_CLEAR_LAG_M
                or info["residual_rad"] > DIK_NOTICE_CLEAR_LAG_RAD)
        else:
            self._dik_near_singular = sigma < DIK_SIGMA_NOTICE and (
                info["residual_m"] > DIK_NOTICE_LAG_M
                or info["residual_rad"] > DIK_NOTICE_LAG_RAD)
        if not self._dik_near_singular:
            return update_idx, None
        return update_idx, {
            "status": "notice",
            "reason": "near_singular",
            "kind": info["kind"],
            "sigma_min": round(sigma, 4),
        }

    def _record_ik_target(self, joints, position, quaternion, mode, submitted_ns):
        publisher = getattr(self, '_recording_telemetry', None)
        if publisher is not None:
            from teleop.recording_telemetry import emit_ik
            emit_ik(publisher, joints, position, quaternion, mode, submitted_ns, time.time_ns(),
                    tracking_velocity=getattr(self,'_tracking_velocity_sample',None))

    def _restart_cartesian_at_measured_pose(self, target_pose=None):
        """Restart the policy at the measured pose; refuse a target far from it."""
        if getattr(self, '_tracking_impedance_override', None) is not None:
            raise RuntimeError('Tracking policy interrupted; restart as a new experiment')
        self.robot.start_cartesian_impedance()
        self._cartesian_policy_started = True
        self._reset_ik_guard()
        if target_pose is None:
            return None
        from scipy.spatial.transform import Rotation
        position, orientation = self.robot.get_ee_pose()
        offset_m = float(np.linalg.norm(np.asarray(target_pose[:3], dtype=float) - position.numpy()))
        offset_rad = float((Rotation.from_euler("xyz", target_pose[3:])
                            * Rotation.from_quat(orientation.numpy()).inv()).magnitude())
        if offset_m <= RECOVERY_MAX_TARGET_OFFSET_M and offset_rad <= RECOVERY_MAX_TARGET_OFFSET_RAD:
            return None
        result = {"status": "hold", "reason": "recovered_target_far",
                  "offset_m": round(offset_m, 4), "offset_rad": round(offset_rad, 3)}
        LOGGER.warning("Cartesian policy restarted at measured pose; stale target refused: %s", result)
        return result

    @timed("cartesian_guard")
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
        return self._restart_cartesian_at_measured_pose(target_pose)

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
        # Wait for the trajectory, but do not download and discard its entire
        # 1 kHz state log before restarting the holding controller.
        self.robot.move_to_joint_positions(desired_joints, blocking=True, return_log=False)

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

    def set_replay_mode(self, mode: str) -> None:
        """Start a measured-pose replay policy, including after lazy startup."""
        if mode not in ("ee", "joint"):
            raise ValueError("unsupported replay control mode")
        self._cartesian_policy_started = False
        running = bool(self.robot.is_running_policy())
        if running:
            try:
                self.robot.terminate_current_policy(return_log=False)
            except Exception as exc:
                # Only the policy-ending race is benign. Preserve all transport,
                # robot and controller failures instead of reporting success.
                details = exc.details() if callable(getattr(exc, "details", None)) else str(exc)
                if details != "Tried to terminate controller with no controller running.":
                    raise
        self._start_control_mode(mode)
        self.control_mode = mode

    def ensure_cartesian_teleop(self) -> None:
        """PICO recovery v2: restore measured-pose hold without terminating an absent policy."""
        try:
            running = bool(self.robot.is_running_policy())
        except Exception:
            self._cartesian_policy_started = False
            raise
        if self.control_mode == "ee" and self._cartesian_policy_started and running:
            return
        self._cartesian_policy_started = False
        if running:
            try:
                self.robot.terminate_current_policy(return_log=False)
            except Exception as exc:
                # The policy can end between the state query and termination.
                # Only this exact benign race is recoverable here; communication
                # errors and other termination failures remain visible.
                details = exc.details() if callable(getattr(exc, "details", None)) else str(exc)
                if details != "Tried to terminate controller with no controller running.":
                    raise
        self._start_control_mode("ee")
        self.control_mode = "ee"

    def command_replay_joints(self, joint_positions, gripper_width=None) -> None:
        if self.control_mode != "joint":
            raise RuntimeError("joint replay controller is not active")
        joints = np.asarray(joint_positions, dtype=float).reshape(-1)
        if joints.shape != (7,) or not np.all(np.isfinite(joints)):
            raise ValueError("joint replay requires 7 finite values")
        if np.any(joints < EE_JOINT_LIMITS_LOWER) or np.any(joints > EE_JOINT_LIMITS_UPPER):
            raise ValueError("joint replay target exceeds joint limits")
        if gripper_width is not None:
            width = float(gripper_width)
            if not np.isfinite(width) or not 0.0 <= width <= self._max_gripper_width:
                raise ValueError("invalid replay gripper width")
        # Do not silently restart/retry a failed replay trajectory.
        self.robot.update_desired_joint_positions(torch.tensor(joints, dtype=torch.float32))
        if gripper_width is not None:
            self._queue_gripper_command(width, GRIPPER_SPEED, GRIPPER_FORCE)
            self._remember_gripper_command(1.0 - width / self._max_gripper_width, width, "joint_replay")

    @timed("command_ee_pose")
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
            update_idx, guard_hold = self._update_ee_target(position, orientation)
        except Exception as exc:
            if "no controller running" not in str(exc).lower():
                raise
            # is_running_policy() can report a stale/optimistic state after
            # Polymetis has stopped the policy. Force a restart and retry the
            # exact same target once; the old code restarted/checks but then
            # re-raised without ever retrying the pose update. A target far
            # from the restart pose is the stale goal and is not retried.
            stale = self._restart_cartesian_at_measured_pose(pose)
            if stale is not None:
                return stale
            try:
                update_idx, guard_hold = self._update_ee_target(position, orientation)
            except Exception as retry_exc:
                if "no controller running" not in str(retry_exc).lower():
                    raise
                return {
                    "status": "hold",
                    "reason": "cartesian_controller_restarted",
                    "detail": str(retry_exc),
                }
        notice = None
        if guard_hold is not None and guard_hold.get("status") == "notice":
            notice, guard_hold = guard_hold, None
            self._log_ee_hold(notice)
        if guard_hold is not None:
            self._log_ee_hold(guard_hold)
            if update_gripper:
                width = float(np.clip(gripper_width, 0.0, self._max_gripper_width))
                closedness = 1.0 - width / self._max_gripper_width
                self._remember_gripper_command(closedness, width, "command_ee_pose_hold")
                self._queue_gripper_command(width, gripper_speed, gripper_force)
            return guard_hold
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
        return notice

    def command_gripper(self, gripper_width=None, closedness=None,
                        gripper_speed=GRIPPER_SPEED, gripper_force=40.0):
        """Queue an independent gripper target, without an arm pose/IK update."""
        if (gripper_width is None) == (closedness is None):
            raise ValueError("provide exactly one of gripper_width or closedness")
        if closedness is not None:
            value = float(closedness)
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("closedness must be finite and within [0,1]")
            width = (1.0 - value) * self._max_gripper_width
        else:
            width = float(gripper_width)
            if not np.isfinite(width) or not 0.0 <= width <= self._max_gripper_width:
                raise ValueError("gripper width is outside the hardware range")
            value = 1.0 - width / self._max_gripper_width
        speed, force = float(gripper_speed), float(gripper_force)
        if not np.isfinite(speed) or speed <= 0 or not np.isfinite(force) or not 0 < force <= 80:
            raise ValueError("invalid gripper speed or force")
        self._remember_gripper_command(value, width, "command_gripper")
        self._queue_gripper_command(width, speed, force)
        return {"accepted": True}

    @timed("gripper_enqueue")
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

    def _cartesian_hold_anchor(self):
        """Return a cached accepted goal only for the same confirmed live EE policy.

        This runs on the native feedback worker, never the command/VR receiver.
        No policy is started and no target is sent by an observation.
        """
        goal = getattr(self.robot, "_teleop_accepted_joint_goal", None)
        episode = getattr(self.robot, "_teleop_goal_episode", None)
        if self.control_mode != "ee" or goal is None or episode is None:
            return {}
        try:
            from polymetis_pb2 import Empty
            from scipy.spatial.transform import Rotation
            interval = self.robot.grpc_connection.GetEpisodeInterval(Empty(), timeout=0.25)
            if interval.start != episode or interval.end != -1:
                return {}  # Recovery, homing or replay invalidates the old goal.
            joints = np.asarray(goal, dtype=np.float32).reshape(-1)
            if joints.shape != (7,) or not np.all(np.isfinite(joints)):
                return {}
            position, quaternion = self.robot.robot_model.forward_kinematics(torch.tensor(joints))
            pose = np.concatenate([position.detach().cpu().numpy(),
                                   Rotation.from_quat(quaternion.detach().cpu().numpy()).as_euler("xyz")])
            if not np.all(np.isfinite(pose)):
                return {}
            return {"ee_hold_pose_euler": tuple(pose), "ee_hold_policy_episode": int(episode),
                    "ee_hold_joint_goal": tuple(joints)}
        except Exception:
            # An unverified cached goal is never reused. Preserve the existing
            # measured observation contract and let normal guards handle control.
            return {}

    def _read_control_anchor(self):
        # Conservative RPC-start bound; slow policy queries cannot make cached
        # anchors appear fresh or stall native arm/gripper feedback reception.
        result = dict(state_timestamp_ns=time.time_ns())
        if time.monotonic() < self._fast_observation_requested_until:
            # This FK applies only to an accepted COMMAND anchor, never feedback.
            result.update(self._cartesian_hold_anchor())
        return result

    def get_feedback_observations(self, recording: bool = False):
        """Read native caches without sharing a lock/RPC with motion execution."""
        result = self._native_arm.snapshot()
        try:
            measured = self._native_gripper.snapshot()
            measured.pop("state_timestamp_ns")  # Never replace the arm sample time.
            result.update(measured)
        except RuntimeError as exc:
            result.update(gripper_width=(float("nan"),), gripper_width_source="unavailable",
                          gripper_timestamp_ns=0, gripper_error_code=-1,
                          gripper_observation_error=str(exc), gripper_max_width=self._max_gripper_width)
        # One atomic publication keeps command metadata internally consistent
        # when the independent feedback server reads during a gripper command.
        result.update(self._gripper_command_snapshot)
        if not recording:
            width = float(result['gripper_width'][0])
            result['joint_positions'] = np.append(result['joint_positions'], width / self._max_gripper_width)
            result['gripper_position'] = np.array([width / self._max_gripper_width])
        return result

    @timed("observations")
    def get_observations(self, fast: bool = False, recording: bool = False) -> Dict[str, np.ndarray]:
        # Optional command anchors retain their legacy semantics on the command
        # endpoint. Independent feedback never requests or substitutes an anchor.
        result = self.get_feedback_observations(recording=fast or recording)
        if fast:
            self._fast_observation_requested_until = time.monotonic() + .5
            try:
                anchor = self._native_anchor.snapshot()
                anchor.pop('state_timestamp_ns')
                result.update(anchor)
            except RuntimeError:
                pass  # An unverified command anchor is never reused.
        goal = result.pop('ee_hold_joint_goal', None)
        if (not fast or self.control_mode != 'ee' or goal is None
                or result.get('ee_hold_policy_episode') != getattr(self.robot, '_teleop_goal_episode', None)
                or not np.array_equal(goal, getattr(self.robot, '_teleop_accepted_joint_goal', None))):
            result.pop('ee_hold_pose_euler', None)
            result.pop('ee_hold_policy_episode', None)
        return result

    def _gripper_command_fields(self):
        return dict(gripper_closedness=(self._last_gripper_command_raw,),
            gripper_01closedness=(self._last_gripper_command,),
            gripper_closed=(self._last_gripper_command,),
            gripper_command=(self._last_gripper_command,),
            gripper_command_raw=(self._last_gripper_command_raw,),
            gripper_target_width=(self._last_gripper_target_width,),
            gripper_command_timestamp=(self._last_gripper_command_timestamp,),
            gripper_command_source=self._last_gripper_command_source)

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
        self._gripper_command_snapshot = self._gripper_command_fields()


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
