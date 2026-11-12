import pickle
import os
import logging
import threading
import time
from typing import Any, Dict

import numpy as np
import zmq

from teleop.robots.robot import Robot
from polymetis._mozi_timing import begin_request, end_request

DEFAULT_ROBOT_PORT = 6000
LOGGER = logging.getLogger("franka_zmq_node")


def with_replay_feedback(robot, result):
    """Report the accepted policy target, never substitute measured joints for IK."""
    if result is not None and (not isinstance(result, dict) or result.get('status') == 'hold' or 'error' in result):
        return result
    goal = getattr(getattr(robot, 'robot', None), '_teleop_accepted_joint_goal', None)
    try:
        q = np.asarray(goal, dtype=float)
        available = q.shape == (7,) and np.isfinite(q).all()
    except (TypeError, ValueError):
        available = False
    return dict(result or {}, replay_feedback=dict(ik_joints=q.tolist() if available else None,
                source='accepted_policy_joint_target' if available else 'unavailable'))


class ZMQServerRobot:
    def __init__(
        self,
        robot: Robot,
        port: int = DEFAULT_ROBOT_PORT,
        host: str = "127.0.0.1",
    ):
        self._robot = robot
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        self._socket.setsockopt(zmq.LINGER, 0)
        if hasattr(zmq, "TCP_NODELAY"):
            self._socket.setsockopt(zmq.TCP_NODELAY, 1)
        addr = f"tcp://{host}:{port}"
        debug_message = f"Robot Sever Binding to {addr}, Robot: {robot}"
        print(debug_message)
        self._timout_message = f"Timeout in Robot Server, Robot: {robot}"
        self._last_idle_log = 0.0
        self._metrics_lock = threading.Lock()
        self._request_count = 0
        self._command_count = 0
        self._command_timestamps: list[float] = []
        self._last_command_latency_ms = 0.0
        self._socket.bind(addr)
        self._stop_event = threading.Event()
        self._replay_token = None
        self._replay_original_mode = None
        self._feedback = None
        if os.environ.get('MOZI_RPC_FEEDBACK', '0') == '1':
            from teleop.zmq_core.feedback_server import FeedbackServer
            if not callable(getattr(robot, 'get_feedback_observations', None)):
                raise TypeError('robot lacks memory-only feedback snapshots')
            self._feedback = FeedbackServer(robot, int(port) + 100)
            self._feedback.start()

    def serve(self) -> None:
        """Serve the leader robot state over ZMQ."""
        self._socket.setsockopt(zmq.RCVTIMEO, 1000)  # Set timeout to 1000 ms
        while not self._stop_event.is_set():
            try:
                # Wait for next request from client
                message = self._socket.recv()
                method = "<decode>"
                timing_token = None
                try:
                    # A malformed or incompatible pickle is a failed request,
                    # not a reason to terminate the robot command server.
                    request = pickle.loads(message)
                    if not isinstance(request, dict):
                        raise ValueError("robot request must be a dictionary")
                    trace_id = request.get("mozi_trace_id")
                    if isinstance(trace_id, str) and len(trace_id) <= 100:
                        timing_token = begin_request(trace_id)
                    method = request.get("method")
                    args = request.get("args", {})
                    if not isinstance(args, dict):
                        raise ValueError("robot request args must be a dictionary")
                    result: Any
                    request_started = time.monotonic()
                    token = args.pop("replay_token", None)
                    motion_methods = {"command_joint_state", "command_ee_pose", "move_to_joint_positions",
                                      "command_replay_joints", "set_replay_mode", "release_replay",
                                      "ensure_cartesian_teleop", "command_gripper", "configure_tracking_impedance",
                                      "configure_tracking_static", "freeze_tracking_target", "get_tracking_static_sample",
                                      "configure_tracking_native", "get_tracking_native", "start_tracking_native", "finish_tracking_native", "configure_tracking_record"}
                    if method in motion_methods:
                        if self._replay_token is not None and token != self._replay_token:
                            raise RuntimeError("robot is reserved for episode replay")
                        if token is not None and token != self._replay_token:
                            raise RuntimeError("replay reservation is no longer valid")
                        if method in {"set_replay_mode", "command_replay_joints", "release_replay", "configure_tracking_impedance", "configure_tracking_static", "freeze_tracking_target", "get_tracking_static_sample", "configure_tracking_native", "get_tracking_native", "start_tracking_native", "finish_tracking_native", "configure_tracking_record"} and not self._replay_token:
                            raise RuntimeError("replay reservation required")
                    if method == "get_replay_capabilities":
                        result = {"version": 1, "recording_observations": True, "reserved": self._replay_token is not None, "replay_feedback": True, "tracking_impedance": 3, "static_sampling": 1, "native_tracking": 1}
                    elif method == "reserve_replay":
                        if not isinstance(token, str) or not token:
                            raise ValueError("replay token required")
                        if self._replay_token not in (None, token):
                            raise RuntimeError("robot is already reserved for replay")
                        if self._replay_token is None:
                            self._replay_original_mode = self._robot.get_control_mode()
                            self._replay_token = token
                        result = {"ok": True}
                    elif method == "get_tracking_impedance":
                        if args:
                            raise ValueError("get_tracking_impedance accepts no arguments")
                        from teleop.tracking_impedance import snapshot
                        result = snapshot(self._robot)
                    elif method == "configure_tracking_impedance":
                        if set(args) != {"gains","reference","velocity_mode"}:
                            raise ValueError("configure_tracking_impedance requires gains, fixed trajectory reference and velocity_mode")
                        from teleop.tracking_impedance import configure
                        result = configure(self._robot, args["gains"], args["reference"], args["velocity_mode"])
                    elif method == "configure_tracking_static":
                        if set(args) != {"joint_target", "gripper_width"}:
                            raise ValueError("Static configuration requires joint_target and gripper_width")
                        from teleop.tracking_static import configure
                        result = configure(self._robot, **args)
                    elif method in {"freeze_tracking_target", "get_tracking_static_sample"}:
                        if args:
                            raise ValueError("Static snapshot accepts no arguments")
                        from teleop.tracking_static import freeze, sample
                        result = (freeze if method == "freeze_tracking_target" else sample)(self._robot)
                    elif method == "configure_tracking_record":
                        value=args.get("record_id")
                        if set(args)!={"record_id"} or type(value) is not int or not 0<value<2**48:
                            raise ValueError("Invalid tracking recording ID")
                        if getattr(self._robot,"_tracking_impedance_applied",None) is not None:
                            raise RuntimeError("Configure recording before policy preparation")
                        self._robot._tracking_record_id=value
                        result={"record_id":value}
                    elif method == "configure_tracking_native":
                        if set(args)!={"artifact","sha256","path","record_id","stiffness"}:
                            raise ValueError("Native configuration requires the verified preload and explicit mode")
                        from teleop.tracking_native import configure
                        result=configure(self._robot,**args)
                    elif method == "get_tracking_native":
                        if set(args)-{"heartbeat"} or type(args.get("heartbeat",True)) is not bool:
                            raise ValueError("Native status accepts only a heartbeat flag")
                        from teleop.tracking_native import status
                        result=status(self._robot,**args)
                    elif method == "start_tracking_native":
                        if set(args)!={"start_ns"}:raise ValueError("Native start requires a shared start time")
                        from teleop.tracking_native import start
                        result=start(self._robot,**args)
                    elif method == "finish_tracking_native":
                        if args:raise ValueError("Native finish accepts no arguments")
                        from teleop.tracking_native import finish
                        result=finish(self._robot)
                    elif method == "set_replay_mode":
                        result = self._robot.set_replay_mode(**args)
                    elif method == "command_replay_joints":
                        result = self._robot.command_replay_joints(**args)
                    elif method == "release_replay":
                        # Restore a controller anchored at the current measured pose.
                        # Keep the reservation if restoring fails: fail closed.
                        from teleop.tracking_impedance import clear
                        if getattr(self._robot,'_tracking_native',None):
                            from teleop.tracking_native import finish
                            finish(self._robot)
                        clear(self._robot)
                        self._robot.set_replay_mode(self._replay_original_mode)
                        self._replay_token = None
                        self._replay_original_mode = None
                        result = {"ok": True}
                    elif method == "num_dofs":
                        result = self._robot.num_dofs()
                    elif method == "get_control_mode":
                        if hasattr(self._robot, "get_control_mode"):
                            result = self._robot.get_control_mode()
                        else:
                            result = getattr(self._robot, "control_mode", None)
                    elif method == "get_joint_state":
                        result = self._robot.get_joint_state()
                    elif method == "command_joint_state":
                        result = self._robot.command_joint_state(**args)
                    elif method == "command_ee_pose":
                        if not hasattr(self._robot, "command_ee_pose"):
                            result = {"error": f"Robot {self._robot} does not support command_ee_pose"}
                        else:
                            result = self._robot.command_ee_pose(**args)
                        if self._replay_token is not None and token == self._replay_token:
                            result = with_replay_feedback(self._robot, result)
                        with self._metrics_lock:
                            now = time.monotonic()
                            self._command_count += 1
                            self._command_timestamps.append(now)
                            cutoff = now - 5.0
                            self._command_timestamps = [t for t in self._command_timestamps if t >= cutoff]
                            self._last_command_latency_ms = (now - request_started) * 1000.0
                    elif method == "get_metrics":
                        with self._metrics_lock:
                            now = time.monotonic()
                            cutoff = now - 5.0
                            self._command_timestamps = [t for t in self._command_timestamps if t >= cutoff]
                            window_s = max(1e-3, min(5.0, now - self._command_timestamps[0])) if self._command_timestamps else 0.0
                            result = {
                                "pid": __import__("os").getpid(),
                                "nice": __import__("os").getpriority(__import__("os").PRIO_PROCESS, 0),
                                "requests_total": self._request_count,
                                "command_ee_pose_total": self._command_count,
                                "command_ee_pose_hz": len(self._command_timestamps) / window_s if window_s else 0.0,
                                "last_command_age_ms": (now - self._command_timestamps[-1]) * 1000.0 if self._command_timestamps else None,
                                "last_command_latency_ms": self._last_command_latency_ms,
                            }
                    elif method == "command_gripper":
                        if not hasattr(self._robot, "command_gripper"):
                            raise RuntimeError("robot does not support command_gripper")
                        result = self._robot.command_gripper(**args)
                    elif method == "ensure_cartesian_teleop":
                        if not hasattr(self._robot, "ensure_cartesian_teleop"):
                            raise RuntimeError("robot does not support ensure_cartesian_teleop")
                        result = self._robot.ensure_cartesian_teleop(**args)
                    elif method == "move_to_joint_positions":
                        if not hasattr(self._robot, "move_to_joint_positions"):
                            result = {"error": f"Robot {self._robot} does not support move_to_joint_positions"}
                        else:
                            result = self._robot.move_to_joint_positions(**args)
                    elif method == "get_observations":
                        try:
                            result = self._robot.get_observations(**args)
                        except TypeError as exc:
                            # Older GELLO/robot adapters do not yet accept the
                            # fast observation flag; fall back to their full
                            # observation contract without dropping the ZMQ
                            # request.
                            if "fast" not in str(exc) or not args.get("fast"):
                                raise
                            result = self._robot.get_observations()
                    else:
                        result = {"error": f"Invalid method: {method}"}
                        LOGGER.error("invalid request result=%s", result)
                except Exception as exc:
                    result = {"error": f"{type(exc).__name__}: {exc}"}
                    LOGGER.warning("robot request failed method=%s error=%s", method, exc)

                if timing_token is not None:
                    result = {"__mozi_trace_v1__": end_request(timing_token), "result": result}
                self._socket.send(pickle.dumps(result))
                with self._metrics_lock:
                    self._request_count += 1
            except zmq.Again:
                now = time.monotonic()
                if now - self._last_idle_log >= 5.0:
                    LOGGER.debug("%s (idle; no client request)", self._timout_message)
                    self._last_idle_log = now

    def stop(self) -> None:
        """Signal the server to stop serving."""
        self._stop_event.set()
        if self._feedback is not None:
            self._feedback.stop()


class ZMQClientRobot(Robot):
    """A class representing a ZMQ client for a leader robot."""

    def __init__(self, port: int = DEFAULT_ROBOT_PORT, host: str = "127.0.0.1"):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect(f"tcp://{host}:{port}")

    def num_dofs(self) -> int:
        """Get the number of joints in the robot.

        Returns:
            int: The number of joints in the robot.
        """
        request = {"method": "num_dofs"}
        send_message = pickle.dumps(request)
        self._socket.send(send_message)
        result = pickle.loads(self._socket.recv())
        return result

    def get_control_mode(self) -> str:
        request = {"method": "get_control_mode"}
        send_message = pickle.dumps(request)
        self._socket.send(send_message)
        result = pickle.loads(self._socket.recv())
        return result

    def get_joint_state(self) -> np.ndarray:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        request = {"method": "get_joint_state"}
        send_message = pickle.dumps(request)
        self._socket.send(send_message)
        result = pickle.loads(self._socket.recv())
        return result

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        """Command the leader robot to the given state.

        Args:
            joint_state (T): The state to command the leader robot to.
        """
        request = {
            "method": "command_joint_state",
            "args": {"joint_state": joint_state},
        }
        send_message = pickle.dumps(request)
        self._socket.send(send_message)
        result = pickle.loads(self._socket.recv())
        return result

    def command_ee_pose(
        self,
        pose_6d: np.ndarray,
        gripper_width: float,
        gripper_speed: float = 0.05,
        gripper_force: float = 40.0,
        update_gripper: bool = True,
    ) -> None:
        """Command an absolute end-effector pose plus gripper width."""
        request = {
            "method": "command_ee_pose",
            "args": {
                "pose_6d": pose_6d,
                "gripper_width": gripper_width,
                "gripper_speed": gripper_speed,
                "gripper_force": gripper_force,
                "update_gripper": update_gripper,
            },
        }
        send_message = pickle.dumps(request)
        self._socket.send(send_message)
        result = pickle.loads(self._socket.recv())
        return result

    def get_observations(self) -> Dict[str, np.ndarray]:
        """Get the current observations of the leader robot.

        Returns:
            Dict[str, np.ndarray]: The current observations of the leader robot.
        """
        request = {"method": "get_observations"}
        send_message = pickle.dumps(request)
        self._socket.send(send_message)
        result = pickle.loads(self._socket.recv())
        return result
