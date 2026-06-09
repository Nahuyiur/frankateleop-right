import pickle
import threading
from typing import Any, Dict

import numpy as np
import zmq

from teleop.robots.robot import Robot

DEFAULT_ROBOT_PORT = 6000


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
        addr = f"tcp://{host}:{port}"
        debug_message = f"Robot Sever Binding to {addr}, Robot: {robot}"
        print(debug_message)
        self._timout_message = f"Timeout in Robot Server, Robot: {robot}"
        self._socket.bind(addr)
        self._stop_event = threading.Event()

    def serve(self) -> None:
        """Serve the leader robot state over ZMQ."""
        self._socket.setsockopt(zmq.RCVTIMEO, 1000)  # Set timeout to 1000 ms
        while not self._stop_event.is_set():
            try:
                # Wait for next request from client
                message = self._socket.recv()
                request = pickle.loads(message)

                try:
                    # Call the appropriate method based on the request
                    method = request.get("method")
                    args = request.get("args", {})
                    result: Any
                    if method == "num_dofs":
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
                            result = {
                                "error": f"Robot {self._robot} does not support command_ee_pose"
                            }
                        else:
                            result = self._robot.command_ee_pose(**args)
                    elif method == "get_observations":
                        result = self._robot.get_observations()
                    else:
                        result = {"error": f"Invalid method: {method}"}
                        print(result)
                except Exception as exc:
                    result = {"error": f"{type(exc).__name__}: {exc}"}
                    print(result)

                self._socket.send(pickle.dumps(result))
            except zmq.Again:
                print(self._timout_message)
                # Timeout occurred, check if the stop event is set

    def stop(self) -> None:
        """Signal the server to stop serving."""
        self._stop_event.set()


class ZMQClientRobot(Robot):
    """A class representing a ZMQ client for a leader robot."""

    def __init__(self, port: int = DEFAULT_ROBOT_PORT, host: str = "127.0.0.1"):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect(f"tcp://{host}:{port}")

    def _recv_result(self) -> Any:
        result = pickle.loads(self._socket.recv())
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(result["error"])
        return result

    def num_dofs(self) -> int:
        """Get the number of joints in the robot.

        Returns:
            int: The number of joints in the robot.
        """
        request = {"method": "num_dofs"}
        send_message = pickle.dumps(request)
        self._socket.send(send_message)
        result = self._recv_result()
        return result

    def get_joint_state(self) -> np.ndarray:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        request = {"method": "get_joint_state"}
        send_message = pickle.dumps(request)
        self._socket.send(send_message)
        result = self._recv_result()
        return result

    def get_control_mode(self) -> str:
        request = {"method": "get_control_mode"}
        send_message = pickle.dumps(request)
        self._socket.send(send_message)
        result = self._recv_result()
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
        result = self._recv_result()
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
        result = self._recv_result()
        return result

    def get_observations(self) -> Dict[str, np.ndarray]:
        """Get the current observations of the leader robot.

        Returns:
            Dict[str, np.ndarray]: The current observations of the leader robot.
        """
        request = {"method": "get_observations"}
        send_message = pickle.dumps(request)
        self._socket.send(send_message)
        result = self._recv_result()
        return result
