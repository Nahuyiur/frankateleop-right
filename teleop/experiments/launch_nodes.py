import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import tyro

from teleop.robots.robot import BimanualRobot, PrintRobot
from teleop.zmq_core.robot_node import ZMQServerRobot


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_initial_joints_file() -> str:
    return str(repo_root() / "config" / "initial_joints.json")


@dataclass
class Args:
    robot: str = "xarm"
    tele_port: int = 6001
    hostname: str = "127.0.0.1"
    robot_ip: str = "192.168.1.10"
    robot_port: int = 50051
    gripper_port: int = 50052
    control_mode: str = "joint"
    home_on_init: bool = True
    open_gripper_on_init: bool = True
    move_to_initial_pose: bool = env_bool("FRANKA_MOVE_TO_INITIAL_POSE", True)
    initial_pose_source: str = os.environ.get("FRANKA_INITIAL_POSE_SOURCE", "auto")
    initial_joints_file: str = os.environ.get(
        "FRANKA_INITIAL_JOINTS_FILE",
        default_initial_joints_file(),
    )
    initial_joints: str = os.environ.get("FRANKA_INITIAL_JOINTS", "")


JOINT_LIMITS_LOWER = (-2.64, -1.68, -2.80, -2.94, -2.70, 0.64, -2.91)
JOINT_LIMITS_UPPER = (2.64, 1.68, 2.80, -0.25, 2.70, 4.41, 2.91)


def _robot_side(args: Args) -> str:
    if args.robot == "fr3_left":
        return "left"
    if args.robot == "fr3_right":
        return "right"
    explicit = os.environ.get("FRANKA_ROBOT_NAME", "").strip().lower()
    if explicit:
        if explicit not in {"left", "right", "single"}:
            raise ValueError(
                "FRANKA_ROBOT_NAME must be left, right, or single; "
                f"got {explicit!r}"
            )
        return explicit
    return "single"


def _parse_joints(value, *, label: str):
    if value is None:
        raise ValueError(f"{label} is empty")
    if isinstance(value, str):
        value = value.strip()
        if not value:
            raise ValueError(f"{label} is empty")
        if value.startswith("["):
            joints = json.loads(value)
        else:
            joints = [item for item in value.replace(",", " ").split() if item]
    else:
        joints = value
    if not isinstance(joints, (list, tuple)) or len(joints) != 7:
        raise ValueError(f"{label} must contain exactly 7 joint values")
    parsed = [float(item) for item in joints]
    for index, joint in enumerate(parsed):
        if not math.isfinite(joint):
            raise ValueError(f"{label}[{index}] is not finite: {joint}")
        lower = JOINT_LIMITS_LOWER[index]
        upper = JOINT_LIMITS_UPPER[index]
        if joint < lower or joint > upper:
            raise ValueError(
                f"{label}[{index}]={joint:.6f} outside configured FR3 limit "
                f"[{lower:.6f}, {upper:.6f}]"
            )
    return parsed


def _load_calibrated_joints(path: Path, side: str):
    if not path.exists():
        raise FileNotFoundError(
            f"Calibrated initial joints file does not exist: {path}"
        )
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("schema_version") != "initial_joints_v1":
        raise ValueError(
            f"Unsupported initial joints schema in {path}: "
            f"{payload.get('schema_version')!r}"
        )
    entry = payload.get(side)
    if entry is None and side == "single":
        entry = payload.get("left")
    if entry is None:
        raise KeyError(f"No calibrated initial joints for {side!r} in {path}")
    if isinstance(entry, dict):
        joints = entry.get("joints")
        name = entry.get("name", side)
    else:
        joints = entry
        name = side
    return _parse_joints(joints, label=f"{path}:{side}.joints"), name


def resolve_initial_joints(args: Args):
    if not args.move_to_initial_pose:
        return None

    side = _robot_side(args)
    side_env_names = {
        "left": "FRANKA_LEFT_INITIAL_JOINTS",
        "right": "FRANKA_RIGHT_INITIAL_JOINTS",
        "single": "FRANKA_SINGLE_INITIAL_JOINTS",
    }
    side_env = side_env_names[side]
    if os.environ.get(side_env, "").strip():
        joints = _parse_joints(os.environ[side_env], label=side_env)
        return joints, side_env

    if args.initial_joints and side == "single":
        joints = _parse_joints(args.initial_joints, label="FRANKA_INITIAL_JOINTS")
        return joints, "FRANKA_INITIAL_JOINTS"

    source = args.initial_pose_source.strip().lower()
    if source in {"", "home", "default", "go_home"}:
        return None
    if source == "auto":
        path = Path(args.initial_joints_file).expanduser()
        if not path.exists():
            return None
        try:
            joints, name = _load_calibrated_joints(path, side)
        except KeyError:
            return None
        return joints, f"{args.initial_joints_file}:{side}:{name}"
    if source != "calibrated":
        raise ValueError(
            "initial_pose_source must be auto, home, or calibrated; "
            f"got {args.initial_pose_source!r}"
        )
    joints, name = _load_calibrated_joints(
        Path(args.initial_joints_file).expanduser(),
        side,
    )
    return joints, f"{args.initial_joints_file}:{side}:{name}"


def _make_fr3_robot(args: Args, robot_cls):
    import torch

    effective_home_on_init = args.home_on_init and args.move_to_initial_pose
    resolved = resolve_initial_joints(args)
    if resolved is None:
        joint_positions_desired = None
        print(
            "Initial pose: "
            f"{'default go_home' if effective_home_on_init else 'disabled'}"
        )
    else:
        joints, source = resolved
        joint_positions_desired = torch.tensor(joints, dtype=torch.float32)
        print(
            "Initial pose: calibrated joints from "
            f"{source}: {joint_positions_desired.tolist()}"
        )
    return robot_cls(
        robot_ip=args.robot_ip,
        franka_port=args.robot_port,
        frankahand_port=args.gripper_port,
        joint_positions_desired=joint_positions_desired,
        control_mode=args.control_mode,
        home_on_init=effective_home_on_init,
        open_gripper_on_init=args.open_gripper_on_init,
    )


def launch_robot_server(args: Args):
    port = args.tele_port
    if args.robot == "sim_ur":
        MENAGERIE_ROOT: Path = (
            Path(__file__).parent.parent / "third_party" / "mujoco_menagerie"
        )
        xml = MENAGERIE_ROOT / "universal_robots_ur5e" / "ur5e.xml"
        gripper_xml = MENAGERIE_ROOT / "robotiq_2f85" / "2f85.xml"
        from teleop.robots.sim_robot import MujocoRobotServer

        server = MujocoRobotServer(
            xml_path=xml, gripper_xml_path=gripper_xml, port=port, host=args.hostname
        )
        server.serve()
    elif args.robot == "sim_fr3":
        from teleop.robots.sim_robot import MujocoRobotServer

        MENAGERIE_ROOT: Path = (
            Path(__file__).parent.parent / "third_party" / "mujoco_menagerie"
        )
        xml = MENAGERIE_ROOT / "franka_emika_fr3" / "fr3.xml"
        gripper_xml = None
        server = MujocoRobotServer(
            xml_path=xml, gripper_xml_path=gripper_xml, port=port, host=args.hostname
        )
        server.serve()
    elif args.robot == "sim_xarm":
        from teleop.robots.sim_robot import MujocoRobotServer

        MENAGERIE_ROOT: Path = (
            Path(__file__).parent.parent / "third_party" / "mujoco_menagerie"
        )
        xml = MENAGERIE_ROOT / "ufactory_xarm7" / "xarm7.xml"
        gripper_xml = None
        server = MujocoRobotServer(
            xml_path=xml, gripper_xml_path=gripper_xml, port=port, host=args.hostname
        )
        server.serve()

    else:
        if args.robot == "xarm":
            from teleop.robots.xarm_robot import XArmRobot

            robot = XArmRobot(ip=args.robot_ip)
        elif args.robot == "ur":
            from teleop.robots.ur import URRobot
            
        elif args.robot == "fr3_left":
            from teleop.robots.fr3 import fr3Robot

            robot = _make_fr3_robot(args, fr3Robot)
        
        elif args.robot == "fr3_right":
            from teleop.robots.fr3 import fr3Robot

            robot = _make_fr3_robot(args, fr3Robot)
       
        elif args.robot == "fr3":
            from teleop.robots.fr3 import fr3Robot

            robot = _make_fr3_robot(args, fr3Robot)
        elif args.robot == "bimanual_ur":
            from teleop.robots.ur import URRobot

            # IP for the bimanual robot setup is hardcoded
            _robot_l = URRobot(robot_ip="192.168.2.10")
            _robot_r = URRobot(robot_ip="192.168.1.10")
            robot = BimanualRobot(_robot_l, _robot_r)
        elif args.robot == "none" or args.robot == "print":
            robot = PrintRobot(8)

        else:
            raise NotImplementedError(
                f"Robot {args.robot} not implemented, choose one of: sim_ur, xarm, ur, bimanual_ur, none"
            )
        server = ZMQServerRobot(robot, port=port, host=args.hostname)
        print(
            f"Starting robot server on port {port}, control_mode={args.control_mode}, "
            f"home_on_init={args.home_on_init}, "
            f"open_gripper_on_init={args.open_gripper_on_init}, "
            f"move_to_initial_pose={args.move_to_initial_pose}, "
            f"initial_pose_source={args.initial_pose_source}"
        )
        server.serve()


def main(args):
    launch_robot_server(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
