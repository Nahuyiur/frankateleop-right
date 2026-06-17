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
        effective_home_on_init = args.home_on_init and args.move_to_initial_pose
        if args.robot == "xarm":
            from teleop.robots.xarm_robot import XArmRobot

            robot = XArmRobot(ip=args.robot_ip)
        elif args.robot == "ur":
            from teleop.robots.ur import URRobot
            
        elif args.robot == "fr3_left":
            from teleop.robots.fr3 import fr3Robot

            robot = fr3Robot(
                robot_ip=args.robot_ip, 
                franka_port=args.robot_port, 
                frankahand_port=args.gripper_port, 
                control_mode=args.control_mode,
                home_on_init=effective_home_on_init,
                open_gripper_on_init=args.open_gripper_on_init,
                )
        
        elif args.robot == "fr3_right":
            from teleop.robots.fr3 import fr3Robot

            robot = fr3Robot(
                robot_ip=args.robot_ip, 
                franka_port=args.robot_port, 
                frankahand_port=args.gripper_port, 
                control_mode=args.control_mode,
                home_on_init=effective_home_on_init,
                open_gripper_on_init=args.open_gripper_on_init,
                )
       
        elif args.robot == "fr3":
            from teleop.robots.fr3 import fr3Robot

            robot = fr3Robot(
                robot_ip=args.robot_ip, 
                franka_port=args.robot_port, 
                frankahand_port=args.gripper_port,
                control_mode=args.control_mode,
                home_on_init=effective_home_on_init,
                open_gripper_on_init=args.open_gripper_on_init,
                )
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
            f"move_to_initial_pose={args.move_to_initial_pose}"
        )
        server.serve()


def main(args):
    launch_robot_server(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
