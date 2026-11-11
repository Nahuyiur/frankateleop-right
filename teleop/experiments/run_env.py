import datetime
import glob
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tyro

from teleop.agents.agent import BimanualAgent, DummyAgent
from teleop.agents.teleop_agent import TeleopAgent
from teleop.data_utils.format_obs import save_frame
from teleop.env import RobotEnv
from teleop.robots.robot import PrintRobot
from teleop.zmq_core.robot_node import ZMQClientRobot


# Damping configuration for GELLO (Current-based Position Control Mode).
# User-selected GELLO arm torque mask and PID values.
DYNAMIXEL_DAMPING_DEFAULTS = {
    # Previous torque mask: [0, 0, 0, 0, 0, 0, 0].
    # Seven arm joints; DynamixelRobot pads the trigger joint with torque off.
    "torque_enable": [1, 1, 1, 1, 1, 1, 1],
    # Previous PID values, retained for reference:
    # "kp_p": [30, 60, 0, 30, 0, 0, 0],
    # "kp_i": [0, 0, 0, 0, 0, 0, 0],
    # "kp_d": [250, 100, 80, 60, 30, 10, 5],
    "kp_p": [12, 20, 0, 12, 0, 0, 0],
    "kp_i": [0, 0, 0, 0, 0, 0, 0],
    "kp_d": [120, 60, 40, 30, 15, 5, 2],
    "goal_position": [0.0, 0.0, 0.0, -1.571, 0.0, 1.571, 0.0],
}

DYNAMIXEL_GOAL_POSITION_MAP = {
    "fr3_left": [
        0.19421,
        0.259328,
        0.192638,
        -1.80504,
        1.2055,
        2.84043,
        -0.736754,
    ],
    "fr3_right": [
        -0.234954,
        0.278439,
        -0.195666,
        -1.76859,
        -0.929469,
        2.8677,
        -1.36821,
    ],
}


def print_color(*args, color=None, attrs=(), **kwargs):
    import termcolor

    if len(args) > 0:
        args = tuple(termcolor.colored(arg, color=color, attrs=attrs) for arg in args)
    print(*args, **kwargs)


def arm_joint_indices(num_dofs, bimanual=False):
    if num_dofs < 2:
        raise ValueError(f"Expected arm joints plus a gripper, got {num_dofs} DOFs")
    if not bimanual:
        return np.arange(num_dofs - 1)
    if num_dofs % 2 != 0:
        raise ValueError(f"Bimanual state must have even size, got {num_dofs}")
    arm_size = num_dofs // 2
    if arm_size < 2:
        raise ValueError(f"Invalid bimanual state size: {num_dofs}")
    return np.concatenate(
        [np.arange(arm_size - 1), np.arange(arm_size, num_dofs - 1)]
    )


def joint_vectors(action, reference):
    action = np.asarray(action, dtype=float).copy()
    reference = np.asarray(reference, dtype=float)
    if action.ndim != 1 or reference.ndim != 1:
        raise ValueError(
            f"Joint states must be 1-D, got {action.shape} and {reference.shape}"
        )
    if action.shape != reference.shape:
        raise ValueError(
            f"Agent output shape {action.shape} does not match robot state "
            f"{reference.shape}"
        )
    return action, reference


def wrap_arm_action_to_nearest(action, reference, bimanual=False):
    action, reference = joint_vectors(action, reference)
    arm_indices = arm_joint_indices(action.size, bimanual=bimanual)
    action[arm_indices] = reference[arm_indices] + (
        (action[arm_indices] - reference[arm_indices] + np.pi) % (2 * np.pi)
        - np.pi
    )
    return action


def limit_arm_step(command, current, max_delta, bimanual=False):
    command, current = joint_vectors(command, current)
    arm_indices = arm_joint_indices(command.size, bimanual=bimanual)
    delta = command[arm_indices] - current[arm_indices]
    largest_delta = float(np.max(np.abs(delta)))
    if largest_delta > max_delta:
        delta = delta / largest_delta * max_delta
    limited = command.copy()
    limited[arm_indices] = current[arm_indices] + delta
    return limited


def _read_start_alignment(env, agent, bimanual=False):
    """Read leader/follower alignment without sending a follower command."""
    obs = env.get_obs()
    joints = np.asarray(obs["joint_positions"], dtype=float)
    start_pos = wrap_arm_action_to_nearest(
        agent.act(obs), joints, bimanual=bimanual
    )
    arm_indices = arm_joint_indices(joints.size, bimanual=bimanual)
    abs_deltas = np.abs(start_pos[arm_indices] - joints[arm_indices])
    return obs, joints, start_pos, arm_indices, abs_deltas


def _print_unsafe_alignment(start_pos, joints, arm_indices, abs_deltas, limit):
    unsafe_mask = abs_deltas > limit
    unsafe_indices = arm_indices[unsafe_mask]
    for i, delta, joint, current_j in zip(
        unsafe_indices,
        abs_deltas[unsafe_mask],
        start_pos[unsafe_indices],
        joints[unsafe_indices],
    ):
        print(
            f"joint[{i}]: \t delta: {delta:4.3f} , leader: \t{joint:4.3f} , "
            f"follower: \t{current_j:4.3f}",
            flush=True,
        )


def wait_for_safe_start_alignment(
    env,
    agent,
    *,
    bimanual=False,
    max_joint_delta=0.8,
    timeout_sec=0.0,
    poll_interval_sec=0.2,
):
    """Wait for a safe leader pose while keeping the follower stationary."""
    started = time.monotonic()
    deadline = started + max(0.0, timeout_sec)
    next_report = started

    while True:
        snapshot = _read_start_alignment(env, agent, bimanual=bimanual)
        obs, joints, start_pos, arm_indices, abs_deltas = snapshot
        largest_delta = float(np.max(abs_deltas))
        if largest_delta <= max_joint_delta:
            elapsed = time.monotonic() - started
            if elapsed > poll_interval_sec:
                print(
                    "Teleop alignment passed: "
                    f"max arm delta={largest_delta:.3f} rad after {elapsed:.1f}s.",
                    flush=True,
                )
            return obs, joints, start_pos

        now = time.monotonic()
        if now >= next_report:
            if next_report == started:
                print()
                print(
                    "同构主臂与机器人当前姿态尚未安全对齐；"
                    f"最多等待 {timeout_sec:.0f}s。请手动摆动主臂，使下列关节差小于 "
                    f"{max_joint_delta:.3f} rad。",
                    flush=True,
                )
                print(
                    "安全等待期间只读取主/从状态，不会向机器人发送运动命令。",
                    flush=True,
                )
            else:
                remaining = max(0.0, deadline - now)
                print(
                    f"等待主臂对齐：max delta={largest_delta:.3f} rad, "
                    f"remaining={remaining:.1f}s",
                    flush=True,
                )
            _print_unsafe_alignment(
                start_pos,
                joints,
                arm_indices,
                abs_deltas,
                max_joint_delta,
            )
            next_report = now + 1.0

        if now >= deadline:
            print(
                "主臂对齐超时，遥操作未启动。请将同构主臂摆到接近机器人当前姿态后重试。",
                flush=True,
            )
            raise SystemExit(1)

        time.sleep(min(poll_interval_sec, max(0.0, deadline - now)))


def env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


@dataclass
class Args:
    agent: str = "none"
    control_port: int = 6001
    wrist_camera_port: int = 5000
    base_camera_port: int = 5001
    hostname: str = "127.0.0.1"
    robot_type: Optional[str] = None  # only needed for quest or spacemouse agents
    hz: int = 100
    start_joints: Optional[Tuple[float, ...]] = None

    teleop_port: Optional[str] = None
    mock: bool = False
    use_save_interface: bool = False
    data_dir: str = "~/bc_data"
    bimanual: bool = False
    verbose: bool = False
    check_only: bool = False

    torque_enable: bool = True
    """Enable GELLO leader damping in Current-based Position Control Mode."""
    dynamixel_goal_position: Optional[Tuple[float, ...]] = None
    """Explicit damping goal in radians; otherwise selected by robot_type."""
    alignment_timeout_sec: float = float(
        os.environ.get("TELEOP_ALIGNMENT_TIMEOUT_SEC", "30")
    )
    """Seconds to wait for safe leader/follower alignment before exiting."""


def main(args):
    if args.mock:
        robot_client = PrintRobot(8, dont_print=True)
        camera_clients = {}
    else:
        camera_clients = {
            # you can optionally add camera nodes here for imitation learning purposes
            # "wrist": ZMQClientCamera(port=args.wrist_camera_port, host=args.hostname),
            # "base": ZMQClientCamera(port=args.base_camera_port, host=args.hostname),
        }
        robot_client = ZMQClientRobot(port=args.control_port, host=args.hostname)
    env = RobotEnv(robot_client, control_rate_hz=args.hz, camera_dict=camera_clients)

    if args.bimanual:
        if args.agent == "teleop":
            # dynamixel control box port map (to distinguish left and right teleop)
            right = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7WBG6A-if00-port0"
            left = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7WBEIA-if00-port0"
            left_agent = TeleopAgent(port=left)
            right_agent = TeleopAgent(port=right)
            agent = BimanualAgent(left_agent, right_agent)
        elif args.agent == "quest":
            from teleop.agents.quest_agent import SingleArmQuestAgent

            left_agent = SingleArmQuestAgent(robot_type=args.robot_type, which_hand="l")
            right_agent = SingleArmQuestAgent(
                robot_type=args.robot_type, which_hand="r"
            )
            agent = BimanualAgent(left_agent, right_agent)
            # raise NotImplementedError
        elif args.agent == "spacemouse":
            from teleop.agents.spacemouse_agent import SpacemouseAgent

            left_path = "/dev/hidraw0"
            right_path = "/dev/hidraw1"
            left_agent = SpacemouseAgent(
                robot_type=args.robot_type, device_path=left_path, verbose=args.verbose
            )
            right_agent = SpacemouseAgent(
                robot_type=args.robot_type,
                device_path=right_path,
                verbose=args.verbose,
                invert_button=True,
            )
            agent = BimanualAgent(left_agent, right_agent)
        else:
            raise ValueError(f"Invalid agent name for bimanual: {args.agent}")

        # System setup specific. This reset configuration works well on our setup. If you are mounting the robot
        # differently, you need a separate reset joint configuration.
        reset_joints_left = np.deg2rad([0, -90, -90, -90, 90, 0, 0])
        reset_joints_right = np.deg2rad([0, -90, 90, -90, -90, 0, 0])
        reset_joints = np.concatenate([reset_joints_left, reset_joints_right])
        curr_joints = env.get_obs()["joint_positions"]
        max_delta = (np.abs(curr_joints - reset_joints)).max()
        steps = min(int(max_delta / 0.01), 100)

        for jnt in np.linspace(curr_joints, reset_joints, steps):
            env.step(jnt)
    else:
        if args.agent == "teleop":
            teleop_port = args.teleop_port
            if teleop_port is None:
                usb_ports = glob.glob("/dev/serial/by-id/*")
                print(f"Found {len(usb_ports)} ports")
                if len(usb_ports) > 0:
                    teleop_port = usb_ports[0]
                    print(f"using port {teleop_port}")
                else:
                    raise ValueError(
                        "No teleop port found, please specify one or plug in teleop"
                    )
            if args.start_joints is None:
                reset_joints = np.deg2rad(
                    [0, -90, 90, -90, -90, 0, 0]
                )  # Change this to your own reset joints
            else:
                reset_joints = args.start_joints
                reset_joints = np.array(reset_joints)
            if args.torque_enable and not args.check_only:
                from teleop.agents.teleop_agent import (
                    DynamixelRobotConfig,
                    PORT_CONFIG_MAP,
                )

                assert teleop_port in PORT_CONFIG_MAP, (
                    f"Port {teleop_port} not found in PORT_CONFIG_MAP"
                )
                base_cfg = PORT_CONFIG_MAP[teleop_port]
                if args.dynamixel_goal_position is not None:
                    goal_pos = list(args.dynamixel_goal_position)
                elif (
                    args.robot_type is not None
                    and args.robot_type in DYNAMIXEL_GOAL_POSITION_MAP
                ):
                    goal_pos = DYNAMIXEL_GOAL_POSITION_MAP[args.robot_type]
                else:
                    goal_pos = DYNAMIXEL_DAMPING_DEFAULTS["goal_position"]

                dyn_config = DynamixelRobotConfig(
                    joint_ids=base_cfg.joint_ids,
                    joint_offsets=base_cfg.joint_offsets,
                    joint_signs=base_cfg.joint_signs,
                    gripper_config=base_cfg.gripper_config,
                    dynamixel_torque_enable=DYNAMIXEL_DAMPING_DEFAULTS[
                        "torque_enable"
                    ],
                    dynamixel_kp_p=DYNAMIXEL_DAMPING_DEFAULTS["kp_p"],
                    dynamixel_kp_i=DYNAMIXEL_DAMPING_DEFAULTS["kp_i"],
                    dynamixel_kp_d=DYNAMIXEL_DAMPING_DEFAULTS["kp_d"],
                    dynamixel_goal_position=goal_pos,
                )
                print(f"  Torque enable ON, goal_position={goal_pos}")
                agent = TeleopAgent(
                    port=teleop_port,
                    dynamixel_config=dyn_config,
                    start_joints=args.start_joints,
                )
            else:
                agent = TeleopAgent(
                    port=teleop_port, start_joints=args.start_joints
                )
            curr_joints = env.get_obs()["joint_positions"]
            curr_joints = np.array(curr_joints)
            if args.check_only:
                print("Skipping reset_joints in check-only mode")
            elif reset_joints.shape == curr_joints.shape:
                max_delta = (np.abs(curr_joints - reset_joints)).max()
                steps = min(int(max_delta / 0.01), 100)

                for jnt in np.linspace(curr_joints, reset_joints, steps):
                    env.step(jnt)
                    time.sleep(0.001)
            elif reset_joints.size + 1 == curr_joints.size:
                print(
                    "Keeping robot-node initial pose: legacy reset target contains "
                    f"{reset_joints.size} arm joints while robot state also includes "
                    "a gripper value"
                )
            else:
                print(
                    "Skipping reset_joints because shape does not match robot state: "
                    f"reset_joints={reset_joints.shape}, robot_joints={curr_joints.shape}"
                )
        elif args.agent == "quest":
            from teleop.agents.quest_agent import SingleArmQuestAgent

            agent = SingleArmQuestAgent(robot_type=args.robot_type, which_hand="l")
        elif args.agent == "spacemouse":
            from teleop.agents.spacemouse_agent import SpacemouseAgent

            agent = SpacemouseAgent(robot_type=args.robot_type, verbose=args.verbose)
        elif args.agent == "dummy" or args.agent == "none":
            agent = DummyAgent(num_dofs=robot_client.num_dofs())
        elif args.agent == "policy":
            raise NotImplementedError("add your imitation policy here if there is one")
        else:
            raise ValueError("Invalid agent name")

    # going to start position
    print("Going to start position")
    max_joint_delta = 0.8
    alignment_timeout_sec = (
        args.alignment_timeout_sec
        if args.agent == "teleop" and not args.check_only
        else 0.0
    )
    obs, joints, start_pos = wait_for_safe_start_alignment(
        env,
        agent,
        bimanual=args.bimanual,
        max_joint_delta=max_joint_delta,
        timeout_sec=alignment_timeout_sec,
    )

    print(f"Start pos: {len(start_pos)}", f"Joints: {len(joints)}")
    assert len(start_pos) == len(
        joints
    ), f"agent output dim = {len(start_pos)}, but env dim = {len(joints)}"

    if args.check_only:
        print("Teleop alignment check passed.")
        return

    max_delta = 0.05
    for _ in range(25):
        obs = env.get_obs()
        current_joints = obs["joint_positions"]
        command_joints = wrap_arm_action_to_nearest(
            agent.act(obs), current_joints, bimanual=args.bimanual
        )
        env.step(
            limit_arm_step(
                command_joints,
                current_joints,
                max_delta,
                bimanual=args.bimanual,
            )
        )

    obs = env.get_obs()
    joints = obs["joint_positions"]
    action = wrap_arm_action_to_nearest(
        agent.act(obs), joints, bimanual=args.bimanual
    )
    arm_indices = arm_joint_indices(len(joints), bimanual=args.bimanual)
    arm_delta = action[arm_indices] - joints[arm_indices]
    if (np.abs(arm_delta) > 0.5).any():
        print("Action is too big")

        # print which joints are too big
        joint_indices = arm_indices[np.abs(arm_delta) > 0.5]
        for j in joint_indices:
            print(
                f"Joint [{j}], leader: {action[j]}, follower: {joints[j]}, diff: {action[j] - joints[j]}"
            )
        exit()

    if args.use_save_interface:
        from teleop.data_utils.keyboard_interface import KBReset

        kb_interface = KBReset()

    print_color("\nStart 🚀🚀🚀", color="green", attrs=("bold",))

    save_path = None
    start_time = time.time()
    debug_action = env_flag("TELEOP_DEBUG_ACTION")
    debug_interval = float(os.environ.get("TELEOP_DEBUG_INTERVAL_SEC", "1.0"))
    debug_next_time = 0.0
    debug_prev_action = None
    if debug_action:
        print(
            "TELEOP_DEBUG_ACTION enabled "
            f"(interval={debug_interval:.2f}s)"
        )
    while True:
        num = time.time() - start_time
        message = f"\rTime passed: {round(num, 2)}          "
        print_color(
            message,
            color="white",
            attrs=("bold",),
            end="",
            flush=True,
        )
        action = wrap_arm_action_to_nearest(
            agent.act(obs), obs["joint_positions"], bimanual=args.bimanual
        )
        now = time.time()
        if debug_action and now >= debug_next_time:
            current = np.asarray(obs["joint_positions"], dtype=float)
            action_array = np.asarray(action, dtype=float)
            delta = action_array - current
            if debug_prev_action is None:
                leader_step = 0.0
            else:
                leader_step = float(np.max(np.abs(action_array - debug_prev_action)))
            debug_prev_action = action_array.copy()
            print()
            print(
                "TELEOP_DEBUG_ACTION "
                f"max_cmd_delta={np.max(np.abs(delta)):.4f} "
                f"max_leader_step={leader_step:.4f} "
                "current[:7]="
                f"{np.array2string(current[:7], precision=3, suppress_small=True)} "
                "action[:7]="
                f"{np.array2string(action_array[:7], precision=3, suppress_small=True)}",
                flush=True,
            )
            debug_next_time = now + debug_interval
        dt = datetime.datetime.now()
        if args.use_save_interface:
            state = kb_interface.update()
            if state == "start":
                dt_time = datetime.datetime.now()
                save_path = (
                    Path(args.data_dir).expanduser()
                    / args.agent
                    / dt_time.strftime("%m%d_%H%M%S")
                )
                save_path.mkdir(parents=True, exist_ok=True)
                print(f"Saving to {save_path}")
            elif state == "save":
                assert save_path is not None, "something went wrong"
                save_frame(save_path, dt, obs, action)
            elif state == "normal":
                save_path = None
            else:
                raise ValueError(f"Invalid state {state}")
        obs = env.step(action)


if __name__ == "__main__":
    main(tyro.cli(Args))
