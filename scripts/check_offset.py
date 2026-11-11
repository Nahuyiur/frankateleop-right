"""Read leader joints and optionally compare them with a follower state.

This tool only reads state. It never sends a joint, gripper, or torque command.

Examples:
    python scripts/check_offset.py --list-ports
    python scripts/check_offset.py --teleop_port=/dev/serial/by-id/usb-...
    python scripts/check_offset.py --teleop_port=/dev/serial/by-id/usb-... \
        --control_port=6001
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from dynamixel_sdk.group_sync_read import GroupSyncRead
from dynamixel_sdk.packet_handler import PacketHandler
from dynamixel_sdk.port_handler import PortHandler
from dynamixel_sdk.robotis_def import COMM_SUCCESS


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teleop.agents.teleop_agent import PORT_CONFIG_MAP
from teleop.zmq_core.robot_node import ZMQClientRobot


ADDR_PRESENT_POSITION = 132
LEN_PRESENT_POSITION = 4
DXL_PROTOCOL_VERSION = 2.0
DXL_BAUDRATE = 57600


def _read_leader_joints(config, port: str) -> np.ndarray:
    """Read present positions without writing any Dynamixel register."""
    joint_ids = tuple(config.joint_ids) + (config.gripper_config[0],)
    offsets = np.asarray(tuple(config.joint_offsets) + (0.0,))
    signs = np.asarray(tuple(config.joint_signs) + (1,))

    port_handler = PortHandler(port)
    packet_handler = PacketHandler(DXL_PROTOCOL_VERSION)
    group_read = GroupSyncRead(
        port_handler,
        packet_handler,
        ADDR_PRESENT_POSITION,
        LEN_PRESENT_POSITION,
    )
    if not port_handler.openPort():
        raise RuntimeError(f"Failed to open leader port: {port}")
    try:
        if not port_handler.setBaudRate(DXL_BAUDRATE):
            raise RuntimeError(f"Failed to set leader baudrate: {DXL_BAUDRATE}")
        for joint_id in joint_ids:
            if not group_read.addParam(joint_id):
                raise RuntimeError(
                    f"Failed to add Dynamixel ID {joint_id} to sync-read"
                )

        comm_result = group_read.txRxPacket()
        if comm_result != COMM_SUCCESS:
            raise RuntimeError(packet_handler.getTxRxResult(comm_result))

        raw_positions = []
        for joint_id in joint_ids:
            if not group_read.isAvailable(
                joint_id,
                ADDR_PRESENT_POSITION,
                LEN_PRESENT_POSITION,
            ):
                raise RuntimeError(
                    f"Present position unavailable for Dynamixel ID {joint_id}"
                )
            raw = group_read.getData(
                joint_id,
                ADDR_PRESENT_POSITION,
                LEN_PRESENT_POSITION,
            )
            if raw >= 2**31:
                raw -= 2**32
            raw_positions.append(raw * np.pi / 2048.0)
    finally:
        group_read.clearParam()
        port_handler.closePort()

    positions = (np.asarray(raw_positions) - offsets) * signs
    gripper_open = config.gripper_config[1] * np.pi / 180.0
    gripper_closed = config.gripper_config[2] * np.pi / 180.0
    positions[-1] = np.clip(
        (positions[-1] - gripper_open) / (gripper_closed - gripper_open),
        0.0,
        1.0,
    )
    return positions


def main():
    parser = argparse.ArgumentParser(
        description="Read leader joint angles, optionally compare with follower."
    )
    parser.add_argument(
        "--teleop_port",
        type=str,
        help="Leader serial port, e.g. /dev/serial/by-id/usb-...",
    )
    parser.add_argument(
        "--control_port",
        type=int,
        default=None,
        help="Follower ZMQ port (e.g. 6001). If set, also reads follower joints.",
    )
    parser.add_argument(
        "--hostname",
        type=str,
        default="127.0.0.1",
        help="Follower ZMQ hostname (default 127.0.0.1).",
    )
    parser.add_argument(
        "--list-ports",
        action="store_true",
        help="List configured leader ports without opening a serial device.",
    )
    args = parser.parse_args()

    if args.list_ports:
        for port in PORT_CONFIG_MAP:
            print(port)
        return

    if not args.teleop_port:
        parser.error("--teleop_port is required unless --list-ports is used")

    port = args.teleop_port
    if port not in PORT_CONFIG_MAP:
        print(f"Error: port '{port}' not found in PORT_CONFIG_MAP", file=sys.stderr)
        print("Available ports:", file=sys.stderr)
        for configured_port in PORT_CONFIG_MAP:
            print(f"  {configured_port}", file=sys.stderr)
        sys.exit(1)

    config = PORT_CONFIG_MAP[port]

    print("Connecting to leader (read-only)...")
    leader_joints = _read_leader_joints(config, port)
    print(f"  Leader DOF: {len(leader_joints)}")

    follower_joints = None
    if args.control_port is not None:
        print(f"Connecting to follower (ZMQ port {args.control_port}, read-only)...")
        try:
            follower_client = ZMQClientRobot(
                port=args.control_port,
                host=args.hostname,
            )
            obs = follower_client.get_observations()
            follower_joints = np.asarray(obs["joint_positions"])
            print(f"  Follower DOF: {len(follower_joints)}")
        except Exception as exc:
            print(f"  Warning: failed to read follower: {exc}")
            print("  Skipping follower data.")

    print()
    print(f"{'j':>5} | {'leader':>10} | {'follower':>10} | {'delta':>10}")
    print("-" * 46)

    for index, leader_value in enumerate(leader_joints):
        if follower_joints is not None and index < len(follower_joints):
            follower_value = follower_joints[index]
            delta = leader_value - follower_value
            print(
                f"  [{index:>2}] | {leader_value:>10.4f} | "
                f"{follower_value:>10.4f} | {delta:>10.4f}"
            )
        else:
            print(
                f"  [{index:>2}] | {leader_value:>10.4f} | "
                f"{'-':>10} | {'-':>10}"
            )

    if follower_joints is not None and len(follower_joints) > len(leader_joints):
        for index in range(len(leader_joints), len(follower_joints)):
            follower_value = follower_joints[index]
            print(
                f"  [{index:>2}] | {'-':>10} | "
                f"{follower_value:>10.4f} | {'-':>10}"
            )

    print(
        f"\nTotal: leader={len(leader_joints)} "
        f"follower={len(follower_joints) if follower_joints is not None else 0}"
    )
    print("Done.\n")


if __name__ == "__main__":
    main()
