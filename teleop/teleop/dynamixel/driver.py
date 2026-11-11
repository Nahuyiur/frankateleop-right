import atexit
import os
import time
from threading import Event, Lock, Thread
from typing import Optional, Protocol, Sequence
import weakref

import numpy as np
from dynamixel_sdk.group_sync_read import GroupSyncRead
from dynamixel_sdk.group_sync_write import GroupSyncWrite
from dynamixel_sdk.packet_handler import PacketHandler
from dynamixel_sdk.port_handler import PortHandler
from dynamixel_sdk.robotis_def import (
    COMM_SUCCESS,
    DXL_HIBYTE,
    DXL_HIWORD,
    DXL_LOBYTE,
    DXL_LOWORD,
)

# Constants
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
LEN_GOAL_POSITION = 4
ADDR_PRESENT_POSITION = 132
LEN_PRESENT_POSITION = 4
ADDR_OPERATING_MODE = 11
LEN_OPERATING_MODE = 1
ADDR_GOAL_CURRENT = 102
LEN_GOAL_CURRENT = 2
ADDR_KP_P = 84
LEN_KP_P = 2
ADDR_KP_I = 82
LEN_KP_I = 2
ADDR_KP_D = 80
LEN_KP_D = 2
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
OPERATING_MODE_CURRENT_POSITION = 5
DEFAULT_GOAL_CURRENT_MA = 600


_driver_refs = []


def _cleanup_all_drivers():
    """Best-effort torque shutdown for live real drivers at process exit."""
    for driver_ref in _driver_refs:
        driver = driver_ref()
        if driver is not None and not getattr(driver, "_closed", True):
            try:
                driver.set_torque_mode(False)
            except Exception:
                pass


atexit.register(_cleanup_all_drivers)


class DynamixelDriverProtocol(Protocol):
    def set_joints(self, joint_angles: Sequence[float]):
        """Set the joint angles for the Dynamixel servos.

        Args:
            joint_angles (Sequence[float]): A list of joint angles.
        """
        ...

    def torque_enabled(self) -> bool:
        """Check if torque is enabled for the Dynamixel servos.

        Returns:
            bool: True if torque is enabled, False if it is disabled.
        """
        ...

    def set_torque_mode(self, enable: bool):
        """Set the torque mode for the Dynamixel servos.

        Args:
            enable (bool): True to enable torque, False to disable.
        """
        ...

    def get_joints(self) -> np.ndarray:
        """Get the current joint angles in radians.

        Returns:
            np.ndarray: An array of joint angles.
        """
        ...

    def configure_damping(
        self,
        torque_enable: Sequence[int],
        kp_p: Optional[Sequence[int]] = None,
        kp_i: Optional[Sequence[int]] = None,
        kp_d: Optional[Sequence[int]] = None,
        goal_position_rad: Optional[Sequence[float]] = None,
    ):
        """Configure current-based position control damping."""
        ...

    def close(self):
        """Close the driver."""


class FakeDynamixelDriver(DynamixelDriverProtocol):
    def __init__(self, ids: Sequence[int]):
        self._ids = ids
        self._joint_angles = np.zeros(len(ids), dtype=int)
        self._torque_enabled = False
        self._config = {
            "operating_mode": [0] * len(ids),
            "goal_current": [0] * len(ids),
            "kp_p": [0] * len(ids),
            "kp_i": [0] * len(ids),
            "kp_d": [0] * len(ids),
            "goal_position_rad": None,
            "torque_enable": [0] * len(ids),
        }

    def set_joints(self, joint_angles: Sequence[float]):
        if len(joint_angles) != len(self._ids):
            raise ValueError(
                "The length of joint_angles must match the number of servos"
            )
        if not self._torque_enabled:
            raise RuntimeError("Torque must be enabled to set joint angles")
        self._joint_angles = np.array(joint_angles)

    def torque_enabled(self) -> bool:
        return self._torque_enabled

    def set_torque_mode(self, enable: bool):
        self._torque_enabled = enable

    def get_joints(self) -> np.ndarray:
        return self._joint_angles.copy()

    def configure_damping(
        self,
        torque_enable: Sequence[int],
        kp_p: Optional[Sequence[int]] = None,
        kp_i: Optional[Sequence[int]] = None,
        kp_d: Optional[Sequence[int]] = None,
        goal_position_rad: Optional[Sequence[float]] = None,
    ):
        n = len(self._ids)
        values = {
            "torque_enable": torque_enable,
            "kp_p": kp_p,
            "kp_i": kp_i,
            "kp_d": kp_d,
            "goal_position_rad": goal_position_rad,
        }
        for name, value in values.items():
            if value is not None and len(value) != n:
                raise ValueError(f"{name} must contain one value per servo")
        if any(value not in (0, 1) for value in torque_enable):
            raise ValueError("torque_enable values must be 0 or 1")

        self._config["operating_mode"] = [OPERATING_MODE_CURRENT_POSITION] * n
        self._config["goal_current"] = [DEFAULT_GOAL_CURRENT_MA] * n
        self._config["torque_enable"] = list(torque_enable)
        for name, value in (("kp_p", kp_p), ("kp_i", kp_i), ("kp_d", kp_d)):
            if value is not None:
                self._config[name] = list(value)
        if goal_position_rad is not None:
            self._config["goal_position_rad"] = list(goal_position_rad)
            self._joint_angles = np.array(goal_position_rad, dtype=float)
        self._torque_enabled = any(value == 1 for value in torque_enable)

    def close(self):
        self.set_torque_mode(False)


class DynamixelDriver(DynamixelDriverProtocol):
    def __init__(
        self, ids: Sequence[int], port: str = "/dev/ttyUSB0", baudrate: int = 57600
    ):
        """Initialize the DynamixelDriver class.

        Args:
            ids (Sequence[int]): A list of IDs for the Dynamixel servos.
            port (str): The USB port to connect to the arm.
            baudrate (int): The baudrate for communication.
        """
        self._ids = ids
        self._joint_angles = None
        self._read_error = None
        self._lock = Lock()
        self._closed = False

        # Initialize the port handler, packet handler, and group sync read/write
        self._portHandler = PortHandler(port)
        self._packetHandler = PacketHandler(2.0)
        self._groupSyncRead = GroupSyncRead(
            self._portHandler,
            self._packetHandler,
            ADDR_PRESENT_POSITION,
            LEN_PRESENT_POSITION,
        )
        self._groupSyncWrite = GroupSyncWrite(
            self._portHandler,
            self._packetHandler,
            ADDR_GOAL_POSITION,
            LEN_GOAL_POSITION,
        )

        # Open the port and set the baudrate
        if not self._portHandler.openPort():
            raise RuntimeError("Failed to open the port")

        if not self._portHandler.setBaudRate(baudrate):
            raise RuntimeError(f"Failed to change the baudrate, {baudrate}")
        packet_timeout_ms = int(os.environ.get("DYNAMIXEL_PACKET_TIMEOUT_MS", "500"))
        self._portHandler.setPacketTimeoutMillis(packet_timeout_ms)

        # Add parameters for each Dynamixel servo to the group sync read
        for dxl_id in self._ids:
            if not self._groupSyncRead.addParam(dxl_id):
                raise RuntimeError(
                    f"Failed to add parameter for Dynamixel with ID {dxl_id}"
                )

        # Disable torque for each Dynamixel servo
        self._torque_enabled = False
        try:
            self.set_torque_mode(self._torque_enabled)
        except Exception as e:
            self._portHandler.closePort()
            raise RuntimeError(f"Failed to initialize Dynamixel bus on {port}: {e}") from e

        self._stop_thread = Event()
        self._start_reading_thread()
        _driver_refs.append(weakref.ref(self))

    @staticmethod
    def _validate_values(name: str, values: Sequence[int], count: int):
        if len(values) != count:
            raise ValueError(f"{name} must contain one value per servo")

    def _write_values_to_addr(
        self, addr: int, value_length: int, values: Sequence[int]
    ):
        """Sync-write one integer field for all configured motors."""
        self._validate_values("values", values, len(self._ids))
        group_sync_write = GroupSyncWrite(
            self._portHandler, self._packetHandler, addr, value_length
        )
        with self._lock:
            try:
                for dxl_id, value in zip(self._ids, values):
                    param = [
                        (int(value) >> (8 * byte_index)) & 0xFF
                        for byte_index in range(value_length)
                    ]
                    if not group_sync_write.addParam(dxl_id, param):
                        raise RuntimeError(
                            f"Failed to add Dynamixel ID {dxl_id} for address {addr}"
                        )
                dxl_comm_result = group_sync_write.txPacket()
                if dxl_comm_result != COMM_SUCCESS:
                    raise RuntimeError(
                        f"Failed to syncwrite address {addr}: {dxl_comm_result}"
                    )
            finally:
                group_sync_write.clearParam()

    def set_joints(self, joint_angles: Sequence[float]):
        if len(joint_angles) != len(self._ids):
            raise ValueError(
                "The length of joint_angles must match the number of servos"
            )
        if not self._torque_enabled:
            raise RuntimeError("Torque must be enabled to set joint angles")

        for dxl_id, angle in zip(self._ids, joint_angles):
            # Convert the angle to the appropriate value for the servo
            position_value = int(angle * 2048 / np.pi)

            # Allocate goal position value into byte array
            param_goal_position = [
                DXL_LOBYTE(DXL_LOWORD(position_value)),
                DXL_HIBYTE(DXL_LOWORD(position_value)),
                DXL_LOBYTE(DXL_HIWORD(position_value)),
                DXL_HIBYTE(DXL_HIWORD(position_value)),
            ]

            # Add goal position value to the Syncwrite parameter storage
            dxl_addparam_result = self._groupSyncWrite.addParam(
                dxl_id, param_goal_position
            )
            if not dxl_addparam_result:
                raise RuntimeError(
                    f"Failed to set joint angle for Dynamixel with ID {dxl_id}"
                )

        # Syncwrite goal position
        dxl_comm_result = self._groupSyncWrite.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            raise RuntimeError("Failed to syncwrite goal position")

        # Clear syncwrite parameter storage
        self._groupSyncWrite.clearParam()

    def torque_enabled(self) -> bool:
        return self._torque_enabled

    def set_torque_mode(self, enable: bool):
        torque_value = TORQUE_ENABLE if enable else TORQUE_DISABLE
        with self._lock:
            for dxl_id in self._ids:
                dxl_comm_result, dxl_error = self._packetHandler.write1ByteTxRx(
                    self._portHandler, dxl_id, ADDR_TORQUE_ENABLE, torque_value
                )
                if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
                    print(dxl_comm_result)
                    print(dxl_error)
                    raise RuntimeError(
                        f"Failed to set torque mode for Dynamixel with ID {dxl_id}"
                    )

        self._torque_enabled = enable

    def configure_damping(
        self,
        torque_enable: Sequence[int],
        kp_p: Optional[Sequence[int]] = None,
        kp_i: Optional[Sequence[int]] = None,
        kp_d: Optional[Sequence[int]] = None,
        goal_position_rad: Optional[Sequence[float]] = None,
    ):
        """Configure current-based position-control damping (operating mode 5)."""
        count = len(self._ids)
        values = {
            "torque_enable": torque_enable,
            "kp_p": kp_p,
            "kp_i": kp_i,
            "kp_d": kp_d,
            "goal_position_rad": goal_position_rad,
        }
        for name, value in values.items():
            if value is not None:
                self._validate_values(name, value, count)
        if any(value not in (0, 1) for value in torque_enable):
            raise ValueError("torque_enable values must be 0 or 1")

        # Operating mode changes require torque off. Keep the device torque-free
        # until all current and gain fields have been written successfully.
        self.set_torque_mode(False)
        try:
            self._write_values_to_addr(
                ADDR_OPERATING_MODE,
                LEN_OPERATING_MODE,
                [OPERATING_MODE_CURRENT_POSITION] * count,
            )
            time.sleep(0.01)
            self._write_values_to_addr(
                ADDR_GOAL_CURRENT,
                LEN_GOAL_CURRENT,
                [DEFAULT_GOAL_CURRENT_MA] * count,
            )
            time.sleep(0.01)

            for addr, length, value in (
                (ADDR_KP_P, LEN_KP_P, kp_p),
                (ADDR_KP_I, LEN_KP_I, kp_i),
                (ADDR_KP_D, LEN_KP_D, kp_d),
            ):
                if value is not None:
                    self._write_values_to_addr(addr, length, value)
            time.sleep(0.01)

            # Enabling torque resets the motor's goal position, so the damping
            # target must be sent only after the per-joint torque mask.
            with self._lock:
                for dxl_id, enabled in zip(self._ids, torque_enable):
                    torque_value = TORQUE_ENABLE if enabled == 1 else TORQUE_DISABLE
                    dxl_comm_result, dxl_error = self._packetHandler.write1ByteTxRx(
                        self._portHandler,
                        dxl_id,
                        ADDR_TORQUE_ENABLE,
                        torque_value,
                    )
                    if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
                        raise RuntimeError(
                            f"Failed to set damping torque for Dynamixel ID {dxl_id}"
                        )
            self._torque_enabled = any(value == 1 for value in torque_enable)
            time.sleep(0.01)

            if goal_position_rad is not None:
                goal_pulses = [
                    int(angle * 2048 / np.pi) for angle in goal_position_rad
                ]
                self._write_values_to_addr(
                    ADDR_GOAL_POSITION, LEN_GOAL_POSITION, goal_pulses
                )
        except Exception:
            # A partially configured bus must never be left energized.
            try:
                self.set_torque_mode(False)
            except Exception:
                pass
            raise

    def _start_reading_thread(self):
        self._reading_thread = Thread(target=self._read_joint_angles)
        self._reading_thread.daemon = True
        self._reading_thread.start()

    def _read_joint_angles(self):
        # Continuously read joint angles and update the joint_angles array
        while not self._stop_thread.is_set():
            time.sleep(0.001)
            try:
                with self._lock:
                    _joint_angles = np.zeros(len(self._ids), dtype=int)
                    dxl_comm_result = self._groupSyncRead.txRxPacket()
                    if dxl_comm_result != COMM_SUCCESS:
                        print(f"warning, comm failed: {dxl_comm_result}", flush=True)
                        continue
                    for i, dxl_id in enumerate(self._ids):
                        if self._groupSyncRead.isAvailable(
                            dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
                        ):
                            angle = self._groupSyncRead.getData(
                                dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
                            )
                            angle = np.int32(np.uint32(angle))
                            _joint_angles[i] = angle
                        else:
                            raise RuntimeError(
                                f"Failed to get joint angles for Dynamixel with ID {dxl_id}"
                            )
                    self._joint_angles = _joint_angles
            except Exception as exc:
                self._read_error = exc
                print(f"error, Dynamixel read thread stopped: {exc}", flush=True)
                self._stop_thread.set()
                return
            # self._groupSyncRead.clearParam() # TODO what does this do? should i add it

    def get_joints(self) -> np.ndarray:
        # Return a copy of the joint_angles array to avoid race conditions
        timeout_sec = float(os.environ.get("DYNAMIXEL_READ_TIMEOUT_SEC", "5"))
        started = time.time()
        while self._joint_angles is None:
            if self._read_error is not None:
                raise RuntimeError(f"Dynamixel read failed: {self._read_error}") from self._read_error
            if time.time() - started >= timeout_sec:
                raise RuntimeError(
                    f"Timed out waiting for first Dynamixel joint read after {timeout_sec:.1f}s"
                )
            time.sleep(0.05)
        # with self._lock:
        _j = self._joint_angles.copy()
        return _j / 2048.0 * np.pi

    def close(self):
        if self._closed:
            return
        torque_error = None
        try:
            self.set_torque_mode(False)
        except Exception as exc:
            torque_error = exc
        self._stop_thread.set()
        self._reading_thread.join(timeout=1.0)
        self._portHandler.closePort()
        self._closed = True
        if torque_error is not None:
            raise RuntimeError(
                "Failed to disable Dynamixel torque while closing"
            ) from torque_error


def main():
    # Set the port, baudrate, and servo IDs
    ids = [1]

    # Create a DynamixelDriver instance
    try:
        driver = DynamixelDriver(ids)
    except FileNotFoundError:
        driver = DynamixelDriver(ids, port="/dev/cu.usbserial-FT7WBMUB")

    # Test setting torque mode
    driver.set_torque_mode(True)
    driver.set_torque_mode(False)

    # Test reading the joint angles
    try:
        while True:
            joint_angles = driver.get_joints()
            print(f"Joint angles for IDs {ids}: {joint_angles}")
            # print(f"Joint angles for IDs {ids[1]}: {joint_angles[1]}")
    except KeyboardInterrupt:
        driver.close()


if __name__ == "__main__":
    main()  # Test the driver
