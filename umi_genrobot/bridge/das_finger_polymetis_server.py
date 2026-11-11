#!/usr/bin/env python3
"""Polymetis-compatible gRPC server for a DAS Finger Controller RV3.

The vendor ``DataBus`` combines encoder polling and position commands in one
``ReadBatch`` packet.  For that reason this bridge deliberately sends no serial
packets until Polymetis receives its first explicit ``Goto`` request.  Merely
starting stage 2 therefore opens the serial port but does not move the finger.
"""

import argparse
import logging
import math
import os
import signal
import struct
import sys
import threading
import time
from concurrent import futures
from pathlib import Path
from typing import Optional

import grpc
import polymetis_pb2
import polymetis_pb2_grpc


LOG = logging.getLogger("das_finger_polymetis")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expose a DAS Finger Controller through the Polymetis gripper API."
    )
    parser.add_argument(
        "--serial-port",
        default=os.environ.get("LEFT_FINGER_COMPORT", "/dev/ttyFingerLeft"),
    )
    parser.add_argument(
        "--sdk-root",
        default=os.environ.get(
            "DAS_FINGER_SDK_ROOT",
            str(
                Path(__file__).resolve().parents[1]
                / "vendor"
                / "gen_finger_con_python_sdk_release"
            ),
        ),
    )
    parser.add_argument("--server-ip", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=50052)
    parser.add_argument("--baudrate", type=int, default=921600)
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--max-width", type=float, default=0.2)
    parser.add_argument("--moving-tolerance", type=float, default=0.002)
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate imports and device permissions without opening the serial port.",
    )
    return parser.parse_args()


def _load_vendor_api(sdk_root: Path):
    if not sdk_root.is_dir():
        raise FileNotFoundError(f"DAS Finger SDK directory does not exist: {sdk_root}")
    sys.path.insert(0, str(sdk_root))
    from scripts.databus import DataBus  # pylint: disable=import-outside-toplevel
    from scripts.pack import (  # pylint: disable=import-outside-toplevel
        CmdPack,
        Opcode,
        RecordType,
    )

    return DataBus, CmdPack, Opcode, RecordType


class DASFingerGripperService(polymetis_pb2_grpc.GripperServerServicer):
    """Serve Polymetis gripper RPCs and drive the vendor serial data bus."""

    def __init__(
        self,
        databus,
        cmd_pack,
        opcode,
        record_type,
        *,
        max_width: float,
        hz: float,
        moving_tolerance: float,
    ) -> None:
        self._databus = databus
        self._cmd_pack = cmd_pack
        self._opcode = opcode
        self._record_type = record_type
        self._max_width = max_width
        self._hz = hz
        self._moving_tolerance = moving_tolerance
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._target_width: Optional[float] = None
        self._measured_width = 0.0
        self._last_feedback_monotonic: Optional[float] = None
        self._command_started_monotonic: Optional[float] = None
        self._last_command = polymetis_pb2.GripperCommand()
        self._metadata = polymetis_pb2.GripperMetadata(
            polymetis_version="das-finger-bridge-v1",
            hz=max(1, int(round(hz))),
            max_width=max_width,
        )

    def start(self) -> None:
        self._worker = threading.Thread(
            target=self._command_loop,
            name="das-finger-command-loop",
            daemon=True,
        )
        self._worker.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._worker is not None:
            self._worker.join(timeout=2.0)

    def encoder_callback(self, record_data: bytes) -> None:
        try:
            if len(record_data) != 4:
                raise ValueError(f"expected 4 encoder bytes, got {len(record_data)}")
            width = float(struct.unpack(">f", record_data)[0])
            if not math.isfinite(width):
                raise ValueError(f"non-finite encoder value: {width}")
            with self._lock:
                self._measured_width = min(self._max_width, max(0.0, width))
                self._last_feedback_monotonic = time.monotonic()
        except Exception as exc:  # keep the vendor parsing thread alive
            LOG.warning("Invalid DAS Finger encoder response: %s", exc)

    def InitRobotClient(self, request, context):  # noqa: N802
        # This bridge owns the hardware client and its metadata.  Accepting this
        # RPC keeps compatibility with Polymetis diagnostics without replacing
        # the verified DAS Finger limits.
        return polymetis_pb2.Empty()

    def GetRobotClientMetadata(self, request, context):  # noqa: N802
        result = polymetis_pb2.GripperMetadata()
        result.CopyFrom(self._metadata)
        return result

    def GetState(self, request, context):  # noqa: N802
        return self._make_state()

    def Goto(self, request, context):  # noqa: N802
        width = float(request.width)
        if not math.isfinite(width):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "width must be finite")
        if width < -1e-6 or width > self._max_width + 1e-6:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"width {width} is outside [0.0, {self._max_width}] m",
            )

        width = min(self._max_width, max(0.0, width))
        with self._lock:
            first_command = self._target_width is None
            self._target_width = width
            self._command_started_monotonic = time.monotonic()
            self._last_command.CopyFrom(request)
        if first_command:
            LOG.info(
                "First explicit Goto received; enabling DAS Finger command/feedback loop."
            )
        LOG.info(
            "DAS Finger target width=%.4f m (speed/force are not supported by the vendor protocol)",
            width,
        )
        return polymetis_pb2.Empty()

    def ControlUpdate(self, request, context):  # noqa: N802
        result = polymetis_pb2.GripperCommand()
        with self._lock:
            result.CopyFrom(self._last_command)
        return result

    def _make_state(self):
        now = time.monotonic()
        with self._lock:
            measured = self._measured_width
            target = self._target_width
            feedback_at = self._last_feedback_monotonic
            command_at = self._command_started_monotonic

        state = polymetis_pb2.GripperState()
        state.timestamp.GetCurrentTime()
        state.width = measured
        state.is_grasped = False
        state.is_moving = target is not None and (
            feedback_at is None or abs(measured - target) > self._moving_tolerance
        )
        state.prev_command_successful = (
            target is not None
            and feedback_at is not None
            and command_at is not None
            and feedback_at >= command_at
        )
        if target is not None and (feedback_at is None or now - feedback_at > 2.0):
            state.error_code = 1
        else:
            state.error_code = 0
        return state

    def _command_loop(self) -> None:
        period = 1.0 / self._hz
        next_tick = time.monotonic()
        while not self._stop_event.is_set():
            with self._lock:
                target = self._target_width

            # The no-target branch is the startup safety guard: no vendor packet
            # is queued until an external caller explicitly invokes Goto.
            if target is not None:
                command = self._cmd_pack.pack(
                    opcode=self._opcode.ReadBatch,
                    record_type=self._record_type.Encoder,
                    record=struct.pack(">f", target),
                )
                if not self._databus.add_cmd(command):
                    LOG.warning("DAS Finger serial command queue is full")

            next_tick += period
            delay = next_tick - time.monotonic()
            if delay <= 0:
                next_tick = time.monotonic()
                delay = 0.0
            self._stop_event.wait(delay)


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )

    if args.hz <= 0:
        raise ValueError("--hz must be positive")
    if args.max_width <= 0 or args.max_width > 0.2:
        raise ValueError("--max-width must be in (0.0, 0.2]")

    serial_path = Path(args.serial_port)
    sdk_root = Path(args.sdk_root).resolve()
    DataBus, CmdPack, Opcode, RecordType = _load_vendor_api(sdk_root)

    if not serial_path.exists():
        raise FileNotFoundError(f"DAS Finger serial port does not exist: {serial_path}")
    if not os.access(str(serial_path), os.R_OK | os.W_OK):
        raise PermissionError(
            f"DAS Finger serial port is not readable/writable: {serial_path}; "
            "the user must belong to dialout"
        )
    if args.validate_config:
        print(
            "DAS Finger configuration valid: "
            f"serial={serial_path}, baud={args.baudrate}, sdk={sdk_root}, "
            f"server={args.server_ip}:{args.server_port}, max_width={args.max_width}"
        )
        print("No serial port was opened and no motion command was sent.")
        return 0

    databus = None
    service = None
    server = None
    stop_event = threading.Event()

    def request_stop(signum, _frame):
        LOG.info("Signal %s received; stopping DAS Finger bridge", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        databus = DataBus(
            tty_port=str(serial_path),
            baudrate=args.baudrate,
            encoder_freq=None,
            encoder_callback=None,
        )
        service = DASFingerGripperService(
            databus,
            CmdPack,
            Opcode,
            RecordType,
            max_width=args.max_width,
            hz=args.hz,
            moving_tolerance=args.moving_tolerance,
        )
        databus.register_encoder_callback(service.encoder_callback)

        server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        polymetis_pb2_grpc.add_GripperServerServicer_to_server(service, server)
        address = f"{args.server_ip}:{args.server_port}"
        if server.add_insecure_port(address) == 0:
            raise RuntimeError(f"Could not bind Polymetis gripper server to {address}")
        server.start()
        service.start()
        LOG.info(
            "DAS Finger bridge ready: serial=%s baud=%d grpc=%s max_width=%.3f m",
            serial_path,
            args.baudrate,
            address,
            args.max_width,
        )
        LOG.info("Startup guard active: waiting for the first explicit Goto command.")

        while not stop_event.wait(0.5):
            pass
        return 0
    finally:
        if service is not None:
            service.stop()
        if server is not None:
            server.stop(grace=1.0)
        if databus is not None:
            databus.stop()


if __name__ == "__main__":
    raise SystemExit(main())
