#!/usr/bin/env python3
"""Strict non-motion MCUID/VERSION query with timeout-aware exit status."""

import argparse
from pathlib import Path
import sys
import time


root = Path(__file__).resolve().parents[2]
sdk = root / "vendor" / "gen_finger_con_python_sdk_release"
sys.path.insert(0, str(sdk))

from scripts.databus import DataBus


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial-port", required=True)
    parser.add_argument("--command", required=True, choices=("MCUID",))
    args = parser.parse_args()

    bus = None
    try:
        bus = DataBus(
            tty_port=args.serial_port,
            baudrate=921600,
            is_calib_cmd=True,
        )
        time.sleep(1.0)
        if not bus.send_camera_calib_cmd(args.command):
            print("IDENTITY_QUERY_SEND_FAILED", file=sys.stderr)
            return 5
        if not bus.wait_for_calib_response(timeout_sec=5.0):
            print(f"IDENTITY_QUERY_NO_RESPONSE command={args.command}", file=sys.stderr)
            return 4
        print(f"IDENTITY_QUERY_RESPONSE_OK command={args.command}")
        return 0
    finally:
        if bus is not None:
            bus.stop()


if __name__ == "__main__":
    raise SystemExit(main())
