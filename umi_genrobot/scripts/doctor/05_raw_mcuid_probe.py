#!/usr/bin/env python3
"""Capture raw serial bytes around the official non-motion MCUID query."""

import argparse
import binascii
import time

import serial


def collect(port, duration):
    data = bytearray()
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        waiting = port.in_waiting
        chunk = port.read(waiting or 1)
        if chunk:
            data.extend(chunk)
    return bytes(data)


def describe(label, data):
    print(f"{label}_COUNT={len(data)}")
    print(f"{label}_HEX={binascii.hexlify(data).decode('ascii')}")
    print(f"{label}_REPR={data!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial-port", required=True)
    args = parser.parse_args()

    with serial.Serial(
        port=args.serial_port,
        baudrate=921600,
        timeout=0.05,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        dsrdtr=False,
        rtscts=False,
    ) as port:
        port.dtr = True
        port.rts = False
        baseline = collect(port, 1.0)
        describe("BASELINE", baseline)

        query = b"MCUID"
        print(f"TX_HEX={query.hex()}")
        port.write(query)
        port.flush()
        response = collect(port, 5.0)
        describe("RESPONSE", response)

    if response:
        print("RAW_MCUID_RESPONSE_PRESENT")
        return 0
    print("RAW_MCUID_NO_RESPONSE")
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
