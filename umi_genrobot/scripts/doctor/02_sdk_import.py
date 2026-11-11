#!/usr/bin/env python3
"""Import-only SDK check. This file must not instantiate hardware classes."""

from pathlib import Path
import importlib
import sys


root = Path(__file__).resolve().parents[2]
sdk = root / "vendor" / "gen_finger_con_python_sdk_release"
sys.path.insert(0, str(sdk))

import cv2
import numpy
import serial

modules = [
    "scripts.camera",
    "scripts.camera_cmd",
    "scripts.das_protocol",
    "scripts.databus",
    "scripts.pack",
    "scripts.system",
]

for name in modules:
    module = importlib.import_module(name)
    print(f"IMPORT_OK {name} {Path(module.__file__).resolve()}")

print(f"numpy={numpy.__version__}")
print(f"opencv={cv2.__version__}")
print(f"pyserial={serial.VERSION}")
print("SDK_IMPORT_OK")
