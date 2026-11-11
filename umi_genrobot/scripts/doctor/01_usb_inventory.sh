#!/usr/bin/env bash
set -euo pipefail

echo "== Candidate USB devices =="
lsusb | grep -E "1a86:7523|1bcf:2d4f|343c:0000|3274:8012" || true

echo
echo "== Serial devices =="
for dev in /dev/ttyUSB*; do
  [[ -e "$dev" ]] || continue
  echo "[$dev]"
  udevadm info -q property -n "$dev" \
    | grep -E "^(ID_VENDOR_ID|ID_MODEL_ID|ID_SERIAL|ID_SERIAL_SHORT|ID_PATH)=" \
    || true
  if command -v fuser >/dev/null; then
    fuser "$dev" 2>/dev/null || true
  fi
done

echo
echo "== Stable serial links =="
find -L /dev/serial/by-id -maxdepth 1 -mindepth 1 -print 2>/dev/null | sort || true

echo
echo "== V4L2 devices =="
v4l2-ctl --list-devices

echo "USB_INVENTORY_OK"
