#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$ROOT/.conda/bin/python"

test -x "$PYTHON"
test -d "$ROOT/vendor/gen_finger_con_python_sdk_release"

echo "ROOT=$ROOT"
"$PYTHON" --version
"$PYTHON" -m pip --version
"$PYTHON" -m pip check
v4l2-ctl --version | head -n 1
sha256sum \
  "$ROOT/vendor/gen_finger_con_python_sdk_release/start_finger.py" \
  "$ROOT/vendor/gen_finger_con_python_sdk_release/scripts/databus.py"

echo "ENVIRONMENT_OK"
