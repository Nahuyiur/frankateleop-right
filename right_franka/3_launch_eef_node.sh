#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODE_PORT="${RIGHT_TELEOP_NODE_PORT:-6001}"
if ss -ltn "sport = :$NODE_PORT" 2>/dev/null | grep -q LISTEN; then
    echo "error: right EEF node port $NODE_PORT already in use; existing services are not stopped." >&2
    exit 1
fi
# Keep libfranka's 1 s communication watchdog quiet during Pico preview.
export FRANKA_LAZY_EE_START="${FRANKA_LAZY_EE_START:-1}"
exec bash "$SCRIPT_DIR/3_launch_node.sh" \
    --hostname 127.0.0.1 \
    --tele-port "$NODE_PORT" \
    --robot-port "${RIGHT_ROBOT_SERVER_PORT:-50051}" \
    --gripper-port "${RIGHT_GRIPPER_SERVER_PORT:-50053}" \
    --control-mode ee --no-home-on-init --no-open-gripper-on-init \
    --no-move-to-initial-pose "$@"
