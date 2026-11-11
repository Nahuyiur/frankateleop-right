#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SIDE="${GELLO_EEF_SIDE:-left}"

source_conda() {
    local conda_base
    if command -v conda >/dev/null 2>&1; then
        conda_base="$(conda info --base)"
    elif [[ -x "$HOME/miniconda3/bin/conda" ]]; then
        conda_base="$HOME/miniconda3"
    elif [[ -x "/home/pnp/miniconda3/bin/conda" ]]; then
        conda_base="/home/pnp/miniconda3"
    else
        echo "error: conda not found" >&2
        exit 1
    fi
    # shellcheck source=/dev/null
    source "$conda_base/etc/profile.d/conda.sh"
}

resolve_teleop_port() {
    local default_port="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBZIB2R-if00-port0"
    local port="${LEFT_TELEOP_PORT:-${FRANKA_TELEOP_PORT:-${TELEOP_PORT:-$default_port}}}"
    if [[ "$SIDE" == "right" ]]; then
        default_port="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBJTKP2-if00-port0"
        port="${RIGHT_TELEOP_PORT:-$default_port}"
    fi
    if [[ ! -e "$port" ]]; then
        echo "error: GELLO serial port does not exist: $port" >&2
        echo "Set LEFT_TELEOP_PORT to override it." >&2
        exit 1
    fi
    printf '%s\n' "$port"
}

# Some conda activation hooks reference optional variables before defining
# them, which is incompatible with nounset while the hook is running.
set +u
source_conda
conda activate polymetis
set -u

POLYMETIS_ROOT="$REPO_ROOT/polymetis"
TORCH_ISOLATION="$POLYMETIS_ROOT/polymetis/build/torch_isolation"
PINOCCHIO_ISOLATION="$TORCH_ISOLATION/pinocchio_isolation"
TORCH_LIB="$(python -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parent / "lib")')"
for required_dir in "$TORCH_ISOLATION" "$PINOCCHIO_ISOLATION" "$TORCH_LIB"; do
    if [[ ! -d "$required_dir" ]]; then
        echo "error: missing Polymetis runtime directory: $required_dir" >&2
        exit 1
    fi
done
export LD_LIBRARY_PATH="$PINOCCHIO_ISOLATION:$TORCH_ISOLATION:$TORCH_LIB:$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

teleop_port="$(resolve_teleop_port)"
node_port="${LEFT_TELEOP_NODE_PORT:-${LEFT_TELEOP_PORT_ZMQ:-6002}}"
robot_port="${LEFT_ROBOT_SERVER_PORT:-${FRANKA_ROBOT_SERVER_PORT:-50052}}"
if [[ "$SIDE" == "right" ]]; then
    node_port="${RIGHT_TELEOP_NODE_PORT:-6001}"
    robot_port="${RIGHT_ROBOT_SERVER_PORT:-50051}"
fi

echo ">>> GELLO -> EEF adapter"
echo "    GELLO serial: $teleop_port"
echo "    robot node:   127.0.0.1:$node_port"
echo "    robot server: 127.0.0.1:$robot_port"
echo "    default mode is MONITOR ONLY; add --execute for robot motion"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/teleop${PYTHONPATH:+:$PYTHONPATH}"
exec python3 teleop/experiments/gello_eef_teleop.py \
    --side "$SIDE" \
    --host 127.0.0.1 \
    --node-port "$node_port" \
    --robot-port "$robot_port" \
    --teleop-port "$teleop_port" \
    "$@"
