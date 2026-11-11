#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIDE="${GELLO_EEF_SIDE:-left}"
case "$SIDE" in
    left)
        ROBOT_PORT="${LEFT_ROBOT_SERVER_PORT:-50052}"
        GRIPPER_PORT="${LEFT_GRIPPER_SERVER_PORT:-50054}"
        NODE_PORT="${LEFT_TELEOP_NODE_PORT:-6002}" ;;
    right)
        ROBOT_PORT="${RIGHT_ROBOT_SERVER_PORT:-50051}"
        GRIPPER_PORT="${RIGHT_GRIPPER_SERVER_PORT:-50053}"
        NODE_PORT="${RIGHT_TELEOP_NODE_PORT:-6001}" ;;
    *) echo "Invalid GELLO_EEF_SIDE=$SIDE" >&2; exit 2 ;;
esac
ARM_DIR="$REPO_ROOT/${SIDE}_franka"
READY_TIMEOUT_SEC="${GELLO_EEF_READY_TIMEOUT_SEC:-120}"
SUDO_PASSWORD_FILE="${FRANKA_SUDO_PASSWORD_FILE:-$HOME/.franka_gui_sudo_password}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${GELLO_EEF_LOG_DIR:-$REPO_ROOT/logs/gello_eef/$RUN_ID}"

EXECUTE=0
CHECK_ONLY=0
ADAPTER_ARGS=()
PIDS=()
NAMES=()
LOGS=()
CLEANING_UP=0

usage() {
    cat <<'EOF'
Usage:
  bash A_run_left_gello_eef.sh --check
  bash A_run_left_gello_eef.sh [adapter options]
  bash A_run_left_gello_eef.sh --execute [adapter options]

This is the single-command launcher for the left GELLO -> EEF test stack. It
starts, checks, and later cleans up:
  1. left_franka/1_launch_robot.sh       robot gRPC :50052
  2. left_franka/2_launch_gripper.sh     gripper gRPC :50054
  3. left_franka/3_launch_eef_node.sh    EE robot node :6002
  4. left_franka/4_run_eef_gello.sh      GELLO EEF adapter

Modes:
  default      Monitor-only: calculate and print EEF targets; robot does not move.
  --execute    Explicitly enable EEF commands after the adapter countdown.
  --check      Validate scripts and unit tests without starting hardware.

Examples:
  bash A_run_left_gello_eef.sh --check
  bash A_run_left_gello_eef.sh
  bash A_run_left_gello_eef.sh --execute \
    --base-alignment-rpy-deg 0 0 0 --calibration-verified \
    --translation-scale 0.5 --rotation-scale 0.5 \
    --max-anchor-translation 0.12 --max-anchor-rotation 0.6

Ctrl+C stops the adapter and all three stack processes started by this command.
The launcher refuses to replace an existing service on ports 50052/50054/6002.
EOF
}

log() {
    printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"
}

while (($#)); do
    case "$1" in
        --execute) EXECUTE=1 ;;
        --check) CHECK_ONLY=1 ;;
        -h|--help) usage; exit 0 ;;
        *) ADAPTER_ARGS+=("$1") ;;
    esac
    shift
done

if ((EXECUTE)); then
    alignment_set=0
    calibration_verified=0
    for arg in "${ADAPTER_ARGS[@]}"; do
        [[ "$arg" != "--base-alignment-rpy-deg" ]] || alignment_set=1
        [[ "$arg" != "--calibration-verified" ]] || calibration_verified=1
    done
    if ((alignment_set == 0 || calibration_verified == 0)); then
        echo "Refusing hardware startup: supply measured --base-alignment-rpy-deg R P Y and --calibration-verified." >&2
        exit 2
    fi
fi

required_files=(
    "$ARM_DIR/1_launch_robot.sh"
    "$ARM_DIR/2_launch_gripper.sh"
    "$ARM_DIR/3_launch_eef_node.sh"
    "$ARM_DIR/4_run_eef_gello.sh"
    "$REPO_ROOT/teleop/experiments/gello_eef_teleop.py"
)
for required_file in "${required_files[@]}"; do
    [[ -f "$required_file" ]] || {
        echo "ERROR: required file is missing: $required_file" >&2
        exit 2
    }
done

source_polymetis() {
    local conda_base
    if command -v conda >/dev/null 2>&1; then
        conda_base="$(conda info --base)"
    elif [[ -x "$HOME/miniconda3/bin/conda" ]]; then
        conda_base="$HOME/miniconda3"
    else
        echo "ERROR: conda not found" >&2
        return 1
    fi
    set +u
    # shellcheck source=/dev/null
    source "$conda_base/etc/profile.d/conda.sh"
    conda activate polymetis
    set -u
}

setup_runtime() {
    local polymetis_root torch_isolation pinocchio_isolation torch_lib
    source_polymetis
    polymetis_root="$REPO_ROOT/polymetis"
    torch_isolation="$polymetis_root/polymetis/build/torch_isolation"
    pinocchio_isolation="$torch_isolation/pinocchio_isolation"
    torch_lib="$(python -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parent / "lib")')"
    export LD_LIBRARY_PATH="$pinocchio_isolation:$torch_isolation:$torch_lib:$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export PYTHONPATH="$REPO_ROOT/teleop${PYTHONPATH:+:$PYTHONPATH}"
    export PYTHONDONTWRITEBYTECODE=1
}

run_check() {
    local script
    log "Checking GELLO EEF stack (no hardware)."
    for script in "${required_files[@]:0:4}" "$REPO_ROOT"/*run*gello_eef.sh "$REPO_ROOT"/right_franka/*eef*.sh; do
        bash -n "$script"
    done
    setup_runtime
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    PYTHONPATH="$REPO_ROOT/teleop${PYTHONPATH:+:$PYTHONPATH}" \
        python -B -m pytest -q -p no:cacheprovider \
        "$REPO_ROOT/tests/test_gello_eef_adapter.py" \
        "$REPO_ROOT/tests/test_fr3_ee_safety.py" \
        "$REPO_ROOT/tests/test_dual_gello_eef.py"
    log "Check passed. No robot service was started."
}

if ((CHECK_ONLY)); then
    ((EXECUTE == 0)) || {
        echo "ERROR: --check and --execute cannot be used together" >&2
        exit 2
    }
    run_check
    exit 0
fi

port_open() {
    local port="$1"
    timeout 1 bash -c "</dev/tcp/127.0.0.1/$port" >/dev/null 2>&1
}

process_alive() {
    kill -0 "$1" 2>/dev/null
}

terminate_group() {
    local pid="$1"
    local name="$2"
    process_alive "$pid" || return 0
    log "Stopping $name (process group $pid)."
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    local deadline=$((SECONDS + 5))
    while process_alive "$pid" && ((SECONDS < deadline)); do
        sleep 0.1
    done
    if process_alive "$pid"; then
        kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    fi
}

cleanup() {
    ((CLEANING_UP == 0)) || return 0
    CLEANING_UP=1
    local index
    for ((index=${#PIDS[@]}-1; index>=0; index--)); do
        terminate_group "${PIDS[$index]}" "${NAMES[$index]}"
    done
    log "Cleanup complete. Logs: $LOG_DIR"
}

on_signal() {
    log "Signal received; stopping GELLO EEF stack."
    cleanup
    exit 130
}

trap 'on_signal' INT TERM
trap 'cleanup' EXIT

validate_sudo_password_file() {
    local mode owner
    [[ -f "$SUDO_PASSWORD_FILE" ]] || {
        echo "ERROR: sudo password file not found: $SUDO_PASSWORD_FILE" >&2
        echo "Create it with mode 600, or set FRANKA_SUDO_PASSWORD_FILE." >&2
        exit 2
    }
    mode="$(stat -c '%a' "$SUDO_PASSWORD_FILE")"
    owner="$(stat -c '%u' "$SUDO_PASSWORD_FILE")"
    [[ "$owner" == "$(id -u)" && ( "$mode" == "600" || "$mode" == "400" ) ]] || {
        echo "ERROR: $SUDO_PASSWORD_FILE must be owned by $(id -un) with mode 600 or 400" >&2
        exit 2
    }
}

assert_port_free() {
    local name="$1"
    local port="$2"
    if port_open "$port"; then
        echo "ERROR: $name port $port is already in use." >&2
        echo "Stop the existing joint/EEF stack first; this launcher will not kill it." >&2
        exit 2
    fi
}

start_and_wait() {
    local name="$1"
    local port="$2"
    shift 2
    local safe_name="${name// /_}"
    local logfile="$LOG_DIR/${safe_name}.log"
    log "Starting $name; log=$logfile"
    setsid "$@" >"$logfile" 2>&1 &
    local pid="$!"
    PIDS+=("$pid")
    NAMES+=("$name")
    LOGS+=("$logfile")

    local deadline=$((SECONDS + READY_TIMEOUT_SEC))
    while true; do
        if port_open "$port"; then
            log "$name ready on port $port."
            return 0
        fi
        if ! process_alive "$pid"; then
            echo "ERROR: $name exited before port $port became ready" >&2
            tail -n 80 "$logfile" >&2 || true
            return 1
        fi
        if ((SECONDS >= deadline)); then
            echo "ERROR: $name did not open port $port within ${READY_TIMEOUT_SEC}s" >&2
            tail -n 80 "$logfile" >&2 || true
            return 1
        fi
        sleep 0.5
    done
}

validate_sudo_password_file
export FRANKA_SUDO_PASSWORD_FILE="$SUDO_PASSWORD_FILE"
export LEFT_ROBOT_SERVER_PORT="$ROBOT_PORT"
export LEFT_GRIPPER_SERVER_PORT="$GRIPPER_PORT"
export LEFT_TELEOP_NODE_PORT="$NODE_PORT"
if [[ "$SIDE" == "right" ]]; then
    export RIGHT_ROBOT_SERVER_PORT="$ROBOT_PORT"
    export RIGHT_GRIPPER_SERVER_PORT="$GRIPPER_PORT"
    export RIGHT_TELEOP_NODE_PORT="$NODE_PORT"
fi
mkdir -p "$LOG_DIR"

assert_port_free "$SIDE robot server" "$ROBOT_PORT"
assert_port_free "$SIDE gripper server" "$GRIPPER_PORT"
assert_port_free "$SIDE robot node" "$NODE_PORT"

# Validate every numerical CLI value before starting any stack process.
setup_runtime
validation_args=(--validate-only --teleop-port /dev/argument-validation-only --side "$SIDE")
((EXECUTE == 0)) || validation_args+=(--execute)
python -B "$REPO_ROOT/teleop/experiments/gello_eef_teleop.py" "${validation_args[@]}" "${ADAPTER_ARGS[@]}"

start_and_wait "$SIDE robot server" "$ROBOT_PORT" \
    bash "$ARM_DIR/1_launch_robot.sh"
start_and_wait "$SIDE gripper server" "$GRIPPER_PORT" \
    bash "$ARM_DIR/2_launch_gripper.sh"
start_and_wait "$SIDE EE robot node" "$NODE_PORT" \
    bash "$ARM_DIR/3_launch_eef_node.sh"

adapter_command=(bash "$ARM_DIR/4_run_eef_gello.sh")
((EXECUTE == 0)) || adapter_command+=(--execute)
adapter_command+=("${ADAPTER_ARGS[@]}")

log "Starting GELLO EEF adapter in $([[ $EXECUTE == 1 ]] && echo EXECUTE || echo MONITOR-ONLY) mode."
log "Press Ctrl+C once to stop the complete stack."
set +e
"${adapter_command[@]}" 2>&1 | tee "$LOG_DIR/gello_eef_adapter.log"
adapter_status="${PIPESTATUS[0]}"
set -e
if ((adapter_status != 0)); then
    echo "ERROR: GELLO EEF adapter exited with status $adapter_status" >&2
fi
exit "$adapter_status"
