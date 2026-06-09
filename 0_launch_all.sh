#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="root"
SCRIPT_SET_DIR="$ROOT_DIR"
COUNTDOWN_SECONDS="${COUNTDOWN_SECONDS:-0}"
ROBOT_TIMEOUT_SECONDS="${ROBOT_TIMEOUT_SECONDS:-90}"
GRIPPER_TIMEOUT_SECONDS="${GRIPPER_TIMEOUT_SECONDS:-60}"
NODE_TIMEOUT_SECONDS="${NODE_TIMEOUT_SECONDS:-45}"
DRY_RUN=0

declare -a ENV_ARGS=()
declare -a STARTED_PIDS=()
LAST_STARTED_PID=""

usage() {
    cat <<EOF
Usage:
  ./0_launch_all.sh [root|right|left] [--countdown SECONDS] [--dry-run] [-- ENV_ARGS...]

Examples:
  ./0_launch_all.sh
  ./0_launch_all.sh right
  ./0_launch_all.sh left
  ./0_launch_all.sh -- --use-save-interface

Notes:
  - Starts 1_launch_robot.sh, 2_launch_gripper.sh, 3_launch_node.sh in the background.
  - Starts 4_run_env.sh in the foreground so Ctrl-C stops this launch session.
  - Logs are written to logs/launch_YYYYmmdd_HHMMSS/.
EOF
}

log() {
    printf '[INFO] %s\n' "$*"
}

warn() {
    printf '[WARN] %s\n' "$*" >&2
}

error() {
    printf '[ERROR] %s\n' "$*" >&2
}

die() {
    error "$*"
    exit 1
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            root|right|left)
                PROFILE="$1"
                shift
                ;;
            --profile)
                [[ $# -ge 2 ]] || die "--profile needs one value: root, right, or left"
                PROFILE="$2"
                shift 2
                ;;
            --dir)
                [[ $# -ge 2 ]] || die "--dir needs a script directory"
                SCRIPT_SET_DIR="$2"
                PROFILE="custom"
                shift 2
                ;;
            --no-countdown)
                COUNTDOWN_SECONDS=0
                shift
                ;;
            --countdown)
                [[ $# -ge 2 ]] || die "--countdown needs a number of seconds"
                COUNTDOWN_SECONDS="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=1
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            --)
                shift
                ENV_ARGS=("$@")
                break
                ;;
            *)
                ENV_ARGS+=("$1")
                shift
                ;;
        esac
    done

    case "$PROFILE" in
        root)
            SCRIPT_SET_DIR="$ROOT_DIR"
            ;;
        right)
            SCRIPT_SET_DIR="$ROOT_DIR/right_franka"
            ;;
        left)
            SCRIPT_SET_DIR="$ROOT_DIR/left_franka"
            ;;
        custom)
            ;;
        *)
            die "Unknown profile: $PROFILE. Use root, right, or left."
            ;;
    esac

    SCRIPT_SET_DIR="$(cd "$SCRIPT_SET_DIR" && pwd)"
}

script_path() {
    printf '%s/%s' "$SCRIPT_SET_DIR" "$1"
}

extract_cli_int() {
    local file="$1"
    local option="$2"
    local value
    value="$(grep -oE -- "${option}(=| )[0-9]+" "$file" 2>/dev/null | head -n 1 | sed -E "s/${option}(=| )//" || true)"
    printf '%s' "$value"
}

extract_teleop_port() {
    local file="$1"
    local value
    value="$(sed -nE 's/^[[:space:]]*TELEOP_PORT="?([^"[:space:]]+)"?.*/\1/p' "$file" | tail -n 1)"
    if [[ -z "$value" ]]; then
        value="$(grep -oE -- '--teleop_port(=| )[^[:space:]]+' "$file" 2>/dev/null | head -n 1 | sed -E 's/--teleop_port(=| )//; s/"//g' || true)"
    fi
    printf '%s' "$value"
}

default_robot_port() {
    if [[ "$PROFILE" == "left" ]]; then
        printf '50052'
    else
        printf '50051'
    fi
}

default_gripper_port() {
    if [[ "$PROFILE" == "left" ]]; then
        printf '50054'
    else
        printf '50053'
    fi
}

default_tele_port() {
    if [[ "$PROFILE" == "left" ]]; then
        printf '6002'
    else
        printf '6001'
    fi
}

port_is_open() {
    local host="$1"
    local port="$2"
    nc -z -w 1 "$host" "$port" >/dev/null 2>&1
}

show_port_owner() {
    local port="$1"
    if command -v lsof >/dev/null 2>&1; then
        sudo lsof -nP -iTCP:"$port" -sTCP:LISTEN 2>/dev/null || lsof -nP -iTCP:"$port" -sTCP:LISTEN 2>/dev/null || true
    elif command -v ss >/dev/null 2>&1; then
        ss -ltnp "sport = :$port" 2>/dev/null || true
    fi
}

print_tail() {
    local log_file="$1"
    if [[ -f "$log_file" ]]; then
        printf '\n----- Last 80 log lines: %s -----\n' "$log_file" >&2
        tail -n 80 "$log_file" >&2 || true
        printf '%s\n\n' '----------------------------------------' >&2
    fi
}

print_step_tips() {
    local step="$1"
    local port="${2:-}"
    case "$step" in
        robot)
            cat >&2 <<EOF
Tips for step 1:
  - Check the Franka Desk is unlocked and FCI is active.
  - Check the robot IP in polymetis config, usually 172.16.0.2 for right and 172.16.0.3 for left.
  - If port ${port:-50051} is occupied by an old server, stop it first or reboot the old launch session.
EOF
            ;;
        gripper)
            cat >&2 <<EOF
Tips for step 2:
  - Check the hand is powered and reachable from the same Franka Desk/FCI network.
  - If port ${port:-50053} is occupied, stop the stale gripper process first.
EOF
            ;;
        node)
            cat >&2 <<EOF
Tips for step 3:
  - Step 1 and step 2 must both stay alive before the ZMQ robot node can connect.
  - If port ${port:-6001} is occupied, kill the old launch_nodes.py process or run teleop/kill_nodes.sh.
EOF
            ;;
        env)
            cat >&2 <<EOF
Tips for step 4:
  - Check the teleop USB serial device exists and belongs to the correct arm.
  - Check the leader and follower joint positions are close enough before enabling teleop.
  - Check step 3 is still running and listening on the teleop port.
EOF
            ;;
    esac
}

fail_step() {
    local step="$1"
    local log_file="$2"
    local message="$3"
    local port="${4:-}"
    error "$message"
    print_tail "$log_file"
    print_step_tips "$step" "$port"
    exit 1
}

process_is_running() {
    local pid="$1"
    local stat
    stat="$(ps -p "$pid" -o stat= 2>/dev/null | awk '{print $1}' || true)"
    [[ -n "$stat" && "$stat" != Z* ]]
}

kill_tree() {
    local pid="$1"
    local sig="$2"
    local child
    while read -r child; do
        [[ -n "$child" ]] || continue
        kill_tree "$child" "$sig"
    done < <(pgrep -P "$pid" 2>/dev/null || true)
    kill "-$sig" "$pid" >/dev/null 2>&1 || true
}

cleanup() {
    local status=$?
    trap - EXIT INT TERM

    if [[ ${#STARTED_PIDS[@]} -gt 0 ]]; then
        log "Stopping background launch processes..."
        local pid
        for pid in "${STARTED_PIDS[@]}"; do
            kill_tree "$pid" TERM
        done
        sleep 2
        for pid in "${STARTED_PIDS[@]}"; do
            kill_tree "$pid" KILL
        done
    fi

    if [[ ${status:-0} -eq 130 ]]; then
        warn "Interrupted by Ctrl-C. Background processes from this launcher were stopped."
    fi
    exit "$status"
}

trap cleanup EXIT
trap 'exit 130' INT TERM

start_background_step() {
    local step_key="$1"
    local step_name="$2"
    local script_file="$3"
    local log_file="$4"
    shift 4

    log "Starting $step_name"
    log "  script: $script_file"
    log "  log:    $log_file"
    LAST_STARTED_PID=""

    if [[ "$DRY_RUN" -eq 1 ]]; then
        return 0
    fi

    (
        cd "$SCRIPT_SET_DIR" || exit 1
        exec bash "$script_file" "$@"
    ) >"$log_file" 2>&1 &
    local pid=$!
    STARTED_PIDS+=("$pid")
    LAST_STARTED_PID="$pid"
}

robot_rpc_ready() {
    local port="$1"
    conda run -n polymetis python - "$port" <<'PY' >/dev/null 2>&1
import sys
from polymetis import RobotInterface

port = int(sys.argv[1])
robot = RobotInterface(ip_address="127.0.0.1", port=port)
robot.get_robot_state()
PY
}

gripper_rpc_ready() {
    local port="$1"
    conda run -n polymetis python - "$port" <<'PY' >/dev/null 2>&1
import sys
from polymetis import GripperInterface

port = int(sys.argv[1])
gripper = GripperInterface(ip_address="127.0.0.1", port=port)
gripper.get_state()
PY
}

wait_for_port() {
    local step_key="$1"
    local step_name="$2"
    local host="$3"
    local port="$4"
    local pid="$5"
    local log_file="$6"
    local timeout="$7"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        return 0
    fi

    local start_time=$SECONDS
    while (( SECONDS - start_time < timeout )); do
        if ! process_is_running "$pid"; then
            fail_step "$step_key" "$log_file" "$step_name exited before port $port became ready." "$port"
        fi
        if port_is_open "$host" "$port"; then
            sleep 2
            if ! process_is_running "$pid"; then
                fail_step "$step_key" "$log_file" "$step_name exited just after port $port opened." "$port"
            fi
            log "$step_name is ready on $host:$port"
            return 0
        fi
        sleep 1
    done

    show_port_owner "$port" >&2
    fail_step "$step_key" "$log_file" "Timed out waiting for $step_name on $host:$port after ${timeout}s." "$port"
}

wait_for_robot_rpc() {
    local step_name="$1"
    local port="$2"
    local pid="$3"
    local log_file="$4"
    local timeout="$5"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        return 0
    fi

    local start_time=$SECONDS
    while (( SECONDS - start_time < timeout )); do
        if ! process_is_running "$pid"; then
            fail_step robot "$log_file" "$step_name exited before robot RPC became ready." "$port"
        fi
        if robot_rpc_ready "$port"; then
            log "$step_name robot RPC is ready on 127.0.0.1:$port"
            return 0
        fi
        sleep 2
    done

    show_port_owner "$port" >&2
    fail_step robot "$log_file" "Timed out waiting for robot RPC readiness on 127.0.0.1:$port after ${timeout}s." "$port"
}

wait_for_gripper_rpc() {
    local step_name="$1"
    local port="$2"
    local pid="$3"
    local log_file="$4"
    local timeout="$5"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        return 0
    fi

    local start_time=$SECONDS
    while (( SECONDS - start_time < timeout )); do
        if ! process_is_running "$pid"; then
            fail_step gripper "$log_file" "$step_name exited before gripper RPC became ready." "$port"
        fi
        if gripper_rpc_ready "$port"; then
            log "$step_name gripper RPC is ready on 127.0.0.1:$port"
            return 0
        fi
        sleep 2
    done

    show_port_owner "$port" >&2
    fail_step gripper "$log_file" "Timed out waiting for gripper RPC readiness on 127.0.0.1:$port after ${timeout}s." "$port"
}

require_command() {
    local cmd="$1"
    command -v "$cmd" >/dev/null 2>&1 || die "Required command not found: $cmd"
}

preflight() {
    local robot_script
    local gripper_script
    local node_script
    local env_script
    robot_script="$(script_path 1_launch_robot.sh)"
    gripper_script="$(script_path 2_launch_gripper.sh)"
    node_script="$(script_path 3_launch_node.sh)"
    env_script="$(script_path 4_run_env.sh)"

    [[ -f "$robot_script" ]] || die "Missing script: $robot_script"
    [[ -f "$gripper_script" ]] || die "Missing script: $gripper_script"
    [[ -f "$node_script" ]] || die "Missing script: $node_script"
    [[ -f "$env_script" ]] || die "Missing script: $env_script"

    require_command conda
    require_command nc
    require_command sudo

    local conda_base
    conda_base="$(conda info --base)"
    # shellcheck disable=SC1091
    source "$conda_base/etc/profile.d/conda.sh"
    if ! conda env list | awk '{print $1}' | grep -qx 'polymetis'; then
        die "Conda env 'polymetis' was not found. Create it or check conda env list."
    fi

    if [[ "$DRY_RUN" -ne 1 ]]; then
        log "Checking sudo credentials for cleanup/real-time launch commands..."
        sudo -v || die "sudo authentication failed. Existing 1/2 scripts need sudo."
    fi

    local teleop_port
    teleop_port="$(extract_teleop_port "$env_script")"
    if [[ -n "$teleop_port" && "$teleop_port" == /dev/* && ! -e "$teleop_port" ]]; then
        die "Teleop serial device not found: $teleop_port"
    fi
}

safety_countdown() {
    if [[ "$DRY_RUN" -eq 1 || "$COUNTDOWN_SECONDS" -le 0 ]]; then
        return 0
    fi

    local remaining
    for ((remaining = COUNTDOWN_SECONDS; remaining > 0; remaining--)); do
        printf '\rStarting in %ds. Press Ctrl-C to cancel. ' "$remaining"
        sleep 1
    done
    printf '\n'
}

main() {
    parse_args "$@"

    local robot_script
    local gripper_script
    local node_script
    local env_script
    robot_script="$(script_path 1_launch_robot.sh)"
    gripper_script="$(script_path 2_launch_gripper.sh)"
    node_script="$(script_path 3_launch_node.sh)"
    env_script="$(script_path 4_run_env.sh)"

    local robot_port
    local gripper_port
    local tele_port
    robot_port="$(extract_cli_int "$node_script" '--robot_port')"
    gripper_port="$(extract_cli_int "$node_script" '--gripper_port')"
    tele_port="$(extract_cli_int "$node_script" '--tele_port')"
    robot_port="${robot_port:-$(default_robot_port)}"
    gripper_port="${gripper_port:-$(default_gripper_port)}"
    tele_port="${tele_port:-$(default_tele_port)}"

    local log_dir
    if [[ "$DRY_RUN" -eq 1 ]]; then
        log_dir="$ROOT_DIR/logs/dry_run_$(date +%Y%m%d_%H%M%S)"
    else
        log_dir="$ROOT_DIR/logs/launch_$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$log_dir"
    fi

    log "Launch profile: $PROFILE"
    log "Script directory: $SCRIPT_SET_DIR"
    log "Ports: robot=$robot_port gripper=$gripper_port teleop_node=$tele_port"
    log "Log directory: $log_dir"
    if [[ ${#ENV_ARGS[@]} -gt 0 ]]; then
        log "Step 4 extra args: ${ENV_ARGS[*]}"
    fi

    preflight

    if port_is_open 127.0.0.1 "$tele_port"; then
        warn "Teleop node port $tele_port is already in use."
        show_port_owner "$tele_port" >&2
        die "Stop the old launch_nodes.py process first, then rerun this script."
    fi

    safety_countdown

    local robot_pid
    local gripper_pid
    local node_pid
    start_background_step robot "step 1 robot server/client" "$robot_script" "$log_dir/1_launch_robot.log"
    robot_pid="$LAST_STARTED_PID"
    wait_for_robot_rpc "step 1 robot server/client" "$robot_port" "$robot_pid" "$log_dir/1_launch_robot.log" "$ROBOT_TIMEOUT_SECONDS"

    start_background_step gripper "step 2 gripper server/client" "$gripper_script" "$log_dir/2_launch_gripper.log"
    gripper_pid="$LAST_STARTED_PID"
    wait_for_gripper_rpc "step 2 gripper server/client" "$gripper_port" "$gripper_pid" "$log_dir/2_launch_gripper.log" "$GRIPPER_TIMEOUT_SECONDS"

    start_background_step node "step 3 ZMQ robot node" "$node_script" "$log_dir/3_launch_node.log"
    node_pid="$LAST_STARTED_PID"
    wait_for_port node "step 3 ZMQ robot node" 127.0.0.1 "$tele_port" "$node_pid" "$log_dir/3_launch_node.log" "$NODE_TIMEOUT_SECONDS"

    log "Starting step 4 teleop environment in the foreground."
    log "Step 4 log: $log_dir/4_run_env.log"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        log "Dry run complete."
        return 0
    fi

    set +e
    (
        cd "$SCRIPT_SET_DIR" || exit 1
        exec bash "$env_script" "${ENV_ARGS[@]}"
    ) 2>&1 | tee "$log_dir/4_run_env.log"
    local env_status=${PIPESTATUS[0]}
    set -e

    if [[ "$env_status" -ne 0 ]]; then
        fail_step env "$log_dir/4_run_env.log" "Step 4 exited with status $env_status." "$tele_port"
    fi
}

main "$@"
