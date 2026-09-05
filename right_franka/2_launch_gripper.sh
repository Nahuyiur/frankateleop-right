#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source_conda() {
    if command -v conda >/dev/null 2>&1; then
        CONDA_BASE="$(conda info --base)"
    elif [[ -x "$HOME/miniconda3/bin/conda" ]]; then
        CONDA_BASE="$HOME/miniconda3"
    elif [[ -x "/home/pnp/miniconda3/bin/conda" ]]; then
        CONDA_BASE="/home/pnp/miniconda3"
    elif [[ -x "$HOME/anaconda3/bin/conda" ]]; then
        CONDA_BASE="$HOME/anaconda3"
    else
        echo "❌ 错误：未找到 conda"
        exit 1
    fi
    source "$CONDA_BASE/etc/profile.d/conda.sh"
}

sudo_run() {
    if command -v sudo >/dev/null 2>&1; then
        if sudo -n true >/dev/null 2>&1; then
            sudo -n "$@"
            return
        fi
        local sudo_password="${FRANKA_SUDO_PASSWORD:-}"
        if [[ -z "$sudo_password" && -n "${FRANKA_SUDO_PASSWORD_FILE:-}" ]]; then
            validate_sudo_password_file "$FRANKA_SUDO_PASSWORD_FILE" || return 1
            IFS= read -r sudo_password < "$FRANKA_SUDO_PASSWORD_FILE" || true
        fi
        if [[ -n "$sudo_password" ]]; then
            printf '%s\n' "$sudo_password" | sudo -S -p '' "$@"
        else
            sudo "$@"
        fi
    else
        "$@"
    fi
}

validate_sudo_password_file() {
    local file="$1"
    local mode owner
    [[ -f "$file" ]] || { echo "ERROR: sudo credential file not found: $file" >&2; return 1; }
    mode="$(stat -c '%a' "$file")"
    owner="$(stat -c '%u' "$file")"
    [[ "$owner" == "$(id -u)" && ( "$mode" == "600" || "$mode" == "400" ) ]] || {
        echo "ERROR: sudo credential file must be owned by the current user with mode 600 or 400: $file" >&2
        return 1
    }
}

cleanup_gripper_port() {
    local port="$1"
    local pids=()

    if command -v lsof >/dev/null 2>&1; then
        mapfile -t pids < <(
            {
                sudo_run lsof -t -iTCP:"$port" -sTCP:LISTEN 2>/dev/null || true
                sudo_run lsof -t -i:"$port" 2>/dev/null || true
            } | awk '/^[0-9]+$/ && !seen[$0]++'
        )
    fi
    if command -v fuser >/dev/null 2>&1; then
        mapfile -t pids < <(
            {
                printf '%s\n' "${pids[@]}"
                sudo_run fuser -n tcp "$port" 2>/dev/null | tr ' ' '\n' || true
            } | awk '/^[0-9]+$/ && !seen[$0]++'
        )
    fi
    if command -v ss >/dev/null 2>&1; then
        mapfile -t pids < <(
            {
                printf '%s\n' "${pids[@]}"
                ss -ltnp 2>/dev/null | awk -v suffix=":$port" '$4 ~ suffix "$" {print}' | sed -nE 's/.*pid=([0-9]+).*/\1/p'
            } | awk '/^[0-9]+$/ && !seen[$0]++'
        )
    fi

    if [[ "${#pids[@]}" -eq 0 ]]; then
        return 0
    fi

    echo ">>> 清理占用 gripper 端口 $port 的旧进程: ${pids[*]}"
    local pid
    for pid in "${pids[@]}"; do
        [[ "$pid" == "$$" || "$pid" == "$BASHPID" ]] && continue
        sudo_run kill -TERM "$pid" 2>/dev/null || true
    done
    sleep 1
    for pid in "${pids[@]}"; do
        [[ "$pid" == "$$" || "$pid" == "$BASHPID" ]] && continue
        if kill -0 "$pid" 2>/dev/null; then
            sudo_run kill -KILL "$pid" 2>/dev/null || true
        fi
    done

    local started
    started="$(date +%s)"
    while true; do
        if ! gripper_port_in_use "$port"; then
            return 0
        fi
        if (( $(date +%s) - started >= 5 )); then
            echo "⚠️ gripper 端口 $port 清理后仍被占用，继续启动时可能会失败" >&2
            return 0
        fi
        sleep 0.2
    done
}

gripper_port_in_use() {
    local port="$1"
    if command -v lsof >/dev/null 2>&1 && sudo_run lsof -t -i:"$port" >/dev/null 2>&1; then
        return 0
    fi
    if command -v fuser >/dev/null 2>&1 && sudo_run fuser -n tcp "$port" >/dev/null 2>&1; then
        return 0
    fi
    if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | awk -v suffix=":$port" '$4 ~ suffix "$" {found=1} END {exit found ? 0 : 1}'; then
        return 0
    fi
    return 1
}

echo ">>> 激活 conda 环境 polymetis ..."
source_conda
conda activate polymetis || { echo "❌ 激活失败"; exit 1; }

POLY_ROOT="$REPO_ROOT/polymetis"
GRIPPER_SERVER_PORT="${RIGHT_GRIPPER_SERVER_PORT:-50053}"

WORK_DIR="$POLY_ROOT/polymetis/python/polymetis"
if [[ ! -d "$WORK_DIR/conf" ]]; then
    echo "❌ 配置目录 $WORK_DIR/conf 不存在"
    exit 1
fi

resolve_gripper_driver() {
    local requested="${RIGHT_GRIPPER_DRIVER:-${FRANKA_GRIPPER_DRIVER:-auto}}"
    local finger_comport="${RIGHT_FINGER_COMPORT:-/dev/ttyFingerRight}"

    case "${requested,,}" in
        auto)
            if [[ -e "$finger_comport" ]]; then
                printf '%s\n' "das_finger"
            else
                printf '%s\n' "robotiq"
            fi
            ;;
        das|das_finger|finger)
            printf '%s\n' "das_finger"
            ;;
        robotiq)
            printf '%s\n' "robotiq"
            ;;
        *)
            echo "❌ 未知 RIGHT_GRIPPER_DRIVER：$requested（可选 auto/das_finger/robotiq）" >&2
            exit 1
            ;;
    esac
}

GRIPPER_DRIVER_RESOLVED="$(resolve_gripper_driver)"
cleanup_gripper_port "$GRIPPER_SERVER_PORT"

if [[ "$GRIPPER_DRIVER_RESOLVED" == "das_finger" ]]; then
    FINGER_COMPORT_RESOLVED="${RIGHT_FINGER_COMPORT:-/dev/ttyFingerRight}"
    FINGER_BRIDGE="$REPO_ROOT/umi_genrobot/bridge/das_finger_polymetis_server.py"
    FINGER_SDK_ROOT="${DAS_FINGER_SDK_ROOT:-$REPO_ROOT/umi_genrobot/vendor/gen_finger_con_python_sdk_release}"

    if [[ ! -e "$FINGER_COMPORT_RESOLVED" ]]; then
        echo "❌ DAS Finger 串口不存在：$FINGER_COMPORT_RESOLVED" >&2
        echo "   请检查 /dev/ttyFingerRight udev 映射，或设置 RIGHT_FINGER_COMPORT。" >&2
        exit 1
    fi
    if [[ ! -f "$FINGER_BRIDGE" ]]; then
        echo "❌ DAS Finger Polymetis 适配器不存在：$FINGER_BRIDGE" >&2
        exit 1
    fi
    if [[ ! -d "$FINGER_SDK_ROOT" ]]; then
        echo "❌ DAS Finger SDK 不存在：$FINGER_SDK_ROOT" >&2
        exit 1
    fi

    echo ">>> 启动 DAS Finger Controller RV3 ..."
    echo ">>> DAS Finger 串口: $FINGER_COMPORT_RESOLVED (921600 baud)"
    echo ">>> Gripper server port: $GRIPPER_SERVER_PORT"
    echo ">>> 安全策略: 收到首个明确 Goto 命令前不发送夹爪距离指令"
    exec python3 "$FINGER_BRIDGE" \
        --serial-port "$FINGER_COMPORT_RESOLVED" \
        --sdk-root "$FINGER_SDK_ROOT" \
        --server-port "$GRIPPER_SERVER_PORT" \
        --baudrate "${RIGHT_FINGER_BAUDRATE:-921600}" \
        --hz "${RIGHT_FINGER_COMMAND_HZ:-30}" \
        --max-width "${RIGHT_FINGER_MAX_WIDTH:-0.2}" \
        "$@"
fi

resolve_robotiq_comport() {
    local explicit="${RIGHT_ROBOTIQ_COMPORT:-${RIGHT_ROBOTIQ_PORT:-${FRANKA_ROBOTIQ_COMPORT:-${FRANKA_ROBOTIQ_PORT:-${ROBOTIQ_COMPORT:-${ROBOTIQ_PORT:-}}}}}}"
    local default_comport="${RIGHT_ROBOTIQ_DEFAULT_COMPORT:-/dev/serial/by-id/usb-FTDI_USB_TO_RS-485_DAAQM7QD-if00-port0}"

    if [[ -n "$explicit" ]]; then
        if [[ ! -e "$explicit" ]]; then
            echo "❌ 右臂 Robotiq 串口不存在：$explicit" >&2
            exit 1
        fi
        printf '%s\n' "$explicit"
        return 0
    fi

    if [[ -n "$default_comport" && -e "$default_comport" ]]; then
        printf '%s\n' "$default_comport"
        return 0
    fi

    local rs485_ports=()
    if [[ -d /dev/serial/by-id ]]; then
        mapfile -t rs485_ports < <(find /dev/serial/by-id -maxdepth 1 -type l -name 'usb-FTDI_USB_TO_RS-485_*' | sort)
    fi
    if [[ "${#rs485_ports[@]}" -eq 1 ]]; then
        printf '%s\n' "${rs485_ports[0]}"
        return 0
    fi

    echo "❌ 未能唯一确定右臂 Robotiq RS485 串口。" >&2
    if [[ -n "$default_comport" ]]; then
        echo "   默认串口不存在：$default_comport" >&2
    fi
    echo "   检测到 ${#rs485_ports[@]} 个 RS485 串口：" >&2
    printf '     %s\n' "${rs485_ports[@]}" >&2
    echo "   请在右臂机器设置 RIGHT_ROBOTIQ_COMPORT 或 FRANKA_ROBOTIQ_COMPORT。" >&2
    exit 1
}

ROBOTIQ_COMPORT_RESOLVED="$(resolve_robotiq_comport)"

echo ">>> 启动 Robotiq Gripper 客户端 ..."
echo ">>> Robotiq 串口: $ROBOTIQ_COMPORT_RESOLVED"
echo ">>> Gripper server port: $GRIPPER_SERVER_PORT"
cd "$WORK_DIR"
launch_gripper.py \
    --config-name=launch_right_gripper \
    "port=$GRIPPER_SERVER_PORT" \
    "gripper.comport=$ROBOTIQ_COMPORT_RESOLVED" \
    "$@"
