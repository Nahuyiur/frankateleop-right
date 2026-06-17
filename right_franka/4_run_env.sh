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

echo ">>> 激活 conda 环境 polymetis ..."
source_conda
conda activate polymetis || { echo "❌ 激活失败，请确认 polymetis 环境存在"; exit 1; }

echo ">>> 使用仓库根目录 teleop ..."
POLY_ROOT="$REPO_ROOT/teleop"

if [[ -z "$POLY_ROOT" ]]; then
    echo "❌ 当前目录下未找到 teleop 文件夹"
    exit 1
fi

SCRIPT_PATH="$POLY_ROOT/experiments/run_env.py"
if [[ ! -f "$SCRIPT_PATH" ]]; then
    echo "❌ 未找到脚本：$SCRIPT_PATH"
    exit 1
fi

resolve_teleop_port() {
    local default_port="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBJKECV-if00-port0"
    local teleop_port="${RIGHT_TELEOP_PORT:-${FRANKA_TELEOP_PORT:-${TELEOP_PORT:-}}}"

    if [[ -z "$teleop_port" && -e "$default_port" ]]; then
        teleop_port="$default_port"
    fi

    if [[ -z "$teleop_port" ]]; then
        local ftdi_ports=()
        if [[ -d /dev/serial/by-id ]]; then
            mapfile -t ftdi_ports < <(find /dev/serial/by-id -maxdepth 1 -type l -name 'usb-FTDI_USB__-__Serial_Converter_*' | sort)
        fi
        if [[ "${#ftdi_ports[@]}" -eq 1 ]]; then
            teleop_port="${ftdi_ports[0]}"
        else
            echo "❌ 未指定 RIGHT_TELEOP_PORT/FRANKA_TELEOP_PORT，且找到 ${#ftdi_ports[@]} 个 FTDI 串口"
            printf '  %s\n' "${ftdi_ports[@]}"
            exit 1
        fi
    fi

    if [[ ! -e "$teleop_port" ]]; then
        echo "❌ 串口不存在：$teleop_port"
        exit 1
    fi

    printf '%s\n' "$teleop_port"
}

TELEOP_PORT_RESOLVED="$(resolve_teleop_port)"

echo ">>> 启动Gripper 客户端 ..."
echo ">>> 使用同构臂串口：$TELEOP_PORT_RESOLVED"
python3 "$SCRIPT_PATH" --agent=teleop --tele_port=6001 --teleop_port="$TELEOP_PORT_RESOLVED" "$@"
#如果要启用采集数据，需要在后面增加“--use_save_interface”
