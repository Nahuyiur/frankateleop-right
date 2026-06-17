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

SCRIPT_PATH="$POLY_ROOT/experiments/launch_nodes.py"
if [[ ! -f "$SCRIPT_PATH" ]]; then
    echo "❌ 未找到脚本：$SCRIPT_PATH"
    exit 1
fi

echo ">>> 启动Robot Node ..."
echo ">>> move_to_initial_pose=${FRANKA_MOVE_TO_INITIAL_POSE:-1}"
python3 "$SCRIPT_PATH" --robot=fr3 --tele_port=6001 --robot_port=50051 --gripper_port=50053 --robot_ip=127.0.0.1 "$@"
#robot_ip是直接连接机器人的主机的IP，如果是本机直连机器人，可以用127.0.0.1代替
