#!/bin/bash
set -e

echo ">>> 激活 conda 环境 polymetis ..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate polymetis || { echo "激活失败，请确认 polymetis 环境存在"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/teleop/experiments/launch_nodes.py"

if [[ ! -f "$SCRIPT_PATH" ]]; then
    echo "未找到脚本：$SCRIPT_PATH"
    exit 1
fi

echo ">>> 启动右臂 Robot Node ..."
python3 "$SCRIPT_PATH" \
    --robot=fr3 \
    --tele_port=6001 \
    --robot_port=50051 \
    --gripper_port=50053 \
    --robot_ip=127.0.0.1 \
    "$@"
