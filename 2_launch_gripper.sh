#!/bin/bash
set -e

echo ">>> 激活 conda 环境 polymetis ..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate polymetis || { echo "激活失败，请确认 polymetis 环境存在"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$SCRIPT_DIR/polymetis/polymetis/python/polymetis"

if [[ ! -d "$WORK_DIR/conf" ]]; then
    echo "配置目录 $WORK_DIR/conf 不存在"
    exit 1
fi

echo ">>> 清理旧 Gripper client ..."
sudo pkill -9 franka_hand_client || echo "未发现 franka_hand_client 进程或无需清理"

echo ">>> 启动右臂 Franka Hand 客户端 ..."
cd "$WORK_DIR"
python ../scripts/launch_gripper.py --config-name=launch_right_gripper
