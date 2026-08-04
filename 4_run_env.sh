#!/bin/bash
set -e

echo ">>> 激活 conda 环境 polymetis ..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate polymetis || { echo "激活失败，请确认 polymetis 环境存在"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/teleop/experiments/run_env.py"
PORT_RESOLVER="$SCRIPT_DIR/scripts/resolve_right_teleop_port.sh"

if [[ ! -f "$SCRIPT_PATH" ]]; then
    echo "未找到脚本：$SCRIPT_PATH"
    exit 1
fi

if [[ ! -x "$PORT_RESOLVER" ]]; then
    echo "未找到可执行的右臂同构臂串口解析器：$PORT_RESOLVER"
    exit 1
fi

TELEOP_PORT="$($PORT_RESOLVER)" || exit $?

echo ">>> 使用右臂同构臂串口：$TELEOP_PORT"
echo ">>> 启动右臂同构臂 Teleop 客户端 ..."
python3 "$SCRIPT_PATH" \
    --agent=teleop \
    --tele_port=6001 \
    --teleop_port="$TELEOP_PORT" \
    "$@"
