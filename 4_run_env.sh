#!/bin/bash
set -e

echo ">>> 激活 conda 环境 polymetis ..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate polymetis || { echo "激活失败，请确认 polymetis 环境存在"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/teleop/experiments/run_env.py"
TELEOP_PORT="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBJKECV-if00-port0"

if [[ ! -f "$SCRIPT_PATH" ]]; then
    echo "未找到脚本：$SCRIPT_PATH"
    exit 1
fi

if [[ ! -e "$TELEOP_PORT" ]]; then
    echo "未找到右臂同构臂串口：$TELEOP_PORT"
    exit 1
fi

echo ">>> 启动右臂遥操作客户端 ..."
python3 "$SCRIPT_PATH" \
    --agent=teleop \
    --tele_port=6001 \
    --teleop_port="$TELEOP_PORT" \
    "$@"
