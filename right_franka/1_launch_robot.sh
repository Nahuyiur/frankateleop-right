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
        if [[ -n "${FRANKA_SUDO_PASSWORD:-}" ]]; then
            printf '%s\n' "$FRANKA_SUDO_PASSWORD" | sudo -S -p '' "$@"
        else
            sudo "$@"
        fi
    else
        "$@"
    fi
}

install_sudo_wrapper() {
    [[ -z "${FRANKA_SUDO_PASSWORD:-}" ]] && return 0
    local sudo_bin
    sudo_bin="$(command -v sudo || true)"
    [[ -z "$sudo_bin" ]] && return 0

    local wrapper_dir
    wrapper_dir="$(mktemp -d /tmp/franka-sudo-wrapper.XXXXXX)"
    cat > "$wrapper_dir/sudo" <<EOF
#!/usr/bin/env bash
printf '%s\n' "\${FRANKA_SUDO_PASSWORD}" | "$sudo_bin" -S -p '' "\$@"
EOF
    chmod 700 "$wrapper_dir/sudo"
    export PATH="$wrapper_dir:$PATH"
}

echo ">>> 激活 conda 环境 polymetis ..."
source_conda
conda activate polymetis || { echo "❌ 激活失败"; exit 1; }
echo "✅ conda环境激活成功"

echo ">>> 清理占用50051端口的进程 ..."

# 检查lsof是否安装
if ! command -v lsof &> /dev/null; then
    echo "❌ 错误：未找到lsof命令，请先安装（例如：sudo apt install lsof）"
    exit 1
fi

echo "🔍 正在查找占用50051端口的进程..."
PID=$(sudo_run lsof -t -i:50051 || true)
echo "📝 调试信息：查找到的PID=$PID"

if [[ -n "$PID" ]]; then
    echo "发现占用50051端口的进程，PID: $PID，正在终止..."
    if sudo_run kill -9 "$PID"; then
        echo "✅ 进程$PID已成功终止"
    else
        echo "❌ 终止进程$PID失败"
        exit 1
    fi
else
    echo "⚠️ 未发现占用50051端口的进程，无需清理"
fi

echo "✅ 端口清理步骤完成"

POLY_ROOT="$REPO_ROOT/polymetis"
echo "📝 调试信息：POLY_ROOT=$POLY_ROOT"

WORK_DIR="$POLY_ROOT/polymetis/python/polymetis"
echo "📝 调试信息：WORK_DIR=$WORK_DIR"

if [[ ! -d "$WORK_DIR/conf" ]]; then
    echo "❌ 配置目录 $WORK_DIR/conf 不存在"
    exit 1
fi
echo "✅ 配置目录检查通过"

echo ">>> 启动Franka 客户端 ..."
cd "$WORK_DIR" || { echo "❌ 无法进入工作目录 $WORK_DIR"; exit 1; }
echo "📝 调试信息：当前工作目录=$(pwd)"

install_sudo_wrapper
launch_robot.py --config-name=launch_right_robot
