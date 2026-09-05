#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# Fixed switch topology: Host PC B (172.16.0.3) -> FR3 B (172.16.0.4).
# The value is overridable for maintenance, but never discovered by scanning.
FRANKA_RIGHT_ROBOT_IP="${FRANKA_RIGHT_ROBOT_IP:-${FRANKA_ROBOT_IP:-172.16.0.4}}"
export FRANKA_RIGHT_ROBOT_IP

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

install_sudo_wrapper() {
    # Keep the explicit credential wrapper active even when another process
    # has left a cached sudo timestamp.  launch_robot.py invokes sudo directly
    # and otherwise may hit a non-interactive SSH shell without a password.
    if [[ -z "${FRANKA_SUDO_PASSWORD_FILE:-}" ]] && command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
        return 0
    fi
    if [[ -z "${FRANKA_SUDO_PASSWORD_FILE:-}" && -n "${FRANKA_SUDO_PASSWORD:-}" ]]; then
        FRANKA_SUDO_PASSWORD_FILE="$(mktemp /tmp/franka-sudo-password.XXXXXX)"
        chmod 600 "$FRANKA_SUDO_PASSWORD_FILE"
        printf '%s\n' "$FRANKA_SUDO_PASSWORD" > "$FRANKA_SUDO_PASSWORD_FILE"
        export FRANKA_SUDO_PASSWORD_FILE
        unset FRANKA_SUDO_PASSWORD
    fi
    [[ -z "${FRANKA_SUDO_PASSWORD_FILE:-}" ]] && return 0
    validate_sudo_password_file "$FRANKA_SUDO_PASSWORD_FILE"
    local sudo_bin
    sudo_bin="$(command -v sudo || true)"
    [[ -z "$sudo_bin" ]] && return 0

    local wrapper_dir
    wrapper_dir="$(mktemp -d /tmp/franka-sudo-wrapper.XXXXXX)"
    umask 077
    cat > "$wrapper_dir/sudo" <<EOF
#!/usr/bin/env bash
mode="\$(stat -c '%a' "\${FRANKA_SUDO_PASSWORD_FILE}")"
owner="\$(stat -c '%u' "\${FRANKA_SUDO_PASSWORD_FILE}")"
[[ "\$owner" == "\$(id -u)" && ( "\$mode" == "600" || "\$mode" == "400" ) ]] || exit 1
IFS= read -r sudo_password < "\${FRANKA_SUDO_PASSWORD_FILE}"
printf '%s\n' "\$sudo_password" | "$sudo_bin" -S -p '' "\$@"
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
POLYMETIS_TORCH_ISOLATION="$POLY_ROOT/polymetis/build/torch_isolation"
POLYMETIS_TORCH_LIB="$(python -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parent / "lib")')"
export LD_LIBRARY_PATH="$POLYMETIS_TORCH_ISOLATION/pinocchio_isolation:$POLYMETIS_TORCH_ISOLATION:$POLYMETIS_TORCH_LIB:$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "📝 调试信息：POLY_ROOT=$POLY_ROOT"

WORK_DIR="$POLY_ROOT/polymetis/python/polymetis"
echo "📝 调试信息：WORK_DIR=$WORK_DIR"

if [[ ! -d "$WORK_DIR/conf" ]]; then
    echo "❌ 配置目录 $WORK_DIR/conf 不存在"
    exit 1
fi
echo "✅ 配置目录检查通过"

echo ">>> 启动Franka 客户端 ..."
echo "    FR3 B / right Desk: $FRANKA_RIGHT_ROBOT_IP:1337 (FCI)"
cd "$WORK_DIR" || { echo "❌ 无法进入工作目录 $WORK_DIR"; exit 1; }
echo "📝 调试信息：当前工作目录=$(pwd)"

install_sudo_wrapper
launch_robot.py --config-name=launch_right_robot
