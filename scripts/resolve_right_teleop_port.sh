#!/bin/bash
set -euo pipefail

DEFAULT_PRIMARY_PORT="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBJKECV-if00-port0"
KNOWN_ALTERNATE_PORT="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAUMOPA-if00-port0"
PRIMARY_PORT="${RIGHT_TELEOP_PRIMARY_PORT:-$DEFAULT_PRIMARY_PORT}"
EXPLICIT_PORT="${RIGHT_TELEOP_PORT:-${FRANKA_TELEOP_PORT:-${TELEOP_PORT:-}}}"

print_detected_ports() {
    local by_id_dir="${SERIAL_BY_ID_DIR:-/dev/serial/by-id}"
    local found=0

    echo ">>> 当前检测到的 FTDI 串口：" >&2
    if [[ -d "$by_id_dir" ]]; then
        local port
        for port in "$by_id_dir"/*FTDI*; do
            [[ -e "$port" ]] || continue
            echo "    $port -> $(readlink -f "$port" 2>/dev/null || true)" >&2
            found=1
        done
    fi
    if [[ "$found" -eq 0 ]]; then
        echo "    (无)" >&2
    fi
}

validate_port() {
    local port="$1"
    local source="$2"

    if [[ ! -e "$port" ]]; then
        echo "ERROR: $source 指定的右臂同构臂串口不存在：$port" >&2
        print_detected_ports
        return 1
    fi
    if [[ ! -r "$port" || ! -w "$port" ]]; then
        echo "ERROR: 当前用户对右臂同构臂串口没有读写权限：$port" >&2
        echo ">>> 设备权限：$(ls -l "$port" 2>/dev/null || true)" >&2
        echo ">>> 当前用户：$(id 2>/dev/null || true)" >&2
        return 1
    fi
    printf '%s\n' "$port"
}

if [[ -n "$EXPLICIT_PORT" ]]; then
    validate_port "$EXPLICIT_PORT" "环境变量"
    exit $?
fi

if [[ -e "$PRIMARY_PORT" ]]; then
    validate_port "$PRIMARY_PORT" "默认配置"
    exit $?
fi

echo "ERROR: 未找到右臂同构臂默认串口：$PRIMARY_PORT" >&2
print_detected_ports
if [[ -e "$KNOWN_ALTERNATE_PORT" ]]; then
    echo ">>> 检测到备用右臂 FTDI：$KNOWN_ALTERNATE_PORT" >&2
fi
echo "ERROR: 为避免使用错误的标定映射，拒绝自动选择其他 FTDI 设备。" >&2
echo "请确认硬件连接，或显式设置 RIGHT_TELEOP_PORT。" >&2
exit 1
