#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$ROOT/.conda/bin/python"
COMMAND="${1:-}"
SERIAL_PORT="${SERIAL_PORT:-}"

if [[ "$COMMAND" != "MCUID" ]]; then
  echo "Only the non-motion MCUID query is permitted by this Finger SDK." >&2
  exit 2
fi

if [[ -z "$SERIAL_PORT" || ! -e "$SERIAL_PORT" ]]; then
  echo "Set SERIAL_PORT to the explicitly confirmed Controller serial node." >&2
  exit 2
fi

if command -v fuser >/dev/null && fuser "$SERIAL_PORT" >/dev/null 2>&1; then
  echo "Refusing query: serial port is already in use: $SERIAL_PORT" >&2
  fuser -v "$SERIAL_PORT" >&2 || true
  exit 3
fi

echo "QUERY=$COMMAND"
echo "SERIAL_PORT=$SERIAL_PORT"

timeout 8s "$PYTHON" "$ROOT/scripts/doctor/04_query_identity_strict.py" \
  --serial-port "$SERIAL_PORT" \
  --command "$COMMAND"
