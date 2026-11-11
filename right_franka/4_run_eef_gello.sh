#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GELLO_EEF_SIDE=right
exec bash "$ROOT/left_franka/4_run_eef_gello.sh" "$@"
