#!/usr/bin/env bash
set -euo pipefail
# Local right stack: the dual launcher invokes this on .86, not on .98.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GELLO_EEF_SIDE=right
exec bash "$ROOT/A_run_left_gello_eef.sh" "$@"
