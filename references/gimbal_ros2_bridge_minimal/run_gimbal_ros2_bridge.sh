#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
BIN="${BUILD_DIR}/gimbal_ros2_bridge_minimal"
CONFIG_PATH="${1:-${SCRIPT_DIR}/bridge.yaml}"

if [[ $# -gt 0 ]]; then
  shift
fi

if [[ ! -x "${BIN}" ]]; then
  echo "[gimbal_ros2_bridge_minimal] binary not found: ${BIN}" >&2
  echo "[gimbal_ros2_bridge_minimal] run ./build.sh first" >&2
  exit 1
fi

exec "${BIN}" "${CONFIG_PATH}" "$@"
