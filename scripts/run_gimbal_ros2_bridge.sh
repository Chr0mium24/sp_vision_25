#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_PATH="${1:-${ROOT_DIR}/configs/standard3.yaml}"
if [[ $# -gt 0 ]]; then
  shift
fi

BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
BRIDGE_BIN="${BUILD_DIR}/bin/diag/gimbal/gimbal_ros2_bridge"

if [[ ! -x "${BRIDGE_BIN}" ]]; then
  echo "[run_gimbal_ros2_bridge] bridge binary not found: ${BRIDGE_BIN}" >&2
  echo "[run_gimbal_ros2_bridge] build the project first, for example: ./build.sh" >&2
  exit 1
fi

if [[ -z "${ROS_DISTRO:-}" ]]; then
  echo "[run_gimbal_ros2_bridge] ROS2 environment does not look sourced. Continuing anyway." >&2
fi

exec "${BRIDGE_BIN}" "${CONFIG_PATH}" "$@"
