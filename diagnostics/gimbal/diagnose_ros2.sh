#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ACTION="${1:-}"
shift || true

CONFIG="configs/standard3.yaml"
if [[ $# -gt 0 && "${1}" == *.yaml ]]; then
  CONFIG="${1}"
  shift
fi

DEFAULT_G2V="/gimbal_to_vision"
DEFAULT_V2G="/visionToGimbal"
DEFAULT_STOP_PAYLOAD='[165, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 114, 154]'

usage() {
  cat <<'EOF'
Usage:
  diagnostics/gimbal/diagnose_ros2.sh <action> [config.yaml] [extra args...]

Actions:
  topics        List ROS2 topics and highlight gimbal transport topics
  check         Show resolved topics and whether they currently exist
  info          Show topic type / publisher / subscriber count
  echo-rx       Echo one `/gimbal_to_vision` message
  hz-rx         Measure `/gimbal_to_vision` publish rate
  pub-stop      Publish one valid stop packet to `/visionToGimbal`
  pub-loop      Publish the same stop packet continuously (default 10 Hz)
  help          Show this message

Examples:
  diagnostics/gimbal/diagnose_ros2.sh topics
  diagnostics/gimbal/diagnose_ros2.sh check configs/standard3.yaml
  diagnostics/gimbal/diagnose_ros2.sh echo-rx configs/standard3.yaml
  diagnostics/gimbal/diagnose_ros2.sh hz-rx configs/standard3.yaml
  diagnostics/gimbal/diagnose_ros2.sh pub-stop configs/standard3.yaml
  diagnostics/gimbal/diagnose_ros2.sh pub-loop configs/standard3.yaml --rate=20
EOF
}

ensure_ros2() {
  if ! command -v ros2 >/dev/null 2>&1; then
    echo "[diagnose_ros2] ros2 command not found."
    echo "[diagnose_ros2] Source your ROS2 environment first."
    exit 1
  fi

  if [[ -z "${AMENT_PREFIX_PATH:-}" ]]; then
    echo "[diagnose_ros2] AMENT_PREFIX_PATH is empty; ROS2 may not be sourced." >&2
  fi
}

read_yaml_scalar() {
  local key="$1"
  local file="$2"
  sed -n "s/^${key}:[[:space:]]*\"\{0,1\}\([^\"#[:space:]]\+\)\"\{0,1\}.*/\1/p" "${file}" | head -n 1
}

resolve_topics() {
  local cfg="${1}"
  G2V_TOPIC="$(read_yaml_scalar gimbal_to_vision_topic "${cfg}")"
  V2G_TOPIC="$(read_yaml_scalar vision_to_gimbal_topic "${cfg}")"
  G2V_TOPIC="${G2V_TOPIC:-${DEFAULT_G2V}}"
  V2G_TOPIC="${V2G_TOPIC:-${DEFAULT_V2G}}"
}

topic_exists() {
  local topic="$1"
  ros2 topic list 2>/dev/null | grep -Fx -- "${topic}" >/dev/null 2>&1
}

run_topics() {
  ensure_ros2
  resolve_topics "${CONFIG}"
  echo "[diagnose_ros2] resolved topics:"
  echo "  rx: ${G2V_TOPIC}"
  echo "  tx: ${V2G_TOPIC}"
  echo
  ros2 topic list | grep -E "gimbal|vision|serial|^${G2V_TOPIC}$|^${V2G_TOPIC}$" || true
}

run_check() {
  ensure_ros2
  resolve_topics "${CONFIG}"
  echo "[diagnose_ros2] config: ${CONFIG}"
  echo "[diagnose_ros2] rx topic: ${G2V_TOPIC}"
  echo "[diagnose_ros2] tx topic: ${V2G_TOPIC}"
  echo
  if topic_exists "${G2V_TOPIC}"; then
    echo "[diagnose_ros2] found rx topic: ${G2V_TOPIC}"
  else
    echo "[diagnose_ros2] missing rx topic: ${G2V_TOPIC}"
  fi
  if topic_exists "${V2G_TOPIC}"; then
    echo "[diagnose_ros2] found tx topic: ${V2G_TOPIC}"
  else
    echo "[diagnose_ros2] missing tx topic: ${V2G_TOPIC}"
  fi
}

run_info() {
  ensure_ros2
  resolve_topics "${CONFIG}"
  echo "[diagnose_ros2] topic info: ${G2V_TOPIC}"
  ros2 topic info "${G2V_TOPIC}" || true
  echo
  echo "[diagnose_ros2] topic info: ${V2G_TOPIC}"
  ros2 topic info "${V2G_TOPIC}" || true
}

run_echo_rx() {
  ensure_ros2
  resolve_topics "${CONFIG}"
  ros2 topic echo --once "${G2V_TOPIC}" std_msgs/msg/UInt8MultiArray "$@"
}

run_hz_rx() {
  ensure_ros2
  resolve_topics "${CONFIG}"
  ros2 topic hz "${G2V_TOPIC}" "$@"
}

run_pub_stop() {
  ensure_ros2
  resolve_topics "${CONFIG}"
  echo "[diagnose_ros2] publish one stop packet to ${V2G_TOPIC}"
  echo "[diagnose_ros2] payload: ${DEFAULT_STOP_PAYLOAD}"
  ros2 topic pub --once "${V2G_TOPIC}" std_msgs/msg/UInt8MultiArray "{data: ${DEFAULT_STOP_PAYLOAD}}"
}

run_pub_loop() {
  ensure_ros2
  resolve_topics "${CONFIG}"
  local rate=10
  local passthrough=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --rate=*)
        rate="${1#--rate=}"
        shift
        ;;
      *)
        passthrough+=("$1")
        shift
        ;;
    esac
  done
  echo "[diagnose_ros2] publish stop packet to ${V2G_TOPIC} at ${rate} Hz"
  ros2 topic pub -r "${rate}" "${V2G_TOPIC}" std_msgs/msg/UInt8MultiArray \
    "{data: ${DEFAULT_STOP_PAYLOAD}}" "${passthrough[@]}"
}

if [[ -z "${ACTION}" ]]; then
  usage
  exit 0
fi

case "${ACTION}" in
  topics) run_topics "$@" ;;
  check) run_check "$@" ;;
  info) run_info "$@" ;;
  echo-rx) run_echo_rx "$@" ;;
  hz-rx) run_hz_rx "$@" ;;
  pub-stop) run_pub_stop "$@" ;;
  pub-loop) run_pub_loop "$@" ;;
  help|-h|--help) usage ;;
  *)
    echo "[diagnose_ros2] Unknown action: ${ACTION}"
    usage
    exit 2
    ;;
esac
