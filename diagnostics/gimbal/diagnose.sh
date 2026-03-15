#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

BIN_DIR="${REPO_ROOT}/build/bin/diag/gimbal"
LINK_DIAG="${BIN_DIR}/gimbal_link_diag_test"
UI_TEST="${BIN_DIR}/gimbal_ui_test"
SERIAL_PROBE="${BIN_DIR}/gimbal_serial_probe"

ACTION="${1:-}"
shift || true

CONFIG="configs/standard3.yaml"
if [[ $# -gt 0 && "${1}" == *.yaml ]]; then
  CONFIG="${1}"
  shift
fi

usage() {
  cat <<'EOF'
Usage:
  diagnostics/gimbal/diagnose.sh <action> [config.yaml] [extra args...]

Actions:
  quick          Link quick check (send on, 3s)
  rxonly         RX-only check (send off, 3s)
  proto          Strict protocol check (rx-only + require-rx)
  probe          Byte-stream probe summary (no protocol parse)
  probe-raw      Byte-stream hex sample (short raw dump)
  scan           Scan common serial ports with link diag
  snapshot       One-shot read snapshot (dump-once)
  watch          Continuous read mode (nogui)
  control        Interactive control mode
  script-control Scripted control mode (5s, no-input)
  port-info      Show serial-by-id and udev info of com_port in config
  help           Show this message

Examples:
  diagnostics/gimbal/diagnose.sh quick
  diagnostics/gimbal/diagnose.sh proto
  diagnostics/gimbal/diagnose.sh probe-raw
  diagnostics/gimbal/diagnose.sh scan configs/standard3.yaml
  diagnostics/gimbal/diagnose.sh snapshot configs/standard3.yaml --wait-valid-ms=2500
EOF
}

ensure_bin() {
  local bin="$1"
  if [[ ! -x "${bin}" ]]; then
    echo "[diagnose] Missing binary: ${bin}"
    echo "[diagnose] Run: bash build.sh"
    exit 1
  fi
}

read_com_port() {
  local cfg="$1"
  sed -n 's/^com_port:[[:space:]]*"\{0,1\}\([^"#[:space:]]\+\)"\{0,1\}.*/\1/p' "${cfg}" | head -n 1
}

run_quick() {
  ensure_bin "${LINK_DIAG}"
  "${LINK_DIAG}" "${CONFIG}" --duration-ms=3000 --summary-ms=1000 "$@"
}

run_rxonly() {
  ensure_bin "${LINK_DIAG}"
  "${LINK_DIAG}" "${CONFIG}" --no-send --duration-ms=3000 --summary-ms=1000 "$@"
}

run_proto() {
  ensure_bin "${LINK_DIAG}"
  "${LINK_DIAG}" "${CONFIG}" --no-send --require-rx --duration-ms=2200 --summary-ms=1000 "$@"
}

run_probe() {
  ensure_bin "${SERIAL_PROBE}"
  "${SERIAL_PROBE}" "${CONFIG}" --duration-ms=3000 --summary-ms=1000 "$@"
}

run_probe_raw() {
  ensure_bin "${SERIAL_PROBE}"
  "${SERIAL_PROBE}" "${CONFIG}" --duration-ms=1200 --summary-ms=1200 --raw-log --hex-len=32 "$@"
}

run_scan() {
  ensure_bin "${LINK_DIAG}"
  local ports=()
  local p
  for p in /dev/ttyACM0 /dev/ttyACM1 /dev/ttyACM2 /dev/ttyUSB0 /dev/ttyUSB1 /dev/ttyUSB2 /dev/ttyS0; do
    [[ -e "${p}" ]] && ports+=("${p}")
  done
  local ports_csv
  if [[ ${#ports[@]} -eq 0 ]]; then
    ports_csv="/dev/ttyACM0,/dev/ttyACM1,/dev/ttyUSB0,/dev/ttyUSB1,/dev/ttyS0"
  else
    ports_csv="$(IFS=,; echo "${ports[*]}")"
  fi
  "${LINK_DIAG}" "${CONFIG}" --ports="${ports_csv}" --duration-ms=3000 --summary-ms=1000 "$@"
}

run_snapshot() {
  ensure_bin "${UI_TEST}"
  "${UI_TEST}" "${CONFIG}" --mode=read --dump-once --wait-valid-ms=1500 --nogui "$@"
}

run_watch() {
  ensure_bin "${UI_TEST}"
  "${UI_TEST}" "${CONFIG}" --mode=read --nogui "$@"
}

run_control() {
  ensure_bin "${UI_TEST}"
  "${UI_TEST}" "${CONFIG}" --mode=control "$@"
}

run_script_control() {
  ensure_bin "${UI_TEST}"
  "${UI_TEST}" "${CONFIG}" --mode=control --no-input --duration-ms=5000 \
    --yaw-deg=3 --pitch-deg=-1 --tracking=1 --fric-on=1 --fire-mode=1 "$@"
}

run_port_info() {
  echo "[diagnose] /dev/serial/by-id"
  ls -l /dev/serial/by-id 2>/dev/null || echo "  (no /dev/serial/by-id)"

  local com_port
  com_port="$(read_com_port "${CONFIG}")"
  if [[ -z "${com_port}" ]]; then
    echo "[diagnose] com_port not found in ${CONFIG}"
    return 0
  fi

  echo "[diagnose] com_port from ${CONFIG}: ${com_port}"
  if [[ -e "${com_port}" ]]; then
    udevadm info -a -n "${com_port}" | grep -E 'idVendor|idProduct|serial' || true
  else
    echo "[diagnose] ${com_port} not present"
  fi
}

if [[ -z "${ACTION}" ]]; then
  usage
  exit 0
fi

case "${ACTION}" in
  quick) run_quick "$@" ;;
  rxonly) run_rxonly "$@" ;;
  proto) run_proto "$@" ;;
  probe) run_probe "$@" ;;
  probe-raw) run_probe_raw "$@" ;;
  scan) run_scan "$@" ;;
  snapshot) run_snapshot "$@" ;;
  watch) run_watch "$@" ;;
  control) run_control "$@" ;;
  script-control) run_script_control "$@" ;;
  port-info) run_port_info "$@" ;;
  help|-h|--help) usage ;;
  *)
    echo "[diagnose] Unknown action: ${ACTION}"
    usage
    exit 2
    ;;
esac
