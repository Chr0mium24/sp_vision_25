#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DIAG_DIR="${REPO_ROOT}/build/bin/diag/auto_aim"
TEST_AIM_DIR="${REPO_ROOT}/build/bin/tests/auto_aim"
TEST_BUFF_DIR="${REPO_ROOT}/build/bin/tests/auto_buff"

AUTO_AIM_UI_TEST="${DIAG_DIR}/auto_aim_ui_test"
AUTO_AIM_UI_TUNE="${DIAG_DIR}/auto_aim_ui_tune"
AUTO_AIM_TEST="${TEST_AIM_DIR}/auto_aim_test"
DETECTOR_VIDEO_TEST="${TEST_AIM_DIR}/detector_video_test"
POWER_RUNE_TEST="${TEST_BUFF_DIR}/auto_power_rune_test"
AUTO_BUFF_DEBUG="${REPO_ROOT}/build/auto_buff_debug"
AUTO_BUFF_DEBUG_MPC="${REPO_ROOT}/build/auto_buff_debug_mpc"

ACTION="${1:-}"
if [[ -n "${ACTION}" ]]; then
  shift
fi

CONFIG="configs/standard3.yaml"
if [[ $# -gt 0 && "${1}" == *.yaml ]]; then
  CONFIG="${1}"
  shift
fi

usage() {
  cat <<'EOF'
Usage:
  diagnostics/auto_aim/diagnose.sh <action> [config.yaml] [extra args...]

Actions:
  list            Show auto_aim/power_rune diagnose binary status
  armor-box       Online armor detect + draw boxes (GUI)
  armor-rec       Online armor recognition status (TUI, no GUI)
  armor-tune      Online armor tuning + export yaml (GUI)
  armor-offline   Offline armor replay test (input-prefix -> <prefix>.avi/.txt)
  rune-box        Offline power rune detect + draw boxes (input-prefix)
  rune-rec        Alias of rune-box
  rune-tune       Tune power rune YAML params, then rerun rune-box
  rune-online     Online power rune debug (build/auto_buff_debug)
  rune-online-mpc Online power rune MPC debug (build/auto_buff_debug_mpc)
  help            Show this message

Examples:
  diagnostics/auto_aim/diagnose.sh list
  diagnostics/auto_aim/diagnose.sh armor-box configs/standard3.yaml
  diagnostics/auto_aim/diagnose.sh armor-rec configs/standard3.yaml
  diagnostics/auto_aim/diagnose.sh armor-tune configs/standard3.yaml
  diagnostics/auto_aim/diagnose.sh armor-offline configs/demo.yaml assets/demo/demo --start-index=0 --end-index=0
  diagnostics/auto_aim/diagnose.sh rune-box configs/sentry.yaml assets/demo/power_rune_demo --start-index=0 --end-index=0
  diagnostics/auto_aim/diagnose.sh rune-tune configs/sentry.yaml assets/demo/power_rune_demo --start-index=0 --end-index=0
  diagnostics/auto_aim/diagnose.sh rune-online configs/standard3.yaml
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

read_yaml_value() {
  local cfg="$1"
  local key="$2"
  sed -n "s/^[[:space:]]*${key}:[[:space:]]*\"\\{0,1\\}\\([^\"#[:space:]]\\+\\)\"\\{0,1\\}.*/\\1/p" "${cfg}" | head -n 1
}

is_number() {
  [[ "${1:-}" =~ ^-?[0-9]+([.][0-9]+)?$ ]]
}

float_non_negative() {
  awk -v v="$1" 'BEGIN { if (v < 0) v = 0; printf "%.6f", v }'
}

set_yaml_numeric_first() {
  local cfg="$1"
  local key="$2"
  local value="$3"
  local tmp="${cfg}.tmp.$$"

  awk -v key="${key}" -v value="${value}" '
    BEGIN { done = 0 }
    {
      if (!done && $0 ~ "^[[:space:]]*" key "[[:space:]]*:") {
        sub(":[[:space:]].*$", ": " value)
        done = 1
      }
      print
    }
    END {
      if (!done) print key ": " value
    }
  ' "${cfg}" > "${tmp}"

  mv "${tmp}" "${cfg}"
}

run_list() {
  local bins=(
    "${AUTO_AIM_UI_TEST}"
    "${AUTO_AIM_UI_TUNE}"
    "${AUTO_AIM_TEST}"
    "${DETECTOR_VIDEO_TEST}"
    "${POWER_RUNE_TEST}"
    "${AUTO_BUFF_DEBUG}"
    "${AUTO_BUFF_DEBUG_MPC}"
  )
  local b
  for b in "${bins[@]}"; do
    if [[ -x "${b}" ]]; then
      echo "[ok] ${b}"
    else
      echo "[missing] ${b}"
    fi
  done
}

has_show_arg() {
  local arg
  for arg in "$@"; do
    if [[ "${arg}" == --show=* ]] || [[ "${arg}" == "--show" ]] || [[ "${arg}" == "-s" ]]; then
      return 0
    fi
  done
  return 1
}

run_armor_box() {
  ensure_bin "${AUTO_AIM_UI_TEST}"
  if has_show_arg "$@"; then
    "${AUTO_AIM_UI_TEST}" "${CONFIG}" "$@"
  else
    "${AUTO_AIM_UI_TEST}" "${CONFIG}" --show=true "$@"
  fi
}

run_armor_rec() {
  ensure_bin "${AUTO_AIM_UI_TEST}"
  "${AUTO_AIM_UI_TEST}" "${CONFIG}" "$@"
}

run_armor_tune() {
  ensure_bin "${AUTO_AIM_UI_TUNE}"
  if has_show_arg "$@"; then
    "${AUTO_AIM_UI_TUNE}" "${CONFIG}" "$@"
  else
    "${AUTO_AIM_UI_TUNE}" "${CONFIG}" --show=true "$@"
  fi
}

run_armor_offline() {
  ensure_bin "${AUTO_AIM_TEST}"
  local input="assets/demo/demo"
  if [[ $# -gt 0 && "${1}" != --* ]]; then
    input="${1}"
    shift
  fi
  "${AUTO_AIM_TEST}" --config-path="${CONFIG}" "${input}" "$@"
}

run_rune_box() {
  ensure_bin "${POWER_RUNE_TEST}"
  local input="assets/demo/power_rune_demo"
  if [[ $# -gt 0 && "${1}" != --* ]]; then
    input="${1}"
    shift
  fi
  "${POWER_RUNE_TEST}" --config-path="${CONFIG}" "${input}" "$@"
}

run_rune_online() {
  ensure_bin "${AUTO_BUFF_DEBUG}"
  "${AUTO_BUFF_DEBUG}" "${CONFIG}" "$@"
}

run_rune_online_mpc() {
  ensure_bin "${AUTO_BUFF_DEBUG_MPC}"
  "${AUTO_BUFF_DEBUG_MPC}" "${CONFIG}" "$@"
}

run_rune_tune() {
  ensure_bin "${POWER_RUNE_TEST}"

  local input="assets/demo/power_rune_demo"
  if [[ $# -gt 0 && "${1}" != --* ]]; then
    input="${1}"
    shift
  fi
  local -a rune_args=("$@")

  local yaw pitch fire_gap predict_time
  yaw="$(read_yaml_value "${CONFIG}" "yaw_offset")"
  pitch="$(read_yaml_value "${CONFIG}" "pitch_offset")"
  fire_gap="$(read_yaml_value "${CONFIG}" "fire_gap_time")"
  predict_time="$(read_yaml_value "${CONFIG}" "predict_time")"

  if ! is_number "${yaw}"; then yaw="0"; fi
  if ! is_number "${pitch}"; then pitch="0"; fi
  if ! is_number "${fire_gap}"; then fire_gap="0.7"; fi
  if ! is_number "${predict_time}"; then predict_time="0.12"; fi

  apply_rune_config() {
    set_yaml_numeric_first "${CONFIG}" "yaw_offset" "${yaw}"
    set_yaml_numeric_first "${CONFIG}" "pitch_offset" "${pitch}"
    set_yaml_numeric_first "${CONFIG}" "fire_gap_time" "${fire_gap}"
    set_yaml_numeric_first "${CONFIG}" "predict_time" "${predict_time}"
  }

  print_rune_config() {
    echo "[rune-tune] config=${CONFIG}"
    echo "[rune-tune] yaw_offset=${yaw} pitch_offset=${pitch} fire_gap_time=${fire_gap} predict_time=${predict_time}"
  }

  run_preview() {
    echo "[rune-tune] run: ${POWER_RUNE_TEST} --config-path=${CONFIG} ${input} ${rune_args[*]}"
    "${POWER_RUNE_TEST}" --config-path="${CONFIG}" "${input}" "${rune_args[@]}"
  }

  print_rune_config
  echo "[rune-tune] commands:"
  echo "  y [num]: set yaw_offset(deg)"
  echo "  i [num]: set pitch_offset(deg)"
  echo "  f [num]: set fire_gap_time(s, >=0)"
  echo "  t [num]: set predict_time(s, >=0)"
  echo "  r: rerun power rune visualize"
  echo "  p: print current params"
  echo "  q: quit"

  while true; do
    printf "rune-tune> "
    IFS= read -r line || break
    case "${line}" in
      q)
        break
        ;;
      p)
        print_rune_config
        ;;
      r)
        run_preview
        ;;
      y|y\ *)
        local val_y=""
        if [[ "${line}" == "y" ]]; then
          printf "yaw_offset(deg)> "
          IFS= read -r val_y || break
        else
          val_y="${line#y }"
        fi
        if is_number "${val_y}"; then
          yaw="${val_y}"
          apply_rune_config
          print_rune_config
        else
          echo "[rune-tune] invalid yaw_offset: ${val_y}"
        fi
        ;;
      i|i\ *)
        local val_p=""
        if [[ "${line}" == "i" ]]; then
          printf "pitch_offset(deg)> "
          IFS= read -r val_p || break
        else
          val_p="${line#i }"
        fi
        if is_number "${val_p}"; then
          pitch="${val_p}"
          apply_rune_config
          print_rune_config
        else
          echo "[rune-tune] invalid pitch_offset: ${val_p}"
        fi
        ;;
      f|f\ *)
        local val_f=""
        if [[ "${line}" == "f" ]]; then
          printf "fire_gap_time(s)> "
          IFS= read -r val_f || break
        else
          val_f="${line#f }"
        fi
        if is_number "${val_f}"; then
          fire_gap="$(float_non_negative "${val_f}")"
          apply_rune_config
          print_rune_config
        else
          echo "[rune-tune] invalid fire_gap_time: ${val_f}"
        fi
        ;;
      t|t\ *)
        local val_t=""
        if [[ "${line}" == "t" ]]; then
          printf "predict_time(s)> "
          IFS= read -r val_t || break
        else
          val_t="${line#t }"
        fi
        if is_number "${val_t}"; then
          predict_time="$(float_non_negative "${val_t}")"
          apply_rune_config
          print_rune_config
        else
          echo "[rune-tune] invalid predict_time: ${val_t}"
        fi
        ;;
      "")
        ;;
      *)
        echo "[rune-tune] unknown command: ${line}"
        ;;
    esac
  done
}

if [[ -z "${ACTION}" ]]; then
  usage
  exit 0
fi

case "${ACTION}" in
  list) run_list "$@" ;;
  armor-box) run_armor_box "$@" ;;
  armor-rec) run_armor_rec "$@" ;;
  armor-tune) run_armor_tune "$@" ;;
  armor-offline) run_armor_offline "$@" ;;
  rune-box) run_rune_box "$@" ;;
  rune-rec) run_rune_box "$@" ;;
  rune-tune) run_rune_tune "$@" ;;
  rune-online) run_rune_online "$@" ;;
  rune-online-mpc) run_rune_online_mpc "$@" ;;
  help|-h|--help) usage ;;
  *)
    echo "[diagnose] Unknown action: ${ACTION}"
    usage
    exit 2
    ;;
esac
