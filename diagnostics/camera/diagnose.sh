#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

BIN_DIR="${REPO_ROOT}/build/bin/tests/camera"
CAMERA_TEST="${BIN_DIR}/camera_test"
CAMERA_DETECT_TEST="${BIN_DIR}/camera_detect_test"
CAMERA_WINDOW_TEST="${BIN_DIR}/camera_window_test"
CAMERA_SAVE_TEST="${BIN_DIR}/camera_save_test"
USBCAMERA_TEST="${BIN_DIR}/usbcamera_test"
USBCAMERA_DETECT_TEST="${BIN_DIR}/usbcamera_detect_test"
CAMERA_THREAD_TEST="${BIN_DIR}/camera_thread_test"
HANDEYE_TEST="${BIN_DIR}/handeye_test"

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
  diagnostics/camera/diagnose.sh <action> [config.yaml] [extra args...]

Actions:
  info         Show /dev/video* and v4l2 device info
  list         Show camera test binaries status
  release      Release Hik camera occupancy (docker/ros2/process/fuser)
  tune         Interactive exposure/gain tuning + window reload
  quick        Basic camera fps test (camera_test)
  detect       Camera + detector integration test (camera_detect_test)
  window       Camera window preview (camera_window_test)
  save         Save captured images by key(s) (camera_save_test)
  usb          USB camera basic test (usbcamera_test)
  usb-detect   USB camera + detector test (usbcamera_detect_test)
  thread       Multi-thread camera/detect test (camera_thread_test)
  handeye      Handeye projection test (handeye_test)
  help         Show this message

Examples:
  diagnostics/camera/diagnose.sh info
  sudo diagnostics/camera/diagnose.sh release
  sudo diagnostics/camera/diagnose.sh release --vidpid=2bdf:0001 --force
  diagnostics/camera/diagnose.sh tune configs/standard3.yaml --scale=0.7
  diagnostics/camera/diagnose.sh quick configs/standard3.yaml
  diagnostics/camera/diagnose.sh window configs/standard3.yaml --scale=0.7
  diagnostics/camera/diagnose.sh save configs/standard3.yaml --output-folder=assets/camera_captures
  diagnostics/camera/diagnose.sh usb configs/sentry.yaml --name=video0 -d
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

run_info() {
  echo "[camera] /dev/video*"
  ls -l /dev/video* 2>/dev/null || echo "  (no /dev/video*)"
  if command -v v4l2-ctl >/dev/null 2>&1; then
    echo "[camera] v4l2-ctl --list-devices"
    v4l2-ctl --list-devices || true
  else
    echo "[camera] v4l2-ctl not found (sudo apt install v4l-utils)"
  fi
}

run_list() {
  local bins=(
    "${CAMERA_TEST}"
    "${CAMERA_DETECT_TEST}"
    "${CAMERA_WINDOW_TEST}"
    "${CAMERA_SAVE_TEST}"
    "${USBCAMERA_TEST}"
    "${USBCAMERA_DETECT_TEST}"
    "${CAMERA_THREAD_TEST}"
    "${HANDEYE_TEST}"
  )
  for b in "${bins[@]}"; do
    if [[ -x "${b}" ]]; then
      echo "[ok] ${b}"
    else
      echo "[missing] ${b}"
    fi
  done
}

read_yaml_value() {
  local cfg="$1"
  local key="$2"
  sed -n "s/^[[:space:]]*${key}:[[:space:]]*\"\\{0,1\\}\\([^\"#[:space:]]\\+\\)\"\\{0,1\\}.*/\\1/p" "${cfg}" | head -n 1
}

print_camera_config() {
  local cfg="$1"
  local camera_name exposure_ms gain vid_pid
  camera_name="$(read_yaml_value "${cfg}" "camera_name")"
  exposure_ms="$(read_yaml_value "${cfg}" "exposure_ms")"
  gain="$(read_yaml_value "${cfg}" "gain")"
  vid_pid="$(read_yaml_value "${cfg}" "vid_pid")"

  echo "[camera] config: ${cfg}"
  echo "[camera] camera_name=${camera_name:-<missing>} exposure_ms=${exposure_ms:-<missing>} gain=${gain:-<missing>} vid_pid=${vid_pid:-<missing>}"
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

run_release() {
  if [[ "${EUID}" -ne 0 ]]; then
    echo "[release] Please run with sudo."
    echo "  sudo diagnostics/camera/diagnose.sh release"
    exit 1
  fi

  local vidpid="2bdf:0001"
  local force=false
  local arg
  for arg in "$@"; do
    case "${arg}" in
      --vidpid=*)
        vidpid="${arg#*=}"
        ;;
      --force)
        force=true
        ;;
      *)
        ;;
    esac
  done

  echo "[release] target vid:pid=${vidpid}"

  local matches
  matches="$(lsusb | grep -i "${vidpid}" || true)"
  if [[ -z "${matches}" ]]; then
    echo "[release] No USB device matched ${vidpid} in lsusb."
  else
    echo "[release] matched devices:"
    echo "${matches}"
  fi

  local usb_nodes=()
  while IFS= read -r line; do
    [[ -z "${line}" ]] && continue
    local bus dev
    bus="$(echo "${line}" | awk '{print $2}')"
    dev="$(echo "${line}" | awk '{print $4}' | tr -d ':')"
    usb_nodes+=("/dev/bus/usb/${bus}/${dev}")
  done <<< "${matches}"

  if command -v docker >/dev/null 2>&1; then
    local docker_rows docker_ids
    docker_rows="$(docker ps --format '{{.ID}} {{.Names}} {{.Image}} {{.Command}}' | grep -E 'rm_bringup|foxglove|ros2|camera_detector' || true)"
    if [[ -n "${docker_rows}" ]]; then
      echo "[release] stopping docker containers:"
      echo "${docker_rows}"
      docker_ids="$(echo "${docker_rows}" | awk '{print $1}' | sort -u)"
      local id
      for id in ${docker_ids}; do
        docker update --restart=no "${id}" >/dev/null 2>&1 || true
      done
      docker stop ${docker_ids} >/dev/null 2>&1 || docker kill ${docker_ids} >/dev/null 2>&1 || true
    else
      echo "[release] no matching running docker container found."
    fi
  else
    echo "[release] docker not found, skip container stop."
  fi

  local patterns=(
    "component_container_mt"
    "ros2 launch rm_bringup"
    "ros2 launch foxglove_bridge"
    "rm_serial_driver_node"
    "armor_solver_node"
  )

  local pat pids
  for pat in "${patterns[@]}"; do
    pids="$(pgrep -f "${pat}" || true)"
    [[ -n "${pids}" ]] && kill ${pids} >/dev/null 2>&1 || true
  done

  sleep 1

  for pat in "${patterns[@]}"; do
    pids="$(pgrep -f "${pat}" || true)"
    [[ -n "${pids}" ]] && kill -9 ${pids} >/dev/null 2>&1 || true
  done

  local node
  for node in "${usb_nodes[@]}"; do
    [[ -e "${node}" ]] || continue
    echo "[release] usb holder before: ${node}"
    fuser -v "${node}" 2>/dev/null || true
    if ${force}; then
      fuser -k -9 "${node}" >/dev/null 2>&1 || true
    else
      fuser -k "${node}" >/dev/null 2>&1 || true
    fi
  done

  echo "[release] remaining related processes:"
  pgrep -af "component_container_mt|rm_bringup|ros2 launch|foxglove_bridge|rm_serial_driver_node|armor_solver_node|containerd-shim" || echo "  (none)"

  if [[ ${#usb_nodes[@]} -eq 0 ]]; then
    echo "[release] no usb node derived from lsusb ${vidpid}."
  else
    for node in "${usb_nodes[@]}"; do
      [[ -e "${node}" ]] || continue
      echo "[release] usb holder after: ${node}"
      if ! fuser -v "${node}" 2>/dev/null; then
        echo "  (free)"
      fi
    done
  fi
}

run_tune() {
  ensure_bin "${CAMERA_WINDOW_TEST}"

  local scale="0.5"
  local quiet=true
  local arg
  for arg in "$@"; do
    case "${arg}" in
      --scale=*)
        scale="${arg#*=}"
        ;;
      --show-log)
        quiet=false
        ;;
      *)
        ;;
    esac
  done

  local exposure gain
  exposure="$(read_yaml_value "${CONFIG}" "exposure_ms")"
  gain="$(read_yaml_value "${CONFIG}" "gain")"
  if ! is_number "${exposure}"; then exposure="2.5"; fi
  if ! is_number "${gain}"; then gain="16.9"; fi

  local win_pid=""
  stop_window_process() {
    if [[ -n "${win_pid}" ]] && kill -0 "${win_pid}" >/dev/null 2>&1; then
      kill "${win_pid}" >/dev/null 2>&1 || true
      wait "${win_pid}" >/dev/null 2>&1 || true
    fi
    win_pid=""
  }

  start_window_process() {
    stop_window_process
    echo "[tune] reload window: exposure_ms=${exposure} gain=${gain} scale=${scale}"
    if ${quiet}; then
      "${CAMERA_WINDOW_TEST}" "${CONFIG}" "--scale=${scale}" >/dev/null 2>&1 &
    else
      "${CAMERA_WINDOW_TEST}" "${CONFIG}" "--scale=${scale}" &
    fi
    win_pid="$!"
    sleep 0.2
  }

  apply_config() {
    set_yaml_numeric_first "${CONFIG}" "exposure_ms" "${exposure}"
    set_yaml_numeric_first "${CONFIG}" "gain" "${gain}"
    print_camera_config "${CONFIG}"
  }

  print_camera_config "${CONFIG}"
  echo "[tune] commands:"
  echo "  e [num]: set exposure_ms (no num -> prompt)"
  echo "  g [num]: set gain (no num -> prompt)"
  echo "  r: reload window, p: print current values, q: quit"

  start_window_process

  while true; do
    printf "tune> "
    IFS= read -r line || break

    case "${line}" in
      q)
        break
        ;;
      r)
        start_window_process
        ;;
      p)
        print_camera_config "${CONFIG}"
        ;;
      e|e\ *)
        local val_e=""
        if [[ "${line}" == "e" ]]; then
          printf "exposure_ms> "
          IFS= read -r val_e || break
        else
          val_e="${line#e }"
        fi
        if is_number "${val_e}"; then
          exposure="$(float_non_negative "${val_e}")"
          apply_config
          start_window_process
        else
          echo "[tune] invalid exposure value: ${val_e}"
        fi
        ;;
      g|g\ *)
        local val_g=""
        if [[ "${line}" == "g" ]]; then
          printf "gain> "
          IFS= read -r val_g || break
        else
          val_g="${line#g }"
        fi
        if is_number "${val_g}"; then
          gain="$(float_non_negative "${val_g}")"
          apply_config
          start_window_process
        else
          echo "[tune] invalid gain value: ${val_g}"
        fi
        ;;
      "")
        ;;
      *)
        echo "[tune] unknown command: ${line} (supported: r p q e g)"
        ;;
    esac
  done

  stop_window_process
  echo "[tune] exit."
}

run_quick() {
  ensure_bin "${CAMERA_TEST}"
  "${CAMERA_TEST}" "--config-path=${CONFIG}" "$@"
}

run_detect() {
  ensure_bin "${CAMERA_DETECT_TEST}"
  "${CAMERA_DETECT_TEST}" "${CONFIG}" "$@"
}

run_window() {
  ensure_bin "${CAMERA_WINDOW_TEST}"
  "${CAMERA_WINDOW_TEST}" "${CONFIG}" "$@"
}

run_save() {
  ensure_bin "${CAMERA_SAVE_TEST}"
  "${CAMERA_SAVE_TEST}" "${CONFIG}" "$@"
}

run_usb() {
  ensure_bin "${USBCAMERA_TEST}"
  "${USBCAMERA_TEST}" "${CONFIG}" "$@"
}

run_usb_detect() {
  ensure_bin "${USBCAMERA_DETECT_TEST}"
  "${USBCAMERA_DETECT_TEST}" "${CONFIG}" "$@"
}

run_thread() {
  ensure_bin "${CAMERA_THREAD_TEST}"
  "${CAMERA_THREAD_TEST}" "${CONFIG}" "$@"
}

run_handeye() {
  ensure_bin "${HANDEYE_TEST}"
  local handeye_config="${CONFIG}"
  "${HANDEYE_TEST}" "--config-path=${handeye_config}" "$@"
}

if [[ -z "${ACTION}" ]]; then
  usage
  exit 0
fi

case "${ACTION}" in
  info) run_info "$@" ;;
  list) run_list "$@" ;;
  release) run_release "$@" ;;
  tune) run_tune "$@" ;;
  quick) run_quick "$@" ;;
  detect) run_detect "$@" ;;
  window) run_window "$@" ;;
  save) run_save "$@" ;;
  usb) run_usb "$@" ;;
  usb-detect) run_usb_detect "$@" ;;
  thread) run_thread "$@" ;;
  handeye) run_handeye "$@" ;;
  help|-h|--help) usage ;;
  *)
    echo "[diagnose] Unknown action: ${ACTION}"
    usage
    exit 2
    ;;
esac
