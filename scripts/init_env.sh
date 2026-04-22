#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/init_env.sh [options]

Prepare the project environment for local development.

Options:
  --no-apt        Skip apt package installation/checks.
  --no-build      Skip the C++ build step.
  --no-tests      Skip the Python test step.
  --dry-run       Print the actions without executing them.
  -h, --help      Show this help.

Notes:
  - System packages are optional because they require sudo.
  - OpenVINO, MindVision SDK, and HikRobot SDK are external dependencies and
    cannot be installed automatically here.
EOF
}

run() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[dry-run] %q' "$1"
    shift
    for arg in "$@"; do
      printf ' %q' "$arg"
    done
    printf '\n'
    return 0
  fi
  "$@"
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

APT_PACKAGES=(
  git
  g++
  cmake
  can-utils
  libopencv-dev
  libfmt-dev
  libeigen3-dev
  libspdlog-dev
  libyaml-cpp-dev
  libusb-1.0-0-dev
  nlohmann-json3-dev
  openssh-server
  screen
)

INSTALL_APT=1
RUN_BUILD=1
RUN_TESTS=1
DRY_RUN=0

while (($#)); do
  case "$1" in
    --no-apt)
      INSTALL_APT=0
      ;;
    --no-build)
      RUN_BUILD=0
      ;;
    --no-tests)
      RUN_TESTS=0
      ;;
    --dry-run)
      DRY_RUN=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "[init_env] repo_root=${REPO_ROOT}"

if [[ "${INSTALL_APT}" == "1" ]]; then
  echo "[init_env] checking system packages"
  if have_cmd apt-get && have_cmd sudo; then
    run sudo apt-get update
    run sudo apt-get install -y "${APT_PACKAGES[@]}"
  else
    echo "[init_env] apt-get/sudo not available, skip system package install"
    echo "[init_env] required packages: ${APT_PACKAGES[*]}"
  fi
fi

if ! have_cmd uv; then
  echo "[init_env] uv not found"
  echo "[init_env] install uv first, then re-run this script"
  exit 1
fi

echo "[init_env] syncing python environment"
run env UV_CACHE_DIR=/tmp/uv-cache uv sync

if [[ "${RUN_TESTS}" == "1" ]]; then
  echo "[init_env] running python tests"
  run env UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q
fi

if [[ "${RUN_BUILD}" == "1" ]]; then
  echo "[init_env] building C++ targets"
  run bash ./build.sh
fi

echo "[init_env] done"
