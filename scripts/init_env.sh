#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOT'
Usage: scripts/init_env.sh [options]

Prepare the project environment for local development.

Options:
  --no-apt                 Skip apt package installation/checks.
  --openvino-package PKG    OpenVINO package to install from Intel APT repo
                           (default: openvino).
  --mindvision-installer PATH
                           Install MindVision SDK from a local installer
                           script, executable, or .deb.
  --hikrobot-installer PATH
                           Install HikRobot SDK from a local installer
                           script, executable, or .deb.
  --no-build               Skip the C++ build step.
  --no-tests               Skip the Python test step.
  --dry-run                Print the actions without executing them.
  -h, --help               Show this help.

Notes:
  - System packages are optional because they require sudo.
  - OpenVINO is installed automatically through Intel's APT repository.
  - MindVision/HikRobot SDKs must be provided as local installers because the
    project does not ship vendor download artifacts.
EOT
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

install_openvino_repo() {
  local version_id=""
  if [[ -f /etc/os-release ]]; then
    # shellcheck disable=SC1091
    source /etc/os-release
    version_id="${VERSION_ID:-}"
  fi

  local repo_name=""
  case "${version_id}" in
    24.*)
      repo_name="ubuntu24"
      ;;
    22.*)
      repo_name="ubuntu22"
      ;;
    20.*)
      repo_name="ubuntu20"
      ;;
    *)
      echo "[init_env] unsupported or unknown Ubuntu version: ${version_id:-unknown}"
      echo "[init_env] skip automatic OpenVINO repo setup"
      return 1
      ;;
  esac

  echo "[init_env] configuring Intel OpenVINO APT repo (${repo_name})"
  run sudo apt-get install -y --no-install-recommends gnupg wget ca-certificates
  run env bash -lc 'wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | sudo gpg --dearmor --yes --output /etc/apt/trusted.gpg.d/intel.gpg'
  run env bash -lc "echo 'deb https://apt.repos.intel.com/openvino ${repo_name} main' | sudo tee /etc/apt/sources.list.d/intel-openvino.list >/dev/null"
  run sudo apt-get update
}

install_local_sdk() {
  local label="$1"
  local installer="$2"
  if [[ -z "${installer}" ]]; then
    echo "[init_env] ${label} installer not provided, skip"
    return 0
  fi
  if [[ ! -e "${installer}" ]]; then
    echo "[init_env] ${label} installer not found: ${installer}"
    return 1
  fi

  echo "[init_env] installing ${label} from ${installer}"
  case "${installer}" in
    *.deb)
      run sudo apt-get install -y "${installer}"
      ;;
    *.sh)
      run bash "${installer}"
      ;;
    *)
      if [[ -x "${installer}" ]]; then
        run "${installer}"
      else
        echo "[init_env] unsupported ${label} installer format: ${installer}"
        echo "[init_env] expected .deb, .sh, or executable installer"
        return 1
      fi
      ;;
  esac
}

APT_PACKAGES=(
  git
  g++
  cmake
  can-utils
  ca-certificates
  gnupg
  wget
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
OPENVINO_PACKAGE="${OPENVINO_PACKAGE:-openvino}"
MINDVISION_INSTALLER="${MINDVISION_INSTALLER:-}"
HIKROBOT_INSTALLER="${HIKROBOT_INSTALLER:-}"

while (($#)); do
  case "$1" in
    --no-apt)
      INSTALL_APT=0
      ;;
    --openvino-package)
      shift
      [[ $# -gt 0 ]] || { echo "--openvino-package expects a value" >&2; exit 1; }
      OPENVINO_PACKAGE="$1"
      ;;
    --mindvision-installer)
      shift
      [[ $# -gt 0 ]] || { echo "--mindvision-installer expects a path" >&2; exit 1; }
      MINDVISION_INSTALLER="$1"
      ;;
    --hikrobot-installer)
      shift
      [[ $# -gt 0 ]] || { echo "--hikrobot-installer expects a path" >&2; exit 1; }
      HIKROBOT_INSTALLER="$1"
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
    if install_openvino_repo; then
      echo "[init_env] installing OpenVINO package: ${OPENVINO_PACKAGE}"
      run sudo apt-get install -y --no-install-recommends "${OPENVINO_PACKAGE}"
    fi
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
run env UV_CACHE_DIR=/tmp/uv-cache uv --project python sync

install_local_sdk "MindVision SDK" "${MINDVISION_INSTALLER}"
install_local_sdk "HikRobot SDK" "${HIKROBOT_INSTALLER}"

if [[ "${RUN_BUILD}" == "1" ]]; then
  echo "[init_env] building C++ targets"
  run bash ./build.sh
fi

if [[ "${RUN_TESTS}" == "1" ]]; then
  echo "[init_env] running python tests"
  run env UV_CACHE_DIR=/tmp/uv-cache uv --project python run pytest python/tests -q
fi

echo "[init_env] done"
