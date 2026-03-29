#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

BUILD_DIR="build"

# Common typo fallback: rmv_fastrtps_cpp -> rmw_fastrtps_cpp
if [[ "${RMW_IMPLEMENTATION:-}" == "rmv_fastrtps_cpp" ]]; then
  echo "[build.sh] Detected invalid RMW_IMPLEMENTATION=rmv_fastrtps_cpp; using rmw_fastrtps_cpp."
  export RMW_IMPLEMENTATION="rmw_fastrtps_cpp"
fi

cmake_args=(
  -U RMW_IMPLEMENTATION
)

if [[ -n "${RMW_IMPLEMENTATION:-}" ]]; then
  cmake_args+=("-DRMW_IMPLEMENTATION=${RMW_IMPLEMENTATION}")
fi

cmake -S . -B "${BUILD_DIR}" "${cmake_args[@]}"
cmake --build "${BUILD_DIR}" -j"$(nproc)"
