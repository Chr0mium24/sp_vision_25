#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

BUILD_DIR="build"
cmake -S cpp -B "${BUILD_DIR}"
cmake --build "${BUILD_DIR}" -j"$(nproc)"
