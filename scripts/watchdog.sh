#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

APP_PATH="${APP_PATH:-./build/bin/apps/standard/standard}"
CONFIG_PATH="${CONFIG_PATH:-configs/standard3.yaml}"
RESTART_DELAY_SECONDS="${RESTART_DELAY_SECONDS:-2}"

mkdir -p logs

while true; do
    echo "[watchdog] launching ${APP_PATH} ${CONFIG_PATH}"
    "${APP_PATH}" "${CONFIG_PATH}"
    exit_code=$?
    echo "[watchdog] process exited with code ${exit_code}, restarting in ${RESTART_DELAY_SECONDS}s"
    sleep "${RESTART_DELAY_SECONDS}"
done
