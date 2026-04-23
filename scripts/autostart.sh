#!/usr/bin/env bash
set -euo pipefail

sleep 5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs

screen \
    -L \
    -Logfile "logs/$(date "+%Y-%m-%d_%H-%M-%S").screenlog" \
    -d \
    -m \
    bash -lc "./scripts/watchdog.sh"
