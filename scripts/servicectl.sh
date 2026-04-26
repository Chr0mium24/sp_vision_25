#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="${SERVICE_NAME:-sp-vision}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
UNIT_PATH="/etc/systemd/system/${SERVICE_NAME}.service"

APP_PATH_DEFAULT="${REPO_ROOT}/build/bin/apps/standard/standard"
CONFIG_PATH_DEFAULT="${REPO_ROOT}/configs/standard3.yaml"
APP_PATH="${APP_PATH:-${APP_PATH_DEFAULT}}"
CONFIG_PATH="${CONFIG_PATH:-${CONFIG_PATH_DEFAULT}}"
RESTART_DELAY_SECONDS="${RESTART_DELAY_SECONDS:-2}"

usage() {
    cat <<EOF
Usage: $0 [command]

Run without command to open the interactive menu.

Commands:
  install     Register and enable the systemd service
  uninstall   Stop, disable, and remove the systemd service
  status      Show registration and runtime status
  start       Start the service
  stop        Stop the service
  restart     Restart the service
  logs        Follow service logs

Environment:
  SERVICE_NAME             Service name, default: sp-vision
  SP_VISION_USER           Runtime user, default: sudo user/current user
  APP_PATH                 Binary path, default: ${APP_PATH_DEFAULT}
  CONFIG_PATH              Config path, default: ${CONFIG_PATH_DEFAULT}
  RESTART_DELAY_SECONDS    Watchdog restart delay, default: 2
EOF
}

need_root() {
    if [[ "${EUID}" -ne 0 ]]; then
        echo "This command needs root. Please run: sudo $0 $*" >&2
        exit 1
    fi
}

runtime_user() {
    if [[ -n "${SP_VISION_USER:-}" ]]; then
        printf '%s\n' "${SP_VISION_USER}"
    elif [[ -n "${SUDO_USER:-}" && "${SUDO_USER}" != "root" ]]; then
        printf '%s\n' "${SUDO_USER}"
    else
        id -un
    fi
}

unit_exists() {
    [[ -f "${UNIT_PATH}" ]]
}

install_service() {
    need_root "$@"

    local user
    user="$(runtime_user)"

    if ! id "${user}" >/dev/null 2>&1; then
        echo "Runtime user does not exist: ${user}" >&2
        exit 1
    fi

    install -d -m 0755 "$(dirname "${UNIT_PATH}")"

    cat >"${UNIT_PATH}" <<EOF
[Unit]
Description=SP Vision watchdog service
After=network.target

[Service]
Type=simple
User=${user}
WorkingDirectory=${REPO_ROOT}
Environment=APP_PATH=${APP_PATH}
Environment=CONFIG_PATH=${CONFIG_PATH}
Environment=RESTART_DELAY_SECONDS=${RESTART_DELAY_SECONDS}
ExecStart=${REPO_ROOT}/scripts/watchdog.sh
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

    chmod 0644 "${UNIT_PATH}"
    systemctl daemon-reload
    systemctl enable "${SERVICE_NAME}.service"

    echo "Installed ${SERVICE_NAME}.service"
    echo "Start it now with: sudo $0 start"
}

uninstall_service() {
    need_root "$@"

    if systemctl list-unit-files "${SERVICE_NAME}.service" --no-legend | grep -q .; then
        systemctl stop "${SERVICE_NAME}.service" || true
        systemctl disable "${SERVICE_NAME}.service" || true
    fi

    rm -f "${UNIT_PATH}"
    systemctl daemon-reload
    systemctl reset-failed "${SERVICE_NAME}.service" || true

    echo "Uninstalled ${SERVICE_NAME}.service"
}

show_status() {
    echo "Service name: ${SERVICE_NAME}.service"
    echo "Unit path:    ${UNIT_PATH}"
    echo "Repo root:    ${REPO_ROOT}"
    echo

    if unit_exists; then
        echo "Registered:   yes"
    else
        echo "Registered:   no"
    fi

    if systemctl is-enabled "${SERVICE_NAME}.service" >/dev/null 2>&1; then
        echo "Boot enabled: yes"
    else
        echo "Boot enabled: no"
    fi

    if systemctl is-active "${SERVICE_NAME}.service" >/dev/null 2>&1; then
        echo "Running:      yes"
    else
        echo "Running:      no"
    fi

    echo
    systemctl --no-pager --full status "${SERVICE_NAME}.service" || true
}

control_service() {
    need_root "$@"
    local action="$1"
    systemctl "${action}" "${SERVICE_NAME}.service"
}

follow_logs() {
    journalctl -u "${SERVICE_NAME}.service" -f
}

run_root_command() {
    local command="$1"
    if [[ "${EUID}" -eq 0 ]]; then
        "$0" "${command}"
    else
        sudo \
            SERVICE_NAME="${SERVICE_NAME}" \
            SP_VISION_USER="${SP_VISION_USER:-}" \
            APP_PATH="${APP_PATH}" \
            CONFIG_PATH="${CONFIG_PATH}" \
            RESTART_DELAY_SECONDS="${RESTART_DELAY_SECONDS}" \
            "$0" "${command}"
    fi
}

pause_menu() {
    echo
    read -r -p "Press Enter to continue..."
}

interactive_menu() {
    while true; do
        clear || true
        cat <<EOF
SP Vision Service Manager

Service: ${SERVICE_NAME}.service
Repo:    ${REPO_ROOT}
App:     ${APP_PATH}
Config:  ${CONFIG_PATH}

1) Show status
2) Install and enable boot autostart
3) Start service
4) Stop service
5) Restart service
6) Follow logs
7) Uninstall service
0) Exit

EOF

        read -r -p "Choose an option: " choice
        echo

        case "${choice}" in
            1)
                show_status
                pause_menu
                ;;
            2)
                run_root_command install
                pause_menu
                ;;
            3)
                run_root_command start
                pause_menu
                ;;
            4)
                run_root_command stop
                pause_menu
                ;;
            5)
                run_root_command restart
                pause_menu
                ;;
            6)
                echo "Press Ctrl+C to stop following logs."
                follow_logs
                ;;
            7)
                read -r -p "Uninstall ${SERVICE_NAME}.service? [y/N] " answer
                case "${answer}" in
                    y | Y | yes | YES)
                        run_root_command uninstall
                        ;;
                    *)
                        echo "Canceled."
                        ;;
                esac
                pause_menu
                ;;
            0 | q | Q)
                exit 0
                ;;
            *)
                echo "Unknown option: ${choice}"
                pause_menu
                ;;
        esac
    done
}

main() {
    local command="${1:-}"
    case "${command}" in
        install)
            install_service "$@"
            ;;
        uninstall)
            uninstall_service "$@"
            ;;
        status)
            show_status
            ;;
        start | stop | restart)
            control_service "${command}" "$@"
            ;;
        logs)
            follow_logs
            ;;
        -h | --help | help)
            usage
            ;;
        "")
            interactive_menu
            ;;
        *)
            echo "Unknown command: ${command}" >&2
            usage >&2
            exit 2
            ;;
    esac
}

main "$@"
