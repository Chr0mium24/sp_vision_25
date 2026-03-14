#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ts="$(date +%Y%m%d_%H%M%S)"
backup_dir="$ROOT_DIR/backups/prechange_${ts}"
mkdir -p "$backup_dir"

echo "backup_dir=$backup_dir"

{
  echo "timestamp=$ts"
  echo "pwd=$(pwd)"
  echo "git_head=$(git rev-parse HEAD 2>/dev/null || echo N/A)"
  echo "git_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo N/A)"
} > "$backup_dir/meta.txt"

git status --short > "$backup_dir/git_status_short.txt" || true
git diff > "$backup_dir/git_diff.patch" || true
git submodule status > "$backup_dir/git_submodule_status.txt" || true

if [ -d build ]; then
  find build/bin/tests -maxdepth 2 -type f 2>/dev/null | sort > "$backup_dir/test_binaries.txt" || true
else
  : > "$backup_dir/test_binaries.txt"
fi

{
  echo "# smoke check"
  echo "time=$(date '+%F %T')"

  run_help() {
    local exe="$1"
    if [ -x "$exe" ]; then
      echo "=== $exe --help ==="
      set +e
      "$exe" --help >/tmp/prechange_help.out 2>&1
      rc=$?
      set -e
      echo "exit_code=$rc"
      sed -n '1,40p' /tmp/prechange_help.out
      echo
    else
      echo "=== $exe ==="
      echo "missing"
      echo
    fi
  }

  run_help "./build/bin/tests/gimbal/gimbal_ui_test"
  run_help "./build/bin/tests/auto_aim/auto_aim_ui_test"
  run_help "./build/bin/tests/auto_aim/auto_aim_ui_tune"
  run_help "./build/bin/tests/auto_buff/auto_power_rune_test"
} > "$backup_dir/smoke_help.log"

echo "done"
