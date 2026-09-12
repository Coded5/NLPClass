#!/usr/bin/env bash
set -uo pipefail

mode=${1:?Usage: run_learning_curve_tmux.sh MODE THREAD_ID}
thread_id=${2:?Usage: run_learning_curve_tmux.sh MODE THREAD_ID}

case "$mode" in
  benchmark|run|report) ;;
  *)
    echo "Unsupported learning-curve mode: $mode" >&2
    exit 2
    ;;
esac

run_dir=artifacts/experiments/learning-curve-v1
log_path="$run_dir/${mode}.log"
status_path="$run_dir/${mode}_exit_code.txt"
mkdir -p "$run_dir"

set +e
PYTHONPATH=src .venv/bin/python scripts/run_learning_curve.py "$mode" 2>&1 | tee "$log_path"
status=${PIPESTATUS[0]}
set -e

printf '%s\n' "$status" > "$status_path"
message="Learning curve ${mode} finished with exit status ${status}. Log: ${log_path}. Results: ${run_dir}."
codex queue --thread "$thread_id" --message "$message" || true
exit "$status"
