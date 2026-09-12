#!/usr/bin/env bash
set -uo pipefail
thread_id=${1:?Usage: run_conflict_augmentation_tmux.sh THREAD_ID}
if [[ -n "${TMUX_PANE:-}" ]]; then
  tmux set-window-option -t "$TMUX_PANE" remain-on-exit off
fi
run_dir=artifacts/experiments/conflict-augmentation-v1
mkdir -p "$run_dir"
printf '%s\n' "$$" > "$run_dir/launcher_pid.txt"
(
  PYTHONPATH=src .venv/bin/python -u scripts/run_conflict_augmentation.py run --smoke-only &&
  PYTHONPATH=src .venv/bin/python -u scripts/run_conflict_augmentation.py run
) 2>&1 | tee -a "$run_dir/run.log"
statuses=("${PIPESTATUS[@]}")
status=${statuses[0]}
if (( status == 0 && statuses[1] != 0 )); then status=${statuses[1]}; fi
printf '%s\n' "$status" > "$run_dir/exit_code.txt"
codex queue --thread "$thread_id" --message "Synthetic conflict pilot finished with exit status ${status}. Log: ${run_dir}/run.log. Results: ${run_dir}." || true
exit "$status"
