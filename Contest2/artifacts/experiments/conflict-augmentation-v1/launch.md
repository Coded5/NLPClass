# Launch record

- tmux window: `NLP:6` (`conflict-augmentation`), pane `%22`
- Wrapper PID at launch: `1170757`
- Python PID at initial verification: `1170763` (smoke stage)
- Working directory: `/home/kami/Projects/NLP/Contest2`
- Command: `bash scripts/run_conflict_augmentation_tmux.sh 01a089c6-f972-7a42-9610-a01840c8838c`
- Live log: `artifacts/experiments/conflict-augmentation-v1/run.log`
- Completion status: `artifacts/experiments/conflict-augmentation-v1/exit_code.txt`
- Sequence: three one-epoch smoke fits, then 15 full fits and report generation.
- Initial verification: wrapper and smoke Python process alive; pane alive;
  `remain-on-exit off`; MLflow experiment creation recorded in log.
- Prelaunch verification: 116 unit tests passed, diff and shell syntax checks
  passed, all 400 example reviews/hash matches and fitting-source exclusions
  checked, repeated/synthetic counts matched by aspect and polarity in every fold.

Completion is queued to the originating thread before the pane exits. PIDs and
window numbers describe launch-time state and may be reused after completion.
