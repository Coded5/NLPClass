# Repository Guidelines

## Project Structure & Module Organization

Application code lives in `src/zeroshot_classifier/`. Keep CLI parsing in `cli.py`, orchestration in `runner.py`, model adapters in `backends.py`, and focused logic in the matching modules. Tests live in `tests/` and mirror package responsibilities (for example, `prompt.py` is covered by `tests/test_prompt.py`). Utilities are in `scripts/`; exploratory work belongs in `notebooks/`. Treat `data/` as source input and `artifacts/` as generated output; do not commit secrets or transient runs.

## Build, Test, and Development Commands

Use Python 3.10 or newer and install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Run the complete test suite with:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Common workflows are `zeroshot-classifier split`, `zeroshot-classifier run`, and `zeroshot-classifier evaluate`. The default run requires Ollama serving `qwen2.5:7b`; use a separate `--run-dir` when changing configuration. Use `--overwrite` to regenerate splits and `--retry-failed` to resume failures.

## Coding Style & Naming Conventions

Follow PEP 8: four-space indentation, `snake_case` for functions and modules, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants. Add type hints to public APIs and keep functions focused. Prefer `pathlib.Path`, explicit exception handling, and deterministic data processing. No formatter or linter is configured; match surrounding style and remove unused imports.

## Testing Guidelines

Tests use Python's `unittest` framework. Name files `test_<module>.py`, classes `Test<Behavior>`, and methods `test_<expected_behavior>`. Add regression tests for CLI validation, checkpoint/resume behavior, label parsing, split integrity, and metric changes. Tests should use temporary directories and mocked backends rather than live network services. Run the full suite before opening a pull request.

## Commit & Pull Request Guidelines

Recent history uses short Conventional Commit subjects such as `feat: NLP Contest2`, `fix: logistic regression scoring`, and `chore: update gitignore`. Continue with `feat:`, `fix:`, `test:`, `docs:`, or `chore:` followed by an imperative summary. Pull requests should explain the problem and solution, list verification commands, link relevant issues, and call out changes to data formats, prompts, or model settings. Include metric comparisons when classification or evaluation behavior changes; screenshots are only needed for notebook visualizations.

## Security & Configuration

Pass API credentials through environment variables such as `OPENAI_API_KEY`. Never store keys in notebooks, manifests, checkpoints, or committed configuration. Do not modify `data/contest2_test.csv` or expose gold labels to model prompts.

## Long-running experiments

Codex is normally running inside a tmux session.

For training, evaluation, benchmarking, or other long-running jobs:

- First check whether `$TMUX` is set.
- If already inside tmux, launch the experiment in a new detached tmux
  window or pane in the current session.
- Do NOT start another nested tmux session.
- Prefer a detached tmux window for long ML experiments.
- Redirect verbose stdout/stderr to a log file.
- Keep the log visibly streaming in the tmux window or pane Codex creates.
  Launch through `tee` with pipeline exit-status handling, or automatically run
  `tail -F` in a dedicated pane; do not leave the created tmux view blank and
  require the user to start log monitoring manually.
- After launching, verify that the process started successfully.
- Record the tmux window/pane, PID, command, and log path.
- Make every tmux window or pane created for a job close automatically after
  the command finishes and its exit status has been written. Do not append an
  interactive shell such as `exec bash` merely to keep a completed pane open.
- Before the pane exits, use `codex queue --thread <thread> --message <message>`
  to notify the originating Codex thread that the job finished. Include the
  job name, exit status, and log or result paths so Codex can inspect the final
  output and continue automatically. Write the exit status to an artifact
  before invoking `codex queue` so completion remains discoverable if the
  notification fails.
- Do not repeatedly poll the job or read its logs.
- Return control to me after the experiment is launched.
- When the queued completion message arrives, inspect the final metrics and
  resume without requiring me to report that the job is finished.
