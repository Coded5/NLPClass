from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import cast

from . import __version__
from .backends import BackendSettings, create_backend
from .checkpoint import ResumeMismatchError, RunLock, RunLockedError, RunStore
from .data import DataValidationError, read_input_items, write_split
from .evaluation import EvaluationError, evaluate
from .io_utils import atomic_write_json, file_sha256, text_sha256
from .prompt import PROMPT_VERSION, SYSTEM_PROMPT
from .runner import RunnerSettings, run_classification


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='zeroshot-classifier',
        description='Resumable zero-shot aspect-based sentiment classifier.',
    )
    parser.add_argument('--version', action='version', version=__version__)
    commands = parser.add_subparsers(dest='command', required=True)

    split_parser = commands.add_parser('split', help='Create a grouped, jointly stratified 90:10 split')
    split_parser.add_argument('--input', type=Path, default=Path('data/contest2_train.csv'))
    split_parser.add_argument('--output-dir', type=Path, default=Path('artifacts/splits'))
    split_parser.add_argument('--test-ratio', type=float, default=0.1)
    split_parser.add_argument('--seed', type=int, default=42)
    split_parser.add_argument('--overwrite', action='store_true')

    run_parser = commands.add_parser('run', help='Classify the complete labeled dataset')
    run_parser.add_argument('--input', type=Path, default=Path('data/contest2_train.csv'))
    run_parser.add_argument('--run-dir', type=Path, default=Path('artifacts/runs/default'))
    run_parser.add_argument(
        '--backend',
        choices=('ollama', 'openai-compatible'),
        default='ollama',
    )
    run_parser.add_argument('--model', default='qwen2.5:7b')
    run_parser.add_argument('--base-url')
    run_parser.add_argument('--api-key-env', default='OPENAI_API_KEY')
    run_parser.add_argument('--timeout', type=float, default=120.0)
    run_parser.add_argument('--temperature', type=float, default=0.0)
    run_parser.add_argument('--max-attempts', type=int, default=3)
    run_parser.add_argument('--initial-backoff', type=float, default=2.0)
    run_parser.add_argument('--max-backoff', type=float, default=30.0)
    run_parser.add_argument('--retry-failed', action='store_true')

    evaluate_parser = commands.add_parser('evaluate', help='Run the official evaluator and extra accuracy metrics')
    evaluate_parser.add_argument('--gold', type=Path, default=Path('data/contest2_train.csv'))
    evaluate_parser.add_argument(
        '--predictions',
        type=Path,
        default=Path('artifacts/runs/default/predictions.csv'),
    )
    evaluate_parser.add_argument('--evaluator', type=Path, default=Path('scripts/evaluate.py'))
    evaluate_parser.add_argument(
        '--metrics-output',
        type=Path,
        default=Path('artifacts/runs/default/metrics.json'),
    )
    evaluate_parser.add_argument(
        '--report-output',
        type=Path,
        default=Path('artifacts/runs/default/official_evaluation.txt'),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == 'split':
            return _split(args)
        if args.command == 'run':
            return _run(args)
        if args.command == 'evaluate':
            return _evaluate(args)
    except KeyboardInterrupt:
        print('\nInterrupted. Completed predictions were checkpointed.', file=sys.stderr)
        return 130
    except (
        DataValidationError,
        EvaluationError,
        FileExistsError,
        ResumeMismatchError,
        RunLockedError,
        ValueError,
    ) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 1
    return 0


def _split(args: argparse.Namespace) -> int:
    _reject_protected_test_file(args.input)
    manifest = write_split(
        args.input,
        args.output_dir,
        test_ratio=args.test_ratio,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _run(args: argparse.Namespace) -> int:
    _reject_protected_test_file(args.input)
    if args.timeout <= 0:
        raise ValueError('--timeout must be greater than 0')
    if args.max_attempts < 1:
        raise ValueError('--max-attempts must be at least 1')
    if args.initial_backoff < 0 or args.max_backoff < 0:
        raise ValueError('Backoff values cannot be negative')
    items = read_input_items(args.input)
    base_url = args.base_url or _default_base_url(args.backend)
    backend_settings = BackendSettings(
        backend=args.backend,
        model=args.model,
        base_url=base_url,
        api_key_env=args.api_key_env,
        timeout=args.timeout,
        temperature=args.temperature,
    )
    run_manifest = {
        'format_version': 1,
        'input_sha256': file_sha256(args.input),
        'input_items': len(items),
        'backend': args.backend,
        'model': args.model,
        'base_url': base_url,
        'temperature': args.temperature,
        'prompt_version': PROMPT_VERSION,
        'prompt_sha256': text_sha256(SYSTEM_PROMPT),
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with RunLock(args.run_dir / '.run.lock'):
        with RunStore(args.run_dir / 'checkpoint.sqlite3', run_manifest) as store:
            atomic_write_json(args.run_dir / 'manifest.json', run_manifest)
            summary = run_classification(
                items,
                create_backend(backend_settings),
                store,
                args.run_dir,
                RunnerSettings(
                    max_attempts=args.max_attempts,
                    initial_backoff=args.initial_backoff,
                    max_backoff=args.max_backoff,
                    retry_failed=args.retry_failed,
                ),
            )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if summary['failed'] else 0


def _evaluate(args: argparse.Namespace) -> int:
    _reject_protected_test_file(args.gold)
    metrics = evaluate(
        args.gold,
        args.predictions,
        args.evaluator,
        args.metrics_output,
        args.report_output,
    )
    accuracy = cast(dict[str, float], metrics['supplemental_exact_match_accuracy'])
    print('\n=== EXACT-MATCH ACCURACY ===')
    print(f'Aspect:  {accuracy["aspect"]:.3f}')
    print(f'Polarity: {accuracy["polarity"]:.3f}')
    print(f'Overall: {accuracy["overall"]:.3f}')
    print(f'\nMetrics written to {args.metrics_output}')
    return 0


def _default_base_url(backend: str) -> str:
    if backend == 'ollama':
        return os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    return os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')


def _reject_protected_test_file(path: Path) -> None:
    protected_path = Path(__file__).resolve().parents[2] / 'data' / 'contest2_test.csv'
    if path.resolve() == protected_path.resolve():
        raise DataValidationError(
            'data/contest2_test.csv is protected and cannot be used by this validation workflow'
        )


if __name__ == '__main__':
    raise SystemExit(main())
