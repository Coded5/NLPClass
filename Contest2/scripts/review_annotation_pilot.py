#!/usr/bin/env python3
"""Prepare a frozen blinded pilot; import review responses without changing labels."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from zeroshot_classifier.data import read_labeled_csv
from zeroshot_classifier.io_utils import atomic_write_json, atomic_write_text, file_sha256
from zeroshot_classifier.labels import ASPECTS, POLARITIES

ROOT = Path('artifacts/experiments/annotation-review-pilot-v1')
AUDIT = Path('artifacts/experiments/oof-label-audit-v1/review_cases.json')
TRAIN = Path('artifacts/training/roberta-aspect-exp1/splits/train.csv')


def select_pilot(cases: list[dict], seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    groups = [
        [v for v in cases if v['bucket'] in ('aspect_multi_misc', 'aspect_single_misc')],
        [v for v in cases if v['bucket'] == 'polarity_conflict'],
        [v for v in cases if v['bucket'] == 'polarity_neutral'],
        [v for v in cases if v['bucket'] == 'aspect_other'],
    ]
    chosen = []
    used = set()
    for group in groups:
        queues = defaultdict(list)
        for case in sorted(group, key=lambda v: v['case_key']):
            queues[case['assessment']].append(case)
        labels = sorted(queues)
        rng.shuffle(labels)
        for values in queues.values():
            rng.shuffle(values)
        picked = []
        while len(picked) < 5:
            progressed = False
            for label in labels:
                while queues[label] and queues[label][-1]['id'] in used:
                    queues[label].pop()
                if queues[label] and len(picked) < 5:
                    case = queues[label].pop()
                    used.add(case['id']); picked.append(case); progressed = True
            if not progressed:
                raise ValueError('Not enough unique reviews to fill pilot strata')
        chosen.extend(picked)
    rng.shuffle(chosen)
    return chosen


def prepare(root: Path, audit: Path, train: Path, seed: int) -> None:
    cases = json.loads(audit.read_text())
    rows = read_labeled_csv(train)
    by_id = defaultdict(list)
    for row in rows:
        by_id[row.item_id].append(row)
    selected = select_pilot(cases, seed)
    manifest = {'seed': seed, 'audit_sha256': file_sha256(audit),
                'train_sha256': file_sha256(train), 'case_keys': [v['case_key'] for v in selected],
                'selection': '5 per stratum; round-robin assessment categories; seeded shuffle',
                'prior_exposure': 'User previously saw audit examples; blinding removes displayed labels but cannot erase recall.'}
    manifest_path = root / 'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Frozen pilot inputs differ; use a new output directory')
        return
    blind = []; key = []
    for i, case in enumerate(selected, 1):
        if case['id'] not in by_id or case['text'] != by_id[case['id']][0].text:
            raise ValueError('Pilot contains a non-training or mismatched review')
        token = f'P{i:02d}'
        blind.append({'review_id': token, 'text': case['text']})
        key.append({'review_id': token, 'case': case,
                    'contest_pairs': sorted({(r.aspect, r.polarity) for r in by_id[case['id']]})})
    atomic_write_json(root / 'private/key.json', key)
    atomic_write_json(root / 'manifest.json', manifest)
    atomic_write_json(root / 'blinded/reviews.json', blind)
    atomic_write_json(root / 'blinded/responses-template.json', [
        {'review_id': v['review_id'], 'status': 'pending', 'pairs': [], 'note': ''} for v in blind])
    for batch in range(4):
        lines = [f'# Review batch {batch + 1}', '',
                 'Use GUIDE.md. Supply all aspect/polarity pairs with supporting phrases. '
                 'Mark uncertain or insufficient_context when needed; do not guess missing context.', '']
        for v in blind[batch * 5:(batch + 1) * 5]:
            lines.extend([f"## {v['review_id']}", '', v['text'], '',
                          'Response: aspect / polarity — supporting phrase; status; optional note.', ''])
        atomic_write_text(root / f'blinded/batch-{batch + 1}.md', '\n'.join(lines))
    atomic_write_json(root / 'private/source_comparison.json', {
        'status': 'unverified_login_required', 'matched': None, 'unmatched': None,
        'reason': 'Official META-SHARE v2.0 download redirects to login; no local source XML found.'})


def validate_responses(values: list[dict], allowed: set[str]) -> dict[str, dict]:
    output = {}
    for value in values:
        token = value['review_id']
        if token not in allowed or token in output:
            raise ValueError('Unknown or duplicate review ID')
        if value['status'] not in ('pending', 'complete', 'uncertain', 'insufficient_context'):
            raise ValueError('Unknown review status')
        seen = set()
        for pair in value['pairs']:
            if pair['aspect'] not in ASPECTS or pair['polarity'] not in POLARITIES:
                raise ValueError('Unknown aspect or polarity')
            if pair['aspect'] in seen:
                raise ValueError('Each aspect needs one polarity; use conflict for opposed opinions')
            if not pair.get('evidence', '').strip():
                raise ValueError('A supporting phrase is required')
            seen.add(pair['aspect'])
        if value['status'] == 'complete' and not seen:
            raise ValueError('Complete responses require an aspect set')
        output[token] = value
    return output


def import_responses(root: Path, path: Path) -> None:
    key = json.loads((root / 'private/key.json').read_text())
    destination = root / 'responses.json'
    old = json.loads(destination.read_text()) if destination.exists() else []
    allowed = {v['review_id'] for v in key}
    merged = validate_responses(old, allowed)
    incoming = validate_responses(json.loads(path.read_text()), allowed)
    for token, response in incoming.items():
        if token in merged and merged[token] != response and merged[token]['status'] != 'pending':
            raise ValueError('Existing submitted response differs; preserve it and use a new revision directory')
        merged[token] = response
    atomic_write_json(destination, list(merged.values()))
    terminal = [v for v in merged.values() if v['status'] != 'pending']
    if len(terminal) != len(key):
        atomic_write_json(root / 'progress.json', {'submitted': len(terminal), 'total': len(key)})
        return
    comparisons = []; exact = total = correct = shared = 0
    for entry in key:
        response = merged[entry['review_id']]
        human = {v['aspect']: v['polarity'] for v in response['pairs']}
        gold = dict(entry['contest_pairs'])
        record = {'review_id': entry['review_id'], 'response': response,
                  'contest_pairs': entry['contest_pairs'], 'prior_audit': entry['case']}
        if response['status'] == 'complete':
            total += 1; exact += human.keys() == gold.keys()
            common = human.keys() & gold.keys()
            correct += sum(human[a] == gold[a] for a in common); shared += len(common)
            record.update(added=sorted(human.keys() - gold.keys()), removed=sorted(gold.keys() - human.keys()))
        comparisons.append(record)
    atomic_write_json(root / 'comparison.json', {
        'complete_reviews': total, 'exact_aspect_sets': exact,
        'aspect_set_agreement': exact / total if total else None,
        'matching_polarities': correct, 'shared_aspects': shared,
        'polarity_agreement': correct / shared if shared else None,
        'statuses': dict(Counter(v['status'] for v in terminal)), 'cases': comparisons,
        'interpretation': 'Selected pilot agreement only; uncertain cases excluded from agreement denominators.'})


def compare_source(root: Path, train: Path, xml_path: Path) -> None:
    source = defaultdict(list)
    for sentence in ET.parse(xml_path).getroot().findall('.//sentence'):
        text = sentence.findtext('text', '').strip()
        pairs = sorted({(v.attrib['category'], v.attrib['polarity'])
                        for v in sentence.findall('./aspectCategories/aspectCategory')})
        source[text].append({'id': sentence.attrib.get('id'), 'pairs': pairs})
    grouped = defaultdict(list)
    for row in read_labeled_csv(train):
        grouped[row.item_id].append(row)
    records = []
    for item_id, rows in grouped.items():
        matches = source.get(rows[0].text, [])
        gold = sorted({(v.aspect, v.polarity) for v in rows})
        status = ('unmatched' if not matches else 'multiple_source_matches' if len(matches) > 1
                  else 'same_annotations' if matches[0]['pairs'] == gold else 'different_annotations')
        records.append({'id': item_id, 'status': status, 'contest_pairs': gold, 'source_matches': matches})
    atomic_write_json(root / 'private/source_comparison.json', {
        'status': 'compared_user_supplied_xml', 'xml_sha256': file_sha256(xml_path),
        'train_sha256': file_sha256(train), 'counts': dict(Counter(v['status'] for v in records)),
        'matching': 'Exact text after stripping outer whitespace; IDs are supporting evidence only.',
        'cases': records})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'import', 'compare-source'])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--audit', type=Path, default=AUDIT)
    parser.add_argument('--train', type=Path, default=TRAIN)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--input', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.root, args.audit, args.train, args.seed)
    elif args.input is None:
        parser.error('--input is required for import and compare-source')
    elif args.action == 'import':
        import_responses(args.root, args.input)
    else:
        compare_source(args.root, args.train, args.input)


if __name__ == '__main__':
    main()
