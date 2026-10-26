"""Read-only integrity and selection checks for a retained formal run."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from source_contract import check_equal, read_json, safe_path, sha256


def verify(root):
    manifest = read_json(root / 'manifest.json')
    for relative, entry in manifest['files'].items():
        path = safe_path(root, relative)
        check_equal(relative, entry['sha256'], sha256(path))
        check_equal(relative + '.bytes', entry['bytes'], path.stat().st_size)
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    check_equal('retained file set', set(manifest['files']) | {'manifest.json'}, actual)
    audit = read_json(root / 'job/COMPLETION_AUDIT.json')
    check_equal('status', 'FORMAL_A_COMPLETE', audit['status'])
    check_equal('test isolation', False, audit['final_test_executed'])
    check_equal('night isolation', False, audit['night_data_used'])
    check_equal('exit', '0', (root / 'job/exit').read_text(encoding='utf-8').strip())
    with (root / 'train/results.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    check_equal('epoch count', audit['epochs_completed'], len(rows))
    for i, row in enumerate(rows, 1):
        clean = {k.strip(): v.strip() for k, v in row.items()}
        check_equal('epoch sequence', i, int(clean['epoch']))
        if not all(math.isfinite(float(v)) for k, v in clean.items() if 'loss' in k):
            raise ValueError(f'Non-finite loss in epoch {i}')
    scores = [json.loads(line) for line in (root / 'train/source_metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    check_equal('score count', len(rows), len(scores))
    best_score, best_epoch = -math.inf, None
    for i, score in enumerate(scores, 1):
        check_equal('score epoch', i, score['epoch'])
        primary = (score['per_class_AP50-95']['pedestrian'] + score['per_class_AP50-95']['car'])/2
        check_equal('raw primary metric', primary, score['fitness'])
        improved = primary > best_score
        check_equal('strict improvement', improved, score['strict_improvement'])
        if improved:
            best_score, best_epoch = primary, i
    check_equal('selected epoch', audit['best_epoch'], best_epoch)
    check_equal('selected score', audit['selection_dev_primary'], best_score)
    check_equal('best identity', audit['weights']['best.pt']['sha256'], sha256(root / 'weights/best.pt'))
    check_equal('last identity', audit['weights']['last.pt']['sha256'], sha256(root / 'weights/last.pt'))
    metric = read_json(root / 'train/best_dev_metrics.json')
    check_equal('metric checkpoint', metric['checkpoint_sha256'], sha256(root / 'weights/best.pt'))
    classes = metric['metrics']
    check_equal('two-class metric', (classes['metrics/pedestrian_AP50-95'] + classes['metrics/car_AP50-95'])/2, metric['fitness'])
    return {'status': 'passed', 'files': len(manifest['files']), 'epochs': len(rows),
            'best_epoch': audit['best_epoch'], 'source_dev_primary': metric['fitness'],
            'final_test_executed': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('runs/a_001'))
    print(json.dumps(verify(parser.parse_args().run.resolve()), indent=2))
