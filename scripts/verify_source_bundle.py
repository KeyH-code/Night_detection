"""Pure stdlib verifier, runnable before installing Linux training dependencies."""
import argparse
import json
from pathlib import Path

from source_contract import ContractError, check_equal, read_json, safe_path, sha256, verify_source


def verify(root):
    manifest = read_json(root / 'PACKAGE_MANIFEST.json')
    required = {'scripts/'+n for n in ('source_policy.py', 'source_sampling.py', 'detection_metrics.py',
                'train_monitor.py', 'evaluate.py', 'source_trainer.py', 'train_source.py', 'source_control.py')}
    required |= {'configs/source_s3.yaml', 'configs/source_s3_control.json', 'RULES.md', 'specs/s3/plan.md',
                 'runs/a_001/train/best_dev_metrics.json', 'runs/a_001/train/args.yaml', 'runs/a_001_diag/dev/metrics.json', 'runs/a_001_diag/slices.csv'}
    if not required <= set(manifest['files']):
        raise ContractError('Training package lacks required evaluation/policy dependencies')
    if any('diagnostic_review' in name or name.startswith('scripts/diagnose') for name in manifest['files']):
        raise ContractError('Local review UI is not a training package dependency')
    for name, entry in manifest['files'].items():
        path = safe_path(root, name)
        check_equal(path, entry['sha256'], sha256(path))
        check_equal(str(path) + '.bytes', entry['bytes'], path.stat().st_size)
    expected_data = {name for name in manifest['files'] if name.startswith('data/')}
    actual_data = {p.relative_to(root).as_posix() for p in (root/'data').rglob('*') if p.is_file()}
    check_equal('package data contains no extra files', expected_data, actual_data)
    if any('source_test' in name or 'night_' in name or 'cross_scene' in name for name in actual_data):
        raise ContractError('Forbidden held-out file in package')
    contract = verify_source(root/'data/source_v1', manifest['ready_sha256'])
    print(json.dumps({'status': 'PACKAGE_VERIFIED', 'files': len(manifest['files']),
                      'data_files': len(contract['files']), 'ready_sha256': manifest['ready_sha256']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    verify(p.parse_args().root.resolve())
