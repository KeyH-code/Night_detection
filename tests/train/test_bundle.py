"""Package regressions: current layout, no historical smoke, no held-out files."""
from pathlib import Path
import json
import shutil
import sys
import tarfile
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import bundle_source as bundle
from source_contract import ContractError, sha256, write_json


@pytest.mark.parametrize('config_name', ['source_a.yaml', 'source_s3.yaml'])
def test_package_uses_current_layout_and_exact_data_whitelist(tmp_path, monkeypatch, config_name):
    project = Path(__file__).resolve().parents[2]
    root = tmp_path / '中文项目'
    for folder in ('scripts', 'configs', 'environment', 'tests/train'):
        (root / folder).mkdir(parents=True, exist_ok=True)
        for p in (project / folder).iterdir():
            if p.is_file():
                shutil.copyfile(p, root / folder / p.name)
    for rel in ('RULES.md', 'specs/s3/plan.md', 'runs/a_001/train/best_dev_metrics.json',
                'runs/a_001/train/args.yaml', 'runs/a_001_diag/dev/metrics.json', 'runs/a_001_diag/slices.csv'):
        target = root/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project/rel, target)
    (root / 'weights').mkdir()
    for p in (project / 'weights').iterdir():
        if p.is_file():
            (root / 'weights' / p.name).write_bytes(b'fake pretrained fixture')
    data = root / 'data/source_v1'
    records = ['READY.json', 'categories.json', 'protocol.json', 'manifests/train.json',
               'manifests/dev.json', 'images/source_train/a.jpg', 'labels/source_train/a.txt']
    for name in records + ['labels/source_test/private.txt']:
        p = data / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(name.encode())
    ready = {'categories': {'path': 'categories.json'}, 'grouping_protocol': {'path': 'protocol.json'},
             'manifests': {'source_train': {'path': 'manifests/train.json'}, 'source_dev': {'path': 'manifests/dev.json'}},
             'files': [{'path': name} for name in records[-2:]],
             'counts': {'source_train': 1, 'source_dev': 0, 'source_test': 1}, 'data_fingerprint': 'fixture'}
    contract = {'ready': ready, 'files': ready['files']}
    monkeypatch.setattr(bundle, 'ROOT', root)
    monkeypatch.setattr(bundle, 'verify_source', lambda *_: contract)
    monkeypatch.chdir(root)
    output = tmp_path / 'package.tar.gz'
    bundle.build(data, sha256(data / 'READY.json'), output, config_path=Path('configs')/config_name)
    with tarfile.open(output) as archive:
        names = set(archive.getnames())
        manifest = json.load(archive.extractfile('source_a_kit/PACKAGE_MANIFEST.json'))
    assert {p.removeprefix('source_a_kit/data/source_v1/') for p in names if p.startswith('source_a_kit/data/source_v1/')} == set(records)
    assert not any('source_test' in p or 'artifacts' in p or 'local_smoke' in p for p in names)
    assert 'source_a_kit/scripts/run_source_cloud_job.py' in names
    assert 'source_a_kit/tests/train/test_bundle.py' in names
    assert 'source_a_kit/scripts/source_sampling.py' in names
    assert 'source_a_kit/scripts/source_policy.py' in names
    assert 'source_a_kit/scripts/detection_metrics.py' in names
    assert 'source_a_kit/configs/source_s3.yaml' in names
    assert not any('diagnostic_review.py' in p or '/scripts/diagnose.py' in p for p in names)
    assert manifest['status'] == 'PACKAGED_PENDING_VALIDATION'
    assert manifest['active_config'] == 'configs/'+config_name


def test_verifier_rejects_missing_policy_dependency_before_any_data_access(tmp_path, monkeypatch):
    import verify_source_bundle as verifier
    write_json(tmp_path/'PACKAGE_MANIFEST.json', {'files': {'scripts/train_source.py': {}}})
    monkeypatch.setattr(verifier, 'verify_source', lambda *_: pytest.fail('Data should not be reached'))
    with pytest.raises(ContractError, match='dependencies'): verifier.verify(tmp_path)
