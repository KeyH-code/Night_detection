"""Package regressions: current layout, no historical smoke, no held-out files."""
from pathlib import Path
import json
import shutil
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import bundle_source as bundle
from source_contract import sha256


def test_package_uses_current_layout_and_exact_data_whitelist(tmp_path, monkeypatch):
    project = Path(__file__).resolve().parents[2]
    root = tmp_path / '中文项目'
    for folder in ('scripts', 'configs', 'environment', 'tests/train'):
        (root / folder).mkdir(parents=True, exist_ok=True)
        for p in (project / folder).iterdir():
            if p.is_file():
                shutil.copyfile(p, root / folder / p.name)
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
    output = tmp_path / 'package.tar.gz'
    bundle.build(data, sha256(data / 'READY.json'), output)
    with tarfile.open(output) as archive:
        names = set(archive.getnames())
        manifest = json.load(archive.extractfile('source_a_kit/PACKAGE_MANIFEST.json'))
    assert {p.removeprefix('source_a_kit/data/source_v1/') for p in names if p.startswith('source_a_kit/data/source_v1/')} == set(records)
    assert not any('source_test' in p or 'artifacts' in p or 'local_smoke' in p for p in names)
    assert 'source_a_kit/scripts/run_source_cloud_job.py' in names
    assert 'source_a_kit/tests/train/test_bundle.py' in names
    assert manifest['status'] == 'PACKAGED_PENDING_VALIDATION'
