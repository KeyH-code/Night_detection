"""Build a private, relocatable train/dev-only kit from the exact frozen whitelist."""
import argparse
import io
import json
import tarfile
from pathlib import Path

from source_contract import (ContractError, canonical_hash, check_equal, read_json, safe_path,
                             sha256, verify_source, write_json)

ROOT = Path(__file__).resolve().parents[1]


def build(root, ready_hash, output, smoke_path):
    contract = verify_source(root, ready_hash)
    smoke = read_json(smoke_path)
    check_equal('smoke.ready_sha256', ready_hash, smoke['identity']['ready_sha256'])
    from train_source import code_identity, config_at
    check_equal('smoke.code', code_identity(), smoke['identity']['code'])
    config = config_at(ROOT / 'configs/source_a.yaml')
    check_equal('smoke.config', canonical_hash(config), smoke['identity']['config_sha256'])
    if output.exists():
        raise ContractError(f'Never overwrite published kit: {output}')
    ready = contract['ready']
    paths = {'data/source_v1/READY.json': root / 'READY.json'}
    for entry in [ready['categories'], ready['grouping_protocol'], *ready['manifests'].values(), *ready['files']]:
        paths['data/source_v1/' + entry['path']] = safe_path(root, entry['path'])
    code_files = ['source_contract.py', 'source_trainer.py', 'train_source.py',
                  'bundle_source.py', 'verify_source_bundle.py']
    for name in code_files:
        paths['scripts/' + name] = ROOT / 'scripts' / name
    for relative in ['configs/source_a.yaml', 'environment/training.in',
                     'environment/requirements-linux.lock', 'environment/requirements-windows.lock',
                     'environment/windows-installed.txt', 'environment/setup-linux.sh',
                     'environment/setup-windows.ps1', 'reports/source_a_cloud.md',
                     'weights/yolo11m.pt', 'weights/yolo11m.provenance.json',
                     'weights/yolo26n.pt', 'weights/yolo26n.provenance.json',
                     'training_tests/test_source_training.py']:
        paths[relative] = ROOT / relative
    paths['evidence/local_SMOKE_PASSED.json'] = smoke_path
    paths['evidence/local_environment.json'] = ROOT / 'artifacts/source_a/environment/environment.json'
    paths['evidence/training_tests.xml'] = ROOT / 'reports/source_training_tests.xml'
    manifest = {'schema_version': 1, 'status': 'PACKAGED_PENDING_VALIDATION',
                'scope': 'private source_train/source_dev kit; no source_test or target data',
                'ready_sha256': ready_hash, 'data_fingerprint': ready['data_fingerprint'],
                'counts': {p: ready['counts'][p] for p in ('source_train', 'source_dev')},
                'smoke_formal_A': False, 'formal_A_completed': False,
                'files': {name: {'sha256': sha256(path), 'bytes': path.stat().st_size}
                          for name, path in sorted(paths.items())}}
    output.parent.mkdir(parents=True, exist_ok=True)
    part = output.with_suffix(output.suffix + '.part')
    with tarfile.open(part, 'w:gz', compresslevel=1) as tar:
        for name, path in sorted(paths.items()):
            info = tarfile.TarInfo('source_a_kit/' + name)
            info.size = path.stat().st_size
            info.mode = 0o755 if name.endswith('.sh') else 0o644
            with path.open('rb') as f:
                tar.addfile(info, f)
        body = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
        info = tarfile.TarInfo('source_a_kit/PACKAGE_MANIFEST.json')
        info.size, info.mode = len(body), 0o644
        tar.addfile(info, io.BytesIO(body))
    check_equal('READY unchanged during packaging', ready_hash, sha256(root/'READY.json'))
    part.replace(output)
    summary = {'path': str(output), 'sha256': sha256(output), 'bytes': output.stat().st_size,
               'archive_members': len(paths) + 1, 'training_data_files': len(contract['files']),
               'ready_sha256': ready_hash, 'status': 'PACKAGED_PENDING_VALIDATION'}
    write_json(output.with_suffix(output.suffix + '.json'), summary)
    write_json(output.parent / 'source_a_package_manifest.json', manifest)
    print(json.dumps(summary))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=ROOT/'data/soda10m/splits/source_v1')
    p.add_argument('--ready-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--smoke', type=Path, required=True)
    a = p.parse_args()
    build(a.data_root, a.ready_sha256, a.output, a.smoke)
