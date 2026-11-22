"""Cloud gate binding the current source recipe, runtime and observation code."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from source_contract import ContractError, canonical_hash, check_equal, read_json, sha256, write_json
from source_resources import snapshot, shared_memory_risk

ROOT = Path(__file__).resolve().parents[1]
BASE_ARCHIVE = '1581bd3cc6b238acf8874c9661650dad38259d2047077fb38615e3dc073ca605'
READY = '72e75ebd99014c830e0227061e390e6aed768ae724b532a5c2c90fd8dec93e85'
BASE_CONFIG = '415f66f470103dfa78532d80085b1955bb3a1c3eb3f42a2ec990a7d67ceedb69'
PRETRAINED = 'd5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95'
BASE_CODE = {
    'scripts/train_source.py': 'e226d5f24bab3bbaaf2cb122d968eb56782f1611c6e6b0b9578707d47f7aaf86',
    'scripts/source_trainer.py': 'da9a5b0b0f9c1b2afb98fc8ba9aabfe752e442af0f91dbc78fff904f53be9c81',
    'scripts/source_contract.py': '329f1761dceab98552645d92eb12ea75469b658c974f84b4a06aeee6493b55a3',
    'scripts/evaluate.py': '69d4a8ac9bd1ee75ee1f604bbf10158bdfbefb664d7fd118a5cc7d2b04a633d9',
    'scripts/train_monitor.py': '979497ccbd656a9e2cae656de4824da826a62ecda520fb508e32b1c60d57306c',
}


def resource_rejections(resources, min_cpus=8, min_ram_bytes=30*1024**3):
    reasons = []
    if resources['cpu_count'] < min_cpus:
        reasons.append(f'Effective CPU allocation {resources["cpu_count"]} < {min_cpus}')
    if resources['ram_bytes'] < min_ram_bytes:
        reasons.append(f'Effective RAM allocation {resources["ram_bytes"]} < {min_ram_bytes}')
    if shared_memory_risk(resources):
        reasons.append(f'/dev/shm risk gate: {resources["shared_memory_total_bytes"]} bytes < 1 GiB '
                       '(workers=4 unchanged; no DataLoader crash is implied)')
    return reasons


def runtime_identity(root=ROOT, config_path=None, arm='R', seed=17):
    config_path = Path(config_path) if config_path else root/'configs/source_a.yaml'
    if config_path.name == 'source_s3.yaml':
        import importlib.metadata as metadata
        import platform
        import yaml
        from train_source import config_at
        config = config_at(config_path, arm, seed)
        check_equal('new runtime experiment', 'source_s3', config['experiment'])
        if arm not in ('R', 'S', 'U') or seed not in (17, 29, 43) or (arm == 'U' and seed != 17):
            raise ContractError('Unregistered runtime arm/seed')
        config['sampling']['arm'], config['training']['seed'] = arm, seed
        ready = root/'data/source_v1/READY.json'
        if not ready.exists():
            ready = root/'data/soda10m/splits/source_v1/READY.json'
        check_equal('S3 frozen READY', READY, sha256(ready))
        check_equal('S3 official pretrained', PRETRAINED, sha256(root/'weights/yolo11m.pt'))
        code_names = ('source_contract.py', 'source_trainer.py', 'train_source.py', 'evaluate.py',
                      'detection_metrics.py', 'train_monitor.py', 'source_sampling.py', 'source_policy.py',
                      'source_resources.py', 'source_cloud_gate.py', 'run_source_cloud_job.py',
                      'source_release.py', 'bundle_source.py', 'verify_source_bundle.py')
        paths = ['scripts/'+n for n in code_names]+['RULES.md', 'specs/s3/plan.md', 'runs/a_001/train/args.yaml',
                 'runs/a_001/train/best_dev_metrics.json', 'runs/a_001_diag/dev/metrics.json',
                 'runs/a_001_diag/slices.csv', 'environment/requirements-linux.lock']
        packages = {name: metadata.version(name) for name in config['environment'] if name != 'python'}
        check_equal('S3 runtime Python', config['environment']['python'], '.'.join(platform.python_version_tuple()[:2]))
        for name, version in packages.items():
            check_equal('S3 runtime '+name, config['environment'][name], version)
        # Pin implementation bytes as well as package versions.
        distribution = Path(metadata.distribution('ultralytics').locate_file(''))
        library = {rel: sha256(distribution/rel) for rel in (
            'ultralytics/engine/trainer.py', 'ultralytics/engine/validator.py',
            'ultralytics/data/augment.py', 'ultralytics/data/base.py', 'ultralytics/data/build.py',
            'ultralytics/data/dataset.py', 'ultralytics/utils/nms.py', 'ultralytics/utils/metrics.py')}
        body = {'schema_version': 3, 'experiment': 'source_s3', 'source_ready_sha256': READY,
                'config_file_sha256': sha256(config_path), 'resolved_config_sha256': canonical_hash(config),
                'arm': arm, 'seed': seed, 'pretrained_sha256': PRETRAINED,
                'code_identity': {p: sha256(root/p) for p in paths}, 'packages': packages,
                'platform': platform.system(), 'machine': platform.machine(),
                'library_code': library, 'training_recipe_changed': True, 'frozen_data_changed': False}
        if arm == 'U':
            from source_control import validate_build, validate_u_runtime
            profile = validate_build(root)
            check_equal('U runtime config', profile['U_config_sha256'], canonical_hash(config))
            for rel in ('scripts/source_control.py', 'configs/source_s3_control.json'):
                body['code_identity'][rel] = sha256(root/rel)
            body['control_profile_sha256'] = sha256(root/'configs/source_s3_control.json')
            validate_u_runtime(body, profile)
            import subprocess
            import torch
            driver = subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version',
                        '--format=csv,noheader'], text=True, encoding='utf-8').strip()
            body['gate_numerics'] = {'driver': driver, 'cuda': torch.version.cuda,
                'cudnn': torch.backends.cudnn.version(),
                'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                'cudnn_benchmark': torch.backends.cudnn.benchmark,
                'cudnn_deterministic': torch.backends.cudnn.deterministic,
                'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
                'scope': 'at gate, before trainer initializes deterministic settings; legacy values not recorded'}
        return {**body, 'identity_sha256': canonical_hash(body)}
    for rel, expected in BASE_CODE.items():
        check_equal('v2 baseline ' + rel, expected, sha256(root/rel))
    ready = root/'data/source_v1/READY.json'
    if not ready.is_file():
        ready = root/'data/soda10m/splits/source_v1/READY.json'
    check_equal('frozen READY', READY, sha256(ready))
    check_equal('unchanged source-A recipe', BASE_CONFIG, sha256(root/'configs/source_a.yaml'))
    provenance = read_json(root/'weights/yolo11m.provenance.json')
    check_equal('official pretrained identity', PRETRAINED, provenance['sha256'])
    check_equal('pretrained weight', PRETRAINED, sha256(root/'weights/yolo11m.pt'))
    code = {**BASE_CODE, **{f'scripts/{name}': sha256(root/'scripts'/name)
                          for name in ('source_resources.py', 'source_cloud_gate.py')}}
    identity = {'base_archive_sha256': BASE_ARCHIVE, 'source_ready_sha256': READY,
                'config_file_sha256': sha256(root/'configs/source_a.yaml'),
                'pretrained_sha256': provenance['sha256'], 'code_identity': code,
                'training_recipe_changed': False, 'frozen_data_changed': False}
    return {**identity, 'identity_sha256': canonical_hash(identity)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--enforce', action='store_true')
    p.add_argument('--expected-identity-sha256')
    p.add_argument('--identity-output', type=Path)
    p.add_argument('--config', type=Path)
    p.add_argument('--arm', choices=['R', 'S', 'U'], default='R')
    p.add_argument('--seed', type=int, choices=[17, 29, 43], default=17)
    args = p.parse_args()
    host_memory = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') if hasattr(os, 'sysconf') else 0
    resources = snapshot(os.cpu_count(), host_memory)
    reasons = resource_rejections(resources)
    identity = runtime_identity(args.root.resolve(), args.config, args.arm, args.seed)
    if identity.get('experiment') == 'source_s3' and identity['platform'] != 'Linux':
        reasons.append('S3 cloud preflight requires Linux')
    if args.expected_identity_sha256:
        check_equal('cloud runtime identity', args.expected_identity_sha256, identity['identity_sha256'])
    state = {'status': 'passed' if not reasons else 'rejected', 'resources': resources,
             'rejections': reasons, 'runtime_identity': identity,
             'note': 'Underlying v2 environment.json host view is supplemental; this gate uses cgroup+affinity.'}
    write_json(args.output, state)
    if args.identity_output:
        if args.identity_output.exists():
            check_equal('existing cloud runtime identity', read_json(args.identity_output), identity)
        else:
            pending = args.identity_output.with_suffix(args.identity_output.suffix + '.part')
            write_json(pending, identity)
            pending.replace(args.identity_output)
    print(__import__('json').dumps(state))
    if args.enforce and reasons:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
