"""Source A runner. All real data access is gated by immutable READY + exact whitelist."""
from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import tempfile
import urllib.request
from pathlib import Path

from source_contract import (ContractError, NAMES, canonical_hash, check_equal, read_json,
                             safe_path, sha256, stage_dataset, verify_source, write_json)

ROOT = Path(__file__).resolve().parents[1]


def configure_paths():
    cache = Path(os.environ.get('NIGHT_SOURCE_SCRATCH', tempfile.gettempdir())) / 'night_source_runtime'
    if cache.resolve().is_relative_to(ROOT):
        raise ContractError('Runtime scratch must stay outside the project')
    for directory in ('tmp', 'pip', 'ultralytics', 'torch', 'matplotlib'):
        (cache / directory).mkdir(parents=True, exist_ok=True)
    for name, directory in {'TEMP': 'tmp', 'TMP': 'tmp', 'TMPDIR': 'tmp', 'PIP_CACHE_DIR': 'pip',
                            'YOLO_CONFIG_DIR': 'ultralytics', 'TORCH_HOME': 'torch',
                            'MPLCONFIGDIR': 'matplotlib'}.items():
        os.environ[name] = str(cache / directory)
    os.environ['PYTHONUTF8'] = '1'
    os.environ['PYTHONIOENCODING'] = 'utf-8:surrogateescape'
    os.environ['WANDB_MODE'] = 'disabled'
    os.environ['YOLO_AUTOINSTALL'] = 'false'


def config_at(path):
    import yaml
    config = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    check_equal('config dataset version', 'source_v1', config['dataset']['version'])
    check_equal('config names', NAMES, config['dataset']['names'])
    check_equal('config purpose train', 'source_train', config['dataset']['train_purpose'])
    check_equal('config purpose dev', 'source_dev', config['dataset']['dev_purpose'])
    check_equal('selection', [0, 2], config['selection']['class_ids'])
    check_equal('tie policy', 'retain_earlier', config['selection']['tie_policy'])
    if config.get('experiment') == 'source_s3':
        from source_contract import S3_EVALUATION
        check_equal('S3 evaluation profile', S3_EVALUATION, config['monitor']['profile'])
        check_equal('S3 shared AMP initial scale', 1., config['numerics']['amp_initial_scale'])
        check_equal('S3 sampling algorithm', 'small_thin_union_weighted_v1', config['sampling']['algorithm'])
        if config['sampling']['arm'] not in ('R', 'S') or config['training']['seed'] not in (17, 29, 43):
            raise ContractError('Unregistered S3 arm/seed')
        for key, value in {'epochs': 100, 'batch': 16, 'workers': 4, 'imgsz': 640,
                           'nbs': 64, 'close_mosaic': 10, 'amp': True, 'cache': False,
                           'time': None, 'fraction': 1., 'compile': False, 'multi_scale': 0.}.items():
            check_equal('S3 locked '+key, value, config['training'][key])
        check_equal('S3 fixed weights', (2, 1, 4000),
                    tuple(config['sampling'][k] for k in ('target_weight', 'other_weight', 'slots')))
        check_equal('S3 membership counts', (922, 1617),
                    tuple(config['sampling'][k] for k in ('target_images', 'target_gt')))
        check_equal('S3 full preflight schedule', (2, 100, 4000, 500),
                    tuple(config['preflight'][k] for k in ('epochs', 'schedule_epochs', 'train_images', 'dev_images')))
        check_equal('S3 role checkpoints only', 0, config['selection']['checkpoint_period'])
        check_equal('S3 policy paths', {'rules': 'RULES.md', 'reference': 'runs/a_001/train/best_dev_metrics.json',
                    'diagnostic_reference': 'runs/a_001_diag/dev/metrics.json', 'slice_reference': 'runs/a_001_diag/slices.csv'}, config['policy'])
        check_equal('S3 full monitor', (True, 1, 1, 0),
                    tuple(config['monitor'][k] for k in ('enabled', 'every', 'batch', 'workers')))
        check_equal('S3 policy', 'six_class_rules_v1', config['selection']['policy'])
        check_equal('S3 stopping', 'horizon_only', config['selection']['stop_policy'])
        baseline = yaml.safe_load((ROOT/'runs/a_001/train/args.yaml').read_text(encoding='utf-8'))
        excluded = {'model', 'data', 'project', 'name', 'save_dir', 'resume', 'task', 'mode', 'seed', 'patience', 'plots'}
        check_equal('S3 explicit original fields', set(baseline)-excluded,
                    set(config['training'])-excluded)
        for key in set(baseline)-excluded:
            check_equal('S3 original recipe '+key, baseline[key], config['training'][key])
    return config


def environment(config, output, cloud=False):
    import torch
    import torchvision
    import ultralytics
    import psutil
    if sys.prefix == sys.base_prefix:
        raise ContractError('Use a dedicated venv; refusing a base/learning interpreter for training')
    output.mkdir(parents=True, exist_ok=True)
    actual = {'python': '.'.join(platform.python_version_tuple()[:2]), 'torch': torch.__version__,
              'torchvision': torchvision.__version__, 'ultralytics': ultralytics.__version__}
    for k, v in config['environment'].items():
        check_equal('environment.' + k, v, actual[k])
    check_equal('CUDA toolkit', '13.0', torch.version.cuda)
    if not torch.cuda.is_available():
        raise ContractError('CUDA not available')
    properties = torch.cuda.get_device_properties(0)
    actual.update({'python_full': sys.version, 'executable': sys.executable, 'prefix': sys.prefix,
                   'base_prefix': sys.base_prefix, 'platform': platform.platform(),
                   'cuda_runtime': torch.version.cuda, 'gpu': properties.name,
                   'gpu_memory_bytes': properties.total_memory,
                   'cpu_count': os.cpu_count(), 'ram_bytes': psutil.virtual_memory().total,
                   'disk_free_bytes': shutil.disk_usage(output.resolve()).free,
                   'disk_total_bytes': shutil.disk_usage(output.resolve()).total,
                   'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}})
    if cloud:
        check_equal('cloud operating system', 'Linux', platform.system())
        cfg = config['cloud']
        if cfg['gpu_name_contains'] not in actual['gpu'] or torch.cuda.device_count() != 1:
            raise ContractError(f'Expected single RTX 4090; actual={actual["gpu"]}, count={torch.cuda.device_count()}')
        for label, value, minimum in [('VRAM GiB', actual['gpu_memory_bytes']/2**30, cfg['min_vram_gib']),
                                      ('CPU', actual['cpu_count'], cfg['min_cpus']),
                                      ('RAM GiB', actual['ram_bytes']/2**30, cfg['min_ram_gib']),
                                      ('free disk GiB', actual['disk_free_bytes']/2**30, cfg['min_free_disk_gib']),
                                      ('total disk GiB', actual['disk_total_bytes']/2**30, cfg['min_total_disk_gib'])]:
            if value < minimum:
                raise ContractError(f'{label}: expected>={minimum}; actual={value}')
    # A small forward/backward verifies CUDA kernel and compiled torchvision NMS availability.
    x = torch.randn(8, 8, device='cuda', requires_grad=True)
    x.square().mean().backward()
    torchvision.ops.nms(torch.tensor([[0., 0., 1., 1.]], device='cuda'), torch.ones(1, device='cuda'), .5)
    write_json(output / 'environment.json', actual)
    freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True, encoding='utf-8')
    (output / 'pip-freeze.txt').write_text(freeze, encoding='utf-8')
    return actual


def acquire_weights(config):
    target = ROOT / config['model']['path']
    provenance = ROOT / config['model']['provenance']
    if target.exists() or provenance.exists():
        info = read_json(provenance)
        check_equal(target, info['sha256'], sha256(target))
        check_equal('weight source', config['model']['url'], info['url'])
        return info
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix('.pt.part')
    request = urllib.request.Request(config['model']['url'], headers={'User-Agent': 'NightDetection-sourceA/1'})
    with urllib.request.urlopen(request, timeout=90) as response, part.open('wb') as f:
        shutil.copyfileobj(response, f)
    if part.stat().st_size < 1_000_000:
        raise ContractError('Pretrained download too small')
    part.replace(target)
    info = {'url': config['model']['url'], 'sha256': sha256(target), 'bytes': target.stat().st_size,
            'scope': 'official COCO pretrained YOLO11m; independently downloaded, never smoke checkpoint',
            'downloaded_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    write_json(provenance, info)
    return info


def code_identity():
    return {f'scripts/{name}': sha256(ROOT / 'scripts' / name)
            for name in ('source_contract.py', 'source_trainer.py', 'train_source.py',
                         'evaluate.py', 'train_monitor.py', 'detection_metrics.py',
                         'source_sampling.py', 'source_policy.py')}


def identity(config, contract, weights, batch, scope):
    result = {'ready_sha256': contract['ready_sha256'],
            'data_fingerprint': contract['ready']['data_fingerprint'],
            'normalization_version': contract['ready']['normalization_version'],
            'grouping_protocol_sha256': contract['ready']['grouping_protocol']['sha256'],
            'config_sha256': canonical_hash(config), 'code': code_identity(),
            'pretrained_sha256': weights['sha256'], 'batch': batch, 'scope': scope}
    if config.get('experiment') == 'source_s3':
        from source_policy import load_rules, reference_at
        runtime_hash = os.environ.get('NIGHT_SOURCE_RUNTIME_IDENTITY')
        if not runtime_hash:
            raise ContractError('S3 identity requires the verified new runner runtime identity')
        result['runtime_identity_sha256'] = runtime_hash
        result['s3_policy'] = {'rules_sha256': load_rules(ROOT/'RULES.md')['sha256'],
                               'references': reference_at(ROOT)['identity'],
                               'plan_sha256': sha256(ROOT/'specs/s3/plan.md'),
                               'original_args_sha256': sha256(ROOT/'runs/a_001/train/args.yaml')}
        result['evaluation_policy'] = config['monitor']['profile']
    return result


def run_segment(config, contract, weights, run_dir, dataset_yaml, batch, scope,
                resume=None, stop_after_epoch=None):
    segment_started = time.perf_counter()
    import torch
    from ultralytics import YOLO
    from ultralytics.cfg import DEFAULT_CFG_DICT
    from source_trainer import SourceTrainer
    s3 = config.get('experiment') == 'source_s3'
    args = dict(config['training'])
    if scope in ('smoke', 'preflight') and not s3:
        args.update(epochs=config['smoke']['epochs'], workers=0 if scope == 'smoke' else 4)
    args.update(batch=batch, model=str(ROOT / config['model']['path']), data=str(dataset_yaml.resolve()),
                device=0, project=str(run_dir.parent.resolve()), name=run_dir.name,
                save_dir=str(run_dir.resolve()), exist_ok=True, save_period=-1,
                val=True, split='val', save=True)
    expected = {k: v for k, v in args.items() if k in config['training']}
    current_identity = identity(config, contract, weights, batch, scope)
    audit_dir = run_dir / ('audit_resume' if resume else 'audit_initial')
    audit_dir.mkdir(parents=True, exist_ok=True)
    if resume:
        sidecar = read_json(resume.with_suffix('.pt.sha256.json'))
        check_equal('resume weight hash', sidecar['sha256'], sha256(resume))
        check_equal('resume identity', current_identity, sidecar['identity'])
        ckpt = torch.load(resume, map_location='cpu', weights_only=False)
        check_equal('resume embedded identity', current_identity, ckpt.get('source_identity'))
        if ckpt.get('optimizer') is None or not 0 <= ckpt.get('epoch', -1) < args['epochs'] - 1:
            raise ContractError('Checkpoint is finished or lacks optimizer; cannot resume')
        args.update(model=str(resume.resolve()), resume=str(resume.resolve()))
    elif run_dir.exists() and any(x.name.startswith('audit_') is False for x in run_dir.iterdir()):
        raise ContractError(f'Refusing existing run: {run_dir}')
    write_json(audit_dir / 'config_resolved.json', config)
    write_json(audit_dir / 'defaults_locked.json', DEFAULT_CFG_DICT)
    write_json(audit_dir / 'trainer_passed.json', args)
    if s3:
        from source_policy import load_rules, reference_at
        gate = read_json(Path(os.environ['NIGHT_SOURCE_RESOURCE_GATE']))
        write_json(audit_dir/'resource_gate.json', gate)
        write_json(run_dir/'policy_inputs.json', {'rules': load_rules(ROOT/'RULES.md'), 'reference': reference_at(ROOT)})
        for rel in ['specs/s3/plan.md', 'RULES.md', 'configs/source_s3.yaml',
                    *code_identity()]:
            target = run_dir/'code'/rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                check_equal('immutable run snapshot '+rel, sha256(ROOT/rel), sha256(target))
            else:
                shutil.copyfile(ROOT/rel, target)
    write_json(run_dir / 'source_run.json', {'identity': current_identity, 'scope': scope,
               'formal_A': scope == 'formal', 'final_tests_run': False,
               'dataset_yaml': str(dataset_yaml.resolve())})
    if s3:
        from source_trainer import S3Trainer
        from source_policy import SelectionPolicy, load_rules, reference_at
        trainer = S3Trainer(overrides=args)
        trainer.s3_amp_initial_scale = config['numerics']['amp_initial_scale']
        trainer.s3_contract, trainer.s3_arm = contract, config['sampling']['arm']
        trainer.s3_policy = SelectionPolicy(load_rules(ROOT/'RULES.md'), reference_at(ROOT), args['epochs'])
        trainer.s3_sandbox = dataset_yaml.resolve().parent
    else:
        trainer = SourceTrainer(overrides=args)
    trainer.source_identity = current_identity
    trainer.expected_batch = batch
    trainer.source_expected_args = expected
    trainer.stop_after_epoch = stop_after_epoch
    observer = None
    if scope in ('preflight', 'formal') and not s3:
        from train_monitor import attach
        observer = attach(trainer, run_dir, contract['ready']['counts']['source_train'],
                          config.get('monitor', {}))
    trainer.attach_audits(audit_dir)
    try:
        trainer.train()
    finally:
        if observer is not None:
            observer.close()
    runtime = read_json(audit_dir / 'runtime.json')
    if s3:
        runtime.update(segment_including_setup_final_seconds=time.perf_counter()-segment_started,
                       final_eval_seconds=read_json(run_dir/'final_eval/run.json')['seconds'])
        write_json(audit_dir/'runtime.json', runtime)
    if runtime['optimizer_steps'] < 1 or runtime['batches'] < 1:
        raise ContractError(f'Forward/backward/optimizer evidence missing: {runtime}')
    best = run_dir / 'weights' / 'best.pt'
    if s3:
        roles = read_json(run_dir/'checkpoint_roles.json')
        role = 'best' if 'best' in roles else 'diagnostic'
        best = run_dir/roles[role]['path']
    model = YOLO(str(best))
    check_equal('reloaded model classes', NAMES, list(model.names.values()))
    write_json(audit_dir / 'checkpoint_reload.json', {'best_sha256': sha256(best),
               'selected_role': role if s3 else 'best',
               'names': model.names, 'reloaded': True, 'runtime': runtime})
    if s3:
        from source_release import seal_training
        seal_training(run_dir, scope)
    del trainer, model
    torch.cuda.empty_cache()
    return runtime


def check_checkpoint(path, contract, config, weights):
    info = read_json(path.with_suffix('.pt.sha256.json'))
    check_equal(path, info['sha256'], sha256(path))
    ident = info['identity']
    check_equal('resume config', canonical_hash(config), ident['config_sha256'])
    check_equal('resume READY', contract['ready_sha256'], ident['ready_sha256'])
    check_equal('resume pretrained', weights['sha256'], ident['pretrained_sha256'])
    check_equal('resume code', code_identity(), ident['code'])
    return ident


def verify_recipe(path, expected_sha256, expected_identity):
    check_equal('frozen recipe SHA-256', expected_sha256, sha256(path))
    recipe = read_json(path)
    check_equal('recipe.status', 'frozen', recipe['status'])
    if recipe['identity']['batch'] not in (16, 8):
        raise ContractError(f'Unauthorized formal batch: {recipe["identity"]["batch"]}')
    check_equal('recipe identity', expected_identity, recipe['identity'])
    if 's3_policy' in expected_identity:
        check_equal('S3 recipe batch', 16, recipe['identity']['batch'])
        check_equal('S3 preflight schedule', 100, recipe['schedule_epochs'])
        check_equal('S3 preflight complete epochs', 2, recipe['preflight_runtime']['epoch_1based'])
        check_equal('S3 preflight full counts', [4000, 500], recipe['preflight_counts'])
    return recipe


def recover_s3_outputs(run_dir, epoch):
    """Replay only an incomplete epoch; never append duplicates to completed evidence."""
    import csv
    for name in ('source_metrics.jsonl', 'timings.jsonl', 'train_metrics.csv', 'dev_metrics.csv', 'results.csv'):
        path = safe_path(run_dir, name)
        if not path.exists():
            continue
        if path.suffix == '.jsonl':
            kept = []
            for line in path.read_text(encoding='utf-8').splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    break
                if row['epoch'] <= epoch:
                    kept.append(line)
            path.write_text(''.join(line+'\n' for line in kept), encoding='utf-8')
        else:
            with path.open(encoding='utf-8', newline='') as stream:
                reader = csv.DictReader(stream)
                fields = reader.fieldnames
                epoch_key = next(k for k in fields if k.strip() == 'epoch')
                rows = [r for r in reader if int(r[epoch_key]) <= epoch]
            with path.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader(); writer.writerows(rows)
    for folder in ('observations', 'sampling'):
        for path in (run_dir/folder).glob('epoch*.*'):
            import re
            match = re.fullmatch(r'epoch(\d+)\.(json|npz)', path.name)
            if match and int(match[1]) > epoch:
                safe_path(run_dir, path.relative_to(run_dir).as_posix()).unlink()


def validate_s3_resume(run_dir, checkpoint, ckpt):
    if checkpoint.name != 'last.pt':
        raise ContractError('S3 resumes only the latest complete last.pt, never an earlier selected role')
    if (run_dir/'manifest.json').exists() and read_json(run_dir/'manifest.json')['status'] == 'completed_formal':
        raise ContractError('Completed S3 formal run cannot be truncated/resumed')
    epoch = ckpt['epoch']+1
    check_equal('resumed source metadata', ckpt['source_identity'], read_json(run_dir/'source_run.json')['identity'])
    roles = read_json(run_dir/'checkpoint_roles.json')
    check_equal('resumed last epoch', epoch, roles['last']['epoch'])
    actual = sha256(checkpoint)
    check_equal('resumed last role bytes', actual, roles['last']['sha256'])
    obs = read_json(run_dir/f'observations/epoch{epoch:03d}.json')
    check_equal('resumed committed observation bytes', actual, obs['checkpoint_sha256'])
    check_equal('resumed complete PR bytes', obs['pr_sha256'], sha256(run_dir/f'observations/epoch{epoch:03d}.npz'))
    sample = read_json(run_dir/f'sampling/epoch{epoch:03d}.json')
    check_equal('resumed sampler bytes', sample['consumed_sha256'], sha256(run_dir/f'sampling/epoch{epoch:03d}.npz'))
    for role in ('best', 'diagnostic'):
        state = ckpt['source_resume']['policy'][role]
        if state is None:
            if role in roles:
                raise ContractError('Role appeared after the committed resume point: '+role)
        else:
            check_equal('resumed '+role+' epoch', state['epoch'], roles[role]['epoch'])
            check_equal('resumed '+role+' bytes', roles[role]['sha256'], sha256(safe_path(run_dir, roles[role]['path'])))
    return epoch


def stage_run(contract, destination, limits, s3, stack):
    if s3:
        parent = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='source_s3_data_')))
        destination = parent/'data'
    return stage_dataset(contract, destination, limits)


def _main(stack):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['environment', 'download', 'verify', 'smoke', 'preflight', 'formal', 'resume'])
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data/soda10m/splits/source_v1')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/source_a.yaml')
    parser.add_argument('--ready-sha256')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/source_a')
    parser.add_argument('--recipe', type=Path)
    parser.add_argument('--recipe-sha256')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--hourly-price', type=float)
    parser.add_argument('--currency')
    parser.add_argument('--arm', choices=['R', 'S'])
    parser.add_argument('--seed', type=int, choices=[17, 29, 43])
    parser.add_argument('--resource-gate', type=Path)
    parser.add_argument('--runtime-identity-sha256')
    args = parser.parse_args()
    configure_paths()
    config = config_at(args.config)
    if args.arm is not None or args.seed is not None:
        if config.get('experiment') != 'source_s3':
            parser.error('--arm/--seed apply only to source_s3')
        if args.arm is not None:
            config['sampling']['arm'] = args.arm
        if args.seed is not None:
            config['training']['seed'] = args.seed
    s3 = config.get('experiment') == 'source_s3'
    if s3 and args.mode == 'smoke':
        parser.error('S3 uses full two-epoch preflight at the 100-epoch schedule; no sampled smoke')
    if s3 and args.mode in ('preflight', 'formal', 'resume'):
        if not args.resource_gate or not args.runtime_identity_sha256:
            parser.error('S3 training must enter through run_source_cloud_job.py with its new resource/runtime gate')
        from source_cloud_gate import runtime_identity
        gate = read_json(args.resource_gate)
        check_equal('S3 resource gate status', 'passed', gate['status'])
        current = runtime_identity(ROOT, args.config, config['sampling']['arm'], config['training']['seed'])
        check_equal('S3 external runtime identity', args.runtime_identity_sha256, current['identity_sha256'])
        check_equal('S3 resource gate identity', current, gate['runtime_identity'])
        os.environ['NIGHT_SOURCE_RUNTIME_IDENTITY'] = args.runtime_identity_sha256
        os.environ['NIGHT_SOURCE_RESOURCE_GATE'] = str(args.resource_gate.resolve())
    if args.mode == 'download':
        print(json.dumps(acquire_weights(config), ensure_ascii=False))
        return
    if args.mode == 'environment':
        environment(config, args.output, cloud=False)
        print('ENVIRONMENT_OK')
        return
    if not args.ready_sha256:
        parser.error('--ready-sha256 must be the externally recorded frozen READY SHA-256')
    contract = verify_source(args.data_root, args.ready_sha256)
    if args.mode == 'verify':
        print(json.dumps({'status': 'SOURCE_VERIFIED', 'ready_sha256': args.ready_sha256,
                          'counts': contract['ready']['counts'], 'files': len(contract['files'])}))
        return
    weights = acquire_weights(config)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    environment(config, args.output / 'environment', cloud=args.mode in ('preflight', 'formal'))
    if args.mode == 'resume':
        if not args.checkpoint:
            parser.error('--checkpoint is required')
        ident = check_checkpoint(args.checkpoint, contract, config, weights)
        if s3 and ident['scope'] != 'formal':
            raise ContractError('S3 preflight is a fresh two-epoch check; failed preflight must restart from COCO')
        if ident['scope'] in ('formal', 'preflight'):
            environment(config, args.output / 'environment', cloud=True)
        run_dir = args.checkpoint.resolve().parents[1]
        metadata = read_json(run_dir / 'source_run.json')
        dataset_yaml = Path(metadata['dataset_yaml'])
        # Re-stage from current verified input if a run is relocated; no blind checkpoint data path.
        dataset_yaml, _ = stage_run(contract, run_dir / f'data_resume_{time.time_ns()}',
                        {'source_train': 64, 'source_dev': 32} if ident['scope'] == 'smoke' else None, s3, stack)
        if s3:
            ckpt = __import__('torch').load(args.checkpoint, map_location='cpu', weights_only=False)
            epoch = validate_s3_resume(run_dir, args.checkpoint, ckpt)
            recover_s3_outputs(run_dir, epoch)
        run_segment(config, contract, weights, run_dir, dataset_yaml, ident['batch'], ident['scope'],
                    resume=args.checkpoint, stop_after_epoch=2 if s3 and ident['scope'] == 'preflight' else None)
        print('RESUME_COMPLETE')
        return
    if args.mode == 'formal':
        if not args.recipe or not args.recipe_sha256:
            parser.error('--recipe and --recipe-sha256 from successful preflight are required')
        batch = read_json(args.recipe)['identity']['batch']
        verify_recipe(args.recipe, args.recipe_sha256, identity(config, contract, weights, batch, 'formal'))
        dataset_yaml, _ = stage_run(contract, args.output / 'data', None, s3, stack)
        run_segment(config, contract, weights, args.output / 'train', dataset_yaml, batch, 'formal')
        print('FORMAL_A_TRAINING_COMPLETE_SOURCE_DEV_ONLY')
        return
    import torch
    if args.mode == 'preflight' and args.hourly_price and (args.hourly_price < 0 or not args.currency):
        parser.error('Positive actual --hourly-price requires --currency; omit price while unknown')
    batches = [16] if s3 else [2, 1] if args.mode == 'smoke' else [16, 8]
    for batch in batches:
        run_dir = args.output / f'{args.mode}_batch{batch}'
        dataset_yaml, selections = stage_run(contract, args.output / f'data_{args.mode}_batch{batch}',
                          {'source_train': 64, 'source_dev': 32} if args.mode == 'smoke' else None, s3, stack)
        try:
            runtime = run_segment(config, contract, weights, run_dir, dataset_yaml, batch, args.mode,
                                  stop_after_epoch=2 if s3 else 1 if args.mode == 'smoke' else None)
            if args.mode == 'smoke':
                checkpoint = run_dir / 'weights/last.pt'
                first = read_json(checkpoint.with_suffix('.pt.sha256.json'))
                check_equal('smoke interruption epoch', 0, first['epoch_0based'])
                runtime = run_segment(config, contract, weights, run_dir, dataset_yaml, batch, 'smoke',
                                      resume=checkpoint)
                check_equal('smoke resumed final epoch', 2, runtime['epoch_1based'])
                write_json(args.output / 'SMOKE_PASSED.json', {'scope': 'smoke only; not formal A or performance',
                           'identity': identity(config, contract, weights, batch, 'smoke'),
                           'train_images': 64, 'dev_images': 32, 'epochs': 2,
                           'resumed_start_epoch_0based': runtime['start_epoch_0based'],
                           'class_coverage': {p: sorted({c for r in rs for c in r['classes']})
                                              for p, rs in selections.items()}, 'runtime': runtime})
            else:
                recipe = {'schema_version': 1, 'status': 'frozen',
                          'identity': identity(config, contract, weights, batch, 'formal'),
                          'preflight_runtime': runtime, 'preflight_scope': '2 epochs; not formal A',
                          'hourly_price': args.hourly_price,
                          'currency': args.currency,
                          'estimated_100_epoch_hours': runtime['mean_epoch_seconds'] * 100 / 3600,
                          'estimated_100_epoch_cost': (runtime['mean_epoch_seconds'] * 100 / 3600 *
                                                       args.hourly_price) if args.hourly_price else None}
                if s3:
                    recipe.update(schema_version=2, schedule_epochs=100, preflight_counts=[4000, 500],
                                  preflight_scope='full 4000/500; stop after epoch2 of 100; cold/warmup only',
                                  timing_limit='not steady-state or close-mosaic measurement',
                                  measured_final_eval_seconds=runtime['final_eval_seconds'],
                                  measured_segment_including_setup_final_seconds=runtime['segment_including_setup_final_seconds'])
                write_json(args.output / 'FROZEN_RECIPE.json', recipe)
                write_json(args.output / 'FROZEN_RECIPE.sha256.json', {
                    'sha256': sha256(args.output / 'FROZEN_RECIPE.json'),
                    'instruction': 'Record this SHA-256 outside the recipe; pass --recipe-sha256 to formal'})
            print(args.mode.upper() + '_PASSED')
            return
        except torch.cuda.OutOfMemoryError as e:
            write_json(run_dir / 'OOM.json', {'batch': batch, 'error': str(e), 'scope': args.mode})
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            if batch == batches[-1]:
                raise ContractError(f'{args.mode}: OOM at minimum authorized batch={batch}; stop and diagnose') from e


def main():
    with contextlib.ExitStack() as stack:
        _main(stack)


if __name__ == '__main__':
    try:
        main()
    except (ContractError, FileNotFoundError, KeyError) as e:
        print(f'SOURCE_A_ERROR: {e}', file=sys.stderr)
        raise SystemExit(2)
