"""Source A runner. All real data access is gated by immutable READY + exact whitelist."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from source_contract import (ContractError, NAMES, canonical_hash, check_equal, read_json,
                             sha256, stage_dataset, verify_source, write_json)

ROOT = Path(__file__).resolve().parents[1]


def configure_paths():
    cache = ROOT / '.cache' / 'training'
    for directory in ('tmp', 'pip', 'ultralytics', 'torch', 'matplotlib'):
        (cache / directory).mkdir(parents=True, exist_ok=True)
    for name, directory in {'TEMP': 'tmp', 'TMP': 'tmp', 'TMPDIR': 'tmp', 'PIP_CACHE_DIR': 'pip',
                            'YOLO_CONFIG_DIR': 'ultralytics', 'TORCH_HOME': 'torch',
                            'MPLCONFIGDIR': 'matplotlib'}.items():
        os.environ[name] = str(cache / directory)
    os.environ['PYTHONUTF8'] = '1'
    os.environ['PYTHONIOENCODING'] = 'utf-8:surrogateescape'
    os.environ['WANDB_MODE'] = 'disabled'


def config_at(path):
    import yaml
    config = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    check_equal('config dataset version', 'source_v1', config['dataset']['version'])
    check_equal('config names', NAMES, config['dataset']['names'])
    check_equal('config purpose train', 'source_train', config['dataset']['train_purpose'])
    check_equal('config purpose dev', 'source_dev', config['dataset']['dev_purpose'])
    check_equal('selection', [0, 2], config['selection']['class_ids'])
    check_equal('tie policy', 'retain_earlier', config['selection']['tie_policy'])
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
                         'evaluate.py', 'train_monitor.py')}


def identity(config, contract, weights, batch, scope):
    return {'ready_sha256': contract['ready_sha256'],
            'data_fingerprint': contract['ready']['data_fingerprint'],
            'normalization_version': contract['ready']['normalization_version'],
            'grouping_protocol_sha256': contract['ready']['grouping_protocol']['sha256'],
            'config_sha256': canonical_hash(config), 'code': code_identity(),
            'pretrained_sha256': weights['sha256'], 'batch': batch, 'scope': scope}


def run_segment(config, contract, weights, run_dir, dataset_yaml, batch, scope,
                resume=None, stop_after_epoch=None):
    import torch
    from ultralytics import YOLO
    from ultralytics.cfg import DEFAULT_CFG_DICT
    from source_trainer import SourceTrainer
    args = dict(config['training'])
    if scope in ('smoke', 'preflight'):
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
    write_json(run_dir / 'source_run.json', {'identity': current_identity, 'scope': scope,
               'formal_A': scope == 'formal', 'final_tests_run': False,
               'dataset_yaml': str(dataset_yaml.resolve())})
    trainer = SourceTrainer(overrides=args)
    trainer.source_identity = current_identity
    trainer.expected_batch = batch
    trainer.source_expected_args = expected
    trainer.stop_after_epoch = stop_after_epoch
    observer = None
    if scope in ('preflight', 'formal'):
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
    if runtime['optimizer_steps'] < 1 or runtime['batches'] < 1:
        raise ContractError(f'Forward/backward/optimizer evidence missing: {runtime}')
    best = run_dir / 'weights' / 'best.pt'
    model = YOLO(str(best))
    check_equal('reloaded model classes', NAMES, list(model.names.values()))
    write_json(audit_dir / 'checkpoint_reload.json', {'best_sha256': sha256(best),
               'names': model.names, 'reloaded': True, 'runtime': runtime})
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
    return recipe


def main():
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
    args = parser.parse_args()
    configure_paths()
    config = config_at(args.config)
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
        if ident['scope'] in ('formal', 'preflight'):
            environment(config, args.output / 'environment', cloud=True)
        run_dir = args.checkpoint.resolve().parents[1]
        metadata = read_json(run_dir / 'source_run.json')
        dataset_yaml = Path(metadata['dataset_yaml'])
        # Re-stage from current verified input if a run is relocated; no blind checkpoint data path.
        dataset_yaml, _ = stage_dataset(contract, run_dir / f'data_resume_{time.time_ns()}',
                        {'source_train': 64, 'source_dev': 32} if ident['scope'] == 'smoke' else None)
        run_segment(config, contract, weights, run_dir, dataset_yaml, ident['batch'], ident['scope'],
                    resume=args.checkpoint)
        print('RESUME_COMPLETE')
        return
    if args.mode == 'formal':
        if not args.recipe or not args.recipe_sha256:
            parser.error('--recipe and --recipe-sha256 from successful preflight are required')
        batch = read_json(args.recipe)['identity']['batch']
        verify_recipe(args.recipe, args.recipe_sha256, identity(config, contract, weights, batch, 'formal'))
        dataset_yaml, _ = stage_dataset(contract, args.output / 'data')
        run_segment(config, contract, weights, args.output / 'train', dataset_yaml, batch, 'formal')
        print('FORMAL_A_TRAINING_COMPLETE_SOURCE_DEV_ONLY')
        return
    import torch
    if args.mode == 'preflight' and args.hourly_price and (args.hourly_price < 0 or not args.currency):
        parser.error('Positive actual --hourly-price requires --currency; omit price while unknown')
    batches = [2, 1] if args.mode == 'smoke' else [16, 8]
    for batch in batches:
        run_dir = args.output / f'{args.mode}_batch{batch}'
        dataset_yaml, selections = stage_dataset(contract, args.output / f'data_{args.mode}_batch{batch}',
                          {'source_train': 64, 'source_dev': 32} if args.mode == 'smoke' else None)
        try:
            runtime = run_segment(config, contract, weights, run_dir, dataset_yaml, batch, args.mode,
                                  stop_after_epoch=1 if args.mode == 'smoke' else None)
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


if __name__ == '__main__':
    try:
        main()
    except (ContractError, FileNotFoundError, KeyError) as e:
        print(f'SOURCE_A_ERROR: {e}', file=sys.stderr)
        raise SystemExit(2)
