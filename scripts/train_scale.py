"""Run or transfer a fixed-budget 640/960 fine-tuning comparison."""
from __future__ import annotations

import argparse
from copy import copy, deepcopy
from datetime import datetime
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import time
import warnings

from source_contract import ContractError, NAMES, canonical_hash, check_equal, read_json, sha256, stage_dataset, verify_source, verified_file, write_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'data/soda10m/splits/source_v1'
R = ROOT/'runs/a_002/train'
RULES = ROOT/'RULES.md'
READY = '72e75ebd99014c830e0227061e390e6aed768ae724b532a5c2c90fd8dec93e85'
R_SHA = 'e4c873eda51d1a2b9ab722a9cb8f8bc0de9091a320184a8dc6f49fb84421c6ab'
PREFLIGHT = ROOT/'runs/s3_17/scale_preflight.json'
REFERENCE = ROOT/'runs/s3_17/scale.json'
SIZES = (640, 960)
SEED, EPOCHS, BATCH, ACCUMULATE = 17, 8, 4, 8
LONG_EPOCHS = 24
LONG_OBSERVATIONS = (8, 16, 24)
MOSAIC_OFF_EPOCH = 7
LIBRARIES = {'torch': '2.14.0+cu130', 'torchvision': '0.29.0+cu130', 'ultralytics': '8.4.171'}


def save_json(path, payload):
    scratch = path.with_name(path.name+'.tmp')
    write_json(scratch, payload)
    scratch.replace(path)


def timestamp():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def sealed(root, manifest, filename):
    return verified_file(root, dict(manifest['files'][filename], path=filename))


def prepare_inputs(source, reference):
    manifest = read_json(reference/'manifest.json')
    check_equal('R ready hash', READY, manifest['identity']['ready_sha256'])
    role = read_json(sealed(reference, manifest, 'checkpoint_roles.json'))['diagnostic']
    check_equal('R checkpoint epoch and SHA', (80, R_SHA), (role['epoch'], role['sha256']))
    weight = sealed(reference, manifest, role['path'])
    check_equal('R on disk', R_SHA, sha256(weight))
    preflight = read_json(PREFLIGHT)
    check_equal('preflight microbatch', BATCH, preflight['selected_micro_batch'])
    check_equal('preflight scope', 'local_compute_preflight_passed', preflight['status'])
    check_equal('fixed R evaluation identity', preflight['fixed_inference_reference']['sha256'], sha256(REFERENCE))
    metadata = {n: importlib.metadata.version(n) for n in LIBRARIES}
    check_equal('locked library versions', LIBRARIES, metadata)
    contract = verify_source(source, READY)
    for purpose, count in (('source_train', 4000), ('source_dev', 500)):
        check_equal(purpose+' count', count, len(contract['manifests'][purpose]))
    return weight, sealed(reference, manifest, 'args.yaml'), preflight, metadata, contract


def state_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode('utf-8'))
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def learning_rates(epochs):
    return [.001 * (.1 + .9 * (1 + math.cos(math.pi * epoch / epochs)) / 2) for epoch in range(epochs)]


def run_observations(job):
    if job['identity']['epochs'] == LONG_EPOCHS:
        return LONG_OBSERVATIONS
    return (1,) if job['smoke'] else (EPOCHS,)


def observation_dir(arm, identity, epoch):
    return arm / f'epoch{epoch:02}' if identity['epochs'] == LONG_EPOCHS else arm


def evaluation_sizes(identity):
    return (960,) if identity['epochs'] == LONG_EPOCHS else SIZES


def configure(data, output, weights, raw_args, size, identity):
    import yaml
    raw = yaml.safe_load(raw_args.read_text(encoding='utf-8'))
    shared = {k: raw[k] for k in (
        'hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate', 'scale', 'shear', 'perspective',
        'flipud', 'fliplr', 'bgr', 'mosaic', 'mixup', 'cutmix', 'copy_paste', 'copy_paste_mode',
        'box', 'cls', 'cls_pw', 'dfl', 'momentum', 'weight_decay')}
    return dict(shared, model=str(weights), data=str(data), project=str(output.parent), name=output.name,
                exist_ok=True, imgsz=size, batch=BATCH, nbs=BATCH*ACCUMULATE, epochs=identity['epochs'],
                optimizer='SGD', lr0=.001, lrf=.1, cos_lr=True, warmup_epochs=0.,
                close_mosaic=identity['epochs'] + 1 - MOSAIC_OFF_EPOCH,
                resume=False, pretrained=True, cls_remap=True, seed=identity['seed'], deterministic=True,
                device='0', amp=True, cache=False, workers=0, multi_scale=0., rect=False,
                compile=False, channels_last=False, fraction=1., single_cls=False,
                save=False, plots=False, val=False, verbose=False, patience=0)


def initialize(data, output, weights, raw_args, size, identity, expected_names, expected_state):
    import numpy as np
    import torch
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils.torch_utils import ModelEMA
    from source_sampling import AuditedYOLODataset

    t = DetectionTrainer(overrides=configure(data, output, weights, raw_args, size, identity))
    ckpt = t.setup_model()
    check_equal('initialization is frozen R EMA', expected_state, state_hash(t.model))
    if ckpt.get('ema') is None:
        raise ContractError('R checkpoint is missing EMA')
    del ckpt
    t.model = t.model.to(t.device)
    t.set_model_attributes()
    t.stride = max(int(t.model.stride.max()), 32)
    for name, param in t.model.named_parameters():
        param.requires_grad_('.dfl' not in name)
    t.scaler = torch.amp.GradScaler('cuda', enabled=True, init_scale=1.)
    t.optimizer = t.build_optimizer(t.model, name='SGD', lr=t.args.lr0, momentum=t.args.momentum,
                                   decay=t.args.weight_decay, iterations=identity['epochs'] * 125)
    t._setup_scheduler()
    t.ema = ModelEMA(t.model)
    check_equal('fresh optimizer', 0, len(t.optimizer.state))
    dataset = AuditedYOLODataset(img_path=t.data['train'], imgsz=size, batch_size=BATCH,
                                 augment=True, hyp=t.args, rect=False, cache=False, single_cls=False,
                                 stride=t.stride, pad=0., prefix='scale train: ', task='detect', data=t.data)
    names = [Path(n).name for n in dataset.im_files]
    check_equal('exact gradient image whitelist', sorted(expected_names), sorted(names))
    for name, label in zip(names, dataset.labels, strict=True):
        observed = np.bincount(label['cls'].astype(int).reshape(-1), minlength=6).tolist()
        check_equal('frozen class counts '+name, expected_names[name]['counts'], observed)
    return t, dataset


def seed_epoch(dataset, seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    # Match the finite-loader epoch boundary even with the local workers=0 profile.
    for i in dataset.buffer:
        dataset.ims[i] = dataset.im_hw0[i] = dataset.im_hw[i] = None
    dataset.buffer.clear()

    def seed_transform(transform):
        inner = getattr(transform, 'transform', None)
        if hasattr(inner, 'set_random_seed'):
            inner.set_random_seed(seed % 2**32)
        for child in getattr(transform, 'transforms', ()):
            seed_transform(child)

    seed_transform(dataset.transforms)


def optimizer_step(t):
    import torch
    t.scaler.unscale_(t.optimizer)
    norm = torch.nn.utils.clip_grad_norm_(t.model.parameters(), 10.)
    if not bool(torch.isfinite(norm)):
        raise ContractError('Nonfinite gradient before optimizer step')
    scale = t.scaler.get_scale()
    t.scaler.step(t.optimizer)
    t.scaler.update()
    if t.scaler.get_scale() < scale:
        raise ContractError('AMP skipped an optimizer step')
    t.optimizer.zero_grad(set_to_none=True)
    t.ema.update(t.model)


def train_epoch(t, dataset, size, epoch, output):
    import torch
    import psutil
    from torch.utils.data import DataLoader
    from source_sampling import collate_audited, indices_for, seed_for

    seed = seed_for(t.args.seed, 'scale_augment', epoch)
    seed_epoch(dataset, seed)
    expected = indices_for(range(len(dataset)), t.args.seed, 'R', epoch).tolist()
    loader = DataLoader(dataset, batch_size=BATCH, sampler=expected, num_workers=0,
                        generator=torch.Generator().manual_seed(seed), collate_fn=collate_audited)
    t.model.train()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        t.scheduler.step()
    check_equal('whole accumulation at epoch boundary', 0, len(loader) % ACCUMULATE)
    free_host, free_gpu = psutil.virtual_memory().available, torch.cuda.mem_get_info()[0]
    torch.cuda.reset_peak_memory_stats()
    clock = time.perf_counter()
    # Reproducible main-image and augmentation-read order receipts.
    consumed, reads, sums = [], [], {}
    updates = t.ema.updates
    t.optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader, 1):
        for entry in batch.pop('_source_audit'):
            check_equal('first data read is anchor', entry['anchor'], entry['reads'][0])
            consumed.append(entry['anchor'])
            reads.append(entry['reads'])
        check_equal('gradient tensor shape', (BATCH, 3, size, size), tuple(batch['img'].shape))
        with torch.autocast('cuda', dtype=torch.float16):
            batch = t.preprocess_batch(batch)
            loss, items = t.model(batch)
            loss = loss.sum()
        numbers = torch.stack([loss.detach(), *[v.detach().sum() for v in items.values()]]).cpu().tolist()
        if not all(math.isfinite(x) for x in numbers):
            raise ContractError('Nonfinite training loss')
        for k, value in zip(('loss', *items), numbers, strict=True):
            sums[k] = sums.get(k, 0.) + value
        t.scaler.scale(loss).backward()
        if step % ACCUMULATE == 0:
            optimizer_step(t)
        if step == ACCUMULATE or step % 200 == 0:
            free_host = min(free_host, psutil.virtual_memory().available)
            free_gpu = min(free_gpu, torch.cuda.mem_get_info()[0])
            progress = {'epoch': epoch+1, 'batches': step, 'epoch_batches': len(loader),
                        'updates': t.ema.updates, 'elapsed_seconds': time.perf_counter()-clock,
                        'host_available_bytes': free_host, 'gpu_free_bytes': free_gpu}
            save_json(output/'progress.json', progress)
            print(f'{size} epoch {epoch+1}/{t.args.epochs}, batch {step}/{len(loader)}, updates {t.ema.updates}', flush=True)
    check_equal('actual anchors in registered order', expected, consumed)
    check_equal('exact optimizer update count', len(loader)//ACCUMULATE, t.ema.updates-updates)
    if hasattr(t.model.criterion, 'update'):
        t.model.criterion.update()
    torch.cuda.synchronize()
    return {'epoch': epoch+1, 'batches': len(loader), 'updates': t.ema.updates, 'main_images': len(consumed),
            'order_sha256': canonical_hash(consumed), 'reads_sha256': canonical_hash(reads),
            'extra_reads': sum(map(len, reads))-len(consumed),
            'loss': {k:v/len(loader) for k,v in sums.items()},
            'lr': [g['lr'] for g in t.optimizer.param_groups], 'amp_scale': t.scaler.get_scale(),
            'elapsed_seconds': time.perf_counter()-clock, 'peak_cuda_allocated': torch.cuda.max_memory_allocated(),
            'peak_cuda_reserved': torch.cuda.max_memory_reserved(), 'sampled_host_available': free_host,
            'sampled_gpu_available': free_gpu}


def store_checkpoint(t, history, identity, output):
    import torch
    path = output/'last.pt'
    temp = output/'last.pt.tmp'
    ema = deepcopy(t.ema.ema).half().cpu()
    # The FP16 export uses the normal native loading path. Keep the training states at full precision.
    bundle = {'epoch': len(history)-1, 'ema': ema, 'model': None, 'train_args': vars(t.args),
              'scale_identity': identity, 'history': history, 'updates': t.ema.updates,
              'optimizer': t.optimizer.state_dict(), 'scaler': t.scaler.state_dict(),
              'train_state': t.model.state_dict(), 'ema_state': t.ema.ema.state_dict(),
              'criterion_updates': getattr(t.model.criterion, 'updates', None)}
    torch.save(bundle, temp)
    temp.replace(path)
    return path


def store_observation(t, epoch, identity, arm):
    import torch
    from ultralytics.nn.tasks import load_checkpoint

    directory = observation_dir(arm, identity, epoch)
    directory.mkdir(exist_ok=True)
    path = directory/'ema.pt'
    temp = path.with_suffix('.tmp.pt')
    ema = deepcopy(t.ema.ema).half().cpu()
    expected = state_hash(deepcopy(ema).float())
    torch.save({'epoch': epoch-1, 'ema': ema, 'model': None, 'train_args': vars(t.args),
                'scale_identity': identity, 'updates': t.ema.updates}, temp)
    reloaded, bundle = load_checkpoint(temp)
    check_equal('observation EMA tensors', expected, state_hash(reloaded))
    check_equal('observation budget', t.ema.updates, bundle['updates'])
    temp.replace(path)
    return {'path': path.relative_to(arm).as_posix(), 'sha256': sha256(path), 'updates': t.ema.updates}


def observation_checkpoint(arm, record, epoch, updates_per_epoch):
    identity = record['identity']
    if identity['epochs'] != LONG_EPOCHS or epoch == LONG_EPOCHS:
        entry = {'path': 'last.pt', 'sha256': record['checkpoint_sha256'], 'updates': record['updates']}
    else:
        history = record['history'][epoch-1]
        check_equal('observation epoch', epoch, history['epoch'])
        entry = history['observation']
        check_equal('observation path', f'epoch{epoch:02}/ema.pt', entry['path'])
    check_equal('observation update count', epoch*updates_per_epoch, entry['updates'])
    return verified_file(arm, entry), entry['sha256']


def restore_state(t, bundle, identity, updates_per_epoch):
    check_equal('resume identity', identity, bundle['scale_identity'])
    history = bundle['history']
    check_equal('resume completed epoch', len(history)-1, bundle['epoch'])
    check_equal('resume complete optimizer update count', len(history)*updates_per_epoch, bundle['updates'])
    t.model.load_state_dict(bundle['train_state'])
    t.ema.ema.load_state_dict(bundle['ema_state'])
    t.ema.updates = bundle['updates']
    t.optimizer.load_state_dict(bundle['optimizer'])
    t.scaler.load_state_dict(bundle['scaler'])
    if bundle['criterion_updates'] is not None:
        t.model.criterion = t.model.init_criterion()
        t.model.criterion.updates = bundle['criterion_updates']
    return history


def fit_arm(job, size):
    import torch
    from ultralytics.nn.tasks import load_checkpoint
    torch.set_num_threads(4)
    data, output, weight, raw = (Path(job[k]) for k in ('data', 'output', 'weight', 'raw_args'))
    arm = output/str(size)
    arm.mkdir(exist_ok=True)
    identity = dict(job['identity'], imgsz=size)
    record = {'status': 'running', 'identity': identity, 'started_at': timestamp()}
    history = []
    try:
        t, dataset = initialize(data, arm, weight, raw, size, identity, job['train'], job['init_state'])
        path = arm/'last.pt'
        complete_epochs = run_observations(job)[-1]
        updates_per_epoch = job['updates_per_epoch']
        if path.exists():
            bundle = torch.load(path, weights_only=False, map_location='cpu')
            history = restore_state(t, bundle, identity, updates_per_epoch)
            del bundle
            for epoch in run_observations(job)[:-1]:
                if epoch <= len(history):
                    observation_checkpoint(arm, dict(record, history=history), epoch, updates_per_epoch)
        for epoch in range(len(history), complete_epochs):
            if epoch == max(len(history), MOSAIC_OFF_EPOCH-1) and epoch >= MOSAIC_OFF_EPOCH-1:
                dataset.close_mosaic(hyp=copy(t.args))
            t.scheduler.last_epoch = epoch-1
            history.append(train_epoch(t, dataset, size, epoch, arm))
            expected = job['lr'][epoch]
            if not all(math.isclose(r, expected, rel_tol=1e-6) for r in history[-1]['lr']):
                raise ContractError('Actual SGD learning rate differs from registered schedule')
            if epoch+1 in run_observations(job)[:-1]:
                history[-1]['observation'] = store_observation(t, epoch+1, identity, arm)
            store_checkpoint(t, history, identity, arm)
            record.update(history=history, completed_epochs=len(history), updates=t.ema.updates)
            save_json(arm/'train.json', record)
            print(f'{size} completed epoch {epoch+1}: {history[-1]["elapsed_seconds"]:.1f}s', flush=True)
        expected_export = state_hash(deepcopy(t.ema.ema).half().float())
        del t, dataset
        torch.cuda.empty_cache()
        reloaded, bundle = load_checkpoint(path)
        check_equal('native reloaded EMA tensors', expected_export, state_hash(reloaded))
        want_updates = complete_epochs*updates_per_epoch
        check_equal('sealed budget', want_updates, bundle['updates'])
        record.update(status='complete', history=history, completed_epochs=len(history), updates=want_updates,
                      checkpoint_sha256=sha256(path), final_role='fixed_budget_last', finished_at=timestamp(),
                      gpu=torch.cuda.get_device_name(0))
        save_json(arm/'train.json', record)
        (arm/'progress.json').unlink(missing_ok=True)
        print(f'{size} EMA reload passed, {want_updates} optimizer updates', flush=True)
    except BaseException as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}', history=history)
        save_json(arm/'train.json', record)
        raise


def environment_at(scratch):
    for key in ('TEMP', 'TMP', 'TMPDIR', 'YOLO_CONFIG_DIR', 'MPLCONFIGDIR', 'TORCH_HOME'):
        sub = scratch/key.lower()
        sub.mkdir()
        os.environ[key] = str(sub)
    os.environ.update(YOLO_VERBOSE='false', YOLO_AUTOINSTALL='false', WANDB_MODE='disabled')
    font = Path(os.environ.get('WINDIR', 'C:/Windows'))/'Fonts/arial.ttf'
    if font.is_file():
        cfg = scratch/'yolo_config_dir/Ultralytics'
        cfg.mkdir()
        shutil.copyfile(font, cfg/'Arial.ttf')


def child(mode, job, *sizes):
    subprocess.run([sys.executable, '-B', '-u', str(Path(__file__)), mode, str(job), *map(str, sizes)], check=True)


def file_inventory(output, omit):
    return {p.relative_to(output).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha256(p)}
            for p in sorted(output.rglob('*')) if p.is_file() and p.name not in omit}


def run(output, scratch_base, smoke=False, resume=False, *, seed=SEED, epochs=EPOCHS, source=SOURCE, reference=R,
        train_only=False, evaluate_only=False):
    import psutil
    if epochs not in (EPOCHS, LONG_EPOCHS):
        raise ContractError(f'Unsupported training horizon: {epochs}')
    source, reference = source.resolve(), reference.resolve()
    weight, raw, preflight, versions, contract = prepare_inputs(source, reference)
    output, scratch_base = output.resolve(), scratch_base.resolve()
    if scratch_base.is_relative_to(ROOT) or not scratch_base.is_dir():
        raise ContractError('Use a project-external existing scratch directory')
    if not smoke and output.parent != (ROOT/'runs').resolve():
        raise ContractError('Formal result must live directly under runs/')
    if smoke and output.is_relative_to(ROOT):
        raise ContractError('Smoke output must be project-external')
    limits = {'source_train': 32, 'source_dev': 12} if smoke else None
    code = [Path(__file__), ROOT/'scripts/scale_eval.py', ROOT/'scripts/evaluate.py',
            ROOT/'scripts/detection_metrics.py', ROOT/'scripts/source_contract.py',
            ROOT/'scripts/source_sampling.py', ROOT/'scripts/source_policy.py']
    identity = {'checkpoint': R_SHA, 'ready': READY, 'seed': seed, 'scope': 'smoke' if smoke else 'paired_dev_diagnostic',
                'epochs': epochs, 'batch': BATCH, 'accumulate': ACCUMULATE,
                'plan': sha256(ROOT/'specs/s3/scale.md'), 'rules': sha256(RULES), 'raw_r_args': sha256(raw),
                'preflight': sha256(PREFLIGHT), 'code': {p.name:sha256(p) for p in code},
                'libraries': versions, 'python': platform.python_version(), 'platform': platform.platform()}
    job = {'identity': identity, 'smoke': smoke, 'updates_per_epoch': 1 if smoke else 125}
    observations = run_observations(job)
    training_receipt = output/'training_manifest.json'
    if evaluate_only:
        if (output/'manifest.json').exists():
            raise ContractError('A completed evaluation cannot be overwritten')
        receipt = read_json(training_receipt)
        check_equal('training complete', 'training_complete', receipt['status'])
        frozen = receipt['identity']
        hardware = ('python', 'platform')
        for key in identity.keys() - set(hardware):
            check_equal('transferred '+key, identity[key], frozen[key])
        for rel, item in receipt['files'].items():
            path = verified_file(output, dict(item, path=rel))
            check_equal('transferred file size '+rel, item['bytes'], path.stat().st_size)
        for size in SIZES:
            arm = read_json(output/str(size)/'train.json')
            check_equal('transferred completed arm', 'complete', arm['status'])
            check_equal('transferred arm identity', dict(frozen, imgsz=size), arm['identity'])
            check_equal('transferred arm budget', observations[-1] * job['updates_per_epoch'], arm['updates'])
        eval_host = {x: identity[x] for x in hardware}
        identity = frozen
    elif resume:
        current = read_json(output/'run.json')
        can_resume = current['status'] in ('running', 'failed') or (
            train_only and current['status'] == 'training_complete' and not training_receipt.exists())
        check_equal('only an interrupted or failed run may resume', True, can_resume)
        check_equal('prior run identity', identity, current['identity'])
        check_equal('prior run training phase', train_only, current['train_only'])
    else:
        output.mkdir(exist_ok=False)
        (output/'code').mkdir()
        for p in [*code, raw, ROOT/'specs/s3/scale.md', RULES]:
            shutil.copyfile(p, output/'code'/p.name)
    with (output/'.active').open('x', encoding='utf-8') as lock:
        lock.write(str(os.getpid()))
    status = {'status': 'running', 'identity': identity, 'started_at': timestamp(),
              'host_available_at_start': psutil.virtual_memory().available, 'final_test_executed': False,
              'train_only': train_only}
    if evaluate_only:
        status.update(training_manifest_sha256=sha256(training_receipt), evaluation_host=eval_host)
    save_json(output/'run.json', status)
    try:
        with tempfile.TemporaryDirectory(prefix='night-scale-run-', dir=scratch_base) as name:
            scratch = Path(name)
            environment_at(scratch)
            if evaluate_only:
                from evaluate import stage_subset
                print('Staging source_dev for evaluation only', flush=True)
                records = contract['manifests']['source_dev']
                if smoke:
                    from source_contract import select_records
                    records = select_records(records, limits['source_dev'])
                selected = {'source_train': [], 'source_dev': records}
                data = stage_subset(dict(contract, purpose='source_dev', records=selected['source_dev']), scratch/'data')
            else:
                print('Staging verified source_train / source_dev', flush=True)
                data, selected = stage_dataset(contract, scratch/'data', limits)
            train = {}
            for r in selected['source_train']:
                by_class = {n.lower():v for n, v in r['boxes_by_class'].items()}
                train[Path(r['image']).name] = {'key':r['key'], 'counts':[by_class.get(n, 0) for n in NAMES]}
            smoke_trial = next(r for r in preflight['trials'] if r['imgsz']==960)
            lr = learning_rates(epochs)
            if epochs == EPOCHS:
                check_equal('short-run schedule matches the original preflight', True,
                            all(math.isclose(a, b, rel_tol=1e-6)
                                for a, b in zip(lr, smoke_trial['epoch_lr'], strict=True)))
            job.update(identity=identity, data=str(data), output=str(output), weight=str(weight), raw_args=str(raw),
                       train=train, dev=selected['source_dev'], init_state=smoke_trial['train_start_tensor_sha256'],
                       lr=lr, source_root=str(source))
            job_file = scratch/'job.json'
            write_json(job_file, job)
            sizes = (960,) if smoke and epochs == EPOCHS else SIZES
            for size in () if evaluate_only else sizes:
                history = output/str(size)/'train.json'
                if not history.exists() or read_json(history)['status']!='complete':
                    child('_fit', job_file, size)
            records = [read_json(output/str(s)/'train.json') for s in sizes]
            if len(records) == 2:
                a, b = records
                for k in ('order_sha256', 'reads_sha256', 'extra_reads', 'updates', 'lr'):
                    check_equal('equal paired trajectory '+k, [r[k] for r in a['history']], [r[k] for r in b['history']])
            for fit_size, record in zip(sizes, records, strict=True):
                arm = output/str(fit_size)
                check_equal('completed arm', 'complete', record['status'])
                check_equal('completed epoch count', observations[-1], len(record['history']))
                for epoch in observations:
                    _, weight_hash = observation_checkpoint(arm, record, epoch, job['updates_per_epoch'])
                    if train_only:
                        continue
                    for eval_size in evaluation_sizes(identity):
                        marker = observation_dir(arm, identity, epoch)/f'dev{eval_size}'/'metrics.json'
                        if marker.exists():
                            check_equal('previous evaluation checkpoint', weight_hash, read_json(marker)['checkpoint_sha256'])
                        else:
                            child('_eval', job_file, fit_size, eval_size, epoch)
            if not train_only and (not smoke or epochs == LONG_EPOCHS):
                child('_compare', job_file)
            check_equal('R remains frozen', R_SHA, sha256(weight))
            check_equal('READY remains frozen', READY, sha256(source/'READY.json'))
        status.update(status='training_complete' if train_only else 'complete', finished_at=timestamp())
        save_json(output/'run.json', status)
        manifest = training_receipt if train_only else output/'manifest.json'
        omit = {'.active', manifest.name}
        if train_only:
            omit.add('run.json')  # Replaced by the local evaluation phase.
        save_json(manifest, dict(status, files=file_inventory(output, omit)))
    except BaseException as exc:
        status.update(status='failed', error=f'{type(exc).__name__}: {exc}', finished_at=timestamp())
        save_json(output/'run.json', status)
        raise
    finally:
        (output/'.active').unlink()
    print(f'Completed {output}', flush=True)


def main():
    if len(sys.argv)>1 and sys.argv[1] in ('_fit','_eval','_compare'):
        job = read_json(Path(sys.argv[2]))
        if sys.argv[1]=='_fit':
            fit_arm(job, int(sys.argv[3]))
        else:
            from scale_eval import compare, evaluate
            if sys.argv[1]=='_eval':
                evaluate(job, int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
            else:
                compare(job)
        return
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['smoke','run','resume'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scratch', type=Path, required=True)
    p.add_argument('--seed', type=int, default=SEED, choices=(17, 29, 43))
    p.add_argument('--epochs', type=int, default=EPOCHS, choices=(EPOCHS, LONG_EPOCHS))
    p.add_argument('--source-root', type=Path, default=SOURCE)
    p.add_argument('--reference-dir', type=Path, default=R)
    phases = p.add_mutually_exclusive_group()
    phases.add_argument('--train-only', action='store_true')
    phases.add_argument('--evaluate-only', action='store_true')
    a = p.parse_args()
    if a.evaluate_only and a.mode == 'resume':
        p.error('repeat run or smoke with --evaluate-only to resume evaluation')
    if a.mode == 'smoke' and a.epochs == EPOCHS and (a.train_only or a.evaluate_only):
        p.error('phase splitting is available in the 24-epoch smoke')
    run(a.output, a.scratch, smoke=a.mode=='smoke', resume=a.mode=='resume', seed=a.seed, epochs=a.epochs,
        source=a.source_root, reference=a.reference_dir, train_only=a.train_only, evaluate_only=a.evaluate_only)


if __name__=='__main__':
    main()
