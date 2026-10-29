"""Full unaugmented train metrics after each epoch; never used for model selection."""
from __future__ import annotations

import csv
import contextlib
import hashlib
import json
import random
import tempfile
import time
from pathlib import Path

from evaluate import metric_record, validator_args
from source_contract import ContractError, NAMES, S3_EVALUATION, check_equal, sha256, write_json


def rng_state():
    import numpy as np
    import torch
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def state_digest(value):
    """Hash actual tensor bytes, not rounded scalar summaries or Python object addresses."""
    import torch
    digest = hashlib.sha256()
    def add(x):
        if isinstance(x, torch.Tensor):
            digest.update(str((x.dtype, tuple(x.shape))).encode())
            digest.update(x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x, dict):
            for k in sorted(x, key=str):
                digest.update(str(k).encode()); add(x[k])
        elif isinstance(x, (list, tuple)):
            for child in x:
                add(child)
        else:
            digest.update(repr(x).encode())
    add(value)
    return digest.hexdigest()


@contextlib.contextmanager
def read_only_observation(trainer):
    """Restore all model RNG and reject mutation of training tensors or optimizer state."""
    def state():
        return {'model': trainer.model.state_dict(), 'ema': trainer.ema.ema.state_dict(),
                'ema_updates': trainer.ema.updates,
                'gradients': [p.grad for p in trainer.model.parameters()],
                'optimizer': trainer.optimizer.state_dict(), 'scaler': trainer.scaler.state_dict(),
                'scheduler': trainer.scheduler.state_dict(),
                'model_modes': [m.training for m in trainer.model.modules()],
                'ema_modes': [m.training for m in trainer.ema.ema.modules()]}
    saved_rng, before = rng_state(), state_digest(state())
    try:
        yield
    finally:
        restore_rng(saved_rng)
        check_equal('observation training-state bytes', before, state_digest(state()))


class S3Observation:
    """Locked S2 export on unique full train/dev at one EMA; dev thresholds are shared with train."""
    def __init__(self, trainer, contract, rules):
        self.trainer, self.contract, self.rules = trainer, contract, rules
        self.scratch = tempfile.TemporaryDirectory(prefix='source_s3_observe_')
        self.root = Path(self.scratch.name)

    def evaluate(self, purpose, model=None, points=None):
        if purpose not in ('source_train', 'source_dev'):
            raise ContractError('Forbidden S3 observation purpose: '+str(purpose))
        import numpy as np
        from evaluate import evidence_validator_class
        from source_policy import summarize
        t = self.trainer
        records = self.contract['manifests'][purpose]
        path = Path(t.data['train' if purpose == 'source_train' else 'val']).resolve()
        if path.name != purpose or not path.is_relative_to(t.s3_sandbox):
            raise ContractError('Observation outside verified sandbox: '+str(path))
        data = self.root/(purpose+'.yaml')
        write_json(data, {'path': str(self.root), 'train': str(path), 'val': str(path), 'names': NAMES, 'nc': 6})
        args = validator_args(data, 1, str(t.device), 0, 640)
        args.update(cache=False, fraction=1., single_cls=False, seed=17, deterministic=True)
        cls = evidence_validator_class()
        validator = cls(args=args, save_dir=self.root/purpose, records=records,
                        source_root=self.contract['root'], stream=None)
        if model is None:
            from evaluate import evaluation_ema
            # Match the persisted EMA representation before fuse/cast, so selection
            # and a later checkpoint reload evaluate the same numerical weights.
            model = evaluation_ema(t.ema.ema, t.device)
        validator(model=model)
        check_equal('full '+purpose+' key set', {r['key'] for r in records}, validator.keys)
        check_equal('full '+purpose+' seen', len(records), validator.seen)
        check_equal('unaugmented '+purpose, False, validator.dataloader.dataset.augment)
        check_equal('locked quantize', 16, validator.args.quantize)
        check_equal('locked max_det', 300, validator.args.max_det)
        try:
            summary, matches, curves = summarize(validator.rows, self.rules, points)
        except ContractError as exc:
            write_json(t.save_dir/('failed_observation_'+purpose+'.json'), {
                'epoch': t.epoch+1, 'purpose': purpose, 'error': str(exc),
                'images_seen': validator.seen, 'metrics_accepted': False,
                'nms_limits': [{'key': r['key'], 'gt_count': len(r['gt']['cls']),
                               'audit': r['nms_audit']} for r in validator.rows
                              if r['nms_audit']['at_max_det'] or r['nms_audit']['pre_nms_truncated']]})
            raise
        for c, ap in zip(validator.metrics.box.ap_class_index, validator.metrics.box.all_ap):
            if not np.allclose(ap, summary['metrics']['per_class'][NAMES[int(c)]]['ap_iou'], rtol=0, atol=1e-12):
                raise ContractError('S3 offline/native AP mismatch')
        summary['native_f1'] = metric_record(validator.metrics)
        check_equal('complete NMS audit', len(records), summary['nms']['audited_images'])
        close = getattr(validator.dataloader, 'close', None)
        if close:
            close()
        return summary, matches, curves, validator.rows

    def epoch(self):
        import numpy as np
        t = self.trainer
        start = time.perf_counter()
        dev, matches, dev_curves, dev_rows = self.evaluate('source_dev')
        deferred = t.source_identity.get('evaluation_policy') == S3_EVALUATION
        if deferred:
            train, train_curves = {'status': 'deferred_local', 'images_evaluated': 0}, {}
        else:
            train, _, train_curves, _ = self.evaluate('source_train', points=dev['workpoints'])
        folder = t.save_dir/'observations'
        folder.mkdir(exist_ok=True)
        curves = {**{'dev_'+k: v for k, v in dev_curves.items()}, **{'train_'+k: v for k, v in train_curves.items()}}
        curve_path = folder/f'epoch{t.epoch+1:03d}.npz'
        np.savez_compressed(curve_path, **compact_curves(curves))
        elapsed = time.perf_counter()-start
        epoch = {'epoch': t.epoch+1, 'weights': 'current_epoch_ema', 'train': train, 'dev': dev,
                 'evaluation_seconds': elapsed, 'selection_source': 'source_dev only',
                 'metric_scope': dev['metric_scope'],
                 'pr_sha256': sha256(curve_path),
                 'pr_encoding': 'AP scores + cumulative TP at 10 IoUs; n=1..N, FP=n-TP, GT=per_class.instances; dev working curves filter/rematch'}
        return epoch, dev_rows, matches

    def close(self):
        self.scratch.cleanup()


def compact_curves(curves):
    """Lossless PR data without duplicate derived precision/recall/FP matrices."""
    import numpy as np
    result = {}
    for key, value in curves.items():
        if key.endswith(('_ap_scores', '_ap_tp', '_work_threshold', '_work_tp', '_work_fp', '_work_fn')):
            result[key] = value.astype(np.int32) if np.issubdtype(value.dtype, np.integer) else value
    return result


def finalize_s3_local(run_dir, data_root, ready_hash, device='0'):
    """Export fixed cloud-selected weights locally; never retrain or reselect a checkpoint."""
    from types import SimpleNamespace
    import importlib.metadata
    import sys
    import torch
    from source_contract import read_json, safe_path, stage_dataset, verify_source
    from source_release import audit_training_roles, seal_training
    from source_trainer import S3Trainer
    from train_source import code_identity
    run_dir = Path(run_dir).resolve()
    source = read_json(run_dir/'source_run.json')
    manifest = read_json(run_dir/'manifest.json')
    check_equal('local finalization requires formal training', 'completed_formal', manifest['status'])
    check_equal('local finalization profile', S3_EVALUATION, source['identity'].get('evaluation_policy'))
    check_equal('sealed source identity', source['identity'], manifest['identity'])
    check_equal('local data identity', source['identity']['ready_sha256'], ready_hash)
    check_equal('local algorithm identity', source['identity']['code'], code_identity())
    for rel, entry in manifest['files'].items():
        check_equal('local input '+rel, entry['sha256'], sha256(safe_path(run_dir, rel)))
    roles = audit_training_roles(run_dir)
    check_equal('local evaluation not already completed', 'pending_local_evaluation', roles['evaluation_status'])
    check_equal('formal source scope', 'formal', source['scope'])
    config = read_json(run_dir/'audit_initial/config_resolved.json')
    environment = {'python': '.'.join(sys.version.split()[0].split('.')[:2]),
        **{n: importlib.metadata.version(n) for n in ('torch','torchvision','ultralytics')}}
    for name, expected in config['environment'].items():
        check_equal('local framework '+name, expected, environment[name])
    if device == 'cpu' or not torch.cuda.is_available():
        raise ContractError('Local finalization requires CUDA FP16; no silent CPU/precision substitution')
    contract = verify_source(Path(data_root), ready_hash)
    inputs = read_json(run_dir/'policy_inputs.json')
    check_equal('local RULES identity', source['identity']['s3_policy']['rules_sha256'], inputs['rules']['sha256'])
    observer = None
    try:
        handoff = run_dir/'final_eval/cloud.json'
        pending = run_dir/'final_eval/run.json'
        if handoff.exists():
            check_equal('original cloud handoff', sha256(pending), sha256(handoff))
        else:
            handoff.write_bytes(pending.read_bytes())
        with tempfile.TemporaryDirectory(prefix='source_s3_local_') as scratch:
            data, _ = stage_dataset(contract, Path(scratch)/'data')
            t = SimpleNamespace(save_dir=run_dir, last=run_dir/roles['roles']['last']['path'],
                source_identity=source['identity'], s3_local_export=True,
                device=torch.device('cuda:'+device if device.isdecimal() else device),
                s3_sandbox=data.parent, data={'train':str(data.parent/'images/source_train'),
                                            'val':str(data.parent/'images/source_dev')})
            last_epoch = roles['roles']['last']['epoch']
            t.s3_observation = read_json(run_dir/f'observations/epoch{last_epoch:03d}.json')
            observer = S3Observation(t, contract, inputs['rules'])
            t.s3_observer = observer
            with torch.inference_mode():
                S3Trainer.final_eval(t)
        final = read_json(run_dir/'final_eval/run.json')
        final['local_environment'] = {**environment,'gpu':torch.cuda.get_device_name(t.device),
                                     'cuda':torch.version.cuda, 'device':str(t.device)}
        final['weight_selection'] = 'unchanged cloud role; cloud dev thresholds retained'
        final['cloud_handoff_sha256'] = sha256(handoff)
        from source_policy import SelectionPolicy
        dev = read_json(run_dir/'final_eval/source_dev/metrics.json')
        checks, guards = SelectionPolicy(inputs['rules'], inputs['reference']).assess(dev)
        final['local_numeric_checks'], final['local_protection_checks'] = checks, guards
        final['requires_quality_review'] = roles['selected_role'] == 'best' and not all([*checks.values(), *guards.values()])
        write_json(run_dir/'final_eval/run.json', final)
        check_equal('READY unchanged after local evaluation', ready_hash, sha256(Path(data_root)/'READY.json'))
        seal_training(run_dir, 'formal')
        return final
    except BaseException as exc:
        write_json(run_dir/'final_eval/local_failure.json', {'error':f'{type(exc).__name__}: {exc}',
                   'scope':'local evaluation failure; no training or checkpoint reselection'})
        raise
    finally:
        if observer is not None:
            observer.close()


def evaluate_ema(trainer, validator):
    # The locked validator deep-copies this model in get_model before fusing/casting.
    # It does not receive trainer, so it cannot run training loss, backward or optimizer steps.
    validator(model=trainer.ema.ema)
    return metric_record(validator.metrics)


def append_epoch(path, row):
    if path.exists():
        with path.open(encoding='utf-8', newline='') as stream:
            reader = csv.DictReader(stream)
            check_equal('train metrics columns', list(row), reader.fieldnames)
            rows = list(reader)
        if rows and int(rows[-1]['epoch']) >= row['epoch']:
            raise ContractError('Refusing duplicate/backward train-metrics epoch')
    with path.open('a', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
        stream.flush()


class EpochTrainMetrics:
    def __init__(self, run_dir, expected_images, config, evaluator=evaluate_ema):
        self.run_dir = Path(run_dir)
        self.expected_images = expected_images
        self.config = config
        self.evaluator = evaluator
        self.validator = None
        self.scratch = None
        self.total_seconds = 0.
        self.base_seconds = 0.
        self.epochs = 0
        self.epoch_started = None

    def start_epoch(self, *_):
        self.epoch_started = time.perf_counter()

    def prepare(self, trainer):
        from ultralytics.models.yolo.detect import DetectionValidator
        local_tmp = '/tmp' if Path('/tmp').is_dir() else None
        self.scratch = tempfile.TemporaryDirectory(prefix='train_metrics_', dir=local_tmp)
        root = Path(self.scratch.name)
        train = Path(trainer.data['train']).resolve()
        if not train.is_relative_to(self.run_dir.resolve().parent) or train.name != 'source_train':
            raise ContractError('Train observation must use the verified writable source_train sandbox')
        data = root / 'dataset.yaml'
        write_json(data, {'path': str(self.run_dir.resolve()), 'train': str(train),
                          'val': str(train), 'names': NAMES, 'nc': 6})
        args = validator_args(data, self.config.get('batch', 16), str(trainer.device),
                              self.config.get('workers', 0), int(trainer.args.imgsz))
        # torch device can stringify as cuda:0, accepted by Ultralytics select_device.
        self.validator = DetectionValidator(args=args, save_dir=root / 'validator')

    def __call__(self, trainer):
        started = time.perf_counter()
        if self.epoch_started is None:
            raise ContractError('Missing epoch-start timing callback')
        if self.validator is None:
            self.prepare(trainer)
        previous_fitness = trainer.fitness
        previous_metrics = dict(trainer.metrics)
        ema_training = trainer.ema.ema.training
        record = self.evaluator(trainer, self.validator)
        elapsed = time.perf_counter() - started
        check_equal('unchanged dev fitness', previous_fitness, trainer.fitness)
        check_equal('unchanged dev metrics', previous_metrics, trainer.metrics)
        check_equal('unchanged EMA mode', ema_training, trainer.ema.ema.training)
        check_equal('full train evaluation count', self.expected_images, self.validator.seen)
        check_equal('no train-metrics augmentation', False, self.validator.dataloader.dataset.augment)
        # Ultralytics epoch_time includes the previous fit callback. Use our start hook
        # so the denominator does not already contain the previous train evaluation.
        base = started - self.epoch_started
        row = {'epoch': trainer.epoch + 1, 'images': self.expected_images,
               'weights': 'current_epoch_ema', 'train_primary_ap50_95': record['primary_ap50_95'],
               'dev_primary_ap50_95': float(trainer.validator.source_fitness),
               'train_map50': record['map50'], 'train_map50_95': record['map50_95'],
               'train_precision': record['precision'], 'train_recall': record['recall'],
               'evaluation_seconds': elapsed, 'base_epoch_seconds': base,
               'overhead_ratio': elapsed/base if base > 0 else None}
        for name in NAMES:
            for field in ('images', 'instances', 'precision', 'recall', 'ap50', 'ap50_95'):
                row[f'{name}_{field}'] = record['per_class'][name][field]
            row[f'{name}_dev_ap50_95'] = trainer.validator.source_per_class.get(NAMES.index(name))
        append_epoch(self.run_dir / 'train_metrics.csv', row)
        self.total_seconds += elapsed
        self.base_seconds += base
        self.epochs += 1
        write_json(self.run_dir / 'train_monitor.json', {
            'policy': 'full source_train evaluation each epoch, current EMA, no gradient/augmentation; dev selects best',
            'config': self.config, 'source_identity': trainer.source_identity,
            'epochs_in_segment': self.epochs, 'evaluation_seconds': self.total_seconds,
            'base_epoch_seconds': self.base_seconds,
            'overhead_ratio': self.total_seconds/self.base_seconds if self.base_seconds else None,
            'timing_scope': 'same training run; preflight must include this cost before formal launch',
            'final_tests_run': False})

    def close(self, *_):
        self.validator = None
        if self.scratch is not None:
            self.scratch.cleanup()
            self.scratch = None


def attach(trainer, run_dir, expected_images, config):
    if not config.get('enabled', False):
        return None
    if config.get('every', 1) != 1:
        raise ContractError('Requested full train metrics every epoch; refusing silent frequency reduction')
    observer = EpochTrainMetrics(run_dir, expected_images, config)
    trainer.add_callback('on_train_epoch_start', observer.start_epoch)
    trainer.add_callback('on_fit_epoch_end', observer)
    trainer.add_callback('on_train_end', observer.close)
    return observer


def audit_observation(run_dir):
    """Bind complete per-epoch observations to the same run before cloud publication."""
    from source_contract import read_json, sha256
    run_dir = Path(run_dir)
    source = read_json(run_dir / 'source_run.json')
    deferred = source['identity'].get('evaluation_policy') == S3_EVALUATION
    metrics_name = 'dev_metrics.csv' if deferred else 'train_metrics.csv'
    timing = read_json(run_dir / 'train_monitor.json')
    check_equal('monitor run identity', source['identity'], timing['source_identity'])
    if deferred:
        check_equal('dev-only monitor schema', 3, timing.get('schema_version'))
    with (run_dir / metrics_name).open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    dev_lines = (run_dir / 'source_metrics.jsonl').read_text(encoding='utf-8').splitlines()
    import json
    epochs = [json.loads(line)['epoch'] for line in dev_lines if line]
    check_equal('train/dev epoch coverage', epochs, [int(r['epoch']) for r in rows])
    if timing.get('schema_version') in (2, 3):
        from source_release import audit_training_roles
        check_equal('complete S3 epochs', list(range(1, timing['epoch']+1)), epochs)
        check_equal('S3 observation schema', 3 if deferred else 2, timing['schema_version'])
        check_equal('S3 metric full coverage', True, all(int(r['images']) == (500 if deferred else 4000) for r in rows))
        check_equal('S3 batch1 observation', 1, timing['batch'])
        check_equal('S3 AMP skips', 0, timing['amp_skipped_steps'])
        check_equal('S3 exact gradient batches', 250*len(rows), timing['gradient_batches'])
        check_equal('S3 monitor counts', [0 if deferred else 4000, 500], [timing['full_train_images'], timing['full_dev_images']])
        times = [json.loads(line) for line in (run_dir/'timings.jsonl').read_text(encoding='utf-8').splitlines() if line]
        check_equal('S3 timing coverage', epochs, [r['epoch'] for r in times])
        for epoch in epochs:
            observation = read_json(run_dir/f'observations/epoch{epoch:03d}.json')
            sample = read_json(run_dir/f'sampling/epoch{epoch:03d}.json')
            check_equal('S3 epoch binding', epoch, observation['epoch'])
            for purpose, images in ([('dev', 500)] if deferred else [('train', 4000), ('dev', 500)]):
                check_equal('S3 epoch full '+purpose, images, observation[purpose]['metrics']['images'])
            if deferred:
                check_equal('S3 train evaluation deferred', {'status': 'deferred_local', 'images_evaluated': 0}, observation['train'])
            check_equal('S3 consumed slots', 4000, sample['anchor']['slots'])
            check_equal('S3 consumed index bytes', sample['consumed_sha256'], sha256(run_dir/f'sampling/epoch{epoch:03d}.npz'))
            if len(sample['worker_exit']) != 4 or any(w['alive'] or w['exitcode'] != 0 for w in sample['worker_exit']):
                raise ContractError('S3 worker-exit evidence missing/failed')
            check_equal('S3 complete PR bytes', observation['pr_sha256'], sha256(run_dir/f'observations/epoch{epoch:03d}.npz'))
        roles = audit_training_roles(run_dir)
        return {'epochs': len(rows), 'full_train_images': 0 if deferred else 4000, 'full_dev_images': 500,
                ('dev_metrics_sha256' if deferred else 'train_metrics_sha256'): sha256(run_dir/metrics_name),
                'timing_sha256': sha256(run_dir/'train_monitor.json'), 'roles': roles,
                'last_epoch_overhead_ratio': timing['overhead_ratio'],
                'total_epoch_seconds': sum(r['epoch_seconds'] for r in times),
                'total_evaluation_seconds': sum(r['evaluation_seconds'] for r in times),
                'selection_source': 'source_dev only'}
    return {'epochs': len(rows), 'train_metrics_sha256': sha256(run_dir / 'train_metrics.csv'),
            'timing_sha256': sha256(run_dir / 'train_monitor.json'),
            'last_segment_overhead_ratio': timing['overhead_ratio'],
            'last_segment_evaluation_seconds': timing['evaluation_seconds'],
            'selection_source': 'source_dev only'}
