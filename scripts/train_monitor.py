"""Full unaugmented train metrics after each epoch; never used for model selection."""
from __future__ import annotations

import csv
import tempfile
import time
from pathlib import Path

from evaluate import metric_record, validator_args
from source_contract import ContractError, NAMES, check_equal, write_json


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
    timing = read_json(run_dir / 'train_monitor.json')
    check_equal('monitor run identity', source['identity'], timing['source_identity'])
    with (run_dir / 'train_metrics.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    dev_lines = (run_dir / 'source_metrics.jsonl').read_text(encoding='utf-8').splitlines()
    import json
    epochs = [json.loads(line)['epoch'] for line in dev_lines if line]
    check_equal('train/dev epoch coverage', epochs, [int(r['epoch']) for r in rows])
    return {'epochs': len(rows), 'train_metrics_sha256': sha256(run_dir / 'train_metrics.csv'),
            'timing_sha256': sha256(run_dir / 'train_monitor.json'),
            'last_segment_overhead_ratio': timing['overhead_ratio'],
            'last_segment_evaluation_seconds': timing['evaluation_seconds'],
            'selection_source': 'source_dev only'}
