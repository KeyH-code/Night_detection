"""Ultralytics 8.4.171 adapter: two-class fitness and strict, resumable checkpoints."""
from __future__ import annotations

import math
import shutil
import time
from copy import copy
from pathlib import Path

import torch
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator

from source_contract import ContractError, NAMES, check_equal, primary_fitness, sha256, write_json


class StrictEarlyStopping:
    """Equal scores, including zero, never reset patience."""
    def __init__(self, patience=20):
        self.best_fitness = None
        self.best_epoch = 0
        self.patience = patience
        self.possible_stop = False

    def __call__(self, epoch, fitness):
        if self.best_fitness is None or fitness > self.best_fitness:
            self.best_fitness, self.best_epoch = fitness, epoch
        delta = epoch - self.best_epoch
        self.possible_stop = delta >= self.patience - 1
        return delta >= self.patience


class SourceValidator(DetectionValidator):
    def get_stats(self):
        stats = super().get_stats()
        score, per_class = primary_fitness(self.metrics.box.ap_class_index, self.metrics.box.all_ap)
        self.source_fitness = score
        self.source_per_class = per_class
        stats['fitness'] = score
        stats['metrics/pedestrian_car_AP50-95'] = score
        for i, name in enumerate(NAMES):
            # An absent auxiliary class is explicitly null in the report, not a fabricated AP.
            if i in per_class:
                stats[f'metrics/{name}_AP50-95'] = per_class[i]
        return stats


class SourceTrainer(DetectionTrainer):
    source_identity: dict
    expected_batch: int
    stop_after_epoch: int | None = None

    def optimizer_step(self):
        super().optimizer_step()
        self._source_optimizer_steps += 1

    def _build_train_pipeline(self):
        # Ultralytics auto-halves batch on first-epoch OOM. Our caller owns the only
        # allowed fallback (smoke 2->1, preflight 16->8), restarting from COCO.
        if self.batch_size != self.expected_batch:
            raise torch.cuda.OutOfMemoryError(
                f'Library attempted batch mutation: expected={self.expected_batch}; actual={self.batch_size}')
        return super()._build_train_pipeline()

    def get_validator(self):
        return SourceValidator(self.test_loader, save_dir=self.save_dir,
                               args=copy(self.args), _callbacks=self.callbacks)

    def resume_training(self, ckpt):
        self.stopper = StrictEarlyStopping(self.args.patience)
        if self.resume:
            check_equal('checkpoint.source_identity', self.source_identity, ckpt.get('source_identity'))
        super().resume_training(ckpt)
        if self.resume:
            self.stopper.__dict__.update(ckpt['source_training_state']['stopper'])
            write_json(self.save_dir / 'resume_progress.json', {
                'checkpoint_epoch_0based': ckpt['epoch'], 'start_epoch_0based': self.start_epoch,
                'optimizer_restored': ckpt.get('optimizer') is not None,
                'scaler_restored': bool(ckpt.get('scaler')), 'stopper': self.stopper.__dict__})

    def validate(self):
        metrics = self.validator(self)
        metrics.pop('fitness', None)
        # The upstream validator rounds returned scalars to 5 dp. Select on raw AP.
        fitness = self.validator.source_fitness
        self.source_strict_improvement = self.best_fitness is None or fitness > self.best_fitness
        if self.source_strict_improvement:
            self.best_fitness = fitness
        metrics['metrics/pedestrian_car_AP50-95'] = fitness
        with (self.save_dir / 'source_metrics.jsonl').open('a', encoding='utf-8') as f:
            import json
            f.write(json.dumps({'epoch': self.epoch + 1, 'fitness': fitness,
                                'per_class_AP50-95': {name: self.validator.source_per_class.get(i)
                                                     for i, name in enumerate(NAMES)},
                                'strict_improvement': self.source_strict_improvement}, allow_nan=False) + '\n')
        return metrics, fitness

    def save_model(self):
        best, period = self.best, self.save_period
        temporary = self.wdir / '.upstream-best.pt'
        self.best, self.save_period = temporary, -1
        try:
            saved = super().save_model()
        finally:
            self.best, self.save_period = best, period
        temporary.unlink(missing_ok=True)
        if not saved:
            raise RuntimeError('No checkpoint saved')
        ckpt = torch.load(self.last, map_location='cpu', weights_only=False)
        ckpt['source_identity'] = self.source_identity
        ckpt['source_training_state'] = {'stopper': self.stopper.__dict__.copy()}
        torch.save(ckpt, self.last)
        paths = [self.last]
        if self.source_strict_improvement:
            shutil.copyfile(self.last, best)
            paths.append(best)
        if (self.epoch + 1) % 5 == 0:
            periodic = self.wdir / f'epoch{self.epoch + 1:03d}.pt'
            shutil.copyfile(self.last, periodic)
            paths.append(periodic)
        for path in paths:
            write_json(path.with_suffix('.pt.sha256.json'), {'sha256': sha256(path),
                       'epoch_0based': self.epoch, 'identity': self.source_identity})
        return True

    def final_eval(self):
        # Upstream final_eval strips optimizer from last/best. Preserve both as
        # actual resumable checkpoints and evaluate best only on source_dev.
        if self.best.is_file():
            metrics = self.validator(model=self.best)
            write_json(self.save_dir / 'best_dev_metrics.json', {
                'scope': 'source_dev only', 'fitness': self.validator.source_fitness,
                'metrics': metrics, 'checkpoint_sha256': sha256(self.best)})

    def attach_audits(self, audit_dir: Path):
        def before_training(t):
            if t.device.type != 'cuda' or not bool(t.amp):
                raise ContractError(f'CUDA AMP required: device={t.device}, AMP={t.amp}')
            for k, v in t.source_expected_args.items():
                check_equal(f'effective trainer.{k}', v, getattr(t.args, k))
            write_json(audit_dir / 'trainer_effective.json', vars(t.args))
            write_json(audit_dir / 'optimizer_effective.json', {
                'class': type(t.optimizer).__name__,
                'param_groups': [{k: v for k, v in group.items() if k != 'params'}
                                 for group in t.optimizer.param_groups],
                'accumulate': t.accumulate, 'amp_enabled': bool(t.amp),
                'scheduler_last_epoch': t.scheduler.last_epoch})
            t._source_optimizer_steps = 0
            t._source_batches = 0
            t._source_started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()

        def on_batch(t):
            t._source_batches += 1
            values = t.loss_items.values() if isinstance(t.loss_items, dict) else t.loss_items
            if not torch.isfinite(t.loss).all() or not all(torch.isfinite(x).all() for x in values):
                raise ContractError('Non-finite training loss; abort before treating this run as valid')

        def fit_epoch(t):
            elapsed = time.perf_counter() - t._source_started
            completed = t.epoch - t.start_epoch + 1
            write_json(audit_dir / 'runtime.json', {
                'epoch_1based': t.epoch + 1, 'start_epoch_0based': t.start_epoch,
                'epochs_this_segment': completed, 'elapsed_seconds': elapsed,
                'mean_epoch_seconds': elapsed / completed,
                'train_images_per_second_including_dev': len(t.train_loader.dataset) * completed / elapsed,
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                'batch': t.batch_size, 'batches': t._source_batches,
                'optimizer_steps': t._source_optimizer_steps,
                'finite_loss': True, 'last_epoch_seconds': t.epoch_time})
            if t.stop_after_epoch and t.epoch + 1 >= t.stop_after_epoch:
                t.stop = True

        self.add_callback('on_pretrain_routine_end', before_training)
        self.add_callback('on_train_batch_end', on_batch)
        self.add_callback('on_fit_epoch_end', fit_epoch)
