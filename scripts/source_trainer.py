"""Ultralytics 8.4.171 adapter: two-class fitness and strict, resumable checkpoints."""
from __future__ import annotations

import math
import shutil
import time
from copy import copy
from pathlib import Path

import torch
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator

from source_contract import ContractError, NAMES, S3_EVALUATION, check_equal, primary_fitness, sha256, write_json


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


class ReadOnlyLossValidator(SourceValidator):
    def get_model(self, model, trainer=None):
        from copy import deepcopy
        return deepcopy(trainer.ema.ema if trainer is not None else model)


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
                actual = getattr(t.args, k)
                # select_device canonicalizes integer GPU IDs to strings.
                # Keep the CUDA/AMP guard above and reject any different ID.
                check_equal(f'effective trainer.{k}', str(v) if k == 'device' else v,
                            str(actual) if k == 'device' else actual)
            write_json(audit_dir / 'trainer_effective.json', vars(t.args))
            write_json(audit_dir / 'optimizer_effective.json', {
                'class': type(t.optimizer).__name__,
                'param_groups': [{k: v for k, v in group.items() if k != 'params'}
                                 for group in t.optimizer.param_groups],
                'accumulate': t.accumulate, 'amp_enabled': bool(t.amp),
                'amp_scaler': t.scaler.state_dict(),
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


class S3Trainer(SourceTrainer):
    """Single-GPU S3 adapter; gradient equations mirror pinned 8.4.171, with explicit epoch state.

    The finite loop owns last_opt_step and pending scaled gradients so a completed
    epoch checkpoint resumes the TRAIN model, not the library's rounded EMA restart.
    Legacy source_a continues to use SourceTrainer unchanged.
    """
    def get_validator(self):
        return ReadOnlyLossValidator(self.test_loader, save_dir=self.save_dir,
                                     args=copy(self.args), _callbacks=self.callbacks)

    def build_dataset(self, img_path, mode='train', batch=None):
        if mode != 'train':
            return super().build_dataset(img_path, mode, batch)
        from source_sampling import AuditedYOLODataset
        return AuditedYOLODataset(img_path=img_path, imgsz=self.args.imgsz, batch_size=batch,
                                  augment=True, hyp=self.args, rect=False, cache=False,
                                  single_cls=False, stride=max(int(self.model.stride.max()), 32),
                                  pad=0., prefix='train: ', task='detect', data=self.data, fraction=1.)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        if mode != 'train':
            return super().get_dataloader(dataset_path, batch_size, rank, mode)
        from source_sampling import FiniteEpochLoader, training_index
        if rank != -1 or self.world_size > 1:
            raise ContractError('S3 supports exactly one GPU, no distributed sampler')
        dataset = self.build_dataset(dataset_path, mode, batch_size)
        index = training_index(self.s3_contract, dataset.im_files)
        for r, label in zip(index, dataset.labels):
            import numpy as np
            check_equal('loader retains frozen GT counts '+r['key'], r['classes'],
                        np.bincount(label['cls'].reshape(-1).astype(int), minlength=6).tolist())
        target_images = sum(r['target'] for r in index)
        target_gt = sum(r['target_gt'] for r in index)
        check_equal('S3 target image count', 922, target_images)
        check_equal('S3 target GT count', 1617, target_gt)
        write_json(self.save_dir/'sampling/index.json', {'algorithm': 'small_thin_union_weighted_v1',
                   'records': index, 'target_images': target_images, 'target_gt': target_gt})
        return FiniteEpochLoader(dataset, index, self.args.seed, self.s3_arm,
                                 batch_size, self.args.workers, pin_memory=True)

    def resume_training(self, ckpt):
        from source_policy import HorizonStopping
        self.stopper = HorizonStopping(self.args.epochs)
        self._s3_resume_state = None
        if not self.resume:
            # The pinned default and former 1024 scale overflow real source batches.
            # Both arms start at the registered lower scale; overflow/skip
            # remains fatal and completed-epoch resume restores its own scaler.
            self.scaler = torch.amp.GradScaler(
                self.device.type, enabled=bool(self.amp),
                init_scale=self.s3_amp_initial_scale)
        if self.resume:
            check_equal('S3 embedded resume identity', self.source_identity, ckpt.get('source_identity'))
            self._s3_resume_state = ckpt.get('source_resume')
            if not self._s3_resume_state or not self._s3_resume_state.get('complete_epoch'):
                raise ContractError('S3 resume requires exact completed-epoch training state')
            super(SourceTrainer, self).resume_training(ckpt)
            self.model.load_state_dict(self._s3_resume_state['model'])
            self.ema.ema.load_state_dict(self._s3_resume_state['ema'])
            self.scheduler.load_state_dict(self._s3_resume_state['scheduler'])
            self.s3_policy.state = self._s3_resume_state['policy']

    def _restore_epoch_state(self):
        from train_monitor import restore_rng
        state = self._s3_resume_state
        self.s3_last_opt_step = -1
        self.s3_steps = self.s3_skipped = self.s3_batches = 0
        self.optimizer.zero_grad()
        if state:
            for name, p in self.model.named_parameters():
                grad = state['gradients'][name]
                p.grad = grad.to(p.device, dtype=p.dtype) if grad is not None else None
            self.accumulate = state['accumulate']
            self.s3_last_opt_step = state['last_opt_step']
            self.s3_steps, self.s3_skipped, self.s3_batches = state['steps'], state['skipped'], state['batches']
            self.scheduler.load_state_dict(state['scheduler'])
            if state.get('criterion_updates') is not None:
                if getattr(self.model, 'criterion', None) is None:
                    self.model.criterion = self.model.init_criterion()
                self.model.criterion.updates = state['criterion_updates']
            restore_rng(state['rng'])

    def optimizer_step(self):
        # Same unscale/clip/step/update/zero/EMA order as the locked library.
        # Reject skipped/nonfinite updates rather than silently comparing unequal budgets.
        self.scaler.unscale_(self.optimizer)
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.)
        if not torch.isfinite(norm):
            write_json(self.save_dir/'numerical_failure.json', {
                'status': 'failed_before_optimizer_update', 'epoch': self.epoch+1,
                'batch_0based': self.s3_batches % len(self.train_loader),
                'gradient_batches_before': self.s3_batches, 'optimizer_steps_before': self.s3_steps,
                'amp_scale': self.scaler.get_scale(), 'accumulate': int(self.accumulate),
                'loss': float(self.loss.detach()),
                'loss_items': {k: float(v.detach()) for k, v in self.loss_items.items()},
                'gradient_norm': str(float(norm)), 'update_skipped_or_retried': False,
                'scope': 'failure diagnosis; no accepted update or checkpoint'})
            raise ContractError('Nonfinite gradient; S3 has no automatic recovery/skip')
        scale = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scaler.get_scale() < scale:
            self.s3_skipped += 1
            raise ContractError('AMP skipped an update; pause S3 for investigation')
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)
        self._source_optimizer_steps += 1
        self.s3_steps += 1

    def validate(self):
        from train_monitor import read_only_observation
        from source_policy import recall_intervals
        # Native loss/dev observation retained separately from locked rule evaluation.
        started = time.perf_counter()
        with read_only_observation(self):
            metrics = self.validator(self)
            metrics.pop('fitness', None)
            if any(not math.isfinite(float(value)) for value in metrics.values()):
                raise ContractError('Nonfinite native dev metric/loss; pause S3')
            observation, rows, matches = self.s3_observer.epoch()
            decision = self.s3_policy.update(self.epoch+1, observation['dev'],
                                             lambda: recall_intervals(rows, matches))
        observation.update(selection=decision,
                           evaluation_seconds=time.perf_counter()-started,
                           ema_state_sha256=self.s3_ema_digest())
        self.s3_observation, self.s3_decision = observation, decision
        self.source_strict_improvement = decision['new_best']
        self.best_fitness = self.s3_policy.state['best']['score'] if self.s3_policy.state['best'] else None
        score = observation['dev']['metrics']['primary_ap50_95']
        metrics['metrics/pedestrian_car_AP50-95'] = score
        return metrics, score

    def s3_ema_digest(self):
        from train_monitor import state_digest
        return state_digest(self.ema.ema.state_dict())

    def save_model(self):
        import json
        from copy import deepcopy
        from train_monitor import rng_state
        from source_contract import read_json
        self.wdir.mkdir(parents=True, exist_ok=True)
        # Never clamp or replace poisoned weights. A failed run stays a failed run.
        for module in (self.model, self.ema.ema):
            if not all(torch.isfinite(v).all() for v in module.state_dict().values()):
                raise ContractError('Nonfinite checkpoint tensors')
        ema = deepcopy(self.ema.ema).half().to(memory_format=torch.contiguous_format)
        if hasattr(ema, 'criterion'):
            ema.criterion = None
        if not all(torch.isfinite(v).all() for v in ema.state_dict().values()):
            raise ContractError('FP16 checkpoint overflow; no silent clamping')
        state = {'complete_epoch': True, 'model': {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                 'ema': {k: v.detach().cpu() for k, v in self.ema.ema.state_dict().items()},
                 'gradients': {n: p.grad.detach().cpu() if p.grad is not None else None for n, p in self.model.named_parameters()},
                 'scheduler': self.scheduler.state_dict(), 'accumulate': self.accumulate,
                 'last_opt_step': self.s3_last_opt_step, 'steps': self.s3_steps, 'skipped': self.s3_skipped,
                 'batches': self.s3_batches, 'rng': rng_state(), 'policy': deepcopy(self.s3_policy.state),
                 'criterion_updates': getattr(getattr(self.model, 'criterion', None), 'updates', None)}
        checkpoint = {'epoch': self.epoch, 'best_fitness': self.best_fitness,
                      'model': None, 'ema': ema, 'updates': self.ema.updates,
                      'optimizer': self.optimizer.state_dict(), 'scaler': self.scaler.state_dict(),
                      'train_args': vars(self.args), 'train_metrics': {**self.metrics, 'fitness': self.fitness},
                      'source_identity': self.source_identity, 'source_resume': state,
                      'source_observation': deepcopy(self.s3_observation),
                      'source_ema_state_sha256': self.s3_observation['ema_state_sha256'],
                      'version': '8.4.171', 'date': __import__('datetime').datetime.now().isoformat()}
        roles_path = self.save_dir/'checkpoint_roles.json'
        roles = read_json(roles_path) if roles_path.exists() else {}
        # A prior final_eval may have deduplicated a role to last. Preserve that
        # older role before replacing last during resumed training.
        aliases = [r for r in ('best', 'diagnostic') if r in roles and roles[r]['path'] == 'weights/last.pt']
        if aliases:
            destination = self.best if 'best' in aliases else self.wdir/'diagnostic.pt'
            shutil.copyfile(self.last, destination)
            shutil.copyfile(self.last.with_suffix('.pt.sha256.json'), destination.with_suffix('.pt.sha256.json'))
            for role in aliases:
                roles[role]['path'] = destination.relative_to(self.save_dir).as_posix()
        pending = self.last.with_suffix('.pt.part')
        torch.save(checkpoint, pending)
        pending.replace(self.last)
        writes = [self.last]
        for role, flag in [('best', 'new_best'), ('diagnostic', 'new_diagnostic')]:
            if not self.s3_decision[flag]:
                continue
            path = self.best if role == 'best' else self.wdir/'diagnostic.pt'
            if role == 'diagnostic' and self.s3_decision['new_best']:
                path = self.best
            else:
                shutil.copyfile(self.last, path)
                writes.append(path)
            roles[role] = {'path': path.relative_to(self.save_dir).as_posix(), 'epoch': self.epoch+1,
                           'sha256': sha256(path), 'status': self.s3_policy.state[role]['status']}
        roles['last'] = {'path': self.last.relative_to(self.save_dir).as_posix(), 'epoch': self.epoch+1,
                         'sha256': sha256(self.last), 'status': 'complete_epoch_resumable'}
        for path in set(writes):
            write_json(path.with_suffix('.pt.sha256.json'), {'sha256': sha256(path),
                       'epoch_0based': self.epoch, 'identity': self.source_identity})
        write_json(roles_path, roles)
        self.s3_observation['checkpoint_sha256'] = roles['last']['sha256']
        write_json(self.save_dir/f'observations/epoch{self.epoch+1:03d}.json', self.s3_observation)
        from train_monitor import append_epoch
        deferred = self.source_identity.get('evaluation_policy') == S3_EVALUATION
        observed = self.s3_observation['dev' if deferred else 'train']
        row = {'epoch': self.epoch+1, 'images': observed['metrics']['images'],
               ('observed_dev_primary_ap50_95' if deferred else 'train_primary_ap50_95'): observed['metrics']['primary_ap50_95'],
               'dev_primary_ap50_95': self.fitness, 'evaluation_seconds': self.s3_observation['evaluation_seconds']}
        for name in NAMES:
            for k in ('ap50', 'ap75', 'ap50_95'):
                row[f'{name}_{k}'] = observed['metrics']['per_class'][name][k]
            for k in ('precision', 'recall', 'tp', 'fp', 'fn'):
                row[f'{name}_{k}'] = observed['workpoints'][name][k]
        append_epoch(self.save_dir/('dev_metrics.csv' if deferred else 'train_metrics.csv'), row)
        with (self.save_dir/'source_metrics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'epoch': self.epoch+1, 'fitness': self.fitness,
                                     'selection': self.s3_decision}, allow_nan=False)+'\n')
        return True

    def _do_train(self):
        import warnings
        import numpy as np
        from ultralytics.utils.torch_utils import autocast
        from train_monitor import S3Observation
        if self.world_size > 1 or self.args.time or self.args.compile or self.args.multi_scale:
            raise ContractError('S3 requires single GPU, fixed horizon, eager 640 training')
        self._setup_train()
        self.s3_observer = S3Observation(self, self.s3_contract, self.s3_policy.rules)
        nb = len(self.train_loader)
        nw = self._get_warmup_iterations(nb)
        self.train_time_start = time.time()
        self.run_callbacks('on_train_start')
        # Resume RNG after startup/logging callbacks, which do not run between
        # epochs of an uninterrupted segment.
        self._restore_epoch_state()
        try:
            for epoch in range(self.start_epoch, self.epochs):
                self.epoch = epoch
                started = time.perf_counter()
                self.run_callbacks('on_train_epoch_start')
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    self.scheduler.step()
                self._model_train()
                if epoch >= self.epochs-self.args.close_mosaic:
                    self._close_dataloader_mosaic()
                self.train_loader.set_epoch(epoch)
                self.tloss = None
                for i, batch in enumerate(self.train_loader):
                    self.run_callbacks('on_train_batch_start')
                    ni = i+nb*epoch
                    if ni < nw:
                        xi = [0, nw]
                        self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs/self.batch_size]).round()))
                        for group in self.optimizer.param_groups:
                            group['lr'] = float(np.interp(ni, xi,
                                [self.args.warmup_bias_lr if group.get('param_group') == 'bias' else 0.,
                                 group['initial_lr']*self.lf(epoch)]))
                            if 'momentum' in group:
                                group['momentum'] = float(np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum]))
                    with autocast(self.amp, device=self.device.type):
                        batch = self.preprocess_batch(batch)
                        loss, self.loss_items = self.model(batch)
                        self.loss = loss.sum()
                        if not torch.isfinite(self.loss) or not all(torch.isfinite(v).all() for v in self.loss_items.values()):
                            raise ContractError('Nonfinite S3 training loss')
                        if not self.loss_names:
                            self.loss_names = tuple(self.loss_items)
                        self.tloss = self.loss_items if self.tloss is None else {
                            k: (self.tloss[k]*i+v)/(i+1) for k, v in self.loss_items.items()}
                    self.scaler.scale(self.loss).backward()
                    if ni-self.s3_last_opt_step >= self.accumulate:
                        self.optimizer_step()
                        self.s3_last_opt_step = ni
                    self.s3_batches += 1
                    self.run_callbacks('on_train_batch_end')
                    if self.stop:
                        raise ContractError('Interrupted within epoch; resume previous complete last')
                audit = self.train_loader.finish(self.save_dir/'sampling')
                write_json(self.save_dir/f'sampling/epoch{epoch+1:03d}.json', audit)
                gradient_seconds = time.perf_counter()-started
                if hasattr(self.model.criterion, 'update'):
                    self.model.criterion.update()
                self.lr = {f'lr/pg{i}': g['lr'] for i, g in enumerate(self.optimizer.param_groups)}
                self.run_callbacks('on_train_epoch_end')
                self.ema.update_attr(self.model, include=['yaml', 'nc', 'args', 'names', 'stride', 'class_weights'])
                self.metrics, self.fitness = self.validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop = self.stopper(epoch+1, self.fitness)
                save_started = time.perf_counter()
                self.save_model()
                self.run_callbacks('on_model_save')
                self.epoch_time = time.perf_counter()-started
                import json
                with (self.save_dir/'timings.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps({'epoch': epoch+1, 'epoch_seconds': self.epoch_time,
                        'gradient_loader_seconds': gradient_seconds,
                        'evaluation_seconds': self.s3_observation['evaluation_seconds'],
                        'checkpoint_seconds': time.perf_counter()-save_started}, allow_nan=False)+'\n')
                deferred = self.source_identity.get('evaluation_policy') == S3_EVALUATION
                write_json(self.save_dir/'train_monitor.json', {
                    'schema_version': 3 if deferred else 2, 'source_identity': self.source_identity, 'epoch': epoch+1,
                    'full_train_images': 0 if deferred else 4000, 'full_dev_images': 500, 'batch': 1,
                    'train_evaluation': 'deferred_local' if deferred else 'each_epoch',
                    'evaluation_seconds': self.s3_observation['evaluation_seconds'],
                    'overhead_ratio': self.s3_observation['evaluation_seconds']/max(self.epoch_time-self.s3_observation['evaluation_seconds'], 1e-9),
                    'overhead_ratio_scope': 'rule evaluation / epoch excluding rule evaluation',
                    'epoch_seconds': self.epoch_time, 'optimizer_steps': self.s3_steps,
                    'amp_skipped_steps': self.s3_skipped, 'gradient_batches': self.s3_batches,
                    'stop_policy': 'horizon_only', 'selection_source': 'source_dev only'})
                self.run_callbacks('on_fit_epoch_end')
                if self.stop or (self.stop_after_epoch and epoch+1 >= self.stop_after_epoch):
                    break
            self.final_eval()
            self.run_callbacks('on_train_end')
        finally:
            errors = []
            for close in (self.train_loader.close, getattr(self.test_loader, 'close', lambda: None),
                          self.s3_observer.close):
                try:
                    close()
                except Exception as exc:
                    errors.append(f'{type(exc).__name__}: {exc}')
            if errors:
                raise ContractError('S3 cleanup failed: '+'; '.join(errors))

    def final_eval(self):
        import numpy as np
        import detection_metrics as dm
        from train_monitor import read_only_observation
        from source_contract import read_json
        from source_policy import summarize
        from source_release import deduplicate_roles
        started = time.perf_counter()
        local_export = getattr(self, 's3_local_export', False)
        roles = deduplicate_roles(self.save_dir)
        role = 'best' if 'best' in roles else 'diagnostic'
        selected = roles[role]
        folder = self.save_dir/'final_eval'
        folder.mkdir(exist_ok=True)
        if self.source_identity.get('evaluation_policy') == S3_EVALUATION and not local_export:
            # No additional cloud inference: load and bind every retained role.
            for entry in {r['path']: r for r in roles.values()}.values():
                path = self.save_dir/entry['path']
                checkpoint = torch.load(path, map_location='cpu', weights_only=False)
                check_equal('handoff checkpoint identity', self.source_identity, checkpoint['source_identity'])
                check_equal('handoff checkpoint epoch', entry['epoch'], checkpoint['epoch']+1)
                check_equal('handoff checkpoint names', NAMES, list(checkpoint['ema'].names.values()))
                saved = read_json(self.save_dir/f"observations/epoch{entry['epoch']:03d}.json")
                check_equal('handoff dev metrics', saved['dev']['metrics'], checkpoint['source_observation']['dev']['metrics'])
                del checkpoint
            write_json(folder/'run.json', {
                'status': 'pending_local_evaluation', 'selected_role': role, 'roles': roles,
                'results': {}, 'last_dev_checkpoint_sha256': sha256(self.last),
                'weight_reload_checked': True, 'final_tests_run': False,
                'seconds': time.perf_counter()-started,
                'qualification': 'single_run_pending_repeats' if role == 'best' else 'no_eligible_checkpoint'})
            return
        results = {}
        drift = {}
        from contextlib import nullcontext
        with (nullcontext() if local_export else read_only_observation(self)):
            for purpose in ('source_dev', 'source_train'):
                result, matches, curves, rows = self.s3_observer.evaluate(purpose, self.save_dir/selected['path'],
                                            points=points if purpose == 'source_train' else None)
                if purpose == 'source_dev':
                    previous = read_json(self.save_dir/f"observations/epoch{selected['epoch']:03d}.json")['dev']
                    if local_export:
                        local_points, native_f1 = result['workpoints'], result['native_f1']
                        result, matches, _ = summarize(rows, self.s3_observer.rules, previous['workpoints'])
                        result.update(local_reselected_workpoints=local_points, native_f1=native_f1,
                                      threshold_origin='saved_cloud_dev_workpoints')
                        drift['selected_dev_metrics_equal'] = previous['metrics'] == result['metrics']
                        drift['selected_workpoint_counts_equal'] = all(
                            result['workpoints'][n][k] == previous['workpoints'][n][k]
                            for n in NAMES for k in ('tp','fp','fn'))
                        drift['selected_ap_delta_pp'] = {n: (result['metrics']['per_class'][n]['ap50_95']-
                            previous['metrics']['per_class'][n]['ap50_95'])*100 for n in NAMES}
                    else:
                        check_equal('selected saved EMA metric identity', previous['metrics'], result['metrics'])
                    points = result['workpoints']
                destination = folder/purpose
                destination.mkdir(exist_ok=True)
                dm.write_rows(destination/'objects.jsonl.gz', rows)
                dm.write_rows(destination/'matches.jsonl.gz', matches)
                np.savez_compressed(destination/'pr.npz', **curves)
                write_json(destination/'metrics.json', result)
                results[purpose] = {'images': len(rows), 'checkpoint_sha256': selected['sha256']}
            last_result, _, _, _ = self.s3_observer.evaluate('source_dev', self.last,
                points=self.s3_observation['dev']['workpoints'] if local_export else None)
            expected = self.s3_observation['dev']['metrics']
            if local_export:
                drift['last_dev_metrics_equal'] = expected == last_result['metrics']
                drift['last_workpoint_counts_equal'] = all(
                    last_result['workpoints'][n][k] == self.s3_observation['dev']['workpoints'][n][k]
                    for n in NAMES for k in ('tp','fp','fn'))
            else:
                check_equal('last saved EMA metric identity', expected, last_result['metrics'])
            write_json(folder/'last_dev.json', last_result)
        write_json(folder/'run.json', {'selected_role': role, 'roles': roles, 'results': results,
                   'status': 'complete_local' if local_export else 'complete',
                   'cross_device': drift,
                   'last_dev_checkpoint_sha256': sha256(self.last), 'final_tests_run': False,
                   'seconds': time.perf_counter()-started,
                   'qualification': 'single_run_pending_repeats' if role == 'best' else 'no_eligible_checkpoint'})
