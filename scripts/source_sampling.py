"""S3 source-train sampling, finite worker lifetime and consumed augmentation provenance."""
from __future__ import annotations

from collections import Counter
import functools
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from ultralytics.data.dataset import YOLODataset

from source_contract import ContractError, NAMES, canonical_hash, check_equal, safe_path, sha256, write_json

ALGORITHM = 'small_thin_union_weighted_v1'


def seed_for(seed, *parts):
    payload = json.dumps([ALGORITHM, seed, *parts], separators=(',', ':')).encode('utf-8')
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little')


def target_box(width, height, image_width, image_height):
    """Widths are original pixels derived from frozen normalized labels, as in S2."""
    w, h = round(float(width), 6), round(float(height), 6)
    return w*h < 32**2 or round(w*640/max(image_width, image_height), 6) <= 8


def training_index(contract, image_files=None):
    records = contract['manifests']['source_train']
    # Validate the entire purpose/path envelope BEFORE opening any label.
    for rec in records:
        check_equal('sampling purpose', 'source_train', rec['purpose'])
        for kind, prefix in [('image', 'images'), ('label', 'labels')]:
            rel = rec[kind]
            if not rel.startswith(prefix+'/source_train/') or rel not in contract['files']:
                raise ContractError('Unapproved sampling path: '+rel)
            check_equal('sampling whitelist purpose', 'source_train', contract['files'][rel]['purpose'])
    by_name = {Path(r['image']).name: r for r in records}
    check_equal('unique sampling image names', len(records), len(by_name))
    if image_files is not None:
        names = [Path(f).name for f in image_files]
        check_equal('loader complete train name set', sorted(by_name), sorted(names))
        records = [by_name[n] for n in names]
    else:
        records = sorted(records, key=lambda r: Path(r['image']).name)
    result = []
    for r in records:
        path = safe_path(contract['root'], r['label'])
        check_equal('sampling label SHA', r['label_sha256'], sha256(path))
        counts = [0]*6
        n_target = 0
        for line in path.read_text(encoding='utf-8').splitlines():
            c, x, y, w, h = map(float, line.split())
            if int(c) != c or int(c) not in range(6) or not all(np.isfinite([x, y, w, h])) or min(w, h) <= 0:
                raise ContractError('Invalid sampling GT: '+r['label'])
            counts[int(c)] += 1
            n_target += c == 0 and target_box(w*r['width'], h*r['height'], r['width'], r['height'])
        result.append({'key': r['key'], 'group_id': r['group_id'], 'image': r['image'],
                       'label_sha256': r['label_sha256'], 'target': bool(n_target),
                       'target_gt': int(n_target), 'classes': counts})
    return result


def indices_for(index, seed, arm, epoch):
    if arm not in ('R', 'S', 'U') or epoch < 0 or not index or (arm == 'U' and seed != 17):
        raise ContractError('Invalid sampling arm/epoch/index')
    rng = np.random.Generator(np.random.PCG64(seed_for(seed, 'S' if arm == 'U' else arm, epoch, 'indices')))
    if arm == 'R':
        return rng.permutation(len(index)).astype(np.int32)
    weights = np.ones(len(index)) if arm == 'U' else np.array([2. if r['target'] else 1. for r in index])
    return rng.choice(len(index), len(index), replace=True, p=weights/weights.sum()).astype(np.int32)


def describe(index, indices):
    counts = np.bincount(indices, minlength=len(index))
    groups = Counter()
    classes = np.zeros(6, dtype=np.int64)
    target = empty = 0
    for r, n in zip(index, counts):
        if not n:
            continue
        groups[r['group_id']] += int(n)
        classes += np.asarray(r['classes'])*n
        target += int(n)*r['target']
        empty += int(n)*(not any(r['classes']))
    return {'slots': len(indices), 'target_slots': target, 'other_slots': len(indices)-target,
            'empty_slots': empty, 'unique_images': int((counts > 0).sum()),
            'unique_groups': len(groups), 'max_repeat': int(counts.max()),
            'per_image_counts': counts.tolist(), 'per_group_counts': dict(sorted(groups.items())),
            'class_gt': dict(zip(NAMES, classes.tolist()))}


class AuditedYOLODataset(YOLODataset):
    """Track actual anchor/partner reads without changing labels or consuming RNG."""
    def __init__(self, *args, **kwargs):
        self._source_reads = None
        super().__init__(*args, **kwargs)

    def get_image_and_label(self, index):
        if self._source_reads is not None:
            self._source_reads.append(int(index))
        return super().get_image_and_label(index)

    def __getitem__(self, index):
        self._source_reads = []
        try:
            output = super().__getitem__(index)
            output['_source_audit'] = {'anchor': int(index), 'reads': list(self._source_reads)}
            return output
        finally:
            self._source_reads = None


def collate_audited(batch):
    evidence = [x.pop('_source_audit') for x in batch]
    output = YOLODataset.collate_fn(batch)
    output['_source_audit'] = evidence
    return output


def seed_worker(worker_id, seed, epoch):
    value = seed_for(seed, 'worker', epoch, worker_id)
    random.seed(value)
    np.random.seed(value % 2**32)
    torch.manual_seed(value)
    # Albumentations has an independent generator in the pinned version.
    info = torch.utils.data.get_worker_info()
    def seed_transform(transform):
        inner = getattr(transform, 'transform', None)
        if inner is not None and hasattr(inner, 'set_random_seed'):
            inner.set_random_seed(value % 2**32)
        for child in getattr(transform, 'transforms', []):
            seed_transform(child)
    seed_transform(info.dataset.transforms)


class FiniteEpochLoader:
    """A fresh finite DataLoader per epoch; no sampler or worker prefetch can cross epochs."""
    def __init__(self, dataset, index, seed, arm, batch_size=16, workers=4, pin_memory=True):
        if len(dataset) != len(index) or len(index) % batch_size or workers < 1:
            raise ContractError('Finite loader needs complete divisible data and explicit workers')
        self.dataset, self.index = dataset, index
        self.seed, self.arm = seed, arm
        self.batch_size, self.num_workers = batch_size, workers
        self.pin_memory, self.drop_last = pin_memory, False
        self.epoch = None
        self.iterator = None
        self.sampler = []
        self.worker_exit = None

    def __len__(self):
        return len(self.index)//self.batch_size

    def set_epoch(self, epoch):
        self.close()
        self.epoch = int(epoch)
        self.sampler = indices_for(self.index, self.seed, self.arm, epoch).tolist()
        self.consumed, self.reads = [], []
        self.final_class = np.zeros(6, dtype=np.int64)
        self.final_hist = np.zeros((6, 3, 3), dtype=np.int64)
        self.completed = False

    def reset(self):
        self.close()

    def close(self):
        iterator, self.iterator = self.iterator, None
        if iterator is not None and hasattr(iterator, '_shutdown_workers'):
            workers = list(iterator._workers)
            iterator._shutdown_workers()
            self.worker_exit = [{'alive': w.is_alive(), 'exitcode': w.exitcode} for w in workers]
            if any(w['alive'] or w['exitcode'] != 0 for w in self.worker_exit):
                raise ContractError('Training worker did not exit cleanly: '+str(self.worker_exit))

    def __iter__(self):
        if self.epoch is None or self.iterator is not None or self.completed:
            raise ContractError('Finite loader needs exactly one iteration per registered epoch')
        generator = torch.Generator().manual_seed(seed_for(self.seed, 'loader', self.epoch))
        loader = DataLoader(self.dataset, batch_size=self.batch_size, sampler=self.sampler,
                            num_workers=self.num_workers, drop_last=False, pin_memory=self.pin_memory,
                            persistent_workers=False, in_order=True, prefetch_factor=2, generator=generator,
                            worker_init_fn=functools.partial(seed_worker, seed=self.seed, epoch=self.epoch),
                            collate_fn=collate_audited)
        self.iterator = iter(loader)
        try:
            for batch in self.iterator:
                evidence = batch.pop('_source_audit')
                for entry in evidence:
                    check_equal('first source read is anchor', entry['anchor'], entry['reads'][0])
                    self.consumed.append(entry['anchor'])
                    self.reads.extend(entry['reads'])
                cls = batch['cls'].reshape(-1).numpy().astype(int)
                wh = batch['bboxes'][:, 2:].numpy()*np.array(batch['img'].shape[-2:][::-1])
                wi = np.searchsorted([8., 16.], wh[:, 0], side='left')
                hi = np.searchsorted([32., 64.], wh[:, 1], side='left')
                np.add.at(self.final_class, cls, 1)
                np.add.at(self.final_hist, (cls, wi, hi), 1)
                yield batch
            check_equal('actual consumed sampler order', self.sampler, self.consumed)
            self.completed = True
        finally:
            self.close()

    def finish(self, folder):
        if not self.completed or self.worker_exit is None:
            raise ContractError('Cannot publish incomplete epoch consumption')
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder/f'epoch{self.epoch+1:03d}.npz'
        np.savez_compressed(path, consumed=np.array(self.consumed, dtype=np.int32))
        report = {'epoch': self.epoch+1, 'arm': self.arm, 'seed': self.seed,
                'algorithm': ALGORITHM, 'index_sha256': canonical_hash(self.index),
                'consumed_sha256': sha256(path), 'worker_exit': self.worker_exit,
                'anchor': describe(self.index, self.consumed),
                'all_source_reads': describe(self.index, self.reads),
                'final_class_gt': dict(zip(NAMES, self.final_class.tolist())),
                'final_wh_histogram': self.final_hist.tolist(),
                'histogram': {'width_edges_px': [8, 16], 'height_edges_px': [32, 64],
                              'right_closed': True, 'coordinate_space': 'augmented input pixels'},
                'lineage_limit': 'source reads and final boxes; no per-GT survival attribution'}
        if self.arm == 'U':
            report.update(indices_rng_scope='S', replacement=True, target_weight=1, other_weight=1)
        return report
