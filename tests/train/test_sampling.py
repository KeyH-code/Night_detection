"""Sampling membership, consumed workers and augmentation semantics, using synthetic train-only images."""
import copy
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from source_contract import ContractError, NAMES, sha256
from source_sampling import (AuditedYOLODataset, FiniteEpochLoader, describe, indices_for,
                             target_box, training_index)


def index(n=4000):
    return [{'key': str(i), 'group_id': 'g'+str(i//2), 'target': i < 922,
             'classes': [1, 0, 1, 0, 0, 0] if i < 922 else [0]*6} for i in range(n)]


@pytest.mark.parametrize('wh,size,expected', [
    ((32, 32), (640, 640), False), ((31.999999, 32), (640, 640), True),
    ((24.00000000001, 120), (1920, 1080), True), ((24.000002, 120), (1920, 1080), False),
    ((8, 150), (640, 480), True), ((10, 50), (640, 480), True)])
def test_target_union_pixel_rounding(wh, size, expected):
    assert target_box(*wh, *size) == expected


def test_reference_once_weighted_probabilities_and_positive_background():
    records = index()
    r = indices_for(records, 17, 'R', 0)
    assert sorted(r) == list(range(4000))
    assert np.array_equal(r, indices_for(records, 17, 'R', 0))
    assert not np.array_equal(r, indices_for(records, 17, 'R', 1))
    counts = np.zeros(4000, dtype=int)
    for epoch in range(100):
        s = indices_for(records, 17, 'S', epoch)
        assert len(s) == 4000
        counts += np.bincount(s, minlength=4000)
    assert counts.min() > 0
    assert counts[:922].sum()/counts.sum() == pytest.approx(1844/4922, abs=.003)
    report = describe(records, s)
    assert report['slots'] == 4000 and report['unique_images'] < 4000
    assert report['target_slots']+report['other_slots'] == 4000
    assert sum(report['per_group_counts'].values()) == 4000


@pytest.mark.parametrize('arm', ['R', 'S', 'U'])
def test_sampler_does_not_consume_global_random_state(arm):
    from train_monitor import rng_state, restore_rng
    state = rng_state()
    expected = (random.random(), np.random.random(), torch.rand(3))
    restore_rng(state)
    indices_for(index(), 17, arm, 27)
    actual = (random.random(), np.random.random(), torch.rand(3))
    assert actual[:2] == expected[:2] and torch.equal(actual[2], expected[2])


def test_forbidden_membership_rejected_before_labels_are_opened(tmp_path):
    bad = {'purpose': 'source_dev', 'image': 'images/source_dev/a.jpg', 'label': 'labels/source_dev/a.txt'}
    with pytest.raises(ContractError, match='purpose'):
        training_index({'root': tmp_path, 'manifests': {'source_train': [bad]}, 'files': {}})


def test_train_index_uses_labels_not_dev_errors_and_counts_union_once(tmp_path):
    label = tmp_path/'labels/source_train/a.txt'; label.parent.mkdir(parents=True)
    label.write_text('0 0.5 0.5 0.01 0.1\n0 0.5 0.5 0.5 0.5\n2 0.5 0.5 0.01 0.01\n', encoding='utf-8')
    r = {'purpose': 'source_train', 'key': 'train:a', 'group_id': 'g', 'width': 1920, 'height': 1080,
         'image': 'images/source_train/a.jpg', 'label': 'labels/source_train/a.txt', 'label_sha256': sha256(label)}
    contract = {'root': tmp_path, 'manifests': {'source_train': [r]},
                'files': {r[k]: {'purpose': 'source_train'} for k in ('image', 'label')}}
    records = training_index(contract, ['some/sandbox/a.jpg'])
    assert records[0]['target'] and records[0]['target_gt'] == 1
    assert records[0]['classes'] == [2, 0, 1, 0, 0, 0]
    label.write_text('', encoding='utf-8')
    with pytest.raises(ContractError, match='SHA'):
        training_index(contract)


def dataset(tmp_path):
    from PIL import Image
    from ultralytics.cfg import get_cfg
    images = tmp_path/'images/source_train'; images.mkdir(parents=True)
    labels = tmp_path/'labels/source_train'; labels.mkdir(parents=True)
    for i in range(16):
        Image.new('RGB', (192, 128), (i*11, i*7, i*13)).save(images/f'{i:02d}.jpg')
        (labels/f'{i:02d}.txt').write_text(f'{i%6} 0.5 0.5 0.15 0.4\n', encoding='utf-8')
    hyp = get_cfg(overrides={'imgsz': 64, 'mosaic': 1., 'mixup': 0., 'cutmix': 0., 'copy_paste': 0., 'cache': False})
    return AuditedYOLODataset(str(images), imgsz=64, batch_size=4, augment=True, hyp=hyp,
                              rect=False, cache=False, stride=32, data={'names': NAMES, 'nc': 6})


def test_audit_wrapper_preserves_images_boxes_and_random_draws(tmp_path):
    from ultralytics.data.dataset import YOLODataset
    from train_monitor import rng_state, restore_rng
    audited = dataset(tmp_path)
    plain = copy.deepcopy(audited)
    state = rng_state()
    a = audited[0]; next_a = (random.random(), np.random.random(), torch.rand(2))
    restore_rng(state)
    b = YOLODataset.__getitem__(plain, 0); next_b = (random.random(), np.random.random(), torch.rand(2))
    for key in ('img', 'cls', 'bboxes', 'batch_idx'):
        assert torch.equal(a[key], b[key])
    assert next_a[:2] == next_b[:2] and torch.equal(next_a[2], next_b[2])
    assert len(a['_source_audit']['reads']) == 4


@pytest.mark.parametrize('arm', ['S', 'U'])
def test_real_workers4_consume_exact_finite_epochs_replay_and_close_mosaic(tmp_path, arm):
    from ultralytics.cfg import get_cfg
    ds = dataset(tmp_path/'data')
    records = index(16)
    loader = FiniteEpochLoader(ds, records, 17, arm, batch_size=4, workers=4, pin_memory=False)
    state = torch.get_rng_state().clone()
    loader.set_epoch(0)
    first = [b['img'].clone() for b in loader]
    report = loader.finish(tmp_path/'evidence')
    if arm == 'U':
        assert report['indices_rng_scope'] == 'S' and report['target_weight'] == report['other_weight'] == 1
    assert report['anchor']['slots'] == 16 and report['all_source_reads']['slots'] == 64
    assert len(report['worker_exit']) == 4 and all(not w['alive'] and w['exitcode'] == 0 for w in report['worker_exit'])
    assert torch.equal(state, torch.get_rng_state())
    with pytest.raises(ContractError):
        list(loader)
    loader.set_epoch(0)
    replay = [b['img'] for b in loader]
    assert all(torch.equal(a, b) for a, b in zip(first, replay))
    ds.close_mosaic(get_cfg())
    loader.reset(); loader.set_epoch(90)
    list(loader)
    closed = loader.finish(tmp_path/'closed')
    assert closed['all_source_reads']['slots'] == 16
    assert sum(closed['final_class_gt'].values()) == 16
    assert np.array_equal(np.load(tmp_path/'closed/epoch091.npz')['consumed'], indices_for(records, 17, arm, 90))
    loader.close()


def test_u_uniform_replacement_uses_s_stream_explicit_probabilities_and_keeps_all_images_positive():
    from source_sampling import seed_for
    records = index(); total = np.zeros(4000, dtype=int)
    for epoch in range(100):
        u = indices_for(records, 17, 'U', epoch)
        oracle = np.random.Generator(np.random.PCG64(seed_for(17, 'S', epoch, 'indices'))).choice(
            4000, 4000, replace=True, p=np.ones(4000)/4000).astype(np.int32)
        assert np.array_equal(u, oracle)
        assert len(u) == 4000 and u.min() >= 0 and u.max() < 4000 and len(set(u)) < 4000
        total += np.bincount(u, minlength=4000)
    assert total.min() > 0 and total[:922].sum()/total.sum() == pytest.approx(922/4000, abs=.003)
    assert not np.array_equal(indices_for(records, 17, 'U', 0), indices_for(records, 17, 'U', 1))
    # Target flags cannot influence U, while S deliberately uses them.
    flipped = [{**r, 'target': not r['target']} for r in records]
    assert np.array_equal(indices_for(records, 17, 'U', 0), indices_for(flipped, 17, 'U', 0))
    with pytest.raises(ContractError): indices_for(records, 29, 'U', 0)
