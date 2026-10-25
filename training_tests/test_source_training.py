import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from train_source import configure_paths
configure_paths()
from source_contract import (ContractError, NAMES, canonical_hash, primary_fitness, safe_path,
                             select_records, sha256, stage_dataset, verify_source, write_json)


@pytest.fixture(scope='module')
def frozen(tmp_path_factory):
    root = tmp_path_factory.mktemp('中文_source_v1')
    mapping = [dict(training_id=i, official_id=i+1, name=n) for i, n in enumerate(NAMES)]
    components = {'raw_annotation_sha256': {'train': 'a'*64},
                  'normalized_annotation_sha256': {'train': 'b'*64}}
    write_json(root / 'protocol.json', {'fingerprint_components': components,
               'normalization_version': 'clip_xyxy_to_image_bounds_v1',
               'development_gradient_training': False})
    write_json(root / 'categories.json', {'category_mapping': mapping})
    files, manifests = [], {}
    for purpose, count in [('source_train', 4000), ('source_dev', 500)]:
        records = []
        for i in range(count):
            image = f'images/{purpose}/图{i:04d}.jpg'
            label = f'labels/{purpose}/图{i:04d}.txt'
            for kind, rel, data in [('image', image, b'fake image, hash contract only'),
                                    ('label', label, b'0 0.5 0.5 0.1 0.1\n2 0.5 0.5 0.1 0.1\n')]:
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                files.append(dict(path=rel, sha256=sha256(path), kind=kind, purpose=purpose))
            records.append(dict(key=f'{purpose}:{i}', purpose=purpose, official_split='train',
                                official_period='Daytime', group_id=f'{purpose}-g{i}',
                                image=image, label=label, classes=[0, 2],
                                image_sha256=sha256(root / image), label_sha256=sha256(root / label),
                                raw_annotation_sha256='a'*64, normalized_annotation_sha256='b'*64))
        relative = f'manifests/{purpose}.json'
        write_json(root / relative, dict(purpose=purpose, records=records))
        manifests[purpose] = dict(path=relative, count=count, sha256=sha256(root / relative))
    ready = dict(schema_version=1, version='source_v1', status='frozen',
                 normalization_version='clip_xyxy_to_image_bounds_v1',
                 data_fingerprint=canonical_hash(components), final_test_executed=False,
                 grouping_protocol=dict(path='protocol.json', sha256=sha256(root/'protocol.json')),
                 categories=dict(path='categories.json', sha256=sha256(root/'categories.json'), names=NAMES),
                 category_mapping=mapping, counts=dict(source_train=4000, source_dev=500, source_test=500),
                 manifests=manifests, files=files)
    write_json(root / 'READY.json', ready)
    return root, ready


def test_contract_full_whitelist_and_chinese_path(frozen):
    root, _ = frozen
    result = verify_source(root, sha256(root / 'READY.json'))
    assert len(result['files']) == 9000
    assert set(result['manifests']) == {'source_train', 'source_dev'}


@pytest.mark.parametrize('path', ['../source_test/a.jpg', '/tmp/a.jpg', 'C:/a.jpg', r'labels\a.txt'])
def test_path_traversal_rejected(tmp_path, path):
    with pytest.raises(ContractError):
        safe_path(tmp_path, path)


def test_ready_identity_checked_before_loading(frozen):
    root, _ = frozen
    with pytest.raises(ContractError, match='expected=.*actual='):
        verify_source(root, '0'*64)


@pytest.mark.parametrize('purpose', ['source_test', 'night_test', 'day_cross_scene_final'])
def test_forbidden_purpose(frozen, tmp_path, purpose):
    root, ready = frozen
    changed = dict(ready)
    changed['files'] = [dict(path=f'images/{purpose}/test.jpg', purpose=purpose, kind='image', sha256='0'*64)]
    # Metadata copies only; no real held-out data is opened or synthesized into the real root.
    shutil = __import__('shutil')
    for name in ('protocol.json', 'categories.json'):
        shutil.copyfile(root / name, tmp_path / name)
    write_json(tmp_path / 'READY.json', changed)
    with pytest.raises(ContractError, match='Forbidden purpose'):
        verify_source(tmp_path, sha256(tmp_path / 'READY.json'))


def test_tampered_file_reports_path_expected_actual(frozen):
    root, _ = frozen
    path = root / 'images/source_train/图0000.jpg'
    original = path.read_bytes()
    try:
        path.write_bytes(b'corrupt')
        with pytest.raises(ContractError, match='图0000.jpg: expected=.*actual='):
            verify_source(root, sha256(root/'READY.json'))
    finally:
        path.write_bytes(original)


def test_primary_metric_uses_real_indices_and_rejects_missing():
    score, classes = primary_fitness([2, 5, 0], [[.6]*10, [.99]*10, [.2]*10])
    assert score == pytest.approx(.4)
    assert classes[5] == pytest.approx(.99)
    with pytest.raises(ContractError, match='missing'):
        primary_fitness([2, 5], [[.6]*10, [.99]*10])


def test_subset_covers_classes_is_deterministic_and_includes_empty():
    rows = [dict(key=str(i), classes=[i % 6]) for i in range(80)] + [dict(key='blank', classes=[])]
    selected = select_records(rows, 32)
    assert selected == select_records(list(reversed(rows)), 32)
    assert {c for row in selected for c in row['classes']} == set(range(6))
    assert any(not r['classes'] for r in selected)


def test_strict_ties_and_zero_do_not_reset_patience():
    from source_trainer import StrictEarlyStopping
    stopper = StrictEarlyStopping(20)
    assert not stopper(1, 0.)
    assert not stopper(20, 0.)
    assert stopper(21, 0.)
    assert stopper.best_epoch == 1


def test_best_tie_retained_periodic_and_checkpoint_identity(tmp_path, monkeypatch):
    import torch
    from source_trainer import SourceTrainer, DetectionTrainer, StrictEarlyStopping
    def upstream_save(t):
        t.wdir.mkdir(exist_ok=True)
        ckpt = dict(epoch=t.epoch, optimizer={'sentinel': 1})
        torch.save(ckpt, t.last)
        torch.save(ckpt, t.best)
        return True
    monkeypatch.setattr(DetectionTrainer, 'save_model', upstream_save)
    t = SourceTrainer.__new__(SourceTrainer)
    t.wdir = tmp_path
    t.last, t.best = tmp_path/'last.pt', tmp_path/'best.pt'
    t.save_period = -1
    t.epoch = 0
    t.source_identity = {'ready_sha256': 'pinned'}
    t.stopper = StrictEarlyStopping(20)
    t.source_strict_improvement = True
    t.save_model()
    early = sha256(t.best)
    t.epoch = 4
    t.source_strict_improvement = False
    t.save_model()
    assert sha256(t.best) == early
    assert (tmp_path / 'epoch005.pt').is_file()
    ckpt = torch.load(t.last, weights_only=False)
    assert ckpt['source_identity'] == {'ready_sha256': 'pinned'}
    assert ckpt['epoch'] == 4 and ckpt['optimizer']['sentinel'] == 1


def test_frozen_recipe_rejects_byte_tampering_and_unapproved_batch(tmp_path):
    from train_source import verify_recipe
    path = tmp_path / 'recipe.json'
    ident = {'batch': 16, 'scope': 'formal'}
    write_json(path, {'status': 'frozen', 'identity': ident})
    expected_hash = sha256(path)
    assert verify_recipe(path, expected_hash, ident)['status'] == 'frozen'
    path.write_text(path.read_text(encoding='utf-8') + ' ', encoding='utf-8')
    with pytest.raises(ContractError, match='SHA-256'):
        verify_recipe(path, expected_hash, ident)
    write_json(path, {'status': 'frozen', 'identity': {'batch': 1, 'scope': 'formal'}})
    with pytest.raises(ContractError, match='Unauthorized formal batch'):
        verify_recipe(path, sha256(path), {'batch': 1, 'scope': 'formal'})
