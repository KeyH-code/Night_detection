import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
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


@pytest.mark.parametrize('actual,accepted', [('0', True), (0, True), ('1', False), ('cpu', False)])
def test_effective_gpu_id_accepts_framework_string_only_for_same_device(tmp_path, monkeypatch, actual, accepted):
    import torch
    from source_trainer import SourceTrainer
    trainer = SourceTrainer.__new__(SourceTrainer)
    trainer.callbacks = {'on_pretrain_routine_end': [], 'on_train_batch_end': [], 'on_fit_epoch_end': []}
    trainer.source_expected_args = {'device': 0, 'workers': 4}
    trainer.args = SimpleNamespace(device=actual, workers=4)
    trainer.device, trainer.amp = torch.device('cuda:0'), True
    trainer.optimizer = SimpleNamespace(param_groups=[])
    trainer.accumulate = 4
    trainer.scheduler = SimpleNamespace(last_epoch=0)
    trainer.scaler = SimpleNamespace(state_dict=lambda: {'scale': 1024.})
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda: None)
    trainer.attach_audits(tmp_path)
    callback = trainer.callbacks['on_pretrain_routine_end'][0]
    if accepted:
        callback(trainer)
        assert json.loads((tmp_path/'trainer_effective.json').read_text(encoding='utf-8'))['device'] == actual
    else:
        with pytest.raises(ContractError, match='effective trainer.device'):
            callback(trainer)


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


def test_s3_config_is_explicit_original_gradient_recipe_with_only_registered_controls(tmp_path):
    import yaml
    from train_source import config_at
    root = Path(__file__).resolve().parents[2]
    config = config_at(root/'configs/source_s3.yaml')
    assert config['training']['epochs'] == 100 and config['training']['close_mosaic'] == 10
    assert config['monitor']['batch'] == 1 and 'batch_fallback' not in config['cloud']
    for section, key, value in [('training', 'scale', .6), ('sampling', 'target_weight', 3),
                                 ('monitor', 'every', 2), ('training', 'seed', 19),
                                 ('numerics', 'amp_initial_scale', 65536.),
                                 ('numerics', 'amp_initial_scale', 1024.),
                                 ('monitor', 'profile', 'unregistered_sampled_mode')]:
        changed = __import__('copy').deepcopy(config)
        changed[section][key] = value
        path = tmp_path/'source_s3.yaml'; path.write_text(yaml.safe_dump(changed), encoding='utf-8')
        with pytest.raises(ContractError): config_at(path)


def test_s3_new_scaler_uses_registered_initial_scale():
    import torch
    from source_trainer import S3Trainer
    trainer = S3Trainer.__new__(S3Trainer)
    trainer.args = SimpleNamespace(epochs=100)
    trainer.resume, trainer.amp = False, True
    trainer.device = torch.device('cpu')
    trainer.s3_amp_initial_scale = 1.
    trainer.resume_training(None)
    assert trainer.scaler.state_dict()['scale'] == 1.


def test_nonfinite_gradient_retains_batch_context_without_optimizer_update(tmp_path):
    import torch
    from source_trainer import S3Trainer
    trainer = S3Trainer.__new__(S3Trainer)
    trainer.model = torch.nn.Linear(1, 1)
    before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=.1)
    trainer.scaler = torch.amp.GradScaler('cpu', enabled=False)
    trainer.save_dir, trainer.epoch, trainer.s3_batches, trainer.s3_steps = tmp_path, 0, 7, 7
    trainer.train_loader, trainer.accumulate = range(250), 1
    trainer.loss, trainer.loss_items = torch.tensor(73.), {'box_loss': torch.tensor(1.)}
    for parameter in trainer.model.parameters(): parameter.grad = torch.full_like(parameter, float('inf'))
    with pytest.raises(ContractError, match='Nonfinite gradient'):
        trainer.optimizer_step()
    record = json.loads((tmp_path/'numerical_failure.json').read_text(encoding='utf-8'))
    assert record['epoch'] == 1 and record['batch_0based'] == 7
    assert record['optimizer_steps_before'] == 7 and record['amp_scale'] == 1.
    assert record['update_skipped_or_retried'] is False
    assert all(torch.equal(before[k], v) for k, v in trainer.model.state_dict().items())


class TinyCriterion:
    def __init__(self): self.updates = 0
    def update(self): self.updates += 1


def tiny_model():
    # Defined through a module-level class below, so checkpoints can be unpickled.
    return TinyTrainingModel()


import torch


class TinyTrainingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(.2), torch.nn.Linear(4, 1))
        self.criterion = TinyCriterion()
    def forward(self, batch):
        loss = (self.model(batch['img'])-batch['target']).square().mean()
        return loss, {'box_loss': loss.detach()}


class TinyLoader:
    """Only exercises optimizer/RNG/resume; real workers/mosaic are tested in test_sampling."""
    def __init__(self):
        self.dataset = SimpleNamespace(close_mosaic=lambda **_: None)
    def __len__(self): return 5
    def set_epoch(self, epoch): self.epoch = epoch
    def close(self): pass
    def finish(self, folder): return {'epoch': self.epoch+1, 'synthetic': True}
    def __iter__(self):
        for i in range(5):
            yield {'img': torch.full((4, 3), (self.epoch*5+i+1)/20), 'target': torch.full((4, 1), .2)}


@pytest.mark.parametrize('dev_only', [False, True])
@pytest.mark.parametrize('arm', ['R', 'U'])
def test_actual_s3_loop_completed_epoch_resume_matches_continuous_pending_gradients_and_rng(tmp_path, monkeypatch, dev_only, arm):
    import copy
    import random
    import numpy as np
    from source_trainer import S3Trainer
    from source_policy import SelectionPolicy, load_rules, reference_at
    from train_monitor import state_digest
    from ultralytics.utils.torch_utils import ModelEMA
    import train_monitor
    root = Path(__file__).resolve().parents[2]
    def record():
        return {'metrics': {'images': 4000, 'primary_ap50_95': 0., 'per_class': {
                    n: {'instances': 100, 'groups': 80, 'ap50': 0., 'ap75': 0., 'ap50_95': 0.} for n in NAMES}},
                'workpoints': {n: {'origin': 'rule', 't_rule': .5, 'threshold': .5, 'precision': 0.,
                                  'recall': 0., 'tp': 0, 'fp': 3, 'fn': 100} for n in NAMES},
                'slices': [{'class': 'pedestrian', 'slice': s, 'recall50': 0.} for s in ('area_S','width_le8')]}
    class FakeObservation:
        def __init__(self, t, *args): self.t = t
        def epoch(self):
            random.random(); np.random.random(); torch.rand(7)
            dev = record(); dev['metrics']['images'] = 500
            score = .01*(self.t.epoch+1)
            dev['metrics']['primary_ap50_95'] = score
            for row in dev['metrics']['per_class'].values():
                row.update(ap50=score, ap75=score, ap50_95=score)
            train = {'status':'deferred_local','images_evaluated':0} if dev_only else record()
            return {'epoch': self.t.epoch+1, 'train': train, 'dev': dev, 'evaluation_seconds': 0.}, [], []
        def close(self): pass
    monkeypatch.setattr(train_monitor, 'S3Observation', FakeObservation)
    monkeypatch.setattr(S3Trainer, 'final_eval', lambda self: None)
    monkeypatch.setattr(S3Trainer, 'preprocess_batch', lambda self, b: b)
    def make(folder, stop=None, checkpoint=None):
        random.seed(17); np.random.seed(17); torch.manual_seed(17)
        t = S3Trainer.__new__(S3Trainer)
        t.args = SimpleNamespace(time=None, compile=False, multi_scale=0., epochs=4, close_mosaic=1,
            seed=17, nbs=16, warmup_bias_lr=.1, warmup_momentum=.8, momentum=.9, model='synthetic')
        t.world_size, t.epochs, t.batch_size = 1, 4, 4
        t.save_dir, t.wdir = folder, folder/'weights'; folder.mkdir(exist_ok=True)
        t.last, t.best = t.wdir/'last.pt', t.wdir/'best.pt'
        t.source_identity = {'s3_policy': 'synthetic_regression'}
        if dev_only:
            from source_contract import S3_EVALUATION
            t.source_identity['evaluation_policy'] = S3_EVALUATION
        t.s3_contract, t.s3_arm = {}, arm
        t.s3_policy = SelectionPolicy(load_rules(root/'RULES.md'), reference_at(root), horizon=4)
        t.callbacks = {k: [] for k in ('on_train_start','on_train_epoch_start','on_train_batch_start',
                      'on_train_batch_end','on_train_epoch_end','on_model_save','on_fit_epoch_end','on_train_end')}
        t.stop_after_epoch = stop
        t.s3_amp_initial_scale = 1.
        def setup():
            t.model = tiny_model()
            t.optimizer = torch.optim.SGD(t.model.parameters(), lr=.03, momentum=.9)
            t.scheduler = torch.optim.lr_scheduler.LambdaLR(t.optimizer, lambda e: (1+__import__('math').cos(e*__import__('math').pi/4))*.5)
            t.lf = lambda e: (1+__import__('math').cos(e*__import__('math').pi/4))*.5
            t.scaler = torch.amp.GradScaler('cpu', enabled=False)
            t.ema = ModelEMA(t.model)
            t.start_epoch, t.accumulate, t.best_fitness = 0, 4, None
            t.device, t.amp = torch.device('cpu'), False
            t.train_loader, t.test_loader = TinyLoader(), SimpleNamespace()
            t.freeze_layer_names = []
            t._source_optimizer_steps = 0
            t.loss_names, t.metrics, t.stop = (), {}, False
            t.resume = checkpoint is not None
            t.resume_training(checkpoint)
            t.validator = lambda trainer: {}
        t._setup_train = setup
        t._get_warmup_iterations = lambda _: 0
        t.save_metrics = lambda **_: None
        t._do_train()
        return t
    full = make(tmp_path/'full')
    interrupted = make(tmp_path/'split', stop=2)
    ckpt = torch.load(interrupted.last, map_location='cpu', weights_only=False)
    assert any(v is not None and torch.count_nonzero(v) for v in ckpt['source_resume']['gradients'].values())
    assert not interrupted.best.exists() and (interrupted.wdir/'diagnostic.pt').is_file()
    # Completed final export may alias a selected role to last. Resume must
    # preserve that role before overwriting last at the next epoch.
    from source_release import deduplicate_roles
    write_json(interrupted.save_dir/'source_run.json', {'identity': interrupted.source_identity})
    roles = deduplicate_roles(interrupted.save_dir)
    assert roles['diagnostic']['path'] == 'weights/last.pt'
    assert not (interrupted.wdir/'diagnostic.pt').exists()
    resumed = make(tmp_path/'split', checkpoint=ckpt)
    for attr in ('model', 'optimizer', 'scheduler'):
        assert state_digest(getattr(full, attr).state_dict()) == state_digest(getattr(resumed, attr).state_dict())
    assert state_digest(full.ema.ema.state_dict()) == state_digest(resumed.ema.ema.state_dict())
    assert full.s3_steps == resumed.s3_steps == 5
    assert full.s3_batches == resumed.s3_batches == 20
    assert full.model.criterion.updates == resumed.model.criterion.updates == 4
    assert full.s3_policy.state == resumed.s3_policy.state
    if dev_only:
        assert (full.save_dir/'dev_metrics.csv').is_file()
        assert not (full.save_dir/'train_metrics.csv').exists()


def test_completed_formal_resume_refused_before_any_evidence_truncation(tmp_path):
    from train_source import validate_s3_resume
    marker = tmp_path/'source_metrics.jsonl'; marker.write_text('keep original', encoding='utf-8')
    write_json(tmp_path/'manifest.json', {'status':'completed_formal'})
    with pytest.raises(ContractError, match='Completed'):
        validate_s3_resume(tmp_path, tmp_path/'weights/last.pt', {})
    assert marker.read_text(encoding='utf-8') == 'keep original'
    with pytest.raises(ContractError, match='latest complete last'):
        validate_s3_resume(tmp_path, tmp_path/'weights/diagnostic.pt', {})


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_nonfinite_dev_loss_aborts_before_rule_evaluation_or_checkpoint(bad, monkeypatch):
    import contextlib
    import train_monitor
    from source_trainer import S3Trainer
    monkeypatch.setattr(train_monitor, 'read_only_observation', lambda _: contextlib.nullcontext())
    t = S3Trainer.__new__(S3Trainer)
    t.validator = lambda _: {'val/box_loss': bad}
    t.s3_observer = SimpleNamespace(epoch=lambda: pytest.fail('Invalid dev loss must abort before expensive evaluation'))
    with pytest.raises(ContractError, match='Nonfinite native dev'): t.validate()
