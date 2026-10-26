"""Regression tests for full per-epoch train evaluation and untouched dev selection."""
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from source_contract import ContractError, NAMES, write_json
from train_monitor import EpochTrainMetrics, append_epoch, attach, audit_observation


def metric():
    return {'primary_ap50_95': .4, 'map50': .7, 'map50_95': .5, 'precision': .8, 'recall': .6,
            'per_class': {n: {'images': 4, 'instances': 5, 'precision': .8,
                             'recall': .6, 'ap50': .7, 'ap50_95': .5} for n in NAMES}}


def trainer():
    return NS(epoch=0, epoch_time=999., fitness=.3, metrics={'fitness': .3},
              ema=NS(ema=NS(training=False)), source_identity={'fixture': 'identity'},
              validator=NS(source_per_class={i: .3 for i in range(6)}, source_fitness=.3))


def observer(root, evaluator=lambda *_: metric(), seen=4):
    obj = EpochTrainMetrics(root, 4, {'enabled': True, 'every': 1}, evaluator)
    obj.validator = NS(seen=seen, dataloader=NS(dataset=NS(augment=False)))
    return obj


def test_epoch_records_train_and_dev_without_changing_fitness_and_measures_real_overhead(tmp_path, monkeypatch):
    import train_monitor
    ticks = iter([0., 10., 12.])
    monkeypatch.setattr(train_monitor.time, 'perf_counter', lambda: next(ticks))
    t = trainer(); obj = observer(tmp_path)
    obj.start_epoch(t); obj(t)
    assert t.fitness == .3 and t.metrics == {'fitness': .3} and not t.ema.ema.training
    with (tmp_path/'train_metrics.csv').open() as stream:
        row = next(csv.DictReader(stream))
    assert row['weights'] == 'current_epoch_ema'
    assert float(row['train_primary_ap50_95']) == .4
    assert float(row['dev_primary_ap50_95']) == .3
    assert float(row['base_epoch_seconds']) == 10. and float(row['overhead_ratio']) == .2
    assert float(row['pedestrian_dev_ap50_95']) == .3


@pytest.mark.parametrize('issue', ['fitness', 'count', 'augmentation'])
def test_invalid_observation_cannot_be_reported_as_success(tmp_path, issue):
    t = trainer()
    def evaluate(t, _):
        if issue == 'fitness': t.fitness = 1.
        return metric()
    obj = observer(tmp_path, evaluate, seen=3 if issue == 'count' else 4)
    if issue == 'augmentation': obj.validator.dataloader.dataset.augment = True
    obj.start_epoch(t)
    with pytest.raises(ContractError): obj(t)
    assert not (tmp_path/'train_metrics.csv').exists()


def test_duplicate_epoch_or_reduced_frequency_is_rejected(tmp_path):
    path = tmp_path/'train_metrics.csv'
    append_epoch(path, {'epoch': 1, 'value': .4})
    with pytest.raises(ContractError): append_epoch(path, {'epoch': 1, 'value': .5})
    with pytest.raises(ContractError): attach(None, tmp_path, 4, {'enabled': True, 'every': 2})
    assert attach(None, tmp_path, 4, {'enabled': False}) is None


def test_all_required_callbacks_registered():
    calls = {}
    t = NS(add_callback=lambda event, fn: calls.setdefault(event, fn))
    obj = attach(t, Path('run'), 4, {'enabled': True, 'every': 1})
    assert set(calls) == {'on_train_epoch_start', 'on_fit_epoch_end', 'on_train_end'}
    obj.close()


def test_full_epoch_coverage_is_required_for_cloud_receipt(tmp_path):
    identity = {'fixture': 'identity'}
    write_json(tmp_path/'source_run.json', {'identity': identity})
    write_json(tmp_path/'train_monitor.json', {'source_identity': identity, 'overhead_ratio': .2, 'evaluation_seconds': 4.})
    (tmp_path/'source_metrics.jsonl').write_text('{"epoch":1}\n{"epoch":2}\n', encoding='utf-8')
    append_epoch(tmp_path/'train_metrics.csv', {'epoch': 1, 'images': 4})
    with pytest.raises(ContractError): audit_observation(tmp_path)
    append_epoch(tmp_path/'train_metrics.csv', {'epoch': 2, 'images': 4})
    assert audit_observation(tmp_path)['epochs'] == 2


def test_validator_model_copy_preserves_original_ema_state():
    import torch
    from ultralytics.models.yolo.detect import DetectionValidator
    model = torch.nn.Linear(2, 2).float().train()
    copied = DetectionValidator.get_model(None, model)
    copied.half().eval()
    assert model.training and model.weight.dtype == torch.float32
    assert copied is not model


def test_real_validator_callback_preserves_ema_and_selection_on_synthetic_data(tmp_path):
    import torch
    from PIL import Image
    from ultralytics.nn.tasks import DetectionModel
    torch.set_num_threads(2)
    model = DetectionModel('yolo11n.yaml', nc=6, verbose=False).float().eval()
    model.names = dict(enumerate(NAMES))
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    images = tmp_path/'data/images/source_train'
    labels = tmp_path/'data/labels/source_train'
    images.mkdir(parents=True); labels.mkdir(parents=True)
    Image.new('RGB', (64, 64), 'white').save(images/'sample.jpg')
    (labels/'sample.txt').write_text(''.join(f'{i} 0.5 0.5 0.1 0.1\n' for i in range(6)), encoding='utf-8')
    run = tmp_path/'train'; run.mkdir()
    t = trainer()
    t.device = torch.device('cpu'); t.args = NS(imgsz=64)
    t.data = {'train': str(images)}; t.ema.ema = model
    monitor = EpochTrainMetrics(run, 1, {'enabled': True, 'every': 1, 'batch': 1, 'workers': 0})
    try:
        monitor.start_epoch(t); monitor(t)
        assert t.fitness == .3 and t.metrics == {'fitness': .3}
        assert not model.training
        assert all(torch.equal(initial[k], v) for k, v in model.state_dict().items())
        assert all(p.grad is None for p in model.parameters())
        assert (run/'train_metrics.csv').is_file()
    finally:
        monitor.close()
