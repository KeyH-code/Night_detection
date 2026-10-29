"""Read-only split evaluation, staging and comparison boundaries."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from evaluate import compare_dev, evaluate, stage_subset, validator_args, verify_subset
from source_contract import ContractError, NAMES, NORMALIZATION, sha256, write_json


def test_gpu_ema_matches_serialized_checkpoint_before_fusion_without_touching_live_model(tmp_path):
    import copy
    import torch
    from evaluate import evaluation_ema
    from ultralytics.nn.backends.pytorch import PyTorchBackend
    from ultralytics.nn.tasks import DetectionModel
    from train_monitor import state_digest
    if not torch.cuda.is_available():
        pytest.skip('FP16 S3 requires CUDA; no implicit CPU precision substitution')
    torch.set_num_threads(2); torch.manual_seed(17)
    model = DetectionModel('yolo11n.yaml', nc=6, verbose=False).float().eval().to('cuda:0')
    model.names = dict(enumerate(NAMES))
    # Nontrivial BN state catches changes in fuse/cast ordering.
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.running_mean.copy_(torch.randn_like(layer.running_mean)*.2)
                layer.running_var.copy_(torch.rand_like(layer.running_var)+.2)
    before = state_digest(model.state_dict())
    path = tmp_path/'checkpoint.pt'
    saved = copy.deepcopy(model).half().to(memory_format=torch.contiguous_format)
    torch.save({'ema': saved, 'model': None, 'train_args': {'task': 'detect'}, 'version': '8.4.171'}, path)
    direct = PyTorchBackend(evaluation_ema(model, 'cuda:0'), torch.device('cuda:0'), fp16=True, verbose=False)
    loaded = PyTorchBackend(path, torch.device('cuda:0'), fp16=True, verbose=False)
    image = torch.rand((1,3,64,64), device='cuda:0', dtype=torch.float16)
    with torch.inference_mode():
        assert torch.equal(direct.forward(image)[0], loaded.forward(image)[0])
    assert before == state_digest(model.state_dict())


def fixture(tmp_path):
    root = tmp_path / '中文数据'
    files = []
    for kind, suffix in [('image', 'jpg'), ('label', 'txt')]:
        path = f'{"images" if kind == "image" else "labels"}/source_train/a.{suffix}'
        p = root / path; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b'fixture')
        files.append({'path': path, 'kind': kind, 'purpose': 'source_train', 'sha256': sha256(p)})
    write_json(root / 'protocol.json', {})
    write_json(root / 'categories.json', {})
    rec = {'key': 'train:1', 'purpose': 'source_train', 'image': files[0]['path'], 'label': files[1]['path'],
           'image_sha256': files[0]['sha256'], 'label_sha256': files[1]['sha256']}
    write_json(root / 'train.json', {'purpose': 'source_train', 'records': [rec]})
    (root / 'dev.json').write_bytes(b'Never parse this during train evaluation')
    write_json(root / 'READY.json', {'schema_version': 1, 'version': 'source_v1', 'status': 'frozen',
        'normalization_version': NORMALIZATION, 'categories': {'names': NAMES, 'path': 'categories.json', 'sha256': sha256(root/'categories.json')},
        'grouping_protocol': {'path': 'protocol.json', 'sha256': sha256(root/'protocol.json')},
        'counts': {'source_train': 1}, 'data_fingerprint': 'fixture', 'files': files,
        'manifests': {'source_train': {'path': 'train.json', 'sha256': sha256(root/'train.json')}, 'source_dev': {'path': 'dev.json'}}})
    return root, sha256(root / 'READY.json')


def test_only_explicit_split_is_verified_and_staged(tmp_path):
    root, ready = fixture(tmp_path)
    contract = verify_subset(root, ready, 'source_train')
    output = tmp_path / 'sandbox'
    data = stage_subset(contract, output)
    assert data.is_file()
    assert sorted(p.suffix for p in output.rglob('*') if p.is_file()) == ['.jpg', '.txt', '.yaml']
    assert not (output / 'dev.json').exists()


@pytest.mark.parametrize('purpose', ['source_test', 'night_test', 'night_pool'])
def test_final_or_target_purposes_rejected_before_io(tmp_path, purpose):
    with pytest.raises(ContractError, match='Forbidden'):
        verify_subset(tmp_path, 'bad', purpose)


def test_tampering_and_wrong_checkpoint_rejected(tmp_path):
    root, ready = fixture(tmp_path)
    contract = verify_subset(root, ready, 'source_train')
    weights = tmp_path / 'best.pt'; weights.write_bytes(b'weights')
    with pytest.raises(ContractError, match='checkpoint SHA'):
        evaluate(weights, 'bad', contract, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()
    (root / contract['records'][0]['image']).write_bytes(b'changed')
    with pytest.raises(ContractError):
        verify_subset(root, ready, 'source_train')


def test_validation_settings_have_no_training_augmentation():
    args = validator_args(Path('data.yaml'))
    assert args['mode'] == 'val' and not args['augment'] and not args['plots']
    assert args['conf'] == .001 and args['iou'] == .7 and args['max_det'] == 300


@pytest.mark.parametrize('count,truncated', [(299, False), (300, False), (400, True)])
def test_nms_cap_audit_distinguishes_exact_count_from_actual_truncation(count, truncated):
    from types import SimpleNamespace
    import torch
    import torchvision  # same NMS backend for the capped and replayed calls
    from evaluate import evidence_validator_class
    cls = evidence_validator_class()
    validator = cls.__new__(cls)
    validator.args = SimpleNamespace(conf=.001, iou=.7, task='detect', single_cls=False,
                                    agnostic_nms=False, max_det=300)
    validator.nc, validator.end2end, validator.trace_context = 6, False, None
    raw = torch.zeros(1, 10, count)
    raw[0, 0] = torch.arange(count) % 20 * 3 + 1
    raw[0, 1] = torch.arange(count) // 20 * 3 + 1
    raw[0, 2:4] = 1.
    raw[0, 4] = torch.linspace(.9, .5, count)
    original = raw.clone()
    output = validator.postprocess(raw)
    assert len(output[0]['cls']) == min(300, count)
    assert validator.post_nms_counts == [count]
    assert (validator.post_nms_counts[0] > 300) == truncated
    assert torch.equal(raw, original)


def test_comparison_refuses_another_checkpoint(tmp_path):
    reference = tmp_path / 'dev.json'
    write_json(reference, {'scope': 'source_dev only', 'checkpoint_sha256': 'other'})
    with pytest.raises(ContractError, match='same comparison'):
        compare_dev({}, reference, 'expected', tmp_path)
