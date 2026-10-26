"""Read-only split evaluation, staging and comparison boundaries."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from evaluate import compare_dev, evaluate, stage_subset, validator_args, verify_subset
from source_contract import ContractError, NAMES, NORMALIZATION, sha256, write_json


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


def test_comparison_refuses_another_checkpoint(tmp_path):
    reference = tmp_path / 'dev.json'
    write_json(reference, {'scope': 'source_dev only', 'checkpoint_sha256': 'other'})
    with pytest.raises(ContractError, match='same comparison'):
        compare_dev({}, reference, 'expected', tmp_path)
