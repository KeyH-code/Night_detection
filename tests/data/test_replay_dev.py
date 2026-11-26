"""Development replay contract tests without CUDA or private data."""
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
from zipfile import ZipFile

from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import replay_dev
from replay_dev import DEV_SPLITS, headline_differences, stage_dev


def digest(data):
    return hashlib.sha256(data).hexdigest()


def test_only_locked_images_are_staged_and_labels_rebuilt(tmp_path, monkeypatch):
    monkeypatch.setitem(DEV_SPLITS, 'source_dev', ('train', 1))
    raw = tmp_path / 'raw'
    (raw / 'annotations').mkdir(parents=True)
    (raw / 'train').mkdir()
    img = b'selected image payload'
    (raw / 'train/selected.jpg').write_bytes(img)
    image = {'height': 20, 'width': 20}
    annotation = {
        'images': [dict(image, id=7, file_name='selected.jpg'), dict(image, id=8, file_name='held.jpg')],
        'categories': [{'id': 1, 'name': 'Pedestrian'}],
        'annotations': [{'image_id': 8, 'id': 3, 'category_id': 1, 'bbox': [-9, -9, 1, 1]},
                        {'image_id': 7, 'id': 2, 'category_id': 1, 'bbox': [10, 0, 4, 4]},
                        {'image_id': 7, 'id': 1, 'category_id': 1, 'bbox': [-2, 0, 6, 4]}],
    }
    payload = json.dumps(annotation).encode()
    (raw / 'annotations/instance_train.json').write_bytes(payload)
    expected = (b'0 0.100000000000 0.100000000000 0.200000000000 0.200000000000\n'
                b'0 0.600000000000 0.100000000000 0.200000000000 0.200000000000\n')
    sample = digest(b'selected.jpg')
    split = {'official_split': 'train', 'annotations_sha256': digest(payload),
             'samples': {sample: [digest(img), digest(expected), 3]}}
    mapping = [{'official_id': 1, 'official_name': 'Pedestrian', 'training_id': 0}]
    dest = tmp_path / 'staged'
    contract, yaml = stage_dev(raw, dest, 'source_dev', split, mapping)
    row, = contract['records']
    assert row['key'] == sample and row['group_id'] == '0003'
    assert (dest / row['label']).read_bytes() == expected
    assert [x.name for x in (dest / 'images/source_dev').iterdir()] == ['selected.jpg']
    config = json.loads(yaml.read_text(encoding='utf-8'))
    assert config['val'] == 'images/source_dev'


def test_headline_uses_report_precision_and_displays_count_differences():
    expected = {'day': {'macro_pp': -0.639}, 'night_300': {'car': {'fp': 6}}}
    actual = {'day': {'macro_pp': -0.641}, 'night_300': {'car': {'fp': 7}}}
    assert headline_differences(actual, expected) == [
        {'field': '/night_300/car/fp', 'actual': 7, 'expected': 6},
    ]



@pytest.fixture
def development_bundle(monkeypatch):
    monkeypatch.setattr(replay_dev, 'DEV_SPLITS', {'source_dev': ('train', 1), 'night_dev': ('val', 1)})
    image = BytesIO()
    Image.new('RGB', (20, 16), 'white').save(image, format='JPEG')
    payloads, splits = {}, {}
    for group, purpose in enumerate(replay_dev.DEV_SPLITS):
        name = f'{purpose}.jpg'
        labels = b'0 0.5 0.5 0.2 0.2\n'
        payloads[f'images/{purpose}/{name}'] = image.getvalue()
        payloads[f'labels/{purpose}/{purpose}.txt'] = labels
        splits[purpose] = {'samples': {digest(name.encode()): [digest(image.getvalue()), digest(labels), group]}}
    monkeypatch.setattr(replay_dev, 'DEV_ARCHIVE_BYTES', sum(map(len, payloads.values())))
    return payloads, splits


def make_zip(tmp_path, payloads):
    path = tmp_path / 'development.zip'
    with ZipFile(path, 'w') as archive:
        for name, data in payloads.items():
            archive.writestr(name, data)
    return path


def test_archive_stages_only_the_development_splits(tmp_path, development_bundle):
    payloads, splits = development_bundle
    with ZipFile(make_zip(tmp_path, payloads)) as archive:
        staged = replay_dev.stage_archive(archive, tmp_path/'output', splits)
    assert set(staged) == {'source_dev', 'night_dev'}
    for purpose, (contract, config) in staged.items():
        row, = contract['records']
        assert (row['width'], row['height']) == (20, 16)
        assert (contract['root']/row['label']).read_bytes() == payloads[row['label']]
        assert json.loads(config.read_text(encoding='utf-8'))['val'] == f'images/{purpose}'


def test_archive_rejects_out_of_scope_files_before_copying(tmp_path, development_bundle):
    payloads, splits = development_bundle
    key = 'labels/night_dev/night_dev.txt'
    payloads['labels/night_test/night_dev.txt'] = payloads.pop(key)
    with ZipFile(make_zip(tmp_path, payloads)) as archive:
        with pytest.raises(replay_dev.ContractError, match='Missing night_dev label member'):
            replay_dev.stage_archive(archive, tmp_path/'output', splits)
    assert not (tmp_path/'output').exists()


def test_archive_rejects_changed_labels_even_with_same_byte_count(tmp_path, development_bundle):
    payloads, splits = development_bundle
    key = 'labels/source_dev/source_dev.txt'
    payloads[key] = payloads[key].replace(b'0.5', b'0.4', 1)
    with ZipFile(make_zip(tmp_path, payloads)) as archive:
        with pytest.raises(replay_dev.ContractError, match='source_dev label SHA-256'):
            replay_dev.stage_archive(archive, tmp_path/'output', splits)
