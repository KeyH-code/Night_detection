"""Development replay contract tests without CUDA or private data."""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
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
