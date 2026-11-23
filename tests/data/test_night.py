"""The S5 consumers must never need management, pool-label, or held-out files."""
from pathlib import Path
import os
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from source_contract import ContractError, NAMES, sha256, write_json


def fixture(tmp_path):
    root = tmp_path / '冻结'
    image = root / 'images/night_dev/甲.jpg'
    label = root / 'labels/night_dev/甲.txt'
    for path, content in ((image, b'dev image fixture'), (label, b'')):
        path.parent.mkdir(parents=True)
        path.write_bytes(content)
    record = {'key': 'val:1', 'purpose': 'night_dev', 'group_id': 'g1',
              'image': image.relative_to(root).as_posix(), 'label': label.relative_to(root).as_posix(),
              'image_sha256': sha256(image), 'label_sha256': sha256(label), 'width': 20, 'height': 10}
    return root, record


def seal(root, records):
    rel = 'manifests/night_dev.json'
    write_json(root / rel, {'purpose': 'night_dev', 'records': records})
    write_json(root / 'READY.json', {'schema_version': 1, 'version': 'night_v1', 'status': 'frozen',
                                    'categories': {'names': NAMES}, 'counts': {'night_dev': len(records)},
                                    'held_out': {'night_dev': {'path': rel, 'sha256': sha256(root / rel),
                                                               'count': len(records)},
                                                 'night_test': {'path': 'never-read'},
                                                 'night_pool': {'path': 'never-read'}}})
    return sha256(root / 'READY.json')


def test_night_dev_uses_only_its_receipt_images_and_labels(tmp_path):
    from night_contract import verify_dev
    root, row = fixture(tmp_path)
    result = verify_dev(root, seal(root, [row]), count=1)
    assert result['records'] == [row]
    assert result['reads'] == ['READY.json', 'manifests/night_dev.json', row['image'], row['label']]


@pytest.mark.parametrize('change', [
    {'purpose': 'night_test'}, {'purpose': 'source_train'}, {'label': 'hidden_ground_truth/night_pool/a.txt'},
    {'image': 'images/night_dev/../night_test/a.jpg'}, {'image': '/images/night_dev/a.jpg'},
    {'image': 'C:/images/night_dev/a.jpg'}, {'image': 'images/night_dev//甲.jpg'},
])
def test_invalid_manifest_rejected_before_any_payload_read(tmp_path, change):
    from night_contract import verify_dev
    root, row = fixture(tmp_path)
    (root / row['image']).unlink()
    second = dict(row, key='val:2', image='images/night_dev/乙.jpg', label='labels/night_dev/乙.txt')
    second.update(change)
    with pytest.raises(ContractError, match='purpose|path'):
        verify_dev(root, seal(root, [row, second]), count=2)


def test_duplicate_key_rejected_before_payload_read(tmp_path):
    from night_contract import verify_dev
    root, row = fixture(tmp_path)
    (root / row['image']).unlink()
    with pytest.raises(ContractError, match='Duplicate'):
        verify_dev(root, seal(root, [row, row]), count=2)


def test_night_dev_payload_must_match_sealed_hash(tmp_path):
    from night_contract import verify_dev
    root, row = fixture(tmp_path)
    sealed = seal(root, [row])
    (root / row['label']).write_bytes(b'0 0.5 0.5 1 1\n')
    with pytest.raises(ContractError, match='SHA'):
        verify_dev(root, sealed, count=1)


def selection_fixture(tmp_path):
    root = tmp_path / 'pool'
    image = root / 'images/night_pool/乙.jpg'
    image.parent.mkdir(parents=True)
    image.write_bytes(b'candidate fixture')
    row = {'key': 'val:2', 'group_id': 'g2', 'image': image.relative_to(root).as_posix(),
           'image_sha256': sha256(image)}
    return root, row


def select_seal(root, rows):
    path = root / 'selection/night_pool_images.json'
    write_json(path, {'purpose': 'night_pool_selection_images_only', 'records': rows})
    return sha256(path)


def test_selection_sandbox_has_only_hashed_images_and_narrow_manifest(tmp_path):
    from night_contract import stage_selection
    root, row = selection_fixture(tmp_path)
    label = root / 'hidden_ground_truth/night_pool/乙.txt'
    label.parent.mkdir(parents=True)
    label.write_bytes(b'must not copy')
    dest = tmp_path / 'external' / 'sandbox'
    dest.parent.mkdir()
    result = stage_selection(root, select_seal(root, [row]), dest, count=1)
    paths = sorted(p.relative_to(dest).as_posix() for p in dest.rglob('*') if p.is_file())
    assert paths == [row['image'], 'selection/night_pool_images.json']
    assert sha256(dest / row['image']) == row['image_sha256']
    assert result['reads'] == ['selection/night_pool_images.json', row['image']]
    assert result['root'] == dest


@pytest.mark.parametrize('change', [{'label': 'do not read'}, {'image': 'images/night_test/乙.jpg'}])
def test_selection_rejects_annotations_and_wrong_purpose_before_staging(tmp_path, change):
    from night_contract import stage_selection
    root, row = selection_fixture(tmp_path)
    dest = tmp_path / 'sandbox'
    with pytest.raises(ContractError, match='fields|path'):
        stage_selection(root, select_seal(root, [dict(row, **change)]), dest, count=1)
    assert not dest.exists()


def test_selection_rejects_link_even_if_target_is_an_allowed_image(tmp_path):
    from night_contract import stage_selection
    root, row = selection_fixture(tmp_path)
    image = root / row['image']
    link = image.with_name('link.jpg')
    try:
        link.symlink_to(image)
    except OSError as exc:
        if os.name != 'nt':
            pytest.skip(f'Cannot create a symbolic link: {exc}')
        # A Windows directory junction requires no symlink privilege. Replace the
        # approved image directory with a junction to the same original image.
        real = image.parent.with_name('ordinary')
        image.parent.rename(real)
        def quote(path):
            return "'" + str(path).replace("'", "''") + "'"
        subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command',
                        f'New-Item -ItemType Junction -Path {quote(image.parent)} -Target {quote(real)}'],
                       check=True, capture_output=True)
        assert image.parent.is_junction()
    else:
        row['image'] = link.relative_to(root).as_posix()
    with pytest.raises(ContractError, match='link|junction|reparse'):
        stage_selection(root, select_seal(root, [row]), tmp_path / 'sandbox', count=1)
