import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from source_contract import ContractError, read_json, sha256, write_json
from source_release import publish_ready


def fixture(tmp_path, status='passed', wrong_hash=False):
    archive = tmp_path/'kit.tar.gz'
    archive.write_bytes(b'fixture archive bytes, not real data')
    sidecar = tmp_path/'kit.tar.gz.json'
    qa = tmp_path/'qa.json'
    write_json(sidecar, {'status': 'PACKAGED_PENDING_VALIDATION', 'sha256': sha256(archive),
                        'ready_sha256': 'known frozen READY'})
    write_json(qa, {'status': status, 'archive_sha256': '0'*64 if wrong_hash else sha256(archive),
                   'ready_sha256': 'known frozen READY', 'final_tests_run': False})
    return archive, sidecar, qa


def test_failed_relocation_never_publishes_ready(tmp_path):
    archive, sidecar, qa = fixture(tmp_path, status='failed')
    before = sidecar.read_bytes()
    with pytest.raises(ContractError, match='must pass'):
        publish_ready(archive, sidecar, qa)
    assert sidecar.read_bytes() == before
    assert read_json(sidecar)['status'] == 'PACKAGED_PENDING_VALIDATION'


def test_wrong_archive_hash_never_publishes_ready(tmp_path):
    archive, sidecar, qa = fixture(tmp_path, wrong_hash=True)
    before = sidecar.read_bytes()
    with pytest.raises(ContractError, match='SHA'):
        publish_ready(archive, sidecar, qa)
    assert sidecar.read_bytes() == before


def test_success_atomically_publishes_actual_archive_hash(tmp_path):
    archive, sidecar, qa = fixture(tmp_path)
    result = publish_ready(archive, sidecar, qa)
    assert result['status'] == 'READY_FOR_CLOUD'
    assert result['sha256'] == sha256(archive) == read_json(sidecar)['sha256']
    assert not sidecar.with_suffix('.json.part').exists()
