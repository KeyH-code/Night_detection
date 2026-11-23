"""Lock the blind random arm's budget, replay, and actual read boundary."""
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from source_contract import ContractError, sha256, write_json

POLICY = {'schema_version': 1, 'pool_count': 12, 'budget': 5, 'seeds': [17, 29, 43],
          'algorithm': 'lexicographic_key_pcg64_permutation_prefix'}


def image_sandbox(tmp_path):
    images = tmp_path / 'images/night_pool'
    images.mkdir(parents=True)
    records = []
    for i in range(POLICY['pool_count']):
        p = images / f'img-{i}.jpg'
        p.write_bytes(f'synthetic image {i}'.encode())
        records.append({'key': f'val:{i}', 'group_id': f'g{i // 2}',
                        'image': p.relative_to(tmp_path).as_posix(), 'image_sha256': sha256(p)})
    path = tmp_path / 'selection/night_pool_images.json'
    write_json(path, {'purpose': 'night_pool_selection_images_only', 'records': list(reversed(records))})
    return path, records


def test_random_selection_is_sorted_uniform_without_replacement_and_replayable(tmp_path):
    from night_select import run_worker
    path, records = image_sandbox(tmp_path)
    result = run_worker(tmp_path, sha256(path), POLICY)
    assert result == run_worker(tmp_path, sha256(path), POLICY)
    assert result['reads'] == ['selection/night_pool_images.json']
    ordered = sorted(r['key'] for r in records)
    for seed in POLICY['seeds']:
        arm = result['arms'][str(seed)]
        assert len(arm['records']) == len({r['key'] for r in arm['records']}) == POLICY['budget']
        expected = np.random.Generator(np.random.PCG64(seed)).permutation(len(records))[:POLICY['budget']]
        assert [r['key'] for r in arm['records']] == [ordered[i] for i in expected]
        assert all(set(r) == {'image', 'image_sha256', 'key', 'group_id'} for r in arm['records'])


@pytest.mark.parametrize('forbidden', ['hidden_ground_truth', 'labels', 'images/night_test', 'READY.json'])
def test_worker_rejects_unexpected_sandbox_content(tmp_path, forbidden):
    from night_select import run_worker
    path, _ = image_sandbox(tmp_path)
    (tmp_path / forbidden).mkdir()
    with pytest.raises(ContractError, match='sandbox inventory'):
        run_worker(tmp_path, sha256(path), POLICY)


def test_read_guard_blocks_even_existing_files_outside_the_approved_manifest(tmp_path):
    path, _ = image_sandbox(tmp_path)
    other = tmp_path / 'synthetic-hidden-label.txt'
    other.write_text('not allowed', encoding='utf-8')
    snippet = '''import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from night_select import guard_reads
from source_contract import ContractError
approved, forbidden = map(Path, sys.argv[2:])
with approved.open('rb') as f:
    expected=f.read()
reads = guard_reads(approved)
assert approved.read_bytes() == expected
try: forbidden.read_text(encoding='utf-8')
except ContractError: pass
else: raise AssertionError('Read guard did not block an existing off-list file')
assert reads == [approved]
'''
    done = subprocess.run([sys.executable, '-B', '-c', snippet, str(SCRIPTS), str(path), str(other)],
                          capture_output=True, text=True, encoding='utf-8')
    assert done.returncode == 0, done.stderr


def test_worker_rejects_tampered_manifest_before_draw(tmp_path):
    from night_select import run_worker
    path, records = image_sandbox(tmp_path)
    seal = sha256(path)
    write_json(path, {'purpose': 'night_pool_selection_images_only', 'records': records})
    with pytest.raises(ContractError, match='SHA-256'):
        run_worker(tmp_path, seal, POLICY)
