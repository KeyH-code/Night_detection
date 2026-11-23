from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))


def test_equal_tail_quotas_replay_e1_order_and_unequal_quotas_change_only_capacity():
    from night_strategy import rank_e1
    from night_tail import rank_quota

    rng = np.random.default_rng(43)
    keys = [f'x{i:03d}' for i in range(13)]
    z = rng.normal(size=(13, 4)); z /= np.linalg.norm(z, axis=1)[:, None]
    u = rng.random((13, 5))
    base = rank_e1(keys, z, u, keys[:1], shortlist=12, quota=2)
    assert rank_quota(keys, z, u, keys[:1], 12, [2]*5)['keys'] == base['keys']
    custom = rank_quota(keys, z, u, keys[:1], 12, [1, 1, 2, 3, 3])
    assert custom['per_class'] == [1, 1, 2, 3, 3]
    assert custom['keys'][0] == keys[0] and len(set(custom['keys'])) == 11


def test_tail_quota_does_not_substitute_other_class_if_positive_shortlist_is_empty():
    from night_tail import rank_quota
    from source_contract import ContractError

    keys = list('abcdefg')
    u = np.full((7, 5), .5)
    u[:, 4] = 0
    with pytest.raises(ContractError, match='class 4'):
        rank_quota(keys, np.eye(7), u, ['a'], 6, [1]*5)


def test_blind_rank_can_open_sealed_npz_without_widening_file_guard(tmp_path):
    from night_tail import ROOT

    keys = [f'x{i:03d}' for i in range(84)]
    rng = np.random.default_rng(43)
    scores = rng.random((84, 5))
    np.savez_compressed(tmp_path/'features.npz', keys=keys, vectors=np.eye(84))
    np.savez_compressed(tmp_path/'scores.npz', keys=keys, scores=scores)
    (tmp_path/'metadata.json').write_text(json.dumps({'keys': keys, 'anchors': keys[:1],
                'quota': [16]*5, 'shortlist': 83}), encoding='utf-8')
    command = [sys.executable, '-B', str(ROOT/'scripts/night_tail.py'), 'worker', str(tmp_path)]
    result = subprocess.run(command, capture_output=True, check=False, timeout=30)
    assert result.returncode == 0, result.stderr.decode('utf-8')
    assert json.loads(result.stdout)['reads'] == ['features.npz', 'scores.npz', 'metadata.json']
