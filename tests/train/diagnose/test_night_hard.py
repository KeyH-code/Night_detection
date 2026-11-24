from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
from source_contract import ContractError


def test_rank_uses_only_approved_positive_proxies_and_fresh_groups():
    from night_hard import rank

    keys = list('abcdefg')
    p = np.full((7, 5), .5)
    w = p.copy()
    u = p.copy()
    w[1, 3] = 1.  # Same group as the fixed anchor, always excluded.
    u[2, 3] = 0.  # Cannot consume a hard slot even with the strongest support.
    w[2, 3] = 1.
    w[3, 3] = .6
    got = rank(keys, np.eye(7), p, w, u, dict(zip(keys, 'aabcdef')), ['a'],
               [('hard', 3), ('support', 3), ('general', 0)])
    assert got['keys'] == ['a', 'd', 'c', 'e']
    assert [v['kind'] for v in got['steps']] == ['hard', 'support', 'general']


def test_rank_refuses_to_backfill_when_no_positive_eligible_signal_exists():
    from night_hard import rank

    x = np.ones((2, 5))
    with pytest.raises(ContractError, match='quota'):
        rank(['a', 'b'], np.eye(2), x, x, x * 0, {'a': '1', 'b': '2'}, ['a'], [('hard', 4)])


def test_group_support_does_not_turn_repeated_positive_images_into_new_evidence():
    from night_hard import support

    rows = [{'group_id': 'repeat', 'width': 1920, 'height': 1080},
            {'group_id': 'repeat', 'width': 1920, 'height': 1080},
            {'group_id': 'other', 'width': 1920, 'height': 1080}]
    def label(c, w=.05):
        return (c, .5, .5, w, .1)
    a = support(rows, [[label(3), label(2)], [label(3), label(4, .2)], [label(4), label(2)]])
    assert a['truck'] == {'instances': 2, 'images': 2, 'groups': 1,
                           'medium_groups': 1, 'vehicle_cooccurrence_groups': 1,
                           'other_vehicle_negative_groups': 1}
    assert a['tram']['groups'] == 2 and a['tram']['medium_groups'] == 1
