import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))


def test_reference_requires_five_distinct_groups_not_five_high_scoring_boxes():
    from night_supplement import references
    from source_contract import ContractError
    rows = [('a', 'g0', 2, i, .91) for i in range(25)]
    rows += [(f'p{i}', f'g{i}', 2, 1, .6) for i in range(1,5)]
    assert len(references(rows, 2)) == 5
    with pytest.raises(ContractError): references(rows[:-1], 2)


def test_random_control_matches_folds_without_excluding_treatment_groups():
    from night_supplement import random_control, fold
    records = {f'k{i}': {'group_id': f'g{i}'} for i in range(220)}
    selected = list(records)[:40]
    first = random_control(records, selected)
    assert len(set(first)) == 40
    assert Counter(fold(records[k]['group_id']) for k in first) == Counter(fold(records[k]['group_id']) for k in selected)
    assert random_control(dict(reversed(list(records.items()))), first) == first
