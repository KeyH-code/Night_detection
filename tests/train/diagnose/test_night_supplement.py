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


def test_one_to_one_duplicate_cannot_fill_material_gap_and_other_class_tie_follows_gt_order():
    from night_material import boxes_with_truth, actual_role
    # Two tram boxes on the same car/truck overlap; one also overlaps a real tram.
    # Two truck boxes share the same truck: the duplicate is not a boundary negative.
    row={'gt':{'cls':[2,3,4]},'pred':{'cls':[4,4,3,3],'conf':[.5,.6,.8,.7]},
         'anchors':[10,11,12,13], 'iou_edges':[(0,0,.55),(1,0,.55),(0,1,.8),(2,1,.35),(1,2,.9),(1,3,.81)]}
    cut={'truck_fp':[.5]*5,'truck_low':[.85]*5,'tram_low':[.85]*5}
    assert {p['anchor']:actual_role(p,cut,0) for p in boxes_with_truth(row)}=={10:'tram_car',11:None,12:'truck_low',13:None}
