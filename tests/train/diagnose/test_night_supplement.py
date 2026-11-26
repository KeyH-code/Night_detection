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


def test_340_changes_only_fold_paired_night_inputs_and_keeps_every_shared_exposure_aligned():
    from collections import defaultdict
    from night_paired import schedules
    from night_supplement import fold
    base=[f'base-{i%300}' for i in range(2000)]
    old=[((f'day-{i}',bool(i%3),i/2000),(k,bool(i%2),1-i/5000)) for i,k in enumerate(base)]
    groups={f'new-{i}':f'g-{i}' for i in range(2000)}; buckets=defaultdict(list)
    for k,g in groups.items():buckets[fold(g)].append(k)
    t=[k for v in buckets.values() for k in v[:8]]
    r=t[:4]+[k for v in buckets.values() for k in v[8:16-len(set(v[:8])&set(t[:4]))]]
    plans=schedules(old,{'targeted':t,'random':r},groups); counts=Counter();shared=set(base)|set(t)&set(r)
    for (d,n),(a,x),(b,y) in zip(old,plans['targeted'],plans['random'],strict=True):
        assert d==a==b and n[1:]==x[1:]==y[1:]
        if x[0] in shared or y[0] in shared:assert x[0]==y[0]
        if x[0]!=n[0]:
            counts[x[0]]+=1;assert x[0] in t and y[0] in r and fold(groups[x[0]])==fold(groups[y[0]])
    assert counts==dict.fromkeys(t,6)
