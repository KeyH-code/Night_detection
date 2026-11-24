from collections import Counter
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))


def test_300_plan_preserves_shared_slots_and_200_updates_at_balanced_exposure():
    from night_expand import expanded_plan, align_slots

    day=[((f'd{i}',False,1.),('discard',bool(i%2),1.1)) for i in range(2000)]
    d=[f'r{i}' for i in range(300)]
    e=d[:20]+[f'e{i}' for i in range(200)]+d[220:]
    slots=align_slots(d,e)
    assert set(slots)==set(e)
    assert all(slots[i]==k for i,k in enumerate(d) if k in e)
    a=expanded_plan(day,d,43,20261122); b=expanded_plan(day,slots,43,20261122)
    assert len(b)==2000 and set(Counter(n[0] for _,n in b).values())=={6,7}
    assert all(x[0]==y[0] and x[1][1:]==y[1][1:] for x,y in zip(a,b,strict=True))
    assert [n for _,n in a if n[0] in e]==[n for _,n in b if n[0] in d]


def test_expanded_random_prefix_is_the_same_rng_not_a_redraw():
    import numpy as np
    from night_expand import random_prefix
    from source_contract import ContractError

    k=[f'k{i:04d}' for i in range(994)]
    old=[k[i] for i in np.random.Generator(np.random.PCG64(43)).permutation(len(k))[:100]]
    assert random_prefix(k[::-1],old,43)[:100]==old
    with pytest.raises(ContractError):
        random_prefix(k,old[::-1],43)


def test_gap_change_uses_shared_group_draws_on_all_four_models():
    from night_expand import effects
    from night_accept import CLASSES
    from detection_metrics import aggregate_point

    # One truth for each class in each group. A is identical to B and C to D,
    # so signed changes must cancel at every bootstrap draw, not just on average.
    def arm(present):
        rows=[]
        for i in range(2):
            c=list(range(5))
            hit=present if i==0 else not present
            rows.append({'key':f'x{i}','purpose':'night_dev','group_id':f'g{i}',
                         'gt':{'cls':c},'pred':{'cls':c if hit else [],'conf':[.9]*5 if hit else []},
                         'iou_edges':[[j,j,1.] for j in c] if hit else []})
        return rows,{n:aggregate_point(rows,i,.5) for i,n in enumerate(CLASSES)}

    policy={'interval':{'repeats':32,'seed':43}}
    result=effects([arm(True),arm(True),arm(False),arm(False)],[1,-1,-1,1],policy)
    assert result['effects']['macro']['delta']==0
    assert result['effects']['macro']['ci95']==[0.,0.]


def test_mix_slots_prioritize_balanced_when_a_random_slot_conflicts():
    from night_expand import mix_slots

    random=[f'd{i}' for i in range(300)]
    balanced=random[:20]+['b20','b21']+random[22:]
    proposed=random[:20]+['b20','d20','d21']+random[23:]
    slots,receipt=mix_slots(random,balanced,proposed)
    assert slots[20]=='b20' and slots[21]=='d21' and slots[22]=='d20'
    assert receipt=={'balanced_shared': 298, 'random_shared':299, 'random_slot_changes':['d20']}
    assert slots==mix_slots(random,balanced,proposed)[0]


def test_mix_coverage_weights_groups_and_uses_fixed_reference_low_groups():
    import numpy as np
    from night_expand import coverage_stats

    keys=['a1','a2','b','c','d']
    z=np.array([[1.,0.],[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]])
    groups=['a','a','b','c','d']
    base,extra=coverage_stats(keys,z,groups,['a1'],['a1','c'],['c'])
    assert base['group_mean']==.5 and extra['group_mean']==.75
    assert base['fixed_low_quartile_mean']==0 and extra['fixed_low_quartile_mean']==1


def test_mix_ambiguous_or_wrong_representative_prefix_is_rejected():
    from night_expand import check_mix_prefix
    from source_contract import ContractError

    def row(seq,n):
        return {'steps':[{'kind':'representative' if i<n else 'diverse','key':key} for i,key in enumerate(seq)]}
    arms={2:row(['a','b','c','d'],2),3:row(['a','b','d','c'],3),1:row(['a','c','d','b'],1)}
    check_mix_prefix(arms,4)
    arms[1]=row(['b','a','d','c'],1)
    with pytest.raises(ContractError): check_mix_prefix(arms,4)
