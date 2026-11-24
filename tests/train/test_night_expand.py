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
