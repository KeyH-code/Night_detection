from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))


def test_width_score_rejects_large_stable_and_wrong_class_proposals():
    from night_width import width_score

    a=[[10,0,40,40,.5,3],[50,0,130,40,.9,4],[150,0,190,40,.6,4]]
    b=[[10,0,40,40,.8,3],[50,0,130,40,.9,4],[150,0,190,40,.6,2]]
    assert width_score(a,[b],1.,[12,64]).tolist()==pytest.approx([0,0,0,1,0])


def test_width_selection_never_reuses_a_discarded_parent_or_risk_group():
    from night_width import rank_width

    keys=[f'img{i}' for i in range(7)]
    z=np.eye(7)
    u=np.ones((7,5))*.1
    u[2,3]=u[3,3]=u[4,3]=u[5,4]=.9
    u[6,4]=.8
    groups=dict(zip(keys,['a','b','c','b','d','d','e'],strict=True))
    pick=rank_width(keys,z,u,groups,keys[:3],2,1,7)
    assert pick['keys']==['img0','img1','img4','img6']
