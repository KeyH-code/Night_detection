from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))


def test_pair_loss_cancels_common_shift_and_requires_both_score_directions():
    torch=pytest.importorskip('torch')
    from night_frontier import pair_loss

    p=torch.tensor([1.,2.],requires_grad=True); n=torch.tensor([2.,4.],requires_grad=True)
    value=pair_loss(p,n)
    gp,gn=torch.autograd.grad(value,(p,n))
    assert torch.equal(gp,-gn) and bool((gp<0).all())
    assert torch.equal(value,pair_loss(p+3,n+3))


def test_history_accepts_only_cross_group_native_inversions_with_both_orders():
    torch=pytest.importorskip('torch')
    from night_frontier import Box, Ledger

    f=torch.ones(1)
    p=Box(0,0,1.,'tp','a',('x',0),f)
    n=Box(0,1,2.,'classification','a',None,f)
    not_inverted=Box(0,2,.5,'classification','b',None,f)
    vs=Ledger(); assert vs.push([p,n])==[]
    assert vs.push([not_inverted])==[]
    b=Box(0,1,3.,'classification_localization','b',None,f)
    now=Box(0,0,1.5,'tp','b',('y',0),f)
    pairs=vs.push([b,now])
    assert len(pairs)==2 and {(x.gt,y.kind) for x,y,_ in pairs}=={(('x',0),'classification_localization'),(('y',0),'classification')}
    for _ in range(32):vs.push([])
    assert vs.push([p])==[]


@pytest.mark.parametrize(('gt','groups','wrong','mixed','want'),[(12,8,2,2,[]),(11,8,2,2,['tp_support']),
    (12,7,2,2,['fp_groups']),(12,8,1,2,['classification']),(12,8,2,1,['classification_localization'])])
def test_budget_gate_requires_each_registered_support(gt,groups,wrong,mixed,want):
    pytest.importorskip('torch')
    from night_frontier import failures

    assert failures({'positive_gt':gt,'negative_groups':groups,
                     'groups_by_bucket':{'classification':wrong,'classification_localization':mixed}})==want
