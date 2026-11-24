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


def test_full_history_retains_older_inversion_but_not_same_group_or_future():
    torch=pytest.importorskip('torch')
    from night_frontier import Box, Ledger

    x=torch.ones(1)
    earlier=Box(0,0,2.,'classification','a',None,x)
    same=Box(0,1,1.,'tp','a',('a',1),x)
    other=Box(0,2,1.,'tp','b',('b',2),x)
    local,full=Ledger(),Ledger(history_limit=None)
    for ledger in (local,full):
        assert ledger.push([earlier])==[]
        for _ in range(33): assert ledger.push([])==[]
        assert ledger.push([same])==[]
    assert local.push([other])==[]
    assert full.push([other])==[(other,earlier,True)]


def test_full_history_keeps_distinct_exposures_without_double_pair_directions():
    torch=pytest.importorskip('torch')
    from night_frontier import Box, Ledger

    x=torch.ones(1)
    ledger=Ledger(history_limit=None)
    fp=Box(0,0,2.,'classification','a',None,x)
    tp=Box(0,1,1.,'tp','b',('b',1),x)
    assert ledger.push([fp])==[]
    assert ledger.push([fp])==[]
    assert len(ledger.push([tp]))==2
    assert ledger.push([])==[]
    assert len(ledger.push([tp]))==2


@pytest.mark.parametrize(('gt','groups','wrong','mixed','want'),[(12,8,2,2,[]),(11,8,2,2,['tp_support']),
    (12,7,2,2,['fp_groups']),(12,8,1,2,['classification']),(12,8,2,1,['classification_localization'])])
def test_budget_gate_requires_each_registered_support(gt,groups,wrong,mixed,want):
    pytest.importorskip('torch')
    from night_frontier import failures

    assert failures({'positive_gt':gt,'negative_groups':groups,
                     'groups_by_bucket':{'classification':wrong,'classification_localization':mixed}})==want


def test_frontier_fold_changes_only_three_truck_weight_rows_and_never_bias():
    torch=pytest.importorskip('torch')
    from night_expand import truck_changes
    from source_contract import ContractError

    names=[f'head.{i}.weight' for i in range(3)]
    state={k:torch.zeros(6,2) for k in names}|{'head.0.bias':torch.zeros(6)}
    updated={k:v.clone() for k,v in state.items()}; updated[names[0]][3,1]=.1
    assert truck_changes(state,updated,names,bias=False)==1
    updated['head.0.bias'][3]=.1
    with pytest.raises(ContractError):truck_changes(state,updated,names,bias=False)
