from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))


def test_zero_residual_leaves_every_score_unchanged_and_only_truck_can_move():
    torch=pytest.importorskip('torch')
    from night_truck import score_residual

    logits=torch.randn(2,6,3)
    feats=torch.tensor([[[1.,2.,3.]],[[3.,2.,1.]]])
    delta=torch.tensor([0.,0.],requires_grad=True)
    initial=score_residual(logits,feats,delta)
    assert torch.equal(initial,logits)
    g,=torch.autograd.grad(initial[1,3,0],delta)
    assert g.tolist()==[3.,1.]
    changed=score_residual(logits,feats,torch.tensor([1.,.5]))
    assert torch.equal(changed[:,[0,1,2,4,5]],logits[:,[0,1,2,4,5]])
    assert torch.allclose(changed[:,3]-logits[:,3],feats[:,0]+.5)


@pytest.mark.parametrize(('active','near','bad'),[(12,6,[]),(11,6,['active_support']),(12,5,['near_support'])])
def test_training_gate_rejects_without_near_and_total_support(active,near,bad):
    pytest.importorskip('torch')
    from night_truck import failures

    assert failures({'active_gt':active,'near_gt':near})==bad


def test_truck_teacher_selects_only_matched_same_anchor_not_other_geometry():
    torch=pytest.importorskip('torch'); pytest.importorskip('ultralytics')
    from night_truck import select_pairs

    labels=torch.tensor([[1.,1.,11.,11.],[25.,25.,35.,35.]])
    coords=torch.tensor([[6.,6.,10.,10.],[6.,6.,10.,10.],[30.,30.,10.,10.]]).T
    logits=torch.full((6,3),-12.); logits[3]=torch.tensor([3.,2.,2.])
    teacher=torch.cat((coords,logits.sigmoid()))[None]
    current=teacher.clone(); current[0,0,2]=50.
    anchors,gt,z=select_pairs(teacher,logits,current,labels)
    assert anchors.tolist()==[0] and gt.tolist()==[0] and z.tolist()==[3.]


def test_fold_proof_rejects_any_changed_non_truck_row_or_backbone_value():
    torch=pytest.importorskip('torch')
    from night_expand import truck_changes
    from source_contract import ContractError

    keys=[f'head.{i}' for i in range(6)]
    original={k:torch.zeros(6,1) for k in keys}|{'backbone':torch.zeros(1)}
    modified={k:t.clone() for k,t in original.items()}; modified[keys[2]][3]=1
    assert truck_changes(original,modified,keys)==1
    modified[keys[2]][4]=2
    with pytest.raises(ContractError): truck_changes(original,modified,keys)
