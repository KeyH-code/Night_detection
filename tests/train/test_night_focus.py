from pathlib import Path
import sys

import torch
import torch.nn.functional as f

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))


def test_focus_weight_has_only_registered_night_class_and_anchor_effects():
    from night_focus import focused_bce

    scores=torch.zeros(2,5,6)
    targets=torch.zeros_like(scores)
    scores[1,1,3], scores[1,2,3], scores[1,3,4] = 3., 2., 4.
    targets[1,0,3] = .9
    native = f.binary_cross_entropy_with_logits(scores,targets,reduction='none')
    delta = focused_bce(scores,targets,2.,4.,1) - native
    assert {tuple(x) for x in torch.nonzero(delta).tolist()} == {(1,0,3),(1,1,3),(1,3,4)}
    assert (delta>=0).all()


def test_focus_zero_intervention_matches_native_bce_exactly():
    from night_focus import focused_bce

    logits = torch.randn(2,8,6)
    soft = torch.rand_like(logits)
    assert torch.equal(focused_bce(logits,soft,1.,0.,0),f.binary_cross_entropy_with_logits(logits,soft,reduction='none'))


def test_focus_does_not_backpropagate_through_negative_weight():
    from night_focus import focused_bce

    logits = torch.full((2,2,6),2.,requires_grad=True)
    target = torch.zeros_like(logits)
    target[1,0,3] = 1
    loss = focused_bce(logits,target,2.,4.,2)
    grad, = torch.autograd.grad(loss.sum(),logits)
    positive = logits[1,0,3].sigmoid().item()
    torch.testing.assert_close(grad[1,0,3],torch.tensor(2*(positive-1)))
    torch.testing.assert_close(grad[1,1,3],torch.tensor((1+4*positive**2)*positive))
