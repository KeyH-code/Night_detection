from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))


def test_localization_only_reweights_the_two_night_tail_targets():
    from night_joint import localization_scores

    q = torch.ones(2, 4, 6)
    got = localization_scores(q, 1.5)
    assert torch.equal(got[0], q[0])
    assert torch.equal(got[1, :, :3], q[1, :, :3])
    assert torch.equal(got[1, :, 5], q[1, :, 5])
    assert torch.equal(got[1, :, 3:5], q[1, :, 3:5] * 1.5)
    assert q.eq(1.).all()


def test_confidence_quality_uses_aligned_iou_and_chooses_disjoint_hard_negatives():
    from night_joint import JointLoss

    z = torch.zeros(2, 6, 6, requires_grad=True)
    q = torch.zeros_like(z)
    q[0, 0, 3], q[1, 0, 3], q[1, 1, 3] = 1., .7, .8
    same = torch.zeros(6, 2)
    same[5, 0] = .4  # ambiguous same-class overlap, never treated as negative
    other = torch.zeros_like(same)
    other[2] = .8
    other[3] = .2
    aligned = torch.tensor([.80,.25,0,0,0,0], requires_grad=True)
    loss = JointLoss(1.5, .5, .1, 1)
    extra, counts, negatives = loss.class_terms(z, q, aligned, same, other)
    assert counts.tolist() == [2, 0]
    assert negatives[0].tolist() == [1, 1, 1]
    torch.testing.assert_close(extra[0, 0], .5 * q[1, 0, 3] * F.softplus(torch.tensor(0.)))
    assert extra[5, 0] == 0
    dz, dq = torch.autograd.grad(extra.sum(), (z, aligned), allow_unused=True)
    assert dq is None
    assert dz[1, 0, 3] < 0 and dz[1, 1, 3] > 0 and dz[1, 2:5, 3].gt(0).all()
    assert not dz[0].any() and not dz[1, :, :3].any()


def test_location_only_never_replaces_assignment_or_classification():
    from night_joint import JointLoss
    from ultralytics.utils.loss import BboxLoss

    anchor = torch.tensor([[8., 8.]])
    dist = torch.zeros((2, 1, 64))
    box = torch.tensor([[[5., 5., 9., 9.]], [[5., 5., 9., 9.]]])
    gt = torch.tensor([[[5., 5., 10., 9.]], [[5., 5., 10., 9.]]])
    target = torch.zeros(2, 1, 6)
    target[1, 0, 3] = 1.
    mask = torch.tensor([[False], [True]])
    params = (dist, box, anchor, gt, target, target.sum(), mask, torch.tensor([64., 64.]), torch.ones(1, 1))
    original = BboxLoss(16)
    assigned, bce = object(), torch.nn.BCEWithLogitsLoss(reduction='none')
    loss = SimpleNamespace(assigner=assigned, bce=bce, bbox_loss=original)
    objective = JointLoss(1.5, 0, 0, 8)
    objective.attach(loss)
    assert loss.assigner is assigned and loss.bce is bce
    a = original(*params)
    b = loss.bbox_loss(*params)
    torch.testing.assert_close(torch.stack(b), torch.stack(a) * 1.5)
    assert objective.summary()['positive_anchors'] == [1, 0]
    assert objective.summary()['negative_buckets'] == [[0, 0, 0], [0, 0, 0]]


def test_negative_only_preserves_regression_and_skips_positive_quality():
    from night_joint import JointLoss

    bbox = object()
    native = torch.nn.BCEWithLogitsLoss(reduction='none')
    loss = SimpleNamespace(assigner=object(), bce=native, bbox_loss=bbox)
    objective = JointLoss(1, 0, .1, 1)
    objective.attach(loss)
    assert loss.bbox_loss is bbox
    z = torch.zeros((2, 3, 6), requires_grad=True)
    q = torch.zeros_like(z)
    q[1, 0, 3] = .75
    same = torch.zeros(3, 2)
    other = torch.tensor([[0, 0], [.8, .2], [.15, .25]])
    terms, _, counts = objective.class_terms(z, q, None, same, other)
    assert terms[0, 0] == 0
    assert counts[0].tolist() == [1, 1, 0]
    grad, = torch.autograd.grad(terms.sum(), z)
    assert not grad[0].any() and not grad[1, :, :3].any() and grad[1, 0, 3] == 0
