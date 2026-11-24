from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))


@pytest.mark.parametrize('dtype', ['float32', 'float16'])
def test_only_night_tails_in_registered_width_band_get_beta_eight(dtype):
    torch = pytest.importorskip('torch')
    pytest.importorskip('ultralytics')
    from ultralytics.utils.tal import TaskAlignedAssigner
    from night_tal import TailAssigner

    native = TaskAlignedAssigner(topk=2, num_classes=6, alpha=.5)
    labels = torch.tensor([[3, 4, 3, 4, 2, 3]] * 2).unsqueeze(-1)
    boxes = torch.tensor([[0., 0., w, 20.] for w in [8, 9, 64, 65, 48, 32]]).repeat(2, 1, 1)
    predictions = torch.tensor([[[0., 0., 13., 20.], [0., 0., 35., 19.]]] * 2)
    scores = torch.full((2, 2, 6), .5, dtype=getattr(torch, dtype))
    candidates = torch.ones(2, 6, 2, dtype=torch.bool)
    candidates[1, 5] = False  # Padding or no eligible center.
    n = TailAssigner(native)
    n.bs = native.bs = 2
    n.n_max_boxes = native.n_max_boxes = 6
    original, ciou = native.get_box_metrics(scores, predictions, labels, boxes, candidates)
    actual, adjusted_ciou = n.get_box_metrics(scores, predictions, labels, boxes, candidates)
    assert torch.equal(adjusted_ciou, ciou)
    expected = original.clone()
    expected[1, 1:3] = scores[1, :, 0].pow(.5) * ciou[1, 1:3].pow(8.)
    assert torch.equal(actual, expected)
    assert not torch.equal(actual[1, 1:3], original[1, 1:3])


def test_native_assignments_are_preserved_at_beta_six_and_beta_eight_can_change_night_anchor():
    torch = pytest.importorskip('torch')
    pytest.importorskip('ultralytics')
    from ultralytics.utils.tal import TaskAlignedAssigner
    from night_tal import TailAssigner

    native = TaskAlignedAssigner(topk=1, num_classes=6, alpha=.5)
    # Anchor 0 has better CIoU, anchor 1 compensates with its classification score at beta=6 only.
    pred = torch.tensor([[[0., 0., 20., 20.], [1., 0., 19., 20.]]] * 2)
    score = torch.ones(2, 2, 6)
    score[:, 0, 3] = .23
    labels = torch.full((2, 1, 1), 3.)
    gt = pred[:, :1].clone()
    args = (score, pred, torch.tensor([[5., 5.], [15., 15.]]), labels, gt, torch.ones(2, 1, 1))
    original = native(*args)
    reference = TailAssigner(native, beta=6.)(*args)
    treatment = TailAssigner(native)(*args)
    assert all(torch.equal(a, b) for a, b in zip(original, reference, strict=True))
    assert all(torch.equal(a[0], b[0]) for a, b in zip(original, treatment, strict=True))
    assert original[3][1].tolist() == [False, True]
    assert treatment[3][1].tolist() == [True, False]


@pytest.mark.parametrize(('field', 'value', 'reason'), [
    ('base_positive_gt', 11, 'support'), ('changed_gt', 1, 'anchor_change'),
    ('candidate_mean_ciou', .49, 'ciou'), ('candidate_soft_gt', 11, 'soft_gt'),
    ('align_underflow', 1, 'underflow'), ('native_soft_underflow', 1, 'underflow'),
])
def test_any_failed_signal_for_either_class_blocks_training(field, value, reason):
    pytest.importorskip('ultralytics')
    from night_tal import decision

    one = dict(base_positive_gt=12, changed_gt=2, base_mean_ciou=.5, candidate_mean_ciou=.5,
               base_soft_gt=12, candidate_soft_gt=12, align_underflow=0, native_soft_underflow=0)
    stats = {'truck': one, 'tram': one.copy()}
    assert decision(stats) == {'truck': [], 'tram': []}
    stats['tram'][field] = value
    assert decision(stats) == {'truck': [], 'tram': [reason]}


@pytest.mark.parametrize(('cls', 'width', 'admitted'), [(4, 12, True), (3, 12, False), (4, 8, False), (4, 65, False)])
def test_ring_changes_only_allowed_night_candidates_and_preserves_raw_target(cls, width, admitted):
    torch = pytest.importorskip('torch')
    pytest.importorskip('ultralytics')
    from ultralytics.utils.tal import TaskAlignedAssigner
    from night_tal import RingAssigner

    stock = TaskAlignedAssigner(topk=10, num_classes=6, alpha=.5)
    # Outside the padded native region (width 16 for width 12): x=25 fits,
    # x=27 cannot attain IoU .75, the next x=25 has poor predicted CIoU, x=28 is beyond the ring.
    left, right = 16-width/2, 16+width/2
    gt = torch.tensor([[[left, 10., right, 26.]]] * 2)
    anchors = torch.tensor([[16., 18.], [25., 18.], [27., 18.], [25., 22.], [28., 18.]] + [[120., 120.]]*5)
    pred = torch.tensor([[[left, 10., right, 26.], [left, 10., 25., 26.], [left, 10., 27., 26.],
                          [24., 22., 28., 26.], [left, 10., 28., 26.]] + [[119., 119., 121., 121.]]*5] * 2)
    scores = torch.full((2, 10, 6), .7)
    args = scores, pred, anchors, torch.full((2, 1, 1), cls), gt, torch.ones((2, 1, 1))
    native = stock(*args)
    replay = RingAssigner(stock, enabled=False)(*args)
    assert all(torch.equal(a,b) for a,b in zip(native,replay,strict=True))
    snapshots = []
    result = RingAssigner(stock, observe=lambda *x: snapshots.append(x))(*args)
    assert all(torch.equal(a[0], b[0]) for a, b in zip(native, result, strict=True))
    original, extended, *_ = snapshots[0]
    assert (extended & ~original).nonzero().tolist() == ([[1, 0, 1]] if admitted else [])
    if admitted:
        assert result[3][1, 1] and result[2][1, 1, 4] > 0
        assert torch.equal(result[1][1, 1], gt[1, 0])


@pytest.mark.parametrize(('edit', 'reason'), [
    ({'base_positive_gt': 11}, 'support'), ({'ring_gt': 1}, 'ring_signal'),
    ({'candidate_mean_ciou': .599}, 'ciou'), ({'candidate_soft_gt': [3, 4, 4, 6, 12]}, 'soft_gt'),
])
def test_ring_signal_checks_surviving_supervision_and_all_five_classes(edit, reason):
    pytest.importorskip('ultralytics')
    from night_tal import ring_decision

    stats = dict(base_positive_gt=12, ring_gt=2, base_mean_ciou=.6, candidate_mean_ciou=.6,
                 base_soft_gt=[3, 4, 5, 6, 12], candidate_soft_gt=[3, 4, 5, 6, 12])
    assert ring_decision(stats) == []
    assert ring_decision({**stats, **edit}) == [reason]
