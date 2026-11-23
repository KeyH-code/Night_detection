from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))


def test_rank_samples_keep_matched_positives_and_exclude_uncertain_negatives():
    from night_rank import rank_indices

    iou = torch.tensor([[.7, .55, .4, .1, 0.], [0., 0., 0., .8, .7]])
    logits = torch.tensor([-2., -3., 5., 3., 4.])
    pos, neg = rank_indices(iou[:1], logits, positives=2, negatives=2)
    assert pos.tolist() == [0, 1]
    assert neg.tolist() == [4, 3]


def test_rank_objective_raises_positives_and_penalizes_highest_negatives():
    from night_rank import pair_objective

    positive = torch.tensor([-2., 1.], requires_grad=True)
    negative = torch.tensor([2., -1.], requires_grad=True)
    dp, dn = torch.autograd.grad(pair_objective(positive, negative, 1), (positive, negative))
    assert (dp < 0).all()
    assert dn[0] > 0 and dn[1] == 0


def test_checkpoint_guard_rejects_changes_outside_two_registered_rows():
    from night_rank import assert_head_delta
    from source_contract import ContractError

    before = {'head.bias': torch.zeros(6), 'backbone': torch.zeros(2)}
    after = {k: v.clone() for k, v in before.items()}
    after['head.bias'][3:5] = torch.tensor([.5, -.5])
    assert assert_head_delta(before, after, {'head.bias'}) == [1, 1]
    after['head.bias'][2] = 1
    with pytest.raises(ContractError, match='protected'):
        assert_head_delta(before, after, {'head.bias'})


def test_training_feature_precision_is_scoped_even_if_capture_fails():
    from night_rank import score_capture_precision

    before = torch.backends.cudnn.conv.fp32_precision
    with pytest.raises(ValueError, match='capture error'):
        with score_capture_precision(torch):
            assert torch.backends.cudnn.conv.fp32_precision == 'ieee'
            raise ValueError('capture error')
    assert torch.backends.cudnn.conv.fp32_precision == before
