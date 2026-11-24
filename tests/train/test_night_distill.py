from pathlib import Path
import sys

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from source_contract import ContractError


def test_source_teacher_selects_stable_class_top_k_and_excludes_class_five():
    from night_distill import teacher_mask

    z = torch.full((4, 6), -2.)
    z[:3, 0] = torch.tensor([0.5, .5, .8])
    z[0, 4] = 1.
    z[:, 5] = 2.
    mask = teacher_mask(z, .3, 2)
    assert torch.nonzero(mask).tolist() == [[0, 0], [0, 4], [2, 0]]


def test_distillation_gradient_uses_only_selected_day_logits_and_never_teacher():
    from night_distill import replay_terms

    target = torch.zeros(3, 6, requires_grad=True)
    selected = torch.zeros_like(target, dtype=torch.bool)
    selected[0, 3], selected[1, 4] = True, True
    baseline = torch.zeros(2, 3, 6, requires_grad=True)
    equal = replay_terms(baseline, target, selected, .25)
    assert not torch.autograd.grad(equal.sum(), baseline)[0].any()
    logits = torch.ones_like(baseline, requires_grad=True)
    delta = replay_terms(logits, target, selected, .25)
    grad, teacher_grad = torch.autograd.grad(delta.sum(), (logits,target), allow_unused=True)
    assert teacher_grad is None
    assert torch.nonzero(grad).tolist() == [[0, 0, 3], [0, 1, 4]]
    assert grad[0,0,3] == grad[0,1,4] == .25


def test_teacher_cannot_use_a_batch_that_contains_night_images():
    from night_distill import DayTeacher

    teacher = DayTeacher(None, threshold=.3, per_class=2, weight=.25)
    with pytest.raises(ContractError, match='one white-day image'):
        teacher.capture(torch.zeros(2, 3, 64, 64))
