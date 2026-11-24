from pathlib import Path
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))


def test_coupled_vehicle_loss_preserves_other_locations_and_pushes_rivals_down():
    from night_margin import VehicleMargin

    z = torch.zeros(2, 4, 6, requires_grad=True)
    q = torch.zeros_like(z)
    q[0, 0, 3], q[1, 1, 3], q[1, 2, 0], q[1, 3, 2] = .5, .75, .8, .6
    loss = VehicleMargin(.5, .75)
    got = loss(z, q)
    native = F.binary_cross_entropy_with_logits(z, q, reduction='none')
    extra = got - native
    assert torch.nonzero(extra).tolist() == [[1, 1, 3], [1, 3, 2]]
    torch.testing.assert_close(extra[1, 1, 3], .5 * q[1, 1, 3] * F.softplus(torch.tensor(.75)))
    grad, = torch.autograd.grad(extra.sum(), z)
    assert grad[1, 1, 3] < 0 and grad[1, 1, 2] > 0 and grad[1, 1, 4] > 0
    assert grad[1, 1, 0] == 0 and not grad[0].any()
    assert loss.summary()['positive_anchors'] == [1, 1, 0]


def test_empty_vehicle_supervision_is_exact_native_bce():
    from night_margin import VehicleMargin

    z = torch.zeros(2, 4, 6)
    q = torch.zeros_like(z)
    loss = VehicleMargin(.5, .75)
    assert torch.equal(loss(z, q), F.binary_cross_entropy_with_logits(z, q, reduction='none'))
    assert loss.summary()['positive_anchors'] == [0, 0, 0]
