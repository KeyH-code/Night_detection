"""Add an ordinal vehicle-class objective to night examples only."""
from __future__ import annotations

from source_contract import ContractError


class VehicleMargin:
    def __init__(self, weight, margin):
        self.weight = weight
        self.margin = margin
        self.counts = None
        self.extra = None
        self.calls = 0

    def __call__(self, logits, target):
        import torch
        import torch.nn.functional as f

        if (logits.ndim != 3 or logits.shape != target.shape or
                (logits.shape[0], logits.shape[-1]) != (2, 6)):
            raise ContractError('Expected [day, night] logits in the native six-class order')
        native = f.binary_cross_entropy_with_logits(logits, target, reduction='none')
        z = logits[1, :, 2:5].float()
        q = target[1, :, 2:5].float()
        rivals = ~torch.eye(3, dtype=torch.bool, device=z.device)
        terms = f.softplus(self.margin + z[:, None, :] - z[:, :, None])
        added = self.weight * q * (terms * rivals).sum(dim=-1) / 2
        native[1, :, 2:5] = native[1, :, 2:5] + added.to(native.dtype)
        with torch.no_grad():
            if self.counts is None:
                self.counts = torch.zeros(3, dtype=torch.long, device=z.device)
                self.extra = torch.zeros(3, dtype=torch.float64, device=z.device)
            self.counts.add_((q > 0).sum(dim=0))
            self.extra.add_(added.sum(dim=0))
            self.calls += 1
        return native

    def summary(self):
        if self.counts is None:
            raise ContractError('Vehicle margin has no training calls')
        return {'positive_anchors': self.counts.cpu().tolist(),
                'extra_unnormalized': self.extra.cpu().tolist(), 'calls': self.calls}
