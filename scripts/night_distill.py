"""Frozen-A same-anchor score replay on the white-day training image only."""
from __future__ import annotations

import math
import time

from source_contract import ContractError, NAMES, check_equal


def teacher_mask(logits, threshold, per_class):
    import torch

    if logits.ndim != 2 or logits.shape[1] != 6 or per_class < 1 or not 0 < threshold < 1:
        raise ContractError('Expected source teacher logits in registered six-class order')
    scores = logits.detach()[:, :5].float()
    indices = torch.argsort(scores, dim=0, descending=True, stable=True)[:per_class]
    selected = scores.gather(0, indices) >= math.log(threshold / (1-threshold))
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask[:, :5].scatter_(0, indices, selected)
    return mask


def replay_terms(student, teacher, selected, weight):
    import torch

    if student.ndim != 3 or student.shape[0] != 2 or student.shape[1:] != teacher.shape or selected.shape != teacher.shape:
        raise ContractError('Expected same-anchor [day, night] student logits and one source teacher')
    residual = (student[0].float() - teacher.detach().float()).masked_fill(~selected, 0.)
    return torch.stack((.5 * weight * residual.square(), torch.zeros_like(residual)))


def layout(predictions):
    shapes = tuple(tuple(f.shape[-2:]) for f in predictions['feats'])
    check_equal('native source replay feature geometry', ((80, 80), (40, 40), (20, 20)), shapes)
    check_equal('native source replay logits align to feature geometry', (6, 8400), tuple(predictions['scores'].shape[1:]))
    return shapes


class DayTeacher:
    def __init__(self, model, *, threshold, per_class, weight):
        self.model = model
        self.threshold = threshold
        self.per_class = per_class
        self.weight = weight
        self.calls = 0
        self.anchor_counts = self.image_counts = self.loss_sum = None

    def capture(self, day):
        import torch

        if day.ndim != 4 or day.shape[0] != 1:
            raise ContractError('Teacher may consume exactly one white-day image')
        with torch.no_grad():
            _, native = self.model(day)
            shapes = layout(native)
            logits = native['scores'][0].transpose(0, 1).float()
            return logits, teacher_mask(logits, self.threshold, self.per_class), shapes

    def attach(self, criterion):
        import torch

        original = criterion.get_assigned_targets_and_loss
        native_bce = criterion.bce
        pending = None

        def call_bce(student, target):
            if pending is None:
                raise ContractError('Native BCE called without matched source teacher')
            teacher, mask = pending
            extra = replay_terms(student, teacher, mask, self.weight)
            with torch.no_grad():
                anchors = mask.sum(dim=0)[:5]
                images = (anchors > 0).to(anchors.dtype)
                totals = extra[0].detach().sum(dim=0)[:5]
                if self.anchor_counts is None:
                    self.anchor_counts, self.image_counts, self.loss_sum = anchors.clone(), images.clone(), totals.clone()
                else:
                    self.anchor_counts.add_(anchors)
                    self.image_counts.add_(images)
                    self.loss_sum.add_(totals)
                self.calls += 1
            return native_bce(student, target) + extra

        def with_teacher(predictions, batch):
            nonlocal pending
            check_equal('source then night student batch', 2, batch['img'].shape[0])
            check_equal('no reentrant source teacher loss', None, pending)
            z, picked, shapes = self.capture(batch['img'][:1])
            check_equal('teacher and student share the same native anchor layout', shapes, layout(predictions))
            pending = z, picked
            try:
                return original(predictions, batch)
            finally:
                pending = None

        criterion.bce = call_bce
        criterion.get_assigned_targets_and_loss = with_teacher

    def summary(self):
        return {'calls': self.calls, 'class_anchors': self.anchor_counts.tolist(),
                'class_images': self.image_counts.tolist(), 'extra_loss_sum': self.loss_sum.tolist()}


def probe(torch, cv2, teacher, payload, plan, stride, until):
    import night_train as train

    slots = plan[::stride]
    keys = [src[0] for src, _ in slots]
    check_equal('fixed source-only preflight slots', (40, 40), (len(keys), len(set(keys))))
    anchor_counts = torch.zeros(5, dtype=torch.long, device='cuda')
    image_counts = torch.zeros_like(anchor_counts)
    tested_gradient = False
    start = time.monotonic()
    for src, _ in slots:
        if time.monotonic() >= until:
            raise ContractError('Local source signal probe expired')
        day = train._batch(torch, cv2, payload, [src], 640)['img']
        with torch.autocast('cuda', dtype=torch.float16):
            logits, mask, shapes = teacher.capture(day)
        check_equal('finite frozen source logits', True, bool(logits.isfinite().all()))
        if not tested_gradient and mask.any():
            # The student's head is training-mode; the same head can be checked
            # without updating its BatchNorm buffers or teacher parameters.
            head = teacher.model.model[-1]
            head.training = True
            try:
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
                    check_equal('teacher matches training-mode native head', shapes, layout(teacher.model(day)))
            finally:
                head.training = False
            equal = torch.stack((logits, logits)).requires_grad_()
            zero, = torch.autograd.grad(replay_terms(equal, logits, mask, teacher.weight).sum(), equal)
            check_equal('exact zero gradient at identical source scores', True, not zero.any())
            shifted = (equal.detach() + .125).requires_grad_()
            grad, = torch.autograd.grad(replay_terms(shifted, logits, mask, teacher.weight).sum(), shifted)
            check_equal('only selected day anchors receive positive restoring gradient', (True, True, True),
                        (not grad[1].any(), not grad[0][~mask].any(), bool((grad[0][mask] > 0).all())))
            tested_gradient = True
        count = mask.sum(dim=0)[:5]
        anchor_counts.add_(count)
        image_counts.add_(count > 0)
    receipt = {'source_keys': keys, 'class_anchors': anchor_counts.tolist(), 'class_images': image_counts.tolist(),
               'restoring_gradient_verified': tested_gradient, 'teacher_frozen': not teacher.model.training and not any(
                   p.requires_grad or p.grad is not None for p in teacher.model.parameters()), 'seconds': time.monotonic()-start}
    check_probe(receipt, plan, stride)
    print('Source-only preflight passed:', dict(zip(NAMES[:5], receipt['class_anchors'], strict=True)), flush=True)
    return receipt


def check_probe(receipt, plan, stride):
    check_equal('expected source-only probes and isolated teacher/gradient', ([d[0] for d, _ in plan[::stride]], True, True),
                tuple(receipt[k] for k in ('source_keys', 'restoring_gradient_verified', 'teacher_frozen')))
    for name, minimum in (('class_anchors', 8), ('class_images', 1)):
        counts = receipt[name]
        if len(counts) != 5 or min(counts) < minimum:
            raise ContractError(f'Insufficient source training signal: {name}={counts}')


def check_receipt(receipt):
    check_equal('exactly one source teacher call per historical microbatch', 2000, receipt['calls'])
    anchors, images, losses = (receipt[k] for k in ('class_anchors', 'class_images', 'extra_loss_sum'))
    if (len(anchors) != 5 or len(images) != 5 or len(losses) != 5 or
            not all(0 < b <= 2000 and b <= a <= 32*b for a, b in zip(anchors, images, strict=True)) or
            not all(math.isfinite(v) and v > 0 for v in losses)):
        raise ContractError('Missing or invalid white-day distillation signal')
