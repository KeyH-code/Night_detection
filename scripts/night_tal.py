"""Fixed two-domain TAL contrast: only eligible night GT use a different beta."""
from __future__ import annotations

from collections import Counter, defaultdict
from statistics import fmean

import torch
from ultralytics.utils.tal import TaskAlignedAssigner

from source_contract import ContractError, NAMES, check_equal

BALANCED_SHA = '71c2942af5343795066793488d661b68e4e6d5701027fd0f6f19c8e298722d35'
INDICES = tuple(range(0, 2000, 10))


def target_gt(gt_labels, gt_bboxes):
    """Rows of the native [source, night] batch; padded rows have zero width."""
    check_equal('TAL requires one source and one night image in order', 2, gt_bboxes.shape[0])
    cls = gt_labels[..., 0]
    width = gt_bboxes[..., 2] - gt_bboxes[..., 0]
    target = ((cls == 3) | (cls == 4)) & (width > 8) & (width <= 64)
    target[0] = False
    return target


class TailAssigner(TaskAlignedAssigner):
    def __init__(self, native, *, beta=8., observe=None):
        check_equal('registered native horizontal TAL', (TaskAlignedAssigner, 6, .5, 6.),
                    (type(native), native.num_classes, native.alpha, native.beta))
        if beta not in (6., 8.):
            raise ContractError('Only the registered TAL exponents are permitted')
        super().__init__(topk=native.topk, num_classes=native.num_classes, alpha=native.alpha,
                         beta=native.beta, stride=native.stride, eps=native.eps, topk2=native.topk2)
        self.tail_beta = beta
        self.observe = observe

    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        # Refuse native per-image OOM fallback: its positional domain index is no longer valid.
        check_equal('TAL requires intact source/night batches', 2, pd_scores.shape[0])
        return super()._forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        align, overlaps = super().get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt)
        if self.tail_beta == 8. or self.observe is not None:
            target = target_gt(gt_labels, gt_bboxes)
            if self.tail_beta == 8.:
                b, g, a = (target.unsqueeze(-1) & mask_gt.bool()).nonzero(as_tuple=True)
                cls = gt_labels[b, g, 0].long()
                align[b, g, a] = pd_scores[b, a, cls].pow(self.alpha) * overlaps[b, g, a].pow(self.tail_beta)
            if self.observe is not None:
                # Native normalization mutates these tensors after selecting anchors.
                self.observe(align.clone(), overlaps.clone(), target)
        return align, overlaps


def decision(classes):
    check_equal('registered gate classes', {'truck', 'tram'}, set(classes))
    failures = {n: [] for n in classes}
    for n, s in classes.items():
        gates = {
            'support': s['base_positive_gt'] >= 12,
            'anchor_change': s['changed_gt'] * 10 >= s['base_positive_gt'],
            'ciou': s['base_mean_ciou'] is not None and s['candidate_mean_ciou'] >= s['base_mean_ciou'],
            'soft_gt': s['candidate_soft_gt'] >= s['base_soft_gt'],
            'underflow': s['align_underflow'] == s['native_soft_underflow'] == 0,
        }
        failures[n] = [k for k, passed in gates.items() if not passed]
    return failures


class Probe:
    """Compare native, beta=6 replay and beta=8 using identical native assignment inputs."""
    def __init__(self, native):
        self.native = native
        self.rows = {c: defaultdict(list) for c in (3, 4)}
        self.counts, self.indirect = Counter(), Counter()
        self.dtypes = set()
        self.key = None

    def __call__(self, *args):
        check_equal('probe only accepts a scheduled night key', True, bool(self.key))
        scores, _, _, labels, boxes, valid = args
        snapshots = []
        def capture(*tensors):
            snapshots.append(tensors)
        plain = self.native(*args)
        base = TailAssigner(self.native, beta=6., observe=capture)(*args)
        candidate = TailAssigner(self.native, observe=capture)(*args)
        check_equal('beta six replays every native output', True,
                    all(torch.equal(a, b) for a, b in zip(plain, base, strict=True)))
        check_equal('full day assignment exactly preserved', True,
                    all(torch.equal(a[0], b[0]) for a, b in zip(plain, candidate, strict=True)))
        check_equal('no per-image fallback', False, self.native._oom_warned)
        self.counts['batches'] += 1
        if not boxes.shape[1]:
            check_equal('empty assignments never call box metrics', 0, len(snapshots))
            return plain
        (old_score, overlaps, target), (new_score, new_overlaps, new_target) = snapshots
        check_equal('exact same CIoU and eligible GT', True,
                    torch.equal(overlaps, new_overlaps) and torch.equal(target, new_target))
        check_equal('day and non-target direct alignment exactly preserved', True,
                    torch.equal(old_score[~target], new_score[~target]))
        check_equal('valid direct alignment', True,
                    bool(torch.isfinite(old_score).all() & torch.isfinite(new_score).all()))
        self.dtypes.add((str(scores.dtype), str(boxes.dtype), str(old_score.dtype), str(base[2].dtype)))
        self.counts.update(batches_with_metrics=1, day_direct_values=old_score[0].numel(),
                           non_target_night_direct_values=old_score[1].numel() - int(target[1].sum()) * old_score.shape[-1])

        def assigned(out):
            idx = out[3][1].nonzero().flatten()
            owner = out[4][1, idx]
            soft = out[2][1, idx].amax(-1)
            ciou = overlaps[1, owner, idx]
            rows = defaultdict(dict)
            for g, a, quality, raw, cast in zip(owner.tolist(), idx.tolist(), ciou.tolist(),
                                                soft.tolist(), soft.to(scores.dtype).tolist(), strict=True):
                rows[g][a] = (quality, raw, cast)
            return rows

        base_rows, test_rows = assigned(base), assigned(candidate)
        underflow = ((old_score[1] > 0) & (new_score[1] == 0)).sum(-1).tolist()
        night_labels = labels[1, :, 0].tolist()
        night_target = target[1].tolist()
        for i in valid[1, :, 0].nonzero().flatten().tolist():
            b, c = base_rows[i], test_rows[i]
            if not night_target[i]:
                self.indirect['night_non_target_gt_exposures'] += 1
                self.indirect['anchor_set_changed'] += b.keys() != c.keys()
                self.indirect['soft_or_anchor_changed'] += b != c
                continue
            common = b.keys() & c.keys()
            self.rows[int(night_labels[i])][(self.key, i)].append({
                'base_positive': bool(b), 'changed': b.keys() != c.keys(),
                'base_ciou': fmean(r[0] for r in b.values()) if b else 0.,
                'candidate_ciou': fmean(r[0] for r in c.values()) if c else 0.,
                'base_soft': any(r[2] > 0 for r in b.values()),
                'candidate_soft': any(r[2] > 0 for r in c.values()),
                'align_underflow': underflow[i],
                'native_soft_underflow': sum(b[a][1] > 0 and c[a][1] == 0 for a in common),
                'loss_cast_zeroed': sum(b[a][2] > 0 and c[a][1] > 0 and c[a][2] == 0 for a in common),
            })
        return plain

    def summary(self):
        classes = {}
        for c, gt in self.rows.items():
            supported = [[r for r in views if r['base_positive']] for views in gt.values()]
            supported = [views for views in supported if views]
            raw = [r for views in gt.values() for r in views]
            classes[NAMES[c]] = {
                'eligible_gt': len(gt), 'eligible_gt_exposures': len(raw), 'base_positive_gt': len(supported),
                'base_positive_gt_exposures': sum(map(len, supported)),
                'changed_gt': sum(any(r['changed'] for r in views) for views in supported),
                'changed_gt_exposures': sum(r['changed'] for views in supported for r in views),
                **{k: fmean(fmean(r[k] for r in views) for views in supported) if supported else None
                   for k in ('base_ciou', 'candidate_ciou')},
                **{f'{k}_soft_gt': sum(any(r[f'{k}_soft'] for r in views) for views in gt.values())
                   for k in ('base', 'candidate')},
                **{k: sum(r[k] for r in raw) for k in ('align_underflow', 'native_soft_underflow', 'loss_cast_zeroed')},
            }
            s = classes[NAMES[c]]
            s['base_mean_ciou'] = s.pop('base_ciou')
            s['candidate_mean_ciou'] = s.pop('candidate_ciou')
        return {'classes': classes, 'checks': dict(self.counts), 'indirect': dict(self.indirect),
                'dtype_columns': ['class_input', 'box_input', 'direct_alignment', 'native_soft_target'],
                'dtypes': sorted(map(list, self.dtypes))}


def preflight(scratch, out):
    import inspect
    import os
    from pathlib import Path, PurePosixPath
    from importlib.metadata import version
    import random
    import shutil
    import tempfile
    import time
    from types import SimpleNamespace

    import cv2
    import numpy as np
    from ultralytics.nn.tasks import load_checkpoint
    from ultralytics.utils.loss import v8DetectionLoss

    import night_expand as expand
    import night_train as train
    from night_contract import ROOT, read_sealed_json
    from seed_eval import probe_gpu
    from source_contract import canonical_hash, sha256, write_json

    start = time.monotonic()
    gpu = probe_gpu()
    parent = ROOT/'runs/expand_003'
    seal = read_sealed_json(parent, 'manifest.json', BALANCED_SHA, [])
    state = expand.adapt._json(parent, seal, 'run.json')
    check_equal('sealed untreated balanced recipe', ('sealed', 'complete', 'e2', 43, None, None, None, False, False),
                (seal['status'], state['status'], state['arm'], state['seed'], state.get('margin'), state.get('crop'),
                 state.get('e2_mix'), state['cloud_used'], state['final_test_executed']))
    choice = expand.adapt._json(parent, seal, 'selection.json')
    inputs = expand.adapt._json(parent, seal, '43/inputs.json')
    actual = expand.adapt._json(parent, seal, '43/train.json')
    digest = state['config_sha256']; _, cfg, pair, old = expand.configuration(digest)
    _, identity, pool, inv, _, d300, baseplan, _ = expand.context(cfg, pair, old, 'e2')
    slots = expand.align_slots(d300, choice['keys'])
    plan = expand.expanded_plan(baseplan, slots, 43, old['schedule_salt'])
    plan_sha = canonical_hash(plan)
    check_equal('precisely the original augmented plan and observed domain order',
                (slots, plan_sha, plan_sha, [list((d[0], n[0])) for d, n in plan], [d[0] for d, _ in plan]),
                (choice['slots'], choice['schedule_sha256'], actual['augment_schedule_sha256'], actual['microbatches'], inputs['source_keys']))
    check_equal('unchanged frozen initializer and training code', (identity['A_sha256'], seal['files']['code/night_train.py']['sha256']),
                (sha256(train.CHECKPOINT), sha256(ROOT/'scripts/night_train.py')))
    versions = {n: version(n) for n in train.LIBS}
    check_equal('exact balanced package versions', ('8.4.171', state['versions']), (versions['ultralytics'], versions))
    native_sha = {Path(path).name: sha256(Path(path))
                  for path in map(inspect.getfile, (TaskAlignedAssigner, v8DetectionLoss))}
    check_equal('original day manifest', state['source_train'], identity['source_train'])
    entry = identity['source_train']
    doc = read_sealed_json(train.SOURCE, entry['path'], entry['sha256'], [])
    check_equal('source training metadata only', ('source_train', 4000), (doc['purpose'], len(doc['records'])))
    day_rows = {r['key']: r for r in doc['records']}
    nights = [inv[k] for k in slots]
    check_equal('approved 300 with no label read during selection', (False, pool, nights, canonical_hash(nights), slots),
                (choice['labels_read_in_selection'], choice['pool_sha256'], choice['records'], choice['records_sha256'],
                 [r['key'] for r in inputs['records']]))
    night_rows = {}
    for metadata, item, image, label in zip(nights, inputs['records'], inputs['source_reads'][::2], inputs['source_reads'][1::2], strict=True):
        expected_label = f"hidden_ground_truth/night_pool/{PurePosixPath(metadata['image']).with_suffix('.txt').name}"
        check_equal('exact sealed night payload binding', (metadata['group_id'], metadata['image_sha256'], 'night_pool', metadata['image'], expected_label),
                    (item['group_id'], item['image_sha256'], item['purpose'], image, label))
        night_rows[item['key']] = dict(item, image=image, label=label)

    selected = [plan[i] for i in INDICES]
    source = [day_rows[d[0]] for d, _ in selected]
    night = [night_rows[k] for k in dict.fromkeys(n[0] for _, n in selected)]
    for r in source:
        check_equal('day source purpose', 'source_train', r['purpose'])
        if not (r['image'].startswith('images/source_train/') and r['label'].startswith('labels/source_train/')):
            raise ContractError('Out of scope day payload')
    with tempfile.TemporaryDirectory(dir=scratch, prefix='tal-') as td:
        box = Path(td)
        payload, reads = {}, {}
        for purpose, root, rows in (('source_train', train.SOURCE, source), ('night_pool', train.NIGHT, night)):
            staged, paths = train._stage(root, box/purpose, rows)
            check_equal('only sampled image and label payloads', [r[k] for r in rows for k in ('image', 'label')], paths)
            payload.update(train.training_rows(box/purpose, staged, purpose))
            reads[purpose] = {'images': len(staged), 'files': len(paths), 'paths_sha256': canonical_hash(paths)}
        torch.set_num_threads(4); cv2.setNumThreads(1)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.manual_seed(43); random.seed(43); np.random.seed(43)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        model, _ = load_checkpoint(str(train.CHECKPOINT), device='cuda')
        model.train().requires_grad_(False)
        model.args = SimpleNamespace(**dict(model.args))
        check_equal('frozen class order', NAMES, list(model.names.values()))
        for layer in model.modules():
            if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
                layer.eval()
        frozen = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        criterion = v8DetectionLoss(model)
        probe = Probe(criterion.assigner)
        criterion.assigner = probe
        processed = []
        with torch.no_grad(), train.telemetry() as monitor:
            for step, micro in zip(INDICES, selected, strict=True):
                if time.monotonic() - start >= 600:
                    raise ContractError('Fixed TAL probe exceeded ten minutes')
                batch = train._batch(torch, cv2, payload, micro, old['size'])
                probe.key = micro[1][0]
                with torch.autocast('cuda', dtype=torch.float16):
                    _, loss, _ = criterion.get_assigned_targets_and_loss(model(batch['img']), batch)
                check_equal('finite native loss without gradients', (True, False), (bool(loss.isfinite().all()), loss.requires_grad))
                processed.append(step)
                if len(processed) % 50 == 0:
                    print(f'TAL no-update probe: {len(processed)}/{len(INDICES)} microbatches', flush=True)
        check_equal('exact fixed microbatches', (list(INDICES), len(INDICES)), (processed, probe.counts['batches']))
        check_equal('frozen model state untouched', True,
                    all(torch.equal(v.detach().cpu(), frozen[k]) for k, v in model.state_dict().items()))
        stats = probe.summary()
        failures = decision(stats['classes'])
        receipt = {**stats, 'failures': failures, 'met': not any(failures.values()), 'updates': 0, 'weights_unchanged': True,
                   'balanced_sha256': BALANCED_SHA, 'expand_sha256': digest, 'A_sha256': state['a_sha256'],
                   'plan_sha256': plan_sha, 'sample_plan_sha256': canonical_hash(selected), 'indices': processed,
                   'reads': reads, 'development_read': False, 'holdout_read': False, 'cloud_used': False,
                   'versions': versions, 'native_code_sha256': native_sha,
                   'initial_gpu': gpu, 'gpu': monitor, 'seconds': time.monotonic()-start}
        evidence = box/'evidence'; evidence.mkdir()
        shutil.copyfile(__file__, evidence/'night_tal.py')
        write_json(evidence/'metrics.json', receipt)
        write_json(evidence/'manifest.json', {'status': 'sealed', 'files': train._receipt(evidence)})
        evidence.rename(out)
    return verify(out, sha256(out/'manifest.json'))


def verify(path, digest):
    import json

    from night_contract import read_sealed_bytes, read_sealed_json
    from source_contract import sha256

    seal = read_sealed_json(path, 'manifest.json', digest, [])
    files = seal['files']
    check_equal('sealed TAL file names', ('sealed', {'metrics.json', 'night_tal.py'}), (seal['status'], set(files)))
    for file, entry in files.items():
        raw = read_sealed_bytes(path, file, entry['sha256'], [])
        check_equal('sealed TAL bytes', entry['bytes'], len(raw))
        if file == 'metrics.json':
            v = json.loads(raw)
    check_equal('no-update and training-only scope', (list(INDICES), len(INDICES), 0, True, BALANCED_SHA, False, False, False),
                (v['indices'], v['checks']['batches'], v['updates'], v['weights_unchanged'], v['balanced_sha256'],
                 v['development_read'], v['holdout_read'], v['cloud_used']))
    failures = decision(v['classes'])
    check_equal('recomputed TAL gate', (failures, not any(failures.values())), (v['failures'], v['met']))
    return {'manifest_sha256': sha256(path/'manifest.json'), **v}


def main():
    import argparse
    import json
    from pathlib import Path
    import re

    from night_contract import ROOT, _no_reparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('probe', 'verify'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scratch', type=Path)
    p.add_argument('--manifest-sha256')
    a = p.parse_args()
    out = a.output.absolute()
    existing = a.mode == 'verify'
    if out.parent != ROOT/'runs' or not re.fullmatch(r'tal_\d{3}', out.name) or out.exists() != existing or out.is_symlink():
        p.error('Expected a fresh probe or sealed existing runs/tal_NNN')
    _no_reparse(out if existing else out.parent)
    if existing:
        if a.scratch or not a.manifest_sha256:
            p.error('verify requires a manifest hash and no scratch')
        result = verify(out, a.manifest_sha256)
    else:
        if not a.scratch or a.manifest_sha256:
            p.error('probe requires external scratch')
        scratch = a.scratch.absolute()
        _no_reparse(scratch)
        if not scratch.is_dir() or scratch.resolve() != scratch or scratch.is_relative_to(ROOT) or scratch.drive != out.drive:
            p.error('Expected a real external scratch directory on the same drive')
        result = preflight(scratch, out)
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
