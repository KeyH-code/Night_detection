"""Run and independently replay the matched 100-image native vs joint tail training."""
from __future__ import annotations

import argparse
from collections import Counter
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import night_accept as ac
import night_adapt as adapt
import night_hard as hard
import night_train as train
from night_contract import ROOT, _no_reparse, _records, read_sealed_bytes, read_sealed_json, verify_dev
from source_contract import ContractError, canonical_hash, check_equal, sha256, write_json

CONFIG = 'configs/joint.json'
RUN = 'hard_002'
ARMS = ('control', 'joint')
SEED = 43


def localization_scores(q, multiplier):
    if q.ndim != 3 or q.shape[0] != 2 or q.shape[2] != 6:
        raise ContractError('Expected native [day, night] assignments')
    out = q.clone()
    out[1, :, 3:5] = q[1, :, 3:5] * multiplier
    return out


class JointLoss:
    def __init__(self, localization, quality, negative, per_bucket):
        self.localization = localization
        self.quality = quality
        self.negative = negative
        self.per_bucket = per_bucket
        self.assigned = None
        self.counter = None
        self.negatives = None
        self.calls = 0

    def class_terms(self, logits, target, aligned, same, other):
        import torch
        import torch.nn.functional as f

        if logits.shape != target.shape or logits.ndim != 3 or (logits.shape[0], logits.shape[2]) != (2, 6):
            raise ContractError('Expected original day and night native classification')
        z = logits[1, :, 3:5].float()
        q = target[1, :, 3:5].float()
        with torch.no_grad():
            y = ((aligned.detach() - .30) / .40).clamp(0., 1.)[:, None].expand_as(z)
            safe = (q == 0) & (same < .30) & (z.detach().sigmoid() >= .05)
            wrong = other >= .50
            near = torch.maximum(same, other) >= .10
            ranks = torch.argsort(z.detach(), dim=0, descending=True, stable=True)
            picked, counts = torch.zeros_like(safe), []
            for bucket in (wrong, ~wrong & near, ~wrong & ~near):
                mask = (safe & bucket).gather(0, ranks)
                in_bucket = mask & (mask.cumsum(dim=0) <= self.per_bucket)
                picked |= torch.zeros_like(mask).scatter_(0, ranks, in_bucket)
                counts.append(in_bucket.sum(dim=0))
        extra = self.quality * q * f.binary_cross_entropy_with_logits(z, y, reduction='none')
        extra = extra + self.negative * f.softplus(z) * picked
        return extra, (q > 0).sum(dim=0), torch.stack(counts, dim=1)

    def attach(self, loss):
        import torch
        from torchvision.ops import box_iou

        native_assigner, native_bce, native_bbox = loss.assigner, loss.bce, loss.bbox_loss

        def assign(scores, predicted, centers, classes, boxes, mask):
            result = native_assigner(scores, predicted, centers, classes, boxes, mask)
            if self.assigned is not None:
                raise ContractError('Native loss did not consume its assigned geometry')
            with torch.no_grad():
                p, assigned = predicted[1].float(), result[1][1].float()
                area_p = (p[:, 2:] - p[:, :2]).clamp(min=0).prod(dim=1)
                area_g = (assigned[:, 2:] - assigned[:, :2]).clamp(min=0).prod(dim=1)
                inter = (torch.minimum(p[:, 2:], assigned[:, 2:]) - torch.maximum(p[:, :2], assigned[:, :2])).clamp(min=0).prod(dim=1)
                aligned = inter / (area_p + area_g - inter).clamp(min=1e-7)
                gt_class = classes[1, :, 0].long()
                valid = mask[1, :, 0].bool() & (0 <= gt_class) & (gt_class < 5)
                if len(gt_class):
                    iou = box_iou(boxes[1].float(), p)
                    same = torch.stack([(iou * (valid & (gt_class == k))[:, None]).amax(0) for k in (3, 4)], dim=1)
                    other = torch.stack([(iou * (valid & (gt_class != k))[:, None]).amax(0) for k in (3, 4)], dim=1)
                else:
                    same = other = torch.zeros((len(p), 2), device=p.device, dtype=torch.float32)
                self.assigned = aligned, same, other
            return result

        def classify(logits, target):
            if self.assigned is None:
                raise ContractError('No native assignment captured before classification')
            extra, pos, neg = self.class_terms(logits, target, *self.assigned)
            self.assigned = None
            original = native_bce(logits, target)
            original[1, :, 3:5] = original[1, :, 3:5] + extra.to(original.dtype)
            with torch.no_grad():
                if self.counter is None:
                    self.counter, self.negatives = torch.zeros_like(pos), torch.zeros_like(neg)
                self.counter.add_(pos)
                self.negatives.add_(neg)
                self.calls += 1
            return original

        def locate(pred_dist, pred_boxes, anchors, target_boxes, target, denom, mask, imgsz, stride):
            return native_bbox(pred_dist, pred_boxes, anchors, target_boxes,
                               localization_scores(target, self.localization), denom, mask, imgsz, stride)

        loss.assigner, loss.bce, loss.bbox_loss = assign, classify, locate

    def summary(self):
        return {'calls': self.calls, 'positive_anchors': self.counter.tolist(),
                'negative_buckets': self.negatives.tolist(), 'bucket_order': ['wrong', 'near', 'background']}


def checked(folder, digest=None):
    seal = read_sealed_json(folder, 'manifest.json', digest or sha256(folder/'manifest.json'), [])
    check_equal('complete sealed file inventory', ('sealed', seal['files']), (seal['status'], train._receipt(folder)))
    return seal


def configuration(digest):
    raw = read_sealed_bytes(ROOT, CONFIG, digest, [])
    cfg = json.loads(raw)
    check_equal('registered one-seed joint comparison', (1, SEED, 200, 60, {'localization': 1.5, 'quality': .5, 'negative': .1, 'per_bucket': 8}),
                tuple(cfg[k] for k in ('schema_version', 'seed', 'updates', 'max_minutes', 'loss')))
    hc = hard.configuration(cfg['hard_sha256'])
    seal = checked(ROOT/'runs'/hard.LOCK, cfg['selection_sha256'])
    picked = adapt._json(ROOT/'runs'/hard.LOCK, seal, 'selection.json')
    support_seal = checked(ROOT/'runs'/hard.AUDIT, cfg['support_sha256'])
    supported = adapt._json(ROOT/'runs'/hard.AUDIT, support_seal, 'support.json')
    check_equal('approved post-selection support gate', (cfg['hard_sha256'], cfg['selection_sha256'], 100, True, True, False, False),
                tuple(supported[k] for k in ('protocol_sha256', 'selection_manifest_sha256', 'labels_read', 'support_met',
                                             'no_image_reads', 'development_read', 'holdout_read')))
    check_equal('image-only locked receipt', (cfg['hard_sha256'], 100, False, False, False),
                tuple(picked[k] for k in ('protocol_sha256', 'groups', 'candidate_labels_read', 'development_read', 'holdout_read')))
    locked = _records({'purpose': hard.POOL, 'records': picked['records']}, hard.POOL, 100, images_only=True)
    check_equal('ordered locked keys', picked['keys'], [r['key'] for r in locked])
    _, pair, old = adapt._config(cfg['pair_sha256'])
    _, protocol, identity = adapt.alignment(pair, old, arms=(), seeds=(SEED,))
    check_equal('frozen A and only-image identities', (hc['a_sha256'], hc['images_sha256'], hc['ready_sha256']),
                (identity['A_sha256'], protocol['data']['pool_images_sha256'], protocol['data']['night_ready_sha256']))
    droot = ROOT/'runs/d_001'
    ds = read_sealed_json(droot, 'manifest.json', pair['d_manifest_sha256'], [])
    inputs = adapt._json(droot, ds, '43/inputs.json')
    original = adapt._json(droot, ds, '43/train.json')
    d = [r['key'] for r in inputs['night']]
    task = adapt.ratio_slot_task(SEED, 20, adapt._day_keys(identity['source_train']), locked, old,
                                 {'d_input_slots': d, 'd_schedule_sha256': original['augment_schedule_sha256']})
    check_equal('exact same ordered source slots', inputs['source_keys'], [a[0] for a, _ in task['plan']])
    return raw, cfg, hc, pair, old, protocol, identity, task, supported


def managed(hc, task, audited):
    ready = read_sealed_json(train.NIGHT, 'READY.json', hc['ready_sha256'], [])
    entry = ready['held_out']['night_pool']
    check_equal('same frozen support hash-index', ('manifests/night_pool.json', audited['management_sha256']), (entry['path'], entry['sha256']))
    meta = read_sealed_json(train.NIGHT, entry['path'], entry['sha256'], [])
    rows = train.selected_records(task['rows'], meta['records'])
    by_key = {r['key']: r for r in rows}
    ordered = [by_key[k] for k in task['slots']['selection_order']]
    check_equal('exact label hashes as post-selection audit', audited['selected_label_receipt_sha256'],
                canonical_hash([(r['label'], r['label_sha256']) for r in ordered]))
    return rows


def comparison(policy, anchor, rows, assessments):
    def effect(first, second):
        return ac.paired_recall({SEED: (*first, *second)}, policy)['mean']

    return {'assessments': assessments,
            'selection_vs_random': effect(anchor, rows['control']),
            'joint_vs_same_list': effect(rows['control'], rows['joint']),
            'joint_vs_random_descriptive': effect(anchor, rows['joint'])}


def run(scratch, digest):
    from night_focus import fit_native
    from night_run import CODE
    from seed_eval import probe_gpu

    start = time.monotonic()
    out = ROOT/'runs'/RUN
    if out.exists() or out.is_symlink():
        raise ContractError(f'Refusing to overwrite {out.name}')
    raw, cfg, hc, pair, old, protocol, identity, task, audit = configuration(digest)
    until = start + cfg['max_minutes'] * 60
    check_equal('frozen initialization availability', identity['A_sha256'], sha256(train.CHECKPOINT))
    versions = {name: version(name) for name in train.LIBS}
    check_equal('pinned environment versions', identity['versions'], versions)
    gpu = probe_gpu()
    day = train.verify_subset(train.SOURCE, protocol['data']['source_ready_sha256'], 'source_train')
    dev = train.verify_subset(train.SOURCE, protocol['data']['source_ready_sha256'], 'source_dev')
    night = verify_dev(train.NIGHT, protocol['data']['night_ready_sha256'])
    for purpose, contract in (('source_train', day), ('source_dev', dev), ('night_dev', night)):
        check_equal('same fixed source/target contract', identity[purpose], contract['manifest'])
    plan = task['plan']
    by_day = {r['key']: r for r in day['records']}
    source_keys = [a[0] for a, _ in plan]
    target = managed(hc, task, audit)
    groups = {by_day[k]['group_id'] for k in source_keys} | {r['group_id'] for r in target}
    check_equal('no risk group in both gradient and development scope', set(), groups & {r['group_id'] for r in dev['records']+night['records']})
    out.mkdir()
    completed, error = [], None
    try:
        (out/'protocol.json').write_bytes(raw)
        write_json(out/'slots.json', task['slots'])
        (out/'code').mkdir()
        for name in sorted(set(CODE)|{'night_joint.py','night_hard.py','night_focus.py','night_rank.py','night_train.py',
                                     'night_adapt.py','night_accept.py','night_strategy.py','source_policy.py','tricycle_fusion.py'}):
            shutil.copyfile(ROOT/'scripts'/name, out/'code'/name)
        with tempfile.TemporaryDirectory(dir=scratch, prefix='joint-') as td:
            box = Path(td)
            for k in ('YOLO_CONFIG_DIR','TORCH_HOME','MPLCONFIGDIR','TEMP','TMP','TMPDIR','XDG_CONFIG_HOME'):
                p = box/k.lower(); p.mkdir(); os.environ[k] = str(p)
            os.environ.update(YOLO_AUTOINSTALL='false', WANDB_MODE='disabled', CUBLAS_WORKSPACE_CONFIG=':4096:8')
            import cv2
            import torch

            torch.set_num_threads(4); cv2.setNumThreads(1)
            a, src_reads = train._stage(train.SOURCE, box/'source', [by_day[k] for k in source_keys])
            b, night_reads = train._stage(train.NIGHT, box/'target', target)
            check_equal('only approved night training images and labels', [r[k] for r in target for k in ('image','label')], night_reads)
            payloads = {'day': train.training_rows(box/'source', a, 'source_train'),
                        'night': train.training_rows(box/'target', b, 'night_pool')}
            check_equal('unique training scope', (source_keys, task['slots']['input_slots']),
                        (list(payloads['day']), list(payloads['night'])))
            write_json(out/'inputs.json', {'source_keys': source_keys, 'night_keys': list(payloads['night']),
                                           'source_files': len(src_reads), 'night_payload': night_reads})
            policy, ref, _, anchors, a_day = adapt._reference(pair, protocol, dev, night, [SEED])
            assessments, assessed = {}, {}
            for arm in ARMS:
                phase = out/arm; phase.mkdir()
                print(f'{arm}: seed 43, 100 distinct night images, from frozen A.', flush=True)
                with train.telemetry() as monitor:
                    if arm == 'control':
                        weight, rec = train._fit(torch, cv2, payloads['day']|payloads['night'], plan, SEED,
                                                 old, phase, cfg['updates'], until)
                    else:
                        obj = JointLoss(**cfg['loss'])
                        weight, rec = fit_native(torch, cv2, payloads, plan, SEED, old, cfg, until, phase,
                                                 criterion_setup=obj.attach)
                        rec['joint_signal'] = obj.summary()
                        check_signal(rec['joint_signal'])
                check_consumption(rec, task)
                write_json(phase/'train.json', {**rec, 'gpu': monitor, 'a_sha256': identity['A_sha256']})
                torch.cuda.empty_cache()
                torch.use_deterministic_algorithms(False)
                values, assessment = adapt._evaluate(torch, weight, phase/'dev', box/f'eval-{arm}', night, dev,
                                                     policy, ref, anchors[SEED], a_day, until)
                assessed[arm], assessments[arm] = values[2:], assessment
                completed.append(arm)
                print(f'{arm}: night {assessment["classes_met"]}/5; day {assessment["day_protection"]["met"]}.', flush=True)
            write_json(out/'comparison.json', comparison(policy, anchors[SEED], assessed, assessments))
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        write_json(out/'run.json', {'status': 'complete' if completed == list(ARMS) and not error else 'incomplete',
                   'completed_arms': completed, 'seed': SEED, 'config_sha256': digest, 'a_sha256': identity['A_sha256'],
                   'seconds': time.monotonic()-start, 'gpu': gpu, 'error': error, 'versions': versions,
                   'cloud_used': False, 'final_test_executed': False})
        write_json(out/'manifest.json', {'status': 'sealed', 'files': train._receipt(out)})


def check_consumption(rec, task):
    seen = task['plan']
    check_equal('same exact schedule, all updates, no skips', (200, 0, task['slots']['schedule_sha256'],
                [[a[0], b[0]] for a,b in seen], {'source': dict(Counter(a[0] for a,b in seen)), 'night': dict(Counter(b[0] for a,b in seen))}),
                tuple(rec[k] for k in ('updates','skipped_updates','augment_schedule_sha256','microbatches','counts')))


def check_signal(receipt):
    check_equal('joint loss observed each microbatch', 2000, receipt['calls'])
    check_equal('negative categories precisely identified', ['wrong','near','background'], receipt['bucket_order'])
    if (len(receipt['positive_anchors']) != 2 or min(receipt['positive_anchors']) <= 0 or
            len(receipt['negative_buckets']) != 2 or any(len(a) != 3 or min(a) <= 0 for a in receipt['negative_buckets'])):
        raise ContractError('Joint quality and localization must observe each registered class and negative group')


def verify(digest, manifest_sha):
    _, _, _, pair, _, protocol, identity, task, _ = configuration(digest)
    out = ROOT/'runs'/RUN
    seal = checked(out, manifest_sha)
    state = adapt._json(out, seal, 'run.json')
    check_equal('complete exact comparison', ('complete', list(ARMS), digest, identity['A_sha256'], False, False),
                tuple(state[k] for k in ('status','completed_arms','config_sha256','a_sha256','cloud_used','final_test_executed')))
    check_equal('identical approved slots', task['slots'], adapt._json(out, seal, 'slots.json'))
    def dev_only(purpose, folder):
        entry = identity[purpose]
        doc = read_sealed_json(folder, entry['path'], entry['sha256'], [])
        check_equal('exact development metadata', (purpose, entry['count']), (doc['purpose'], len(doc['records'])))
        return {'manifest': entry, 'records': doc['records']}

    dev, night = dev_only('source_dev', train.SOURCE), dev_only('night_dev', train.NIGHT)
    policy, ref, _, anchors, a_day = adapt._reference(pair, protocol, dev, night, [SEED])
    assessed, assessments = {}, {}
    for arm in ARMS:
        t = adapt._json(out, seal, f'{arm}/train.json')
        check_consumption(t, task)
        check_equal('same initialization and final checkpoint SHA', (identity['A_sha256'], seal['files'][f'{arm}/last.pt']['sha256']),
                    (t['a_sha256'], t['last_sha256']))
        if arm == 'joint': check_signal(t['joint_signal'])
        ev = adapt._json(out, seal, f'{arm}/dev/evaluation.json')
        result, items = {}, {}
        for domain, purpose, contract in (('day', 'source_dev', dev), ('night', 'night_dev', night)):
            rel = f'{arm}/dev/{domain}'
            data = ac.sealed_objects(out, seal, f'{rel}/objects.jsonl.gz', purpose, ac._index(contract['records'], purpose), [])
            score = ac.evaluate_dev(data, policy)
            score['capture'] = ac._audit_views(ev[f'{domain}_views'], ev[f'{domain}_fusion_caps'])
            check_equal('recomputed metrics of '+rel, score, adapt._json(out, seal, f'{rel}/metrics.json'))
            items[domain], result[domain] = data, score
        check_equal('recovered keys and GT agree with fixed A in white-day', ac._pair_identity(a_day), ac._pair_identity(items['day']))
        assess = ac._assessment(ref, result['day'], result['night'], policy)
        check_equal('independently recomputed assessment', ev['assessment'], assess)
        assessed[arm], assessments[arm] = (items['night'], result['night']['workpoints']), assess
    values = comparison(policy, anchors[SEED], assessed, assessments)
    check_equal('paired same-list and random evidence', adapt._json(out, seal, 'comparison.json'), values)
    print(json.dumps({'files_verified': len(seal['files']), 'seconds': state['seconds'],
          'counts': {a: {c: [p['tp'], p['fp']] for c,p in v[1].items()} for a,v in assessed.items()}, **values}, allow_nan=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('run','verify'))
    p.add_argument('--config-sha256', required=True)
    p.add_argument('--scratch', type=Path)
    p.add_argument('--manifest-sha256')
    a = p.parse_args()
    if a.mode == 'verify':
        if a.scratch or not a.manifest_sha256: p.error('verify requires only config and manifest SHA')
        verify(a.config_sha256, a.manifest_sha256)
        return
    if not a.scratch or a.manifest_sha256: p.error('run requires an external scratch directory')
    scratch = a.scratch.absolute(); _no_reparse(scratch); _no_reparse(ROOT/'runs')
    if not scratch.is_dir() or scratch.resolve() != scratch or scratch.is_relative_to(ROOT) or scratch.drive != ROOT.drive:
        p.error('Expected an existing external same-volume scratch directory')
    run(scratch, a.config_sha256)


if __name__ == '__main__':
    main()
