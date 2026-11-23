"""Fit native classification rows on sealed training inputs; evaluate unchanged S5 protocol."""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

import night_accept as accept
import night_adapt as adapt
import night_train as train
from night_contract import ROOT, _no_reparse, read_sealed_bytes, read_sealed_json, verify_dev
from source_contract import ContractError, canonical_hash, check_equal, read_json, sha256, write_json

TARGETS = (3, 4)


def rank_indices(iou, logits, *, positives=3, negatives=24):
    """Only a sufficiently separated candidate may supply a negative target."""
    import torch

    if iou.ndim != 2 or logits.ndim != 1 or iou.shape[1] != logits.numel():
        raise ContractError('Expected an IoU matrix aligned to the dense logits')
    if iou.shape[0]:
        overlaps, indices = iou.topk(min(positives, logits.numel()), dim=1)
        pos = torch.unique(indices[overlaps >= .5], sorted=True)
        safe = iou.amax(dim=0) < .30
    else:
        pos = torch.empty(0, dtype=torch.long, device=logits.device)
        safe = torch.ones_like(logits, dtype=torch.bool)
    hard, ids = logits.masked_fill(~safe, -torch.inf).topk(min(negatives, logits.numel()))
    return pos, ids[hard.isfinite()]


def pair_objective(positive, negative, hard):
    import torch.nn.functional as f

    if not positive.numel() or not negative.numel() or hard <= 0:
        raise ContractError('Ranking requires both positive and negative supervision')
    return f.softplus(1 + negative.topk(min(hard, negative.numel())).values[:, None] - positive).mean()


def assert_head_delta(before, after, writable):
    import torch

    check_equal('identical checkpoint tensor set', set(before), set(after))
    if not writable or not writable <= set(before):
        raise ContractError('Missing native classification layers')
    changes = [0, 0]
    for name, old in before.items():
        new = after[name]
        if name not in writable:
            check_equal('protected tensor '+name, True, torch.equal(new, old))
            continue
        check_equal('six classification rows', (6, old.shape), (new.shape[0], new.shape))
        check_equal('protected classification rows '+name, True,
                    torch.equal(new[:3], old[:3]) and torch.equal(new[5:], old[5:]))
        for i, cls in enumerate(TARGETS):
            changes[i] += int(torch.count_nonzero(new[cls] != old[cls]))
    if not all(changes):
        raise ContractError('Each registered class must change after serialization')
    return changes


@contextmanager
def _features(head):
    captured = {}
    hooks = [tower[-1].register_forward_pre_hook(lambda _, x, i=i: captured.__setitem__(i, x[0]))
             for i, tower in enumerate(head.cv3)]
    try:
        yield captured
    finally:
        for hook in hooks:
            hook.remove()


@contextmanager
def score_capture_precision(torch):
    conv = torch.backends.cudnn.conv
    previous = conv.fp32_precision
    try:
        conv.fp32_precision = 'ieee'
        yield
    finally:
        conv.fp32_precision = previous


def _collect(torch, cv2, model, payloads, until):
    from torchvision.ops import box_iou, box_convert

    head = model.model[-1]
    check_equal('native one-to-many detector and center-size geometry', (6, 3, False, False),
                (head.nc, len(head.cv3), head.end2end, head.xyxy))
    check_equal('three native 1x1 classification projections', True, all(
        isinstance(t[-1], torch.nn.Conv2d) and t[-1].kernel_size == (1, 1) and t[-1].groups == 1
        for t in head.cv3))
    cache = defaultdict(list)
    support = {domain: [dict(images=0, gt=0, signaled_gt=0) for _ in TARGETS] for domain in payloads}
    with score_capture_precision(torch), _features(head) as captured, torch.inference_mode():
        for domain, payload in payloads.items():
            for key, (_, _, _, labels) in payload.items():
                if time.monotonic() >= until:
                    raise ContractError('Local rank budget expired while collecting training features')
                classes = torch.tensor([r[0] for r in labels], device='cuda', dtype=torch.long)
                for flip in ((False, True) if domain == 'night' else (False,)):
                    b = train._batch(torch, cv2, payload, [(key, flip, 1.)], 640)
                    captured.clear()
                    y, native = model(b['img'])
                    features = [captured[i][0].flatten(1).T for i in range(3)]
                    sizes = [len(x) for x in features]
                    boundaries = torch.tensor(np.cumsum(sizes)[:-1], device='cuda')
                    levels = torch.repeat_interleave(torch.arange(3, device='cuda'), torch.tensor(sizes, device='cuda'))
                    features = torch.cat(features)
                    features = torch.cat((features, torch.ones_like(features[:, :1])), dim=1)
                    boxes = box_convert(y[0, :4].T.float(), 'cxcywh', 'xyxy')
                    truth = box_convert(b['bboxes']*640, 'cxcywh', 'xyxy')
                    scores = native['scores'][0]
                    check_equal('feature and box order', len(features), len(boxes))
                    if sum(len(v) for v in cache.values()) == 0:
                        for l, tower in enumerate(head.cv3):
                            layer = tower[-1]
                            w = torch.cat((layer.weight[..., 0, 0], layer.bias[:, None]), dim=1)
                            mask = levels == l
                            torch.testing.assert_close(features[mask] @ w.T, scores[:, mask].T, atol=5e-5, rtol=5e-5)
                    for c, cls in enumerate(TARGETS):
                        overlap = box_iou(truth[classes == cls], boxes)
                        pos, neg = rank_indices(overlap, scores[cls])
                        if not flip:
                            counts = support[domain][c]
                            counts['images'] += int(bool(len(overlap)))
                            counts['gt'] += len(overlap)
                            counts['signaled_gt'] += int((overlap.amax(dim=1) >= .5).sum()) if len(overlap) else 0
                        for role, ids in (('pos', pos), ('neg', neg)):
                            chosen = levels[ids]
                            torch.testing.assert_close(chosen, torch.bucketize(ids, boundaries, right=True))
                            cache[domain, c, role].append((features[ids], chosen, scores[cls, ids]))
    return {k: tuple(torch.cat(column) for column in zip(*v, strict=True)) for k,v in cache.items()}, support


def _fit(torch, cv2, parent, payloads, cfg, until, dest):
    from ultralytics.nn.tasks import load_checkpoint

    model, _ = load_checkpoint(str(parent), device='cuda')
    model.eval().requires_grad_(False)
    cache, support = _collect(torch, cv2, model, payloads, until)
    head = model.model[-1]
    width = head.cv3[0][-1].in_channels
    delta = torch.zeros((3, len(TARGETS), width+1), device='cuda', requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=cfg['learning_rate'])
    trace = []

    def adjustment(cache_row, c):
        x, l, _ = cache_row
        return (x * delta[l, c]).sum(dim=1)

    for step in range(cfg['updates']):
        if time.monotonic() >= until:
            raise ContractError('Local rank budget expired while fitting')
        losses = []
        for c in range(len(TARGETS)):
            pos, neg = (cache['night', c, role] for role in ('pos', 'neg'))
            rank = pair_objective(pos[2]+adjustment(pos, c), neg[2]+adjustment(neg, c), cfg['hard_negatives'])
            penalty = sum(adjustment(cache['day', c, role], c).square().mean() for role in ('pos', 'neg')) / 2
            losses.append(rank + cfg['day_penalty'] * penalty)
        objective = sum(losses)/len(losses) + cfg['ridge']*delta.square().sum(dim=-1).mean()
        if not objective.isfinite():
            raise ContractError('Nonfinite ranking objective')
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_([delta], 10., error_if_nonfinite=True)
        optimizer.step()
        trace.append(float(objective.detach()))
    baseline = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
    layers = [tower[-1] for tower in head.cv3]
    writable = {name for name, p in model.named_parameters()
                if any(p is layer.weight or p is layer.bias for layer in layers)}
    check_equal('exact classification parameter set size', 6, len(writable))
    with torch.no_grad():
        for l, layer in enumerate(layers):
            for c, cls in enumerate(TARGETS):
                layer.weight[cls, :, 0, 0].add_(delta[l, c, :-1])
                layer.bias[cls].add_(delta[l, c, -1])
    args = model.args
    save = model.cpu().half().eval()
    path = dest/'last.pt'
    torch.save({'ema': save, 'model': None, 'train_args': args, 'scope': 'native two-row ranking adjustment',
                'updates': cfg['updates']}, path)
    reloaded, _ = load_checkpoint(str(path), device='cpu')
    changed = assert_head_delta(baseline, reloaded.state_dict(), writable)
    return path, {'trainable': sorted(writable), 'rows': TARGETS, 'changed_values': changed,
                  'capture_conv_fp32_precision': 'ieee',
                  'updates': cfg['updates'], 'objective': trace, 'support_orig_orientation': support,
                  'anchors': {'/'.join(map(str,k)): len(v[0]) for k,v in cache.items()}, 'last_sha256': sha256(path),
                  'checkpoint_roles': {'last': 'last.pt', 'best': {'path': 'last.pt', 'selected_on_dev': False}}}


def _parent(cfg, tasks, pair_cfg, identity):
    parent = ROOT/'runs/e1_001'
    seal = read_sealed_json(parent, 'manifest.json', cfg['parent_manifest_sha256'], [])
    rec = adapt._json(parent, seal, 'run.json')
    check_equal('complete E1 parent and source identity', ('complete', 'e1', list(adapt.SEEDS), cfg['pair_sha256'], False,
                                                        identity['A_sha256'], pair_cfg),
                (rec['status'], rec['arm'], rec['completed_seeds'], rec['protocol_sha256'], rec['final_test_executed'],
                 rec['a_sha256'], rec['inputs']))
    result = {}
    for seed, task in tasks.items():
        data = adapt._json(parent, seal, f'{seed}/inputs.json')
        fitted = adapt._json(parent, seal, f'{seed}/train.json')
        check_equal('locked E1 exposure and order', (200, 0, task['slots']['schedule_sha256'],
                                                     [row[0][0] for row in task['plan']], task['slots']['input_slots']),
                    (fitted['updates'], fitted['skipped_updates'], data['schedule_sha256'], data['source_keys'],
                     [r['key'] for r in data['records']]))
        path = parent/f'{seed}/last.pt'
        check_equal('verified parent checkpoint', (fitted['last_sha256'], seal['files'][f'{seed}/last.pt']['sha256']),
                    (sha256(path), sha256(path)))
        result[seed] = (path, data['source_keys'][:cfg['day_images']])
    return result, parent, seal


def snapshot_sources(out, entrypoint):
    from night_run import CODE

    if entrypoint not in ('night_rank.py', 'night_replay.py', 'night_tail.py', 'night_repeat.py', 'night_presence.py', 'night_focus.py'):
        raise ContractError('Unregistered experiment entrypoint')
    (out/'code').mkdir()
    for name in sorted(set(CODE) | {entrypoint, 'night_rank.py', 'night_train.py', 'night_adapt.py', 'night_accept.py',
                                    'source_policy.py', 'tricycle_fusion.py'}):
        shutil.copyfile(ROOT/'scripts'/name, out/'code'/name)
    if entrypoint in ('night_tail.py', 'night_presence.py'):
        extra = ('night_tail.py', 'night_capture.py') if entrypoint == 'night_presence.py' else ()
        for name in ('night_strategy.py', 'night_select.py', *extra):
            shutil.copyfile(ROOT/'scripts'/name, out/'code'/name)


def fit_run(raw, cfg, scratch, out, *, fit=None, entrypoint='night_rank.py', selected=None):
    from seed_eval import probe_gpu

    start = time.monotonic()
    until = start + cfg['max_minutes']*60
    _, pair, old = adapt._config(cfg['pair_sha256'])
    seeds = cfg['seeds']
    all_tasks, protocol, identity = adapt.alignment(pair, old, arms=('e1',), seeds=seeds)
    tasks = all_tasks['e1']
    parents, parent, seal = _parent(cfg, tasks, pair, identity)
    if selected is not None:
        check_equal('selection seed scope', set(seeds), set(selected['lists']))
        day_keys = adapt._day_keys(identity['source_train'])
        tasks = {seed: adapt.ratio_slot_task(seed, 20, day_keys, selected['lists'][seed]['records'], old,
                                             t['slots']) for seed,t in tasks.items()}
    versions = {k: version(k) for k in train.LIBS}
    check_equal('locked package set', identity['versions'], versions)
    check_equal('frozen root A checkpoint', identity['A_sha256'], sha256(train.CHECKPOINT))
    gpu = probe_gpu()
    day = train.verify_subset(train.SOURCE, protocol['data']['source_ready_sha256'], 'source_train')
    day_by = {r['key']: r for r in day['records']}
    dev = train.verify_subset(train.SOURCE, protocol['data']['source_ready_sha256'], 'source_dev')
    night = verify_dev(train.NIGHT, protocol['data']['night_ready_sha256'])
    for split, split_contract in (('source_train', day), ('source_dev', dev), ('night_dev', night)):
        check_equal(f'D {split} manifest', identity[split], split_contract['manifest'])
    dev_groups = {r['group_id'] for r in dev['records']+night['records']}
    out.mkdir(exist_ok=False)
    done, assessments, pairs, parent_pairs = [], {}, {}, {}
    error = None
    try:
        (out/'protocol.json').write_bytes(raw)
        snapshot_sources(out, entrypoint)
        if selected is not None:
            write_json(out/'selection.json', {**selected, 'slots': {str(s): t['slots'] for s,t in tasks.items()}})
        with tempfile.TemporaryDirectory(prefix='rank-', dir=scratch) as td:
            tmp = Path(td)
            for key in ('YOLO_CONFIG_DIR', 'TORCH_HOME', 'MPLCONFIGDIR', 'TEMP', 'TMP', 'TMPDIR', 'XDG_CONFIG_HOME'):
                p = tmp/key.lower(); p.mkdir(); os.environ[key] = str(p)
            os.environ.update(YOLO_AUTOINSTALL='false', WANDB_MODE='disabled', CUBLAS_WORKSPACE_CONFIG=':4096:8')
            import cv2
            import torch

            torch.set_num_threads(4); cv2.setNumThreads(1)
            for seed in seeds:
                path, day_keys = parents[seed]
                task = tasks[seed]
                gradient_groups = {day_by[k]['group_id'] for k in day_keys} | {r['group_id'] for r in task['rows']}
                check_equal('training and development group separation', set(), gradient_groups & dev_groups)
                phase = out/str(seed); phase.mkdir()
                box = tmp/str(seed); box.mkdir()
                nt = box/'night-input'; nt.mkdir()
                grant = nt/'grant.json'
                write_json(grant, {'arm': 'e1', 'seed': seed, 'records': task['rows']})
                grant_sha = sha256(grant)
                subprocess.run([sys.executable, '-B', str(ROOT/'scripts/night_adapt.py'), 'prepare', '--arm', 'e1',
                                '--seed', str(seed), '--scratch', str(nt), '--protocol-sha256', cfg['pair_sha256'],
                                '--grant-sha256', grant_sha], check=True, timeout=120)
                rec = read_json(nt/'inputs.json')
                check_equal('identical parent night inputs', task['slots']['input_slots'], [r['key'] for r in rec['records']])
                check_equal('narrow grant', (seed, 'e1', grant_sha), (rec['seed'], rec['arm'], rec['grant_sha256']))
                ds, _ = train._stage(train.SOURCE, box/'day-input', [day_by[k] for k in day_keys])
                payloads = {'day': train.training_rows(box/'day-input', ds, 'source_train'),
                            'night': train.training_rows(nt, rec['records'], 'night_pool')}
                check_equal('exact rank fit whitelists', (day_keys, task['slots']['input_slots']),
                            (list(payloads['day']), list(payloads['night'])))
                write_json(phase/'inputs.json', {**rec, 'source_keys': day_keys, 'parent_sha256': sha256(path),
                                                   'gradient_identity': canonical_hash([day_keys, task['slots']['input_slots']])})
                print(f'{out.name} seed {seed}: fit on 100 original night images + {len(day_keys)} source replay images.', flush=True)
                torch.manual_seed(seed)
                torch.use_deterministic_algorithms(True)
                with train.telemetry() as telemetry:
                    if fit is None:
                        weight, result = _fit(torch, cv2, path, payloads, cfg, until, phase)
                    else:
                        weight, result = fit(torch, cv2, payloads, task['plan'], seed, old, cfg, until, phase)
                write_json(phase/'train.json', dict(result, gpu=telemetry, seed=seed))
                del payloads
                torch.cuda.empty_cache()
                torch.use_deterministic_algorithms(False)
                print(f'{out.name} seed {seed}: fixed weight saved; evaluate locked development protocol.', flush=True)
                policy, ref, baseline, anchors, a_day = adapt._reference(pair, protocol, dev, night, [seed])
                pairs[seed], assessments[seed] = adapt._evaluate(torch, weight, phase/'dev', box/'eval', night,
                                                                 dev, policy, ref, anchors[seed], a_day, until)
                rel = f'{seed}/dev/night'
                old_rows = accept.sealed_objects(parent, seal, f'{rel}/objects.jsonl.gz', 'night_dev',
                                                  {r['key']: r for r in night['records']}, [])
                old_points = adapt._json(parent, seal, f'{rel}/metrics.json')['workpoints']
                parent_pairs[seed] = (old_rows, old_points, *pairs[seed][2:])
                done.append(seed)
                a = assessments[seed]
                print(f'{out.name} seed {seed}: night {a["classes_met"]}/5; day protection={a["day_protection"]["met"]}; '
                      f'day macro delta={100*a["day_protection"]["macro_delta"]:.6f} pp; review={a["review"]}', flush=True)
            write_json(out/'comparison.json', {'candidate_minus_parent': accept.paired_recall(parent_pairs, policy),
                       'candidate_minus_historical_d': accept.paired_recall(pairs, policy),
                       'D_is_same_training_mechanism': entrypoint in ('night_tail.py','night_presence.py'), 'assessments': assessments,
                       'candidate_day_gate_relative_to_A': True})
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        write_json(out/'run.json', {'status': 'complete' if not error and done == seeds else 'incomplete',
                                   'completed_seeds': done, 'error': error, 'seconds': time.monotonic()-start,
                                   'entry': entrypoint, 'config_sha256': sha256(out/'protocol.json') if (out/'protocol.json').exists() else None,
                                   'a_sha256': identity['A_sha256'], 'versions': versions, 'initial_gpu': gpu,
                                   'python': sys.version, 'cloud_used': False, 'final_test_executed': False,
                                   'source_train': day['manifest'], 'source_dev': dev['manifest'], 'night_dev': night['manifest']})
        write_json(out/'manifest.json', {'status': 'sealed', 'files': train._receipt(out)})


def main(*, kind='rank', fit=None, validate=None, select=None, description=__doc__):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--config-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scratch', type=Path, required=True)
    a = p.parse_args()
    raw = read_sealed_bytes(ROOT, f'configs/{kind}.json', a.config_sha256, [])
    cfg = json.loads(raw)
    check_equal('registered experiment schema', 1, cfg['schema_version'])
    if cfg['seeds'] not in ([43], [17, 29]) or not (1 <= cfg['day_images'] <= 2000 and 0 < cfg['max_minutes'] <= 60):
        raise ContractError('Unregistered seed batch or local training budget')
    out, scratch = a.output.absolute(), a.scratch.absolute()
    if validate is not None:
        validate(cfg)
    if out.parent != ROOT/'runs' or out.exists() or out.is_symlink() or not re.fullmatch(rf'{kind}_\d{{3}}', out.name):
        raise ContractError(f'Expected a new runs/{kind}_NNN directory')
    if scratch.resolve() != scratch or scratch.is_relative_to(ROOT) or not scratch.is_dir():
        raise ContractError('Expected an existing external scratch directory')
    _no_reparse(out.parent); _no_reparse(scratch)
    selected = select(cfg, scratch) if select is not None else None
    fit_run(raw, cfg, scratch, out, fit=fit, entrypoint=f'night_{kind}.py', selected=selected)


if __name__ == '__main__':
    main()
