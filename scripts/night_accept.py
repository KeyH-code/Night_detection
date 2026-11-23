"""Offline five-class development acceptance; no model, candidate, or holdout loader."""
from __future__ import annotations

import argparse
from fractions import Fraction
import gzip
import io
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile

import numpy as np

import detection_metrics as dm
from night_contract import ROOT, _no_reparse, read_sealed_bytes, read_sealed_json
from source_contract import ContractError, NAMES, check_equal, sha256, write_json

CLASSES = tuple(NAMES[:5])
DEV = {'source_dev': ('source_v1', 500), 'night_dev': ('night_v1', 331)}
SEEDS = (17, 29, 43)
EMPTY = 2.0  # finite sentinel above every valid confidence
VIEWS = ((640, False), (960, False), (1280, False), (960, True), (1280, True))
ID_FIELDS = ('group_id', 'image', 'label', 'image_sha256', 'label_sha256', 'width', 'height')


def _index(rows, purpose=None):
    expected = purpose if purpose is not None else (rows[0].get('purpose') if rows else None)
    if expected not in DEV or not rows or any(r.get('purpose') != expected for r in rows):
        raise ContractError('Expected a single approved development purpose')
    found = {r['key']: r for r in rows}
    if len(found) != len(rows) or any(not isinstance(k, str) or not k for k in found):
        raise ContractError('Expected unique, nonempty development keys')
    return found


def _ap_rows(rows):
    """Keep canonical row order and IoU edges, projecting away the excluded class before AP."""
    for row in rows:
        g_cls, p_cls = row['gt']['cls'], row['pred']['cls']
        conf = np.asarray(row['pred']['conf'], dtype=np.float64)
        if (conf.shape != (len(p_cls),) or not np.isfinite(conf).all() or np.any((conf < 0) | (conf > 1))
                or np.any(conf != conf.astype(np.float32))
                or any(type(c) is not int or c not in range(len(NAMES)) for c in (*g_cls, *p_cls))):
            raise ContractError('Invalid native float32 confidence or class IDs')
        g, p = ({old: i for i, old in enumerate(j for j, c in enumerate(classes) if c < len(CLASSES))}
                for classes in (g_cls, p_cls))
        out = dict(row)
        for name, kept in (('gt', g), ('pred', p)):
            size = len(row[name]['cls'])
            out[name] = {k: [v[i] for i in kept] for k, v in row[name].items() if isinstance(v, list) and len(v) == size}
        if 'iou_edges' in row:
            out['iou_edges'] = [[g[i], p[j], v] for i, j, v in row['iou_edges'] if i in g and j in p]
        out['ap_tp'] = dm.ap_flags(out)[0].tolist()
        yield out


def _choose(curve, images, budget):
    choices = np.flatnonzero(curve['fp'] <= budget)
    if len(choices):
        i = max(choices, key=lambda j: (curve['tp'][j], -curve['fp'][j], curve['threshold'][j]))
        chosen = (int(curve['tp'][i]), int(curve['fp'][i]), int(curve['fn'][i]), float(curve['threshold'][i]))
    else:
        chosen = (0, 0, curve['gt'], EMPTY)
    # Empty competes even when a nonempty feasible event exists, with the same tie rule.
    if chosen[0] == 0:
        chosen = (0, 0, curve['gt'], EMPTY)
    result = dm.counts(*chosen)
    return dict(result, fp_budget=budget, fppi=result['fp']/images, empty_output=result['threshold'] == EMPTY)


def _group_counts(rows, points, groups):
    lookup = {g: i for i, g in enumerate(groups)}
    result = np.zeros((len(groups), len(CLASSES), 3), dtype=np.int64)
    for row in rows:
        for i, name in enumerate(CLASSES):
            v, _, _ = dm.image_point(row, i, points[name]['threshold'])
            result[lookup[row['group_id']], i] += (v['tp'], v['fp'], v['fn'])
    check_equal('fixed threshold rematch counts', [[points[n][k] for k in ('tp', 'fp', 'fn')] for n in CLASSES],
                result.sum(axis=0).tolist())
    return result


def _draws(groups, policy):
    iv = policy['interval']
    return np.stack([m for _, m in dm.cluster_draws(groups, iv['repeats'], iv['seed'])])


def _recall_samples(totals):
    n = totals[..., 0] + totals[..., 2]
    return np.divide(totals[..., 0], n, out=np.full(n.shape, np.nan), where=n > 0)


def _ci(samples):
    good = samples[np.isfinite(samples)]
    return {'ci95': np.quantile(good, [.025, .975]).tolist() if len(good) >= 2 else None,
            'valid': len(good), 'invalid': len(samples)-len(good)}


def evaluate_dev(rows, policy):
    """Choose one exact score event per class and retain fixed-threshold evidence."""
    _index(rows)
    projected = list(_ap_rows(rows))
    aps, _ = dm.summarize_ap(projected)
    groups = sorted({r['group_id'] for r in rows})
    images = len(rows)
    budget = int(Fraction(policy['fppi'])*images)
    points = {}
    for i, name in enumerate(CLASSES):
        curve = dm.work_curve(projected, i)
        p = _choose(curve, images, budget)
        p['budget_sensitivity'] = {label: _choose(curve, images, cap) for label, cap in
                                   (('minus_one', max(0, budget-1)), ('plus_one', budget+1))}
        points[name] = p
    counts = _group_counts(projected, points, groups)
    draws = _draws(groups, policy)
    samples = _recall_samples(np.einsum('rg,gct->rct', draws, counts))
    for i, name in enumerate(CLASSES):
        points[name]['interval'] = _ci(samples[:, i])
    per_class = {name: aps['per_class'][name] for name in CLASSES}
    macro = [p['ap50_95'] for p in per_class.values()]
    return {'purpose': rows[0]['purpose'], 'images': images, 'groups': len(groups), 'empty_images': aps['empty_images'],
            'per_class': per_class, 'macro_ap50_95': float(np.mean(macro)) if None not in macro else None,
            'workpoints': points}


def _recall(p):
    total = p['tp'] + p['fn']
    return Fraction(p['tp'], total) if total else None


def _all(flags):
    return False if False in flags else None if None in flags else True


def day_protection(reference, candidate, policy):
    class_drop = Fraction(policy['day']['class_drop'])
    macro_drop = Fraction(policy['day']['macro_drop'])
    deltas = {}
    for n in CLASSES:
        a, b = reference[n]['ap50_95'], candidate[n]['ap50_95']
        deltas[n] = None if a is None or b is None else Fraction(str(b))-Fraction(str(a))
    flags = {n: None if d is None else d >= -class_drop for n, d in deltas.items()}
    macro = None if None in deltas.values() else sum(deltas.values())/len(CLASSES)
    class_pass = _all(list(flags.values()))
    macro_pass = None if macro is None else macro >= -macro_drop
    return {'macro_delta': None if macro is None else float(macro), 'macro_pass': macro_pass, 'class_pass': class_pass,
            'per_class': {n: {'delta': None if d is None else float(d), 'met': flags[n]} for n, d in deltas.items()},
            'met': _all([class_pass, macro_pass])}


def recovery(reference, night, policy):
    ratio = Fraction(policy['recovery_ratio'])
    width = float(Fraction(policy['interval']['width_review']))
    result = {}
    for name in CLASSES:
        a, b = reference[name], night[name]
        ref, got = _recall(a), _recall(b)
        target = ratio*ref if ref else None
        budget_pass = b['fp'] <= b['fp_budget']
        met = None if got is None or target is None else bool(budget_pass and got >= target)
        notes = []
        if ref is None:
            notes.append('missing_day_support')
        elif ref == 0:
            notes.append('zero_day_reference')
        if got is None:
            notes.append('missing_night_support')
        for domain, p in (('day', a), ('night', b)):
            ci = p['interval']['ci95']
            if ci is None:
                notes.append(f'{domain}_interval_undefined')
            elif ci[1]-ci[0] > width:
                notes.append(f'{domain}_interval_wide')
        if met is not None and budget_pass:
            if any((Fraction(v['tp'], b['tp']+b['fn']) >= target) != met for v in b['budget_sensitivity'].values()):
                notes.append('budget_sensitivity')
        result[name] = {'day_recall': None if ref is None else float(ref), 'night_recall': None if got is None else float(got),
                        'target': None if target is None else float(target),
                        'ratio': float(got/ref) if ref and got is not None else None,
                        'delta_to_target': None if met is None else float(got-target), 'budget_pass': budget_pass,
                        'met': met, 'review': notes, 'evidence_pass': not notes}
    return result


def _pair_identity(rows):
    def truth(gt):
        boxes = gt.get('original_boxes', gt.get('boxes', [None]*len(gt['cls'])))
        # A's legacy source_dev snapshot still contains class 5; D removed it.
        # Keep exact research-class box coordinates and order, without rounding.
        return [(c, box) for c, box in zip(gt['cls'], boxes, strict=True) if c < len(CLASSES)]

    return {k: (r['purpose'], tuple(r.get(f) for f in ID_FIELDS), truth(r['gt'])) for k, r in _index(rows).items()}


def _effect(delta, samples):
    result = {n: {'delta': float(delta[i]) if np.isfinite(delta[i]) else None, **_ci(samples[:, i])}
              for i, n in enumerate(CLASSES)}
    result['macro'] = {'delta': float(delta.mean()) if np.isfinite(delta).all() else None, **_ci(samples.mean(axis=1))}
    return result


def paired_recall(pairs, policy):
    """Pairs are seed -> (reference rows, reference workpoints, candidate rows, candidate workpoints)."""
    if not pairs or not set(pairs) <= set(SEEDS):
        raise ContractError('Expected preregistered paired seeds')
    first = pairs[min(pairs)][0]
    identity = _pair_identity(first)
    groups = sorted({r['group_id'] for r in first})
    draws = _draws(groups, policy)
    per_seed, effects, samples = {}, [], []
    for seed, (a, pa, b, pb) in sorted(pairs.items()):
        # Different serialized row order is allowed. Group and GT identity are not.
        if _pair_identity(a) != identity or _pair_identity(b) != identity:
            raise ContractError('Mismatched paired development key / group / GT identity')
        counts = [_group_counts(rows, workpoints, groups) for rows, workpoints in ((a, pa), (b, pb))]
        delta = _recall_samples(counts[1].sum(axis=0)) - _recall_samples(counts[0].sum(axis=0))
        draws_delta = (_recall_samples(np.einsum('rg,gct->rct', draws, counts[1])) -
                       _recall_samples(np.einsum('rg,gct->rct', draws, counts[0])))
        per_seed[str(seed)] = _effect(delta, draws_delta)
        effects.append(delta)
        samples.append(draws_delta)
    return {'per_seed': per_seed, 'mean': _effect(np.mean(effects, axis=0), np.mean(samples, axis=0)),
            'groups': len(groups), 'repeats': len(draws), 'seed': policy['interval']['seed'], 'fixed_thresholds': True}


def _checked_file(root, seal, rel, reads):
    entry = seal['files'][rel]
    raw = read_sealed_bytes(root, rel, entry['sha256'], reads)
    check_equal(f'{rel} bytes', entry['bytes'], len(raw))
    return raw


def sealed_objects(root, seal, rel, purpose, approved, reads):
    if purpose not in DEV:
        raise ContractError('Sealed objects must belong to an approved development split')
    raw = _checked_file(root, seal, rel, reads)
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as archive, io.TextIOWrapper(archive, encoding='utf-8') as text:
        rows = [json.loads(line) for line in text if line.strip()]
    index = _index(rows, purpose)
    if index.keys() != approved.keys():
        raise ContractError(f'Incomplete or unexpected {purpose} coverage: {rel}')
    for key, row in index.items():
        check_equal(f'{rel} object identity {key}', tuple(approved[key][k] for k in ID_FIELDS),
                    tuple(row[k] for k in ID_FIELDS))
    return rows


def _read_policy(root, expected, reads):
    raw = read_sealed_bytes(root, 'configs/accept.json', expected, reads)
    p = json.loads(raw)
    check_equal('locked five-class protocol', (1, list(range(5)), list(SEEDS), '0.02', '0.80',
                {'macro_drop': '0.01', 'class_drop': '0.02'}, {'repeats': 2000, 'seed': 20261026, 'width_review': '0.20'}),
                tuple(p[k] for k in ('schema_version', 'class_ids', 'seeds', 'fppi', 'recovery_ratio', 'day', 'interval')))
    return raw, p


def _audit_views(views, caps, *, historical_nms=False):
    check_equal('locked five-view order', VIEWS, tuple((v['imgsz'], v['mirror']) for v in views))
    flags = []
    for v in views:
        settings = v['settings']
        check_equal('locked view settings', (v['imgsz'], 1, False, True, 16, .001, .7, False, 300),
                    tuple(settings[k] for k in ('imgsz', 'batch', 'augment', 'rect', 'quantize', 'conf', 'iou',
                                                'agnostic_nms', 'max_det')))
        nms = v['nms']
        if historical_nms:
            event = {'truncated_images': nms['truncated_images'], 'images_at_cap': nms['images_at_cap']}
        else:
            event = {k: len(nms[k]) for k in ('pre_nms_truncated', 'max_det_truncated', 'at_max_det')}
        flags.append({'imgsz': v['imgsz'], 'mirror': v['mirror'], **event})
    truncated = any(v.get('truncated_images', 0) or v.get('pre_nms_truncated', 0) or v.get('max_det_truncated', 0)
                    for v in flags)
    return {'views': flags, 'fused_person_caps': len(caps), 'review': ['output_truncation_review'] if truncated or caps else []}


def _assessment(reference, day, night, policy):
    protection = day_protection(reference['per_class'], day['per_class'], policy)
    classes = recovery(reference['workpoints'], night['workpoints'], policy)
    reviews = sorted(set(reference['capture']['review'] + day['capture']['review'] + night['capture']['review']))
    if protection['met'] is None:
        reviews.append('missing_day_ap')
    if any(not p['evidence_pass'] for p in classes.values()):
        reviews.append('class_evidence_review')
    return {'day_protection': protection, 'recovery': classes,
            'classes_met': sum(v['met'] is True for v in classes.values()),
            'point_met': _all([protection['met'], *(v['met'] for v in classes.values())]), 'review': reviews}


def replay(root, policy, reads):
    """Approved experiment paths are fixed; no arbitrary data directory or purpose flags."""
    seals = {name: read_sealed_json(root, f'runs/{name}/manifest.json', policy['sealed'][name], reads)
             for name in ('align_001', 'night_001', 'd_001')}
    a, n, d = (seals[k] for k in ('align_001', 'night_001', 'd_001'))
    check_equal('sealed A/D states', ('complete', False, 'sealed', 'sealed'),
                (a['status'], a['final_test_executed'], n['status'], d['status']))
    checkpoint = policy['sealed']['checkpoint_a']
    check_equal('sealed A checkpoint identities', (checkpoint, checkpoint),
                (a['files']['17/last.pt']['sha256'], n['checkpoint_sha256']))

    def receipt(name, rel):
        local = []
        blob = _checked_file(root/'runs'/name, seals[name], rel, local)
        reads.extend(f'runs/{name}/{p}' for p in local)
        return json.loads(blob)

    approved, dev_entries = {}, {}
    for purpose, (version, count) in DEV.items():
        rel = f'manifests/{purpose}.json'
        entry = {'path': rel, 'count': count, 'sha256': policy['development'][purpose]}
        manifest = read_sealed_json(root, f'data/soda10m/splits/{version}/{rel}', entry['sha256'], reads)
        check_equal(f'{purpose} manifest receipt', (purpose, count), (manifest['purpose'], len(manifest['records'])))
        approved[purpose] = _index(manifest['records'], purpose)
        dev_entries[purpose] = entry
    check_equal('A source_dev receipt', dev_entries['source_dev'], a['source_dev'])
    old_day = receipt('align_001', '17/dev/metrics.json')
    check_equal('A day checkpoint', checkpoint, old_day['checkpoint_sha256'])
    old_night = receipt('night_001', 'run.json')
    check_equal('A night scope', ('night_dev', checkpoint, False, dev_entries['night_dev']),
                (old_night['purpose'], old_night['checkpoint_sha256'], old_night['final_test_executed'], old_night['input']))
    dr = receipt('d_001', 'run.json')
    check_equal('D fixed baseline', ('S6 D only', 'complete', list(SEEDS), False, checkpoint, policy['sealed']['night_001']),
                tuple(dr[k] for k in ('scope', 'status', 'completed_seeds', 'final_test_executed', 'A_sha256', 'base_sha256')))
    for purpose, entry in dev_entries.items():
        check_equal(f'D {purpose} receipt', entry, dr[purpose])

    def evaluate(name, rel, purpose, weights, capture):
        local = []
        rows = sealed_objects(root/'runs'/name, seals[name], rel, purpose, approved[purpose], local)
        reads.extend(f'runs/{name}/{p}' for p in local)
        result = evaluate_dev(rows, policy)
        check_equal(f'{purpose} integer FP budget', 10 if purpose == 'source_dev' else 6,
                    next(iter(result['workpoints'].values()))['fp_budget'])
        result['source'] = {'run': f'runs/{name}', 'manifest_sha256': policy['sealed'][name], 'objects': rel,
                            'objects_sha256': seals[name]['files'][rel]['sha256'], 'checkpoint_sha256': weights,
                            'development_sha256': dev_entries[purpose]['sha256']}
        result['capture'] = capture
        return rows, result

    day_rows, reference = evaluate('align_001', '17/dev/objects.jsonl.gz', 'source_dev', checkpoint,
                                   _audit_views(old_day['observations'][:5], old_day['caps']['person'], historical_nms=True))
    baseline_rows, baseline = evaluate('night_001', 'objects.jsonl.gz', 'night_dev', checkpoint,
                                       _audit_views(old_night['views'], old_night['fused_person_caps']))
    reference['night_targets'] = {n: float(Fraction(policy['recovery_ratio'])*r) if r else None for n, p in
                                  reference['workpoints'].items() for r in [_recall(p)]}
    baseline['recovery'] = recovery(reference['workpoints'], baseline['workpoints'], policy)
    pairs, arms = {}, {}
    for seed in SEEDS:
        train = receipt('d_001', f'{seed}/train.json')
        weight = d['files'][f'{seed}/last.pt']['sha256']
        check_equal(f'D {seed} fixed updates and checkpoint', (seed, 200, 0, True, weight),
                    tuple(train[k] for k in ('seed', 'updates', 'skipped_updates', 'ema_reload_verified', 'last_sha256')))
        view = receipt('d_001', f'{seed}/dev/evaluation.json')
        day = evaluate('d_001', f'{seed}/dev/day/objects.jsonl.gz', 'source_dev', weight,
                       _audit_views(view['day_views'], view['day_fusion_caps']))
        night = evaluate('d_001', f'{seed}/dev/night/objects.jsonl.gz', 'night_dev', weight,
                         _audit_views(view['night_views'], view['night_fusion_caps']))
        if _pair_identity(day[0]) != _pair_identity(day_rows):
            raise ContractError(f'D {seed} white-day GT differs from frozen A')
        arms[str(seed)] = {'day': day[1], 'night': night[1], 'assessment': _assessment(reference, day[1], night[1], policy)}
        pairs[seed] = (baseline_rows, baseline['workpoints'], night[0], night[1]['workpoints'])
    points = [arm['assessment'] for arm in arms.values()]
    counts = {name: sum(a['recovery'][name]['met'] is True for a in points) for name in CLASSES}
    stable = all(a['point_met'] is True and not a['review'] for a in points)
    return reference, {'purpose': 'development_only', 'A_night': baseline, 'D': arms,
                       'D_minus_A_night_recall': paired_recall(pairs, policy), 'D_point_met_by_class': counts,
                       'D_all_stable_met': stable, 'D_any_review': any(a['review'] for a in points)}


def publish(root, protocol_sha, scratch, output):
    scratch, output = Path(scratch).absolute(), Path(output).absolute()
    if (not scratch.is_dir() or scratch.resolve() != scratch or scratch.is_relative_to(root)
            or scratch.drive != output.drive or output.parent != root/'runs' or output.exists() or output.is_symlink()
            or not re.fullmatch(r'accept_\d{3}', output.name)):
        raise ContractError('Use external same-volume scratch and a new runs/accept_NNN')
    _no_reparse(scratch)
    _no_reparse(output.parent)
    reads = []
    raw, policy = _read_policy(root, protocol_sha, reads)
    code = {f'scripts/{name}': sha256(root/'scripts'/name) for name in
            ('night_accept.py', 'detection_metrics.py', 'night_contract.py', 'source_contract.py')}
    print('Scope: sealed A/D source_dev and night_dev predictions only; no model or raw data opens.', flush=True)
    reference, metrics = replay(root, policy, reads)
    with tempfile.TemporaryDirectory(prefix='accept-', dir=scratch) as temporary:
        folder = Path(temporary)/'evidence'
        folder.mkdir()
        (folder/'protocol.json').write_bytes(raw)
        write_json(folder/'reference.json', reference)
        write_json(folder/'metrics.json', metrics)
        write_json(folder/'run.json', {'status': 'complete', 'purpose': 'development_only', 'protocol_sha256': protocol_sha,
                   'reads': reads, 'code_sha256': code, 'environment': {'python': sys.version, 'numpy': np.__version__},
                   'inference_performed': False, 'raw_payload_read': False, 'candidate_labels_read': False,
                   'training_performed': False, 'held_out_read': False})
        for name, digest in code.items():
            to = folder/'code'/name
            to.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root/name, to)
            check_equal(f'archived code {name}', digest, sha256(to))
        files = {p.relative_to(folder).as_posix(): {'sha256': sha256(p), 'bytes': p.stat().st_size}
                 for p in sorted(folder.rglob('*')) if p.is_file()}
        write_json(folder/'manifest.json', {'status': 'sealed', 'scope': 'A/D five-class development replay', 'files': files})
        if output.exists() or output.is_symlink():
            raise ContractError('Refusing to overwrite an evaluation run')
        folder.rename(output)
    for rel, entry in files.items():
        p = output/rel
        check_equal(f'published {rel}', (entry['bytes'], entry['sha256']), (p.stat().st_size, sha256(p)))
    print(f'Sealed {output.name}; D point passes per class: {metrics["D_point_met_by_class"]}; '
          f'review flags: {metrics["D_any_review"]}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('replay',))
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    publish(ROOT, args.protocol_sha256, args.scratch, args.output)


if __name__ == '__main__':
    main()
