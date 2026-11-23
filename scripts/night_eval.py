"""S5 dual-domain evaluation: fixed FPPI, shared confidence, and classwise recovery."""
from __future__ import annotations

import argparse
from fractions import Fraction
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np

import detection_metrics as dm
import person_fusion as pf
from night_contract import DEV, ROOT, read_sealed_bytes, read_sealed_json, verify_dev
from source_contract import ContractError, NAMES, check_equal

PRIMARY = (0, 2)
RESEARCH = (0, 1, 2, 3, 4)
EMPTY_THRESHOLD = 2.0  # Finite JSON value, strictly outside the confidence range [0, 1].


def load_protocol(path=ROOT / 'configs/night.json'):
    raw = Path(path).read_bytes()
    cfg = json.loads(raw.decode('utf-8'))
    check_equal('night protocol version/classes', (1, list(PRIMARY), list(RESEARCH)),
                (cfg.get('schema_version'), cfg.get('primary_class_ids'), cfg.get('research_class_ids')))
    caps = {k: Fraction(cfg['fppi'][k]) for k in ('primary', 'auxiliary')}
    ratio = Fraction(cfg['recovery_ratio'])
    iv = cfg['interval']
    if not (0 < caps['primary'] < caps['auxiliary'] <= 1 and 0 < ratio <= 1
            and type(iv['repeats']) is int and iv['repeats'] >= 2 and 0 < Fraction(iv['width_review']) <= 1):
        raise ContractError('Invalid night FPPI, recovery, or interval configuration')
    return {**cfg, 'sha256': hashlib.sha256(raw).hexdigest()}


def _index(rows, purposes=('source_dev', DEV)):
    result = {r['key']: r for r in rows}
    if not result or len(result) != len(rows) or len({r['purpose'] for r in rows}) != 1 or rows[0]['purpose'] not in purposes:
        raise ContractError('Expected unique rows from one allowed development split')
    return result


def _select_fields(record, indices, length):
    return {k: [v[i] for i in indices] for k, v in record.items() if isinstance(v, list) and len(v) == length}


def _five(row):
    gc, pc = row['gt']['cls'], row['pred']['cls']
    g = {old: new for new, old in enumerate(i for i, c in enumerate(gc) if c in RESEARCH)}
    p = {old: new for new, old in enumerate(i for i, c in enumerate(pc) if c in RESEARCH)}
    result = dict(row, gt=_select_fields(row['gt'], g, len(gc)), pred=_select_fields(row['pred'], p, len(pc)))
    if 'iou_edges' in row:
        result['iou_edges'] = [[g[i], p[j], v] for i, j, v in row['iou_edges'] if i in g and j in p]
    tp, pairs = dm.ap_flags(result)
    return dict(result, ap_tp=tp.tolist(), ap_pairs=pairs)


def fuse_dev_rows(anchor, views):
    """Retain the native 640 anchor's exact edges; fuse people before projecting to five classes."""
    if len(views) != 4:
        raise ContractError('Expected the four locked pedestrian views')
    keys = _index(anchor, (DEV,))
    maps = [_index(view, (DEV,)) for view in views]
    for lookup in maps:
        check_equal('view key identity', set(keys), set(lookup))
    result, caps = [], []
    for row in anchor:
        key = row['key']
        candidates = [lookup[key] for lookup in maps]
        for other in candidates:
            check_equal(f'view identity {key}', pf._identity(row), pf._identity(other))
        pred, dropped = pf.fuse_predictions(row['pred'], tuple(r['pred'] for r in candidates))
        old = [i for i, c in enumerate(row['pred']['cls']) if c != PRIMARY[0]]
        keep = {j: i for i, j in enumerate(old)}
        overlaps = dm.box_iou(row['gt']['original_boxes'], pred['original_boxes'][len(old):])
        g, p = np.nonzero(overlaps)
        edges = [[i, keep[j], v] for i, j, v in row['iou_edges'] if j in keep]
        edges.extend([int(i), int(j) + len(old), float(overlaps[i, j])] for i, j in zip(g, p, strict=True))
        out = dict(row, gt=dict(row['gt'], boxes=row['gt']['original_boxes']),
                   pred=dict(pred, boxes=pred['original_boxes'], ids=list(range(len(pred['cls'])))), iou_edges=edges)
        result.append(_five(out))
        if dropped:
            caps.append({'key': key, 'person_dropped': dropped})
    return result, caps


def fppi_curve(rows):
    """Align the exact, filter-then-rematch class events on a shared score axis."""
    _index(rows)
    curves = [dm.work_curve(rows, c) for c in PRIMARY]
    scores = np.unique(np.concatenate([[EMPTY_THRESHOLD], *(c['threshold'] for c in curves)]))[::-1]
    positions = [np.searchsorted(-c['threshold'], -scores, side='right') for c in curves]
    tp, fp = (np.column_stack([np.concatenate([[0], c[field]])[idx]
                               for c, idx in zip(curves, positions, strict=True)]) for field in ('tp', 'fp'))
    return {'threshold': scores, 'tp': tp, 'fp': fp, 'gt': [int(c['gt']) for c in curves]}


def choose_workpoint(curve, images, cap):
    budget = int(Fraction(str(cap)) * images)
    truth = curve['gt']
    fp = curve['fp'].sum(axis=1)
    missing = any(n == 0 for n in truth)
    # Compare macro recall without rounding or floating-point tie artifacts.
    merits = curve['tp'] @ np.array(truth[::-1], dtype=np.int64)
    i = 0 if missing else max(np.flatnonzero(fp <= budget), key=lambda j: (merits[j], -fp[j], curve['threshold'][j]))
    return {'threshold': float(curve['threshold'][i]), 'fp_budget': budget, 'primary_fp': int(fp[i]),
            'fppi': float(fp[i] / images), 'macro_recall': None if missing else float(merits[i] / (2 * truth[0] * truth[1]))}


def _point_rows(rows, threshold):
    for row in rows:
        yield {NAMES[c]: dm.image_point(row, c, threshold)[0] for c in RESEARCH}


def fixed_intervals(rows, threshold, *, repeats, seed):
    """Resample risk groups, not images; keep both class thresholds fixed."""
    groups = sorted({r['group_id'] for r in rows})
    indices = {k: i for i, k in enumerate(groups)}
    counts = np.zeros((len(groups), len(PRIMARY), 2), dtype=np.int64)
    for row in rows:
        for i, c in enumerate(PRIMARY):
            stats, _, _ = dm.image_point(row, c, threshold)
            counts[indices[row['group_id']], i] += (stats['tp'], stats['tp'] + stats['fn'])
    valid = []
    for _, mult in dm.cluster_draws(groups, repeats=repeats, seed=seed):
        totals = np.tensordot(mult, counts, axes=1)
        if np.all(totals[:, 1] > 0):
            valid.append(totals[:, 0] / totals[:, 1])
    per_class = dict.fromkeys(NAMES[c] for c in PRIMARY)
    macro = None
    if len(valid) >= 2:
        a = np.asarray(valid)
        macro = np.quantile(a.mean(axis=1), [.025, .975]).tolist()
        per_class = {NAMES[c]: np.quantile(a[:, i], [.025, .975]).tolist() for i, c in enumerate(PRIMARY)}
    return {'repeats': repeats, 'valid': len(valid), 'groups': len(groups), 'seed': seed,
            'macro_recall_95': macro, 'primary_recall_95': per_class}


def evaluate_rows(rows, protocol):
    """Evaluate sealed or synthetic records, without opening data or running a model."""
    _index(rows)
    rows = [_five(row) for row in rows]
    aps, ap_curves = dm.summarize_ap(rows)
    curve = fppi_curve(rows)
    points = {}
    for name in ('primary', 'auxiliary'):
        cap = protocol['fppi'][name]
        point = choose_workpoint(curve, len(rows), cap)
        thresholds = point['threshold']
        totals = np.zeros((len(RESEARCH), 3), dtype=np.int64)
        for counts in _point_rows(rows, thresholds):
            totals += [[counts[NAMES[c]][k] for k in ('tp', 'fp', 'fn')] for c in RESEARCH]
        per_class = {NAMES[c]: dm.counts(*v.tolist(), thresholds) for c, v in zip(RESEARCH, totals, strict=True)}
        check_equal('curve and direct rematch primary FP', point['primary_fp'], sum(per_class[NAMES[c]]['fp'] for c in PRIMARY))
        interval = fixed_intervals(rows, thresholds, repeats=protocol['interval']['repeats'], seed=protocol['interval']['seed'])
        points[name] = dict(point, cap=cap, per_class=per_class, interval=interval)
    primary = points['primary']
    reasons = []
    if primary['macro_recall'] is None:
        reasons.append('missing_primary_support')
    elif primary['macro_recall'] == 0:
        reasons.append('zero_primary_recall')
    ci = primary['interval']['macro_recall_95']
    if ci is None:
        reasons.append('missing_interval')
    elif ci[1] - ci[0] > float(Fraction(protocol['interval']['width_review'])):
        reasons.append('wide_interval')
    result = {k: aps[k] for k in ('images', 'groups', 'empty_images')}
    result['per_class'] = {NAMES[c]: aps['per_class'][NAMES[c]] for c in RESEARCH}
    return dict(result, purpose=rows[0]['purpose'], protocol_sha256=protocol['sha256'],
                workpoints=points, review_reasons=reasons), dict(
        {k: v for k, v in ap_curves.items() if not k.startswith('tricycle_')},
        **{f'fppi_{k}': np.asarray(v) for k, v in curve.items()})


def recovery(day, night, ratio):
    """Compare integer-count recalls exactly, with no cross-class averaging."""
    ratio = Fraction(str(ratio))
    def recall(point):
        gt = point['tp'] + point['fn']
        return Fraction(point['tp'], gt) if gt else None

    per_class = {}
    for c in PRIMARY:
        name = NAMES[c]
        reference, observed = recall(day[name]), recall(night[name])
        target = ratio * reference if reference else None
        met = observed >= target if target is not None and observed is not None else None
        per_class[name] = {'day_recall': float(reference) if reference is not None else None,
                           'night_recall': float(observed) if observed is not None else None,
                           'target': float(target) if target is not None else None,
                           'ratio': float(observed / reference) if reference and observed is not None else None,
                           'delta_to_target': float(observed - target) if met is not None else None,
                           'met': met}
    flags = [v['met'] for v in per_class.values()]
    return {'per_class': per_class, 'both_met': None if None in flags else all(flags)}


def source_reference(project, protocol, *, count=500):
    """Re-score the sealed seed-17 source-dev outputs, without images, labels, or GPU."""
    root = Path(project).absolute()
    run = root / 'runs/align_001'
    dataset = root / 'data/soda10m/splits/source_v1'
    run_reads, data_reads, source = [], [], protocol['source_a']
    manifest = read_sealed_json(run, 'manifest.json', source['manifest_sha256'], run_reads)
    ready_sha = protocol['data']['source_ready_sha256']
    check_equal('source reference run', ('complete', False, ready_sha),
                (manifest['status'], manifest['final_test_executed'], manifest['ready']))
    check_equal('source reference weights', source['checkpoint_sha256'], manifest['files']['17/last.pt']['sha256'])
    ready = read_sealed_json(dataset, 'READY.json', ready_sha, data_reads)
    check_equal('source READY', (1, 'source_v1', 'frozen'),
                (ready['schema_version'], ready['version'], ready['status']))
    entry = ready['manifests']['source_dev']
    manifest_rel = 'manifests/source_dev.json'
    check_equal('source_dev receipt', (manifest_rel, count, manifest['source_dev']),
                (entry['path'], entry['count'], entry))
    approved = read_sealed_json(dataset, manifest_rel, entry['sha256'], data_reads)
    check_equal('source_dev manifest', ('source_dev', count), (approved['purpose'], len(approved['records'])))
    expected = _index(approved['records'], ('source_dev',))
    metrics_rel = '17/dev/metrics.json'
    seal = manifest['files'][metrics_rel]
    old = read_sealed_json(run, metrics_rel, seal['sha256'], run_reads)
    check_equal('sealed reference metrics weights', source['checkpoint_sha256'], old['checkpoint_sha256'])
    objects = '17/dev/objects.jsonl.gz'
    seal = manifest['files'][objects]
    data = read_sealed_bytes(run, objects, seal['sha256'], run_reads)
    check_equal('sealed reference objects size', seal['bytes'], len(data))
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as zipped, io.TextIOWrapper(zipped, encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    index = _index(rows, ('source_dev',))
    check_equal('source_dev exact coverage', set(expected), set(index))
    fields = ('group_id', 'image', 'label', 'image_sha256', 'label_sha256', 'width', 'height')
    for key, row in index.items():
        check_equal(f'source_dev object identity {key}', tuple(expected[key][k] for k in fields),
                    tuple(row[k] for k in fields))
    result, _ = evaluate_rows(rows, protocol)
    main = result['workpoints']['primary']['per_class']
    ratio = Fraction(protocol['recovery_ratio'])
    targets = {name: values['target'] for name, values in recovery(main, main, ratio)['per_class'].items()}
    return dict(result, target_recall=targets,
                source={'checkpoint_sha256': source['checkpoint_sha256'], 'run_sha256': source['manifest_sha256'],
                        'ready_sha256': ready_sha, 'source_dev_sha256': entry['sha256'], 'objects_sha256': seal['sha256']},
                reads={'run': run_reads, 'dataset': data_reads})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('reference', 'verify', 'run'),
                        help='reference: re-score sealed source_dev; verify: hash night_dev; run: evaluate night_dev')
    parser.add_argument('--protocol-sha256', required=True, help='Externally confirmed SHA-256 of configs/night.json')
    parser.add_argument('--scratch', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--reference-sha256')
    args = parser.parse_args(argv)
    protocol = load_protocol()
    check_equal('night protocol SHA-256', args.protocol_sha256, protocol['sha256'])
    targets = (args.scratch, args.output, args.reference, args.reference_sha256)
    if args.mode == 'run':
        if not all(targets):
            parser.error('run requires --scratch, --output, --reference, and --reference-sha256')
        from night_run import run
        run(protocol, *targets)
        return
    if any(targets):
        parser.error('scratch, output and reference options apply only to run')
    if args.mode == 'reference':
        print('Scope: frozen seed 17 / sealed source_dev objects; no image or label re-read.', file=sys.stderr, flush=True)
        result = source_reference(ROOT, protocol)
    else:
        print('Scope: night_dev only; verify approved manifest, images and labels without inference.', file=sys.stderr, flush=True)
        ct = verify_dev(ROOT / 'data/soda10m/splits/night_v1', protocol['data']['night_ready_sha256'])
        result = {'purpose': DEV, 'images': len(ct['records']), 'ready_sha256': ct['ready_hash'],
                  'manifest': ct['manifest'], 'reads': ct['reads'], 'protocol_sha256': protocol['sha256']}
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
