"""Offline, auditable metrics for frozen-checkpoint diagnostics (no model or dataset loading).

The AP matcher/integrator reproduce the locked Ultralytics 8.4.171 semantics.
Operating points filter first and rematch; AP TP flags are deliberately not reused there.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

import numpy as np

from source_contract import ContractError, NAMES

IOUS = np.linspace(.5, .95, 10, dtype=np.float32)


def read_rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_rows(path, rows):
    with gzip.open(path, 'wt', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')


def box_iou(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float32).reshape(-1, 4)
    inter = np.maximum(np.minimum(a[:, None, 2:], b[None, :, 2:]) -
                       np.maximum(a[:, None, :2], b[None, :, :2]), 0).prod(2)
    area_a = (a[:, 2:] - a[:, :2]).prod(1)
    area_b = (b[:, 2:] - b[:, :2]).prod(1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + np.float32(1e-7))


def arrays(row):
    gc = np.asarray(row['gt']['cls'], dtype=np.int64)
    pc = np.asarray(row['pred']['cls'], dtype=np.int64)
    conf = np.asarray(row['pred']['conf'], dtype=np.float32)
    if 'iou_edges' in row:
        iou = np.zeros((len(gc), len(pc)), dtype=np.float32)
        for g, p, v in row['iou_edges']:
            iou[g, p] = v
    else:
        iou = box_iou(row['gt']['boxes'], row['pred']['boxes'])
    return gc, pc, conf, iou


def native_pairs(pred_cls, gt_cls, iou, threshold=.5):
    """Same IoU sort + prediction uniqueness + GT uniqueness as BaseValidator.

    Keep NumPy's native sort/unique order, including its tie semantics. This is
    intentionally not Hungarian or confidence-greedy matching.
    """
    masked = np.asarray(iou, dtype=np.float32) * (np.asarray(gt_cls)[:, None] == np.asarray(pred_cls))
    pairs = np.array(np.nonzero(masked >= threshold)).T
    if len(pairs) > 1:
        pairs = pairs[masked[pairs[:, 0], pairs[:, 1]].argsort()[::-1]]
        pairs = pairs[np.unique(pairs[:, 1], return_index=True)[1]]
        pairs = pairs[np.unique(pairs[:, 0], return_index=True)[1]]
    return pairs.reshape(-1, 2).astype(np.int64)


def ap_flags(row):
    gc, pc, _, iou = arrays(row)
    flags = np.zeros((len(pc), 10), dtype=bool)
    pairs = []
    for j, threshold in enumerate(row.get('iou_thresholds', IOUS)):
        m = native_pairs(pc, gc, iou, threshold)
        flags[m[:, 1], j] = True
        pairs.append(m.tolist())
    return flags, pairs


def integrate_ap(recall, precision):
    if not len(recall):
        return 0.
    mrec = np.concatenate(([0.], recall, [recall[-1]], [1.]))
    mpre = np.concatenate(([1.], precision, [0.], [0.]))
    envelope = np.maximum.accumulate(mpre[::-1])[::-1]
    return float(np.trapezoid(np.interp(np.linspace(0, 1, 101), mrec, envelope), np.linspace(0, 1, 101)))


def ap_from_rank(tp, n_gt, weights=None):
    """AP on a fixed confidence order; integer weights replicate rank entries.

    Weights are cluster multiplicities. Zero-weight entries must be removed:
    leaving them changes interpolation at duplicate recall coordinates.
    """
    if n_gt == 0:
        return np.full(10, np.nan)
    tp = np.asarray(tp, dtype=bool).reshape(-1, 10)
    if weights is None:
        weights = np.ones(len(tp), dtype=np.int64)
    weights = np.asarray(weights, dtype=np.int64)
    valid = weights > 0
    tp, weights = tp[valid], weights[valid]
    if not len(tp):
        return np.zeros(10)
    cumulative_tp = (tp * weights[:, None]).cumsum(0)
    cumulative_all = weights.cumsum()
    recall = cumulative_tp / (float(n_gt) + 1e-16)
    precision = cumulative_tp / cumulative_all[:, None]
    return np.array([integrate_ap(recall[:, j], precision[:, j]) for j in range(10)])


def rank_data(rows):
    """Preserve validator image/prediction order before native np.argsort ties."""
    gc, pc, cf, flags, image_indices = [], [], [], [], []
    for i, row in enumerate(rows):
        gc.extend(row['gt']['cls'])
        pc.extend(row['pred']['cls'])
        cf.extend(row['pred']['conf'])
        flags.append(np.asarray(row['ap_tp'], dtype=bool).reshape(-1, 10))
        image_indices.extend([i] * len(row['pred']['cls']))
    pc, cf = np.asarray(pc, dtype=np.int64), np.asarray(cf, dtype=np.float32)
    tp = np.concatenate(flags, axis=0) if flags else np.zeros((0, 10), dtype=bool)
    order = np.argsort(-cf)
    return (np.asarray(gc, dtype=np.int64), pc[order], cf[order], tp[order],
            np.asarray(image_indices, dtype=np.int64)[order])


def summarize_ap(rows):
    gc, pc, cf, tp, _ = rank_data(rows)
    per_class, curves = {}, {}
    for c, name in enumerate(NAMES):
        use = pc == c
        count = int((gc == c).sum())
        aps = ap_from_rank(tp[use], count)
        n = int(use.sum())
        tpc = tp[use].cumsum(0)
        fpc = (~tp[use]).cumsum(0)
        p = tpc / np.maximum(tpc + fpc, 1)
        r = tpc / count if count else np.zeros_like(tpc, dtype=float)
        curves[f'{name}_ap_scores'] = cf[use]
        curves[f'{name}_ap_precision'] = p
        curves[f'{name}_ap_recall'] = r
        curves[f'{name}_ap_tp'] = tpc
        curves[f'{name}_ap_fp'] = fpc
        per_class[name] = {'images': sum(c in row['gt']['cls'] for row in rows), 'instances': count,
                           'groups': len({row['group_id'] for row in rows if c in row['gt']['cls']}),
                           'predictions': n,
                           'ap50': float(aps[0]) if count else None,
                           'ap75': float(aps[5]) if count else None,
                           'ap50_95': float(aps.mean()) if count else None,
                           'ap_iou': aps.tolist() if count else None}
    valid = [v for v in per_class.values() if v['instances']]
    result = {'images': len(rows), 'empty_images': sum(not row['gt']['cls'] for row in rows),
              'groups': len({row['group_id'] for row in rows}), 'per_class': per_class,
              'map50': float(np.mean([v['ap50'] for v in valid])) if valid else None,
              'map75': float(np.mean([v['ap75'] for v in valid])) if valid else None,
              'map50_95': float(np.mean([v['ap50_95'] for v in valid])) if valid else None,
              'primary_ap50_95': np.mean([per_class[n]['ap50_95'] for n in ('pedestrian', 'car')])
              if all(per_class[n]['instances'] for n in ('pedestrian', 'car')) else None}
    return result, curves


def work_curve(rows, c):
    """Exact filter-then-rematch curve, merged from per-image score events."""
    delta_tp, delta_n = defaultdict(int), defaultdict(int)
    n_gt = 0
    for row in rows:
        gc, pc, cf, iou = arrays(row)
        n_gt += int((gc == c).sum())
        indices = np.flatnonzero(pc == c)
        scores, counts = np.unique(cf[indices], return_counts=True)
        previous = 0
        for score, n in zip(scores[::-1], counts[::-1]):
            chosen = indices[cf[indices] >= score]
            t = len(native_pairs(pc[chosen], gc, iou[:, chosen], .5)) if (gc == c).any() else 0
            delta_tp[float(score)] += t - previous
            delta_n[float(score)] += int(n)
            previous = t
    scores = np.array(sorted(delta_n, reverse=True), dtype=np.float64)
    tp = np.cumsum([delta_tp[float(s)] for s in scores], dtype=np.int64)
    n = np.cumsum([delta_n[float(s)] for s in scores], dtype=np.int64)
    return {'threshold': scores, 'tp': tp, 'fp': n-tp, 'fn': n_gt-tp,
            'precision': tp/n if len(n) else np.zeros(0),
            'recall': tp/n_gt if n_gt else np.full(len(tp), np.nan), 'gt': n_gt}


def counts(tp, fp, fn, threshold=None):
    return {'threshold': threshold, 'tp': int(tp), 'fp': int(fp), 'fn': int(fn),
            'precision': float(tp/(tp+fp)) if tp+fp else None,
            'recall': float(tp/(tp+fn)) if tp+fn else None}


def choose_point(curve, minimum_p, minimum_r):
    scores, p, r = curve['threshold'], curve['precision'], curve['recall']
    good = np.flatnonzero((p >= minimum_p) & np.isfinite(r))
    if len(good):
        best = max(good, key=lambda i: (r[i], p[i], scores[i]))
        origin, status = 'rule', 'passed' if r[best] >= minimum_r else 'recall_below_rule'
    elif len(scores):
        f1 = np.nan_to_num(2*p*r/(p+r+1e-16))
        best = max(range(len(scores)), key=lambda i: (f1[i], p[i], scores[i]))
        origin, status = 'diagnostic_f1', 'no_precision_feasible_point'
    else:
        return dict(counts(0, 0, curve['gt'], .001), t_rule=None, origin='output_floor', status='no_predictions')
    result = counts(curve['tp'][best], curve['fp'][best], curve['fn'][best], float(scores[best]))
    return dict(result, t_rule=result['threshold'] if origin == 'rule' else None, origin=origin, status=status)


def image_point(row, c, threshold, iou_level=.5):
    gc, pc, cf, iou = arrays(row)
    selected = np.flatnonzero((pc == c) & (cf >= threshold))
    m = native_pairs(pc[selected], gc, iou[:, selected], iou_level)
    if len(m):
        m[:, 1] = selected[m[:, 1]]
    return counts(len(m), len(selected)-len(m), int((gc == c).sum())-len(m), threshold), m, selected


def aggregate_point(rows, c, threshold):
    total = np.zeros(3, dtype=np.int64)
    for row in rows:
        v, _, _ = image_point(row, c, threshold)
        total += [v['tp'], v['fp'], v['fn']]
    return counts(*total, threshold=threshold)


def slice_tags(row):
    boxes = np.asarray(row['gt']['original_boxes'], dtype=np.float64).reshape(-1, 4)
    wh = boxes[:, 2:] - boxes[:, :2]
    # Frozen normalized labels have finite decimal precision. Coordinate subtraction
    # can turn a true 24px width into 24.000000000001 and incorrectly move 8px
    # reference objects across a slice boundary. This only removes sub-micropixel noise.
    w, h = np.round(wh[:, 0], 6), np.round(wh[:, 1], 6)
    area = w*h
    scale = 640/max(row['width'], row['height'])
    gg = box_iou(boxes, boxes)
    np.fill_diagonal(gg, 0)
    tags = []
    for i, b in enumerate(boxes):
        a = 'area_S' if area[i] < 32**2 else 'area_M' if area[i] < 96**2 else 'area_L'
        wr, hr = np.round(w[i]*scale, 6), np.round(h[i]*scale, 6)
        width = 'width_le8' if wr <= 8 else 'width_8_16' if wr <= 16 else 'width_gt16'
        height = 'height_le32' if hr <= 32 else 'height_32_64' if hr <= 64 else 'height_gt64'
        c = row['gt']['cls'][i]
        crowd = any(gg[i, j] >= .1 and row['gt']['cls'][j] == c for j in range(len(boxes)))
        edge = min(b[0], b[1], row['width']-b[2], row['height']-b[3]) <= .01
        tags.append(['all', a, width, height, 'crowded' if crowd else 'not_crowded', 'edge' if edge else 'not_edge'])
    return tags


def error_type(row, pred_id, pairs):
    gc, pc, _, iou = arrays(row)
    same = np.flatnonzero(gc == pc[pred_id])
    other = np.flatnonzero(gc != pc[pred_id])
    occupied = set(pairs[:, 0].tolist())
    duplicates = [g for g in same if g in occupied and iou[g, pred_id] >= .5]
    best_same = float(iou[same, pred_id].max()) if len(same) else 0.
    best_other = float(iou[other, pred_id].max()) if len(other) else 0.
    if duplicates:
        kind = 'duplicate'
    elif best_other >= .5:
        kind = 'classification'
    elif .1 <= best_same < .5:
        kind = 'localization'
    elif .1 <= best_other < .5:
        kind = 'classification_localization'
    elif max(best_same, best_other) < .1:
        kind = 'background_candidate'
    else:
        kind = 'matching_conflict'
    best_gt = int(iou[:, pred_id].argmax()) if len(gc) else None
    return {'pred_id': int(pred_id), 'type': kind, 'best_gt': best_gt,
            'best_same_iou': best_same, 'best_other_iou': best_other,
            'duplicate_gt': [int(g) for g in duplicates]}


def match_record(row, points):
    gc, pc, _, _ = arrays(row)
    record = {'key': row['key'], 'group_id': row['group_id'], 'ap_pairs': row['ap_pairs'], 'classes': {}}
    for c, name in enumerate(NAMES):
        threshold = points[name]['threshold']
        stats, m, selected = image_point(row, c, threshold)
        _, m75, _ = image_point(row, c, threshold, .75)
        used_g, used_p = set(m[:, 0]), set(m[:, 1])
        fp = [int(p) for p in selected if p not in used_p]
        fn = [int(g) for g in np.flatnonzero(gc == c) if g not in used_g]
        record['classes'][name] = dict(stats, pairs=m.tolist(), pairs75=m75.tolist(),
                                      fn_ids=fn, fp_items=[error_type(row, p, m) for p in fp])
    return record


def slice_statistics(rows, matches, purpose):
    acc = defaultdict(lambda: {'gt': 0, 'tp50': 0, 'tp75': 0, 'images': set(), 'groups': set()})
    for row, match in zip(rows, matches):
        if row['key'] != match['key']:
            raise ContractError('Slice object/match order mismatch')
        tags = slice_tags(row)
        for c, name in enumerate(NAMES):
            m = match['classes'][name]
            tp50 = {g for g, _ in m['pairs']}
            tp75 = {g for g, _ in m['pairs75']}
            for g, cls in enumerate(row['gt']['cls']):
                if cls != c:
                    continue
                for tag in tags[g]:
                    a = acc[name, tag]
                    a['gt'] += 1
                    a['tp50'] += g in tp50
                    a['tp75'] += g in tp75
                    a['images'].add(row['key']); a['groups'].add(row['group_id'])
    result = []
    for (name, tag), v in sorted(acc.items()):
        result.append({'purpose': purpose, 'class': name, 'slice': tag, 'gt': v['gt'],
                       'images': len(v['images']), 'groups': len(v['groups']), 'tp50': v['tp50'],
                       'tp75': v['tp75'], 'fn50': v['gt']-v['tp50'],
                       'recall50': v['tp50']/v['gt'], 'recall75': v['tp75']/v['gt'],
                       'low_support': v['gt'] < 30 or len(v['groups']) < 10})
    return result


def cluster_draws(group_ids, repeats=2000, seed=20261026):
    groups = sorted(set(group_ids))
    rng = np.random.default_rng(seed)
    for _ in range(repeats):
        yield groups, np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))


def bootstrap(rows, matches, repeats=2000, seed=20261026, progress=None):
    """Fixed-workpoint dev cluster bootstrap; no threshold fitting or model calls."""
    groups = sorted({r['group_id'] for r in rows})
    lookup = {g: i for i, g in enumerate(groups)}
    image_group = np.array([lookup[r['group_id']] for r in rows])
    gc, pc, _, tp, image_index = rank_data(rows)
    pred_groups = image_group[image_index]
    gt_groups = np.concatenate([np.repeat(image_group[i], len(r['gt']['cls'])) for i, r in enumerate(rows)])
    gt_counts = np.zeros((len(groups), 6), dtype=np.int64)
    if len(gc):
        np.add.at(gt_counts, (gt_groups, gc), 1)
    wp = np.zeros((len(groups), 6, 3), dtype=np.int64)
    slice_counts = defaultdict(lambda: np.zeros((len(groups), 3), dtype=np.int64))
    for row, match in zip(rows, matches):
        gi = lookup[row['group_id']]
        tags = slice_tags(row)
        for c, name in enumerate(NAMES):
            m = match['classes'][name]
            wp[gi, c] += [m['tp'], m['fp'], m['fn']]
            good50, good75 = {g for g, _ in m['pairs']}, {g for g, _ in m['pairs75']}
            for g, cls in enumerate(row['gt']['cls']):
                if cls == c:
                    for tag in tags[g]:
                        slice_counts[name, tag][gi] += [1, g in good50, g in good75]
    ranked = [(tp[pc == c], pred_groups[pc == c]) for c in range(6)]
    values = defaultdict(list)
    for iteration, (_, multiplicities) in enumerate(cluster_draws([r['group_id'] for r in rows], repeats, seed)):
        ngt = multiplicities @ gt_counts
        totals = np.tensordot(multiplicities, wp, axes=(0, 0))
        primary = []
        for c, name in enumerate(NAMES):
            ct, pg = ranked[c]
            aps = ap_from_rank(ct, int(ngt[c]), multiplicities[pg])
            for key, value in [('ap50', aps[0]), ('ap75', aps[5]), ('ap50_95', aps.mean())]:
                values[f'{name}/{key}'].append(float(value))
            t, f, n = totals[c]
            values[f'{name}/precision'].append(t/(t+f) if t+f else np.nan)
            values[f'{name}/recall'].append(t/(t+n) if t+n else np.nan)
            if c in (0, 2):
                primary.append(aps.mean())
        values['primary/ap50_95'].append(float(np.mean(primary)))
        for (name, tag), data in slice_counts.items():
            n, t50, t75 = multiplicities @ data
            values[f'{name}/{tag}/recall50'].append(t50/n if n else np.nan)
            values[f'{name}/{tag}/recall75'].append(t75/n if n else np.nan)
        if progress and (iteration+1) % 100 == 0:
            progress(iteration+1, repeats)
    intervals = {}
    for key, data in values.items():
        data = np.asarray(data, dtype=np.float64)
        good = data[np.isfinite(data)]
        intervals[key] = {'valid': len(good), 'undefined': int(repeats-len(good)),
                          'lower': float(np.quantile(good, .025)) if len(good) else None,
                          'upper': float(np.quantile(good, .975)) if len(good) else None,
                          'exploratory': len(good) < .95*repeats}
    return {'method': 'risk-group cluster bootstrap, fixed selected dev thresholds, percentile 95%',
            'repeats': repeats, 'seed': seed, 'groups': groups, 'singletons':
            sum(v == 1 for v in Counter(r['group_id'] for r in rows).values()),
            'rank_policy': 'native original confidence order; integer cluster multiplicities, zero entries removed',
            'limitations': ['conditional on selected thresholds and fixed checkpoint',
                            'does not include threshold selection or training-seed uncertainty',
                            'dev reused for threshold fitting; not independent final evaluation'],
            'intervals': intervals}
