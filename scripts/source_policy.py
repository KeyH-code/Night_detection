"""RULES-backed S3 eligibility, diagnostic role and fixed-horizon stopping state."""
from __future__ import annotations

import csv
import math
from pathlib import Path
import re

import numpy as np

import detection_metrics as dm
from source_contract import ContractError, NAMES, canonical_hash, check_equal, read_json, safe_path, sha256, write_json


def load_rules(path):
    text = Path(path).read_text(encoding='utf-8')
    rows = {}
    for name, ap, ap50, p, r in re.findall(
            r'\|\s*(pedestrian|cyclist|car|truck|tram|tricycle)\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)%\s*\|\s*P≥([\d.]+)%，R≥([\d.]+)%', text):
        if name in rows:
            raise ContractError('Duplicate RULES row: '+name)
        rows[name] = dict(zip(('ap50_95', 'ap50', 'precision', 'recall'),
                             [float(x)/100 for x in (ap, ap50, p, r)]))
    check_equal('six RULES rows', set(NAMES), set(rows))
    patterns = {
        'protection_drop': r'回退不得超过([\d.]+)个百分点',
        'recall_ci_width': r'区间总宽≤([\d.]+)个百分点',
        'slice_drop': r'切片Recall下降超过([\d.]+)个百分点',
        'ap75_drop': r'行人AP75下降超过([\d.]+)个百分点',
        'ap_gain': r'行人AP50:95提高([\d.]+)个百分点',
        'recall_gain': r'Recall提高([\d.]+)个百分点',
        'stale_epochs': r'连续([\d.]+)轮没有上述实质开发收益',
    }
    result = {'sha256': sha256(Path(path)), 'classes': rows}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if not match:
            raise ContractError('Missing RULES field: '+key)
        result[key] = int(match[1]) if key == 'stale_epochs' else float(match[1])/100
    return result


def reference_at(root):
    root = Path(root)
    files = ['runs/a_001/train/best_dev_metrics.json', 'runs/a_001_diag/dev/metrics.json',
             'runs/a_001_diag/slices.csv']
    historical = read_json(root/files[0])
    check_equal('historical source-dev scope', 'source_dev only', historical['scope'])
    result = {'historical_ap': {n: historical['metrics'][f'metrics/{n}_AP50-95'] for n in NAMES},
              'ap75': read_json(root/files[1])['per_class']['pedestrian']['ap75'],
              'slices': {}, 'identity': {p: sha256(root/p) for p in files}}
    with (root/files[2]).open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            if row['purpose'] == 'dev' and row['class'] == 'pedestrian' and row['slice'] in ('area_S', 'width_le8'):
                result['slices'][row['slice']] = float(row['recall50'])
    check_equal('reference slices', {'area_S', 'width_le8'}, set(result['slices']))
    return result


def summarize(rows, rules, points=None):
    if not rows or len({r['key'] for r in rows}) != len(rows):
        raise ContractError('Missing or duplicate evaluation images')
    purposes = {r['purpose'] for r in rows}
    if len(purposes) != 1 or not purposes <= {'source_train', 'source_dev'}:
        raise ContractError('Forbidden/mixed S3 evaluation purposes')
    caps = []
    audited = 0
    for row in rows:
        audit = row.get('nms_audit', {})
        if audit.get('pre_nms_truncated'):
            raise ContractError('NMS candidate truncation: '+row['purpose']+'/'+row['key']+
                                '; audit='+str(audit))
        if audit:
            audited += 1
            check_equal('fixed output max_det', 300, audit.get('max_det', 300))
            if audit.get('at_max_det'):
                if 'post_nms_count' not in audit or 'max_det_truncated' not in audit:
                    raise ContractError('Missing exact NMS cap replay: '+row['key'])
                caps.append({'key': row['key'], 'post_nms_count': audit['post_nms_count'],
                             'removed': audit['post_nms_count']-audit['final_count'],
                             'truncated': audit['max_det_truncated']})
    metrics, curves = dm.summarize_ap(rows)
    if points is None:
        points = {}
        for c, name in enumerate(NAMES):
            rule = rules['classes'][name]
            curve = dm.work_curve(rows, c)
            points[name] = dm.choose_point(curve, rule['precision'], rule['recall'])
            curves.update({f'{name}_work_{k}': v for k, v in curve.items() if k != 'gt'})
    else:
        # Reuse the exact dev thresholds, including explicit diagnostic points.
        points = {n: dict(dm.aggregate_point(rows, c, points[n]['threshold']),
                          t_rule=points[n]['t_rule'], origin=points[n]['origin'],
                          status='dev_threshold_applied') for c, n in enumerate(NAMES)}
    matches = [dm.match_record(r, points) for r in rows]
    slices = dm.slice_statistics(rows, matches, rows[0]['purpose'])
    # The fixed target union is an extra mechanism slice; original S2 slices remain unchanged.
    union = {'gt': 0, 'tp50': 0, 'tp75': 0, 'images': set(), 'groups': set()}
    for row, match in zip(rows, matches):
        tags = dm.slice_tags(row)
        hits = [{g for g, _ in match['classes']['pedestrian'][k]} for k in ('pairs', 'pairs75')]
        for g, c in enumerate(row['gt']['cls']):
            if c == 0 and ('area_S' in tags[g] or 'width_le8' in tags[g]):
                union['gt'] += 1
                union['tp50'] += g in hits[0]
                union['tp75'] += g in hits[1]
                union['images'].add(row['key']); union['groups'].add(row['group_id'])
    if union['gt']:
        slices.append({'purpose': rows[0]['purpose'], 'class': 'pedestrian', 'slice': 'small_thin_union',
                       'gt': union['gt'], 'images': len(union['images']), 'groups': len(union['groups']),
                       'tp50': union['tp50'], 'tp75': union['tp75'], 'fn50': union['gt']-union['tp50'],
                       'recall50': union['tp50']/union['gt'], 'recall75': union['tp75']/union['gt'],
                       'low_support': union['gt'] < 30 or len(union['groups']) < 10})
    nms = {'max_det': 300, 'audited_images': audited, 'images_at_cap': len(caps),
           'truncated_images': sum(r['truncated'] for r in caps),
           'removed_boxes': sum(r['removed'] for r in caps),
           'max_post_nms_count': max((r.get('nms_audit', {}).get('post_nms_count', 0) for r in rows), default=0),
           'cap_images': caps, 'interpretation': 'declared output budget; counts alone do not identify lost GT'}
    return {'metrics': metrics, 'workpoints': points, 'slices': slices, 'nms': nms,
            'metric_scope': 'conf>0.001; class-aware NMS IoU0.7; max_det300 output'}, matches, curves


def recall_intervals(rows, matches, repeats=2000, seed=20261026):
    groups = sorted({r['group_id'] for r in rows})
    lookup = {g: i for i, g in enumerate(groups)}
    values = np.zeros((len(groups), 6, 2), dtype=np.int64)
    check_equal('CI record order', [r['key'] for r in rows], [r['key'] for r in matches])
    for row, match in zip(rows, matches):
        for c, name in enumerate(NAMES):
            m = match['classes'][name]
            values[lookup[row['group_id']], c] += [m['tp'], m['tp']+m['fn']]
    draws = np.array([m for _, m in dm.cluster_draws(groups, repeats, seed)])
    totals = np.einsum('rg,gct->rct', draws, values)
    result = {}
    for c, name in enumerate(NAMES):
        good = totals[:, c, 1] > 0
        r = totals[good, c, 0]/totals[good, c, 1]
        result[name] = {'lower': float(np.quantile(r, .025)) if len(r) else None,
                        'upper': float(np.quantile(r, .975)) if len(r) else None,
                        'valid': len(r), 'undefined': int(repeats-len(r)),
                        'exploratory': len(r) < .95*repeats}
    return {'method': 'fixed-dev-workpoint risk-group percentile 95%', 'groups': groups,
            'repeats': repeats, 'seed': seed, 'intervals': result}


class SelectionPolicy:
    def __init__(self, rules, reference, horizon=100):
        self.rules, self.reference, self.horizon = rules, reference, horizon
        self.state = {'epoch': 0, 'best': None, 'diagnostic': None,
                      'ap_anchor': None, 'recall_anchor': None, 'last_meaningful_epoch': 0}

    def assess(self, record):
        checks, guards = {}, {}
        for name in NAMES:
            result, point = record['metrics']['per_class'][name], record['workpoints'][name]
            if not result['instances'] or not result['groups']:
                raise ContractError('Missing dev class support: '+name)
            for field in ('ap50', 'ap75', 'ap50_95'):
                value = result[field]
                if value is None or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ContractError('Invalid '+name+'/'+field)
            rule = self.rules['classes'][name]
            floor = rule['ap50_95']
            if name in ('cyclist', 'car', 'truck', 'tram'):
                floor = max(floor, self.reference['historical_ap'][name]-self.rules['protection_drop'])
                guards[name+'/ap50_95'] = result['ap50_95'] >= floor
            checks[name+'/ap50_95'] = result['ap50_95'] >= floor
            checks[name+'/ap50'] = result['ap50'] >= rule['ap50']
            checks[name+'/workpoint'] = (point['origin'] == 'rule' and point['t_rule'] is not None
                                          and point['precision'] is not None and point['recall'] is not None
                                          and point['tp']+point['fp'] > 0
                                          and point['precision'] >= rule['precision'] and point['recall'] >= rule['recall'])
        for tag in ('area_S', 'width_le8'):
            row = next((r for r in record['slices'] if r['class'] == 'pedestrian' and r['slice'] == tag), None)
            guards['pedestrian/'+tag] = bool(row and row['recall50'] >= self.reference['slices'][tag]-self.rules['slice_drop'])
        guards['pedestrian/ap75'] = record['metrics']['per_class']['pedestrian']['ap75'] >= self.reference['ap75']-self.rules['ap75_drop']
        return checks, guards

    def update(self, epoch, record, interval_factory):
        check_equal('consecutive policy epoch', self.state['epoch']+1, epoch)
        checks, guards = self.assess(record)
        declared = record['metrics']['primary_ap50_95']
        score = sum(record['metrics']['per_class'][n]['ap50_95'] for n in ('pedestrian', 'car'))/2
        if declared is None or not math.isfinite(declared) or not math.isclose(score, declared, rel_tol=0, abs_tol=1e-15):
            raise ContractError('Invalid primary AP')
        numeric = all(checks.values()) and all(guards.values())
        best = self.state['best']
        contender = numeric and (best is None or score > best['score'])
        intervals, evidence = None, {}
        if contender:
            intervals = interval_factory()
            check_equal('CI repeat identity', 2000, intervals['repeats'])
            check_equal('CI seed identity', 20261026, intervals['seed'])
            for name in NAMES:
                ci = intervals['intervals'][name]
                evidence[name] = (not ci['exploratory'] and ci['valid'] >= .95*intervals['repeats']
                                  and ci['lower'] is not None and ci['upper'] is not None
                                  and math.isfinite(ci['lower']) and math.isfinite(ci['upper'])
                                  and 0 <= ci['lower'] <= ci['upper'] <= 1
                                  and ci['upper']-ci['lower'] <= self.rules['recall_ci_width'])
        eligible = contender and all(evidence.values())
        new_best = bool(eligible)
        diagnostic = self.state['diagnostic']
        new_diagnostic = diagnostic is None or score > diagnostic['score']
        if new_best:
            self.state['best'] = {'epoch': epoch, 'score': score, 'status': 'single_run_eligible_pending_repeats'}
        if new_diagnostic:
            self.state['diagnostic'] = {'epoch': epoch, 'score': score, 'status': 'diagnostic_not_qualification'}
        point = record['workpoints']['pedestrian']
        ap = record['metrics']['per_class']['pedestrian']['ap50_95']
        recall = point['recall'] if point['origin'] == 'rule' else None
        meaningful = False
        if all(guards.values()):
            if self.state['ap_anchor'] is None and self.state['last_meaningful_epoch'] == 0:
                self.state['last_meaningful_epoch'] = epoch
            for field, value, gain in [('ap_anchor', ap, self.rules['ap_gain']),
                                        ('recall_anchor', recall, self.rules['recall_gain'])]:
                if value is None:
                    continue
                if self.state[field] is None:
                    self.state[field] = value
                elif value-self.state[field] >= gain:
                    self.state[field] = value
                    meaningful = True
            if meaningful:
                self.state['last_meaningful_epoch'] = epoch
        self.state['epoch'] = epoch
        stale = epoch-self.state['last_meaningful_epoch']
        return {'epoch': epoch, 'score': score, 'checks': checks, 'guards': guards,
                'numeric_pass': numeric, 'evidence': evidence, 'intervals': intervals,
                'ci_reason': 'numeric_contender' if contender else 'numeric_failure' if not numeric else 'not_strict_best_contender',
                'new_best': new_best, 'new_diagnostic': new_diagnostic,
                'failure_reasons': [k for k, v in {**checks, **guards}.items() if not v]
                                   + ['evidence/'+k for k, v in evidence.items() if not v],
                'best': self.state['best'], 'diagnostic': self.state['diagnostic'],
                'meaningful_improvement': meaningful, 'stale_epochs': stale,
                'stale_limit_reached': stale >= self.rules['stale_epochs'],
                'stop_policy': 'horizon_only', 'stop': epoch >= self.horizon,
                'scope': 'source_dev single-run policy; not recipe acceptance or freeze'}


class HorizonStopping:
    """Explicitly replaces upstream/legacy patience, including the unsafe patience=0 interpretation."""
    def __init__(self, horizon=100):
        self.horizon, self.possible_stop = horizon, False

    def __call__(self, epoch, fitness):
        if fitness is None or not math.isfinite(fitness):
            raise ContractError('Nonfinite fitness; no automatic NaN recovery')
        self.possible_stop = epoch+1 >= self.horizon
        return epoch >= self.horizon


def paired_effects(pairs, rules, repeats=2000, seed=20261026, points_by_seed=None):
    """Paired group uncertainty conditional on one or three preregistered training seeds.

    pairs is [(training_seed, reference_rows, sampled_rows), ...]. Thresholds are
    selected once on each full dev, never inside the bootstrap. All seed/arm
    replicas share each group draw; seed variation is reported separately.
    """
    check_equal('paired seeds', [17] if len(pairs) == 1 else [17, 29, 43], [p[0] for p in pairs])
    first = pairs[0][1]
    identity = [(r['key'], r['group_id'], r['gt']) for r in first]
    groups = sorted({r['group_id'] for r in first})
    group_index = {g: i for i, g in enumerate(groups)}
    per_seed, prepared = [], []
    for training_seed, reference, candidate in pairs:
        arms, point_values = [], []
        for arm_index, rows in enumerate((reference, candidate)):
            if any(r['purpose'] != 'source_dev' for r in rows):
                raise ContractError('Paired effects use source_dev only')
            check_equal('paired exact GT/group/image order', identity, [(r['key'], r['group_id'], r['gt']) for r in rows])
            fixed = points_by_seed[training_seed][arm_index] if points_by_seed is not None else None
            record, matches, _ = summarize(rows, rules, fixed)
            gc, pc, _, flags, image = dm.rank_data(rows)
            image_groups = np.array([group_index[r['group_id']] for r in rows])
            gt_counts = np.zeros((len(groups), 6), dtype=np.int64)
            wp = np.zeros((len(groups), 6, 2), dtype=np.int64)
            slices = np.zeros((len(groups), 3, 2), dtype=np.int64)
            for row, match in zip(rows, matches):
                g = group_index[row['group_id']]
                for c, name in enumerate(NAMES):
                    gt_counts[g, c] += row['gt']['cls'].count(c)
                    m = match['classes'][name]
                    wp[g, c] += [m['tp'], m['tp']+m['fn']]
                hits = {i for i, _ in match['classes']['pedestrian']['pairs']}
                for i, tags in enumerate(dm.slice_tags(row)):
                    if row['gt']['cls'][i] == 0:
                        use = ['area_S' in tags, 'width_le8' in tags,
                               'area_S' in tags or 'width_le8' in tags]
                        for j, included in enumerate(use):
                            if included:
                                slices[g, j] += [i in hits, 1]
            feasible = np.array([record['workpoints'][n]['origin'] == 'rule'
                and record['workpoints'][n]['precision'] is not None
                and record['workpoints'][n]['precision'] >= rules['classes'][n]['precision'] for n in NAMES])
            ranked = [(flags[pc == c], image_groups[image[pc == c]]) for c in range(6)]
            arms.append((gt_counts, wp, slices, feasible, ranked))
            value = {n+'/'+key: record['metrics']['per_class'][n][key]
                     for n in NAMES for key in ('ap50', 'ap75', 'ap50_95')}
            value.update({n+'/recall': record['workpoints'][n]['recall'] if feasible[c] else None for c, n in enumerate(NAMES)})
            value.update({'pedestrian/'+s['slice']+'/recall': s['recall50']
                          if feasible[0] else None for s in record['slices']
                          if s['class'] == 'pedestrian' and s['slice'] in ('area_S', 'width_le8', 'small_thin_union')})
            point_values.append(value)
        delta = {k: point_values[1][k]-v if v is not None and point_values[1].get(k) is not None else None
                 for k, v in point_values[0].items()}
        per_seed.append({'seed': training_seed, 'reference': point_values[0], 'candidate': point_values[1], 'delta': delta})
        prepared.append(arms)
    samples = {k: [] for k in per_seed[0]['delta']}
    for _, mult in dm.cluster_draws(groups, repeats, seed):
        seed_values = []
        for arms in prepared:
            values = []
            for gt, wp, slices, feasible, ranked in arms:
                ngt, totals, subset = mult@gt, np.einsum('g,gct->ct', mult, wp), np.einsum('g,gst->st', mult, slices)
                value = {}
                for c, name in enumerate(NAMES):
                    tp, pg = ranked[c]
                    ap = dm.ap_from_rank(tp, int(ngt[c]), mult[pg])
                    value[name+'/ap50'] = float(ap[0])
                    value[name+'/ap75'] = float(ap[5])
                    value[name+'/ap50_95'] = float(ap.mean())
                    value[name+'/recall'] = totals[c, 0]/totals[c, 1] if feasible[c] and totals[c, 1] else np.nan
                for j, tag in enumerate(('area_S', 'width_le8', 'small_thin_union')):
                    value['pedestrian/'+tag+'/recall'] = subset[j, 0]/subset[j, 1] if feasible[0] and subset[j, 1] else np.nan
                values.append(value)
            seed_values.append({k: values[1].get(k, np.nan)-v for k, v in values[0].items()})
        for key in samples:
            samples[key].append(float(np.mean([v[key] for v in seed_values])))
    effect = {}
    for key, data in samples.items():
        good = np.asarray(data)[np.isfinite(data)]
        deltas = [s['delta'][key] for s in per_seed]
        effect[key] = {'mean_delta': float(np.mean(deltas)) if all(v is not None for v in deltas) else None,
                       'seed_spread': float(np.ptp(deltas)) if all(v is not None for v in deltas) else None,
                       'lower': float(np.quantile(good, .025)) if len(good) else None,
                       'upper': float(np.quantile(good, .975)) if len(good) else None,
                       'valid': len(good), 'undefined': repeats-len(good), 'exploratory': len(good) < .95*repeats}
    ap, recall = effect['pedestrian/ap50_95'], effect['pedestrian/recall']
    excluded = (len(pairs) == 3 and not ap['exploratory'] and not recall['exploratory']
                and ap['upper'] is not None and recall['upper'] is not None
                and ap['upper'] < rules['ap_gain'] and recall['upper'] < rules['recall_gain'])
    return {'per_seed': per_seed, 'effects': effect, 'groups': groups, 'repeats': repeats, 'seed': seed,
            'threshold_policy': 'saved cloud dev thresholds' if points_by_seed is not None else 'full-dev selected thresholds',
            'practical_benefit_excluded': bool(excluded),
            'limitations': ['conditional on selected checkpoints and full-dev thresholds',
                            'three fixed seeds are not a distribution of training seeds',
                            'paired group CI excludes checkpoint/threshold selection and seed-population uncertainty']}


def compare_runs(pairs, output, rules_path):
    """Offline report from sealed formal R/S runs; never infer, train or choose a lucky seed.

    pairs: [(17, Path(reference_run), Path(sampled_run)), ...], either the first
    seed or all three preregistered seeds. Output is created only after validation.
    """
    import copy
    import json
    from source_release import audit_training_roles
    check_equal('comparison seeds', [17] if len(pairs) == 1 else [17,29,43], [r[0] for r in pairs])
    rules = load_rules(rules_path)
    metrics_rows, evidence, common, fixed_points, local_environment = [], [], None, {}, None
    for seed, reference, candidate in pairs:
        objects, points = [], []
        for arm, folder in [('R', reference), ('S', candidate)]:
            folder = Path(folder)
            manifest = read_json(folder/'manifest.json')
            check_equal('comparison requires completed formal run', 'completed_formal', manifest['status'])
            for rel, entry in manifest['files'].items():
                check_equal('comparison evidence '+rel, entry['sha256'], sha256(safe_path(folder, rel)))
            identity = manifest['identity']
            config = read_json(folder/'audit_initial/config_resolved.json')
            check_equal('comparison config bytes', identity['config_sha256'], canonical_hash(config))
            check_equal('comparison arm/seed', [arm, seed], [config['sampling']['arm'], config['training']['seed']])
            check_equal('comparison RULES identity', rules['sha256'], identity['s3_policy']['rules_sha256'])
            runtime = read_json(folder/'audit_initial/resource_gate.json')['runtime_identity']
            check_equal('comparison runtime identity', identity['runtime_identity_sha256'], runtime['identity_sha256'])
            normal_config, normal_identity, normal_runtime = copy.deepcopy(config), copy.deepcopy(identity), copy.deepcopy(runtime)
            normal_config['sampling'].pop('arm'); normal_config['training'].pop('seed')
            for key in ('config_sha256', 'runtime_identity_sha256'):
                normal_identity.pop(key)
            for key in ('arm','seed','resolved_config_sha256','identity_sha256'):
                normal_runtime.pop(key)
            normalized = [normal_config, normal_identity, normal_runtime]
            if common is None:
                common = normalized
            else:
                check_equal('paired common recipe/data/code/environment', common, normalized)
            checked = audit_training_roles(folder)
            if checked['evaluation_status'] == 'pending_local_evaluation':
                raise ContractError('Comparison requires completed local evaluation')
            final = read_json(folder/'final_eval/run.json')
            if checked['evaluation_status'] == 'complete_local':
                if local_environment is None:
                    local_environment = final['local_environment']
                else:
                    check_equal('paired local evaluation environment', local_environment, final['local_environment'])
            monitor = read_json(folder/'train_monitor.json')
            check_equal('comparison full batch budget', 25000, monitor['gradient_batches'])
            selected = checked['roles'][checked['selected_role']]
            dev = read_json(folder/'final_eval/source_dev/metrics.json')
            points.append(dev['workpoints'])
            row = {'seed': seed, 'arm': arm, 'run': str(folder), 'role': checked['selected_role'],
                   'selected_epoch': selected['epoch'], 'qualification': checked['qualification'],
                   'pretrained_sha256': identity['pretrained_sha256'], 'ready_sha256': identity['ready_sha256'],
                   'config_sha256': identity['config_sha256'], 'code_sha256': canonical_hash(identity['code']),
                   'checkpoint_sha256': selected['sha256'], 'gradient_batches': monitor['gradient_batches'],
                   'optimizer_steps': monitor['optimizer_steps'], 'amp_skipped_steps': monitor['amp_skipped_steps']}
            for name in NAMES:
                for key in ('ap50', 'ap75', 'ap50_95', 'instances', 'images', 'groups'):
                    row[name+'/'+key] = dev['metrics']['per_class'][name][key]
                for key in ('threshold','origin','precision','recall','tp','fp','fn'):
                    row[name+'/'+key] = dev['workpoints'][name][key]
            for part in dev['slices']:
                if part['class'] == 'pedestrian' and part['slice'] in ('area_S','width_le8','small_thin_union'):
                    row['pedestrian/'+part['slice']+'/recall'] = part['recall50']
            times = [json.loads(line) for line in (folder/'timings.jsonl').read_text(encoding='utf-8').splitlines() if line]
            row['recorded_epoch_hours'] = sum(v['epoch_seconds'] for v in times)/3600
            metrics_rows.append(row)
            objects.append(dm.read_rows(folder/'final_eval/source_dev/objects.jsonl.gz'))
        evidence.append((seed, *objects))
        fixed_points[seed] = points
    result = paired_effects(evidence, rules, points_by_seed=fixed_points)
    result['runs'] = metrics_rows
    result['cost_scope'] = 'recorded completed epochs only; excludes setup, final export, preflight, failed work and billing idle'
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'comparison.json', result)
    with (output/'comparison.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics_rows[0]))
        writer.writeheader(); writer.writerows(metrics_rows)
    return result
