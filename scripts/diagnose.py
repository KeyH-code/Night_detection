"""Fixed source-train/dev checkpoint diagnostics; never train or open held-out data."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import contextlib
import csv
import gzip
import hashlib
import html
import importlib.metadata as metadata
import inspect
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

from evaluate import metric_record, stage_subset, validator_args
from source_contract import (ContractError, NAMES, canonical_hash, check_equal, read_json,
                             safe_path, select_records, sha256, verify_source, write_json)
import detection_metrics as dm
import diagnostic_review as dr

ROOT = Path(__file__).resolve().parents[1]
PURPOSES = ('source_train', 'source_dev')
READY_SHA = '72e75ebd99014c830e0227061e390e6aed768ae724b532a5c2c90fd8dec93e85'
BEST_SHA = '9db1ad388d28ecf3d912feb8e686056d497acde705197a5156a320579dc18bd7'


from source_policy import load_rules


def isolate(scratch):
    scratch = Path(scratch).resolve()
    if scratch.is_relative_to(ROOT):
        raise ContractError('Diagnostic scratch must be outside the repository')
    scratch.mkdir(parents=True, exist_ok=True)
    for name in ('YOLO_CONFIG_DIR', 'MPLCONFIGDIR', 'TORCH_HOME', 'TEMP', 'TMP', 'TMPDIR'):
        folder = scratch / name.lower()
        folder.mkdir(exist_ok=True)
        os.environ[name] = str(folder)
    os.environ.update(YOLO_AUTOINSTALL='false', WANDB_MODE='disabled', PYTHONDONTWRITEBYTECODE='1')
    sys.dont_write_bytecode = True
    tempfile.tempdir = str(scratch)
    return scratch


def assert_purpose(purpose):
    if purpose not in PURPOSES:
        raise ContractError(f'Forbidden diagnostic purpose: {purpose}')


def subset(contract, purpose, records=None):
    assert_purpose(purpose)
    return {'root': contract['root'], 'ready': contract['ready'], 'ready_hash': contract['ready_sha256'],
            'purpose': purpose, 'records': contract['manifests'][purpose] if records is None else records,
            'manifest': contract['ready']['manifests'][purpose]}


def environment():
    import torch
    expected = {'torch': '2.14.0+cu130', 'torchvision': '0.29.0+cu130', 'ultralytics': '8.4.171'}
    actual = {name: metadata.version(name) for name in (*expected, 'numpy')}
    for name, version in expected.items():
        check_equal(name, version, actual[name])
    check_equal('Python major/minor', (3, 12), sys.version_info[:2])
    if not torch.cuda.is_available():
        raise ContractError('CUDA unavailable; refusing an implicit CPU/precision fallback')
    from ultralytics.utils.torch_utils import init_seeds
    init_seeds(17, deterministic=True)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    free, total = torch.cuda.mem_get_info(0)
    from ultralytics.engine import validator
    from ultralytics.models.yolo.detect import val
    from ultralytics.utils import metrics, nms
    return {'python': sys.version, 'executable': sys.executable, 'packages': actual,
            'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda, 'free_vram': free, 'total_vram': total,
            'deterministic': torch.are_deterministic_algorithms_enabled(),
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'tf32_matmul': torch.backends.cuda.matmul.allow_tf32, 'tf32_cudnn': torch.backends.cudnn.allow_tf32,
            'library_code': {str(Path(m.__file__).relative_to(Path(metadata.distribution('ultralytics').locate_file('')))).replace('\\', '/'):
                             sha256(Path(m.__file__)) for m in (validator, val, metrics, nms)}}


# Reusable export/matcher belongs to evaluate; review/trace tools remain local-only.
from evaluate import original_gt, pack_image


def validator_class():
    from evaluate import evidence_validator_class
    return evidence_validator_class(trace_fn=trace_image)


def run_validator(contract, weights, scratch, evidence=True, stream=None, trace_context=None):
    from ultralytics.models.yolo.detect import DetectionValidator
    data = stage_subset(contract, scratch / 'data')
    args = validator_args(data)
    args.update(cache=False, fraction=1., single_cls=False, seed=17, deterministic=True)
    cls = validator_class() if evidence else DetectionValidator
    extra = dict(records=contract['records'], source_root=contract['root'], stream=stream,
                 trace_context=trace_context) if evidence else {}
    validator = cls(args=args, save_dir=scratch/'validator', **extra)
    begin = time.perf_counter()
    validator(model=str(weights.resolve()))
    check_equal('seen', len(contract['records']), validator.seen)
    check_equal('no augmentation', False, validator.dataloader.dataset.augment)
    check_equal('max_det', 300, validator.args.max_det)
    if evidence:
        check_equal('exact emitted key set', {r['key'] for r in contract['records']}, validator.keys)
        offline, _ = dm.summarize_ap(validator.rows)
        for c, ap in zip(validator.metrics.box.ap_class_index, validator.metrics.box.all_ap):
            if not np.allclose(ap, offline['per_class'][NAMES[int(c)]]['ap_iou'], rtol=0, atol=1e-12):
                raise ContractError('Offline/native aggregate AP mismatch')
    return validator, time.perf_counter()-begin


def snapshot_code(output):
    sources = ['scripts/diagnose.py', 'scripts/detection_metrics.py', 'scripts/diagnostic_review.py', 'scripts/evaluate.py', 'scripts/source_policy.py',
               'scripts/source_contract.py', 'scripts/plot.py', 'tests/train/diagnose/test_diagnose.py',
               'tests/train/diagnose/test_detection_metrics.py', 'tests/train/diagnose/test_review.py',
               'environment/requirements-windows.lock']
    identity = {}
    for rel in sources:
        source = ROOT / rel
        target = output/'code'/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        identity[rel] = sha256(source)
    state = subprocess.run(['git', '-c', 'core.quotepath=false', 'status', '--porcelain'], cwd=ROOT,
                           capture_output=True, text=True, encoding='utf-8', check=True).stdout
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True,
                          text=True, encoding='utf-8', check=True).stdout.strip()
    return {'head': head, 'dirty_status': state, 'code': identity}


def collect(args):
    scratch_root = isolate(args.scratch)
    check_equal('checkpoint SHA', args.weights_sha256, sha256(args.weights))
    contract = verify_source(args.data_root, args.ready_sha256)
    env = environment()
    rules = load_rules(ROOT/'RULES.md')
    output = args.output.resolve()
    if output.is_relative_to(args.data_root.resolve()):
        raise ContractError('Output inside frozen data')
    output.mkdir(parents=True, exist_ok=False)
    before = {rel: (safe_path(contract['root'], rel).stat().st_mtime_ns,
                    safe_path(contract['root'], rel).stat().st_size) for rel in contract['files']}
    protocol = {'schema_version': 1, 'scope': list(PURPOSES), 'rules': rules,
                'data_fingerprint': contract['ready']['data_fingerprint'],
                'grouping_protocol': contract['ready']['grouping_protocol'],
                'manifests': contract['ready']['manifests'],
                'settings': {k: v for k, v in validator_args('sandbox').items() if k != 'data'},
                'matching': 'locked native IoU-sorted matching; workpoints filter then rematch at IoU0.5',
                'workpoint_ties': 'max recall, then max precision, then max confidence; whole score ties retained',
                'bootstrap': {'seed': 20261026, 'repeats': 2000, 'thresholds': 'fixed dev-selected'},
                'size_slices': {'area': [1024, 9216], 'width640': [8, 16], 'height640': [32, 64]},
                'approved_plan_sha256': sha256(ROOT/'specs/s2/plan.md')}
    write_json(output/'protocol.json', protocol)
    run = {'schema_version': 1, 'status': 'collecting', 'training_performed': False, 'final_test_executed': False,
           'protocol_sha256': sha256(output/'protocol.json'),
           'checkpoint_sha256': args.weights_sha256, 'ready_sha256': args.ready_sha256,
           'weights': args.weights.resolve().as_posix(), 'data_root': args.data_root.resolve().as_posix(),
           'environment': env, 'identity': snapshot_code(output), 'splits': {}, 'started_utc': time.time()}
    write_json(output/'run.json', run)
    try:
        with model_log(output/'evaluation.log') as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                # Engineering equivalence only, not a new official model score.
                chosen = select_records(contract['manifests']['source_dev'], 12, seed=17)
                qa = subset(contract, 'source_dev', chosen)
                with tempfile.TemporaryDirectory(prefix='qa_', dir=scratch_root) as tmp:
                    base, _ = run_validator(qa, args.weights, Path(tmp)/'base', evidence=False)
                    extended, _ = run_validator(qa, args.weights, Path(tmp)/'export')
                    if not np.allclose(base.metrics.box.all_ap, extended.metrics.box.all_ap, rtol=0, atol=1e-12):
                        raise ContractError('Native/exporter real-subset AP differs')
                    run['equivalence'] = {'images': len(chosen), 'native_export_ap_max_abs_diff':
                                          float(np.max(np.abs(base.metrics.box.all_ap-extended.metrics.box.all_ap))),
                                          'per_image_native_tp_equal': True, 'scope': 'engineering QA only'}
                    del base, extended
                for purpose in PURPOSES:
                    folder = output/('train' if purpose == 'source_train' else 'dev')
                    folder.mkdir()
                    started = time.perf_counter()
                    with tempfile.TemporaryDirectory(prefix=purpose+'_', dir=scratch_root) as tmp:
                        with gzip.open(folder/'objects.jsonl.gz', 'wt', encoding='utf-8') as stream:
                            val, seconds = run_validator(subset(contract, purpose), args.weights, Path(tmp), stream=stream)
                        record = metric_record(val.metrics)
                        for c, ap in zip(val.metrics.box.ap_class_index, val.metrics.box.all_ap):
                            record['per_class'][NAMES[int(c)]]['ap75'] = float(ap[5])
                        write_json(folder/'native.json', record)
                        run['splits'][purpose] = {'images': val.seen, 'validation_seconds': seconds,
                                                  'stage_and_validation_seconds': time.perf_counter()-started,
                                                  'actual_args': json.loads(json.dumps(vars(val.args), default=str)),
                                                  'objects_sha256': sha256(folder/'objects.jsonl.gz'),
                                                  'native_sha256': sha256(folder/'native.json'),
                                                  'nms_cap_images': [r['key'] for r in val.rows if r['nms_audit']['at_max_det']],
                                                  'pre_nms_truncated_images': [r['key'] for r in val.rows if r['nms_audit']['pre_nms_truncated']]}
                        del val
                    write_json(output/'run.json', run)
        text = (output/'evaluation.log').read_text(encoding='utf-8')
        if 'NMS time limit' in text:
            raise ContractError('Native NMS timeout; collection cannot be promoted')
        check_equal('checkpoint SHA after', args.weights_sha256, sha256(args.weights))
        verify_source(args.data_root, args.ready_sha256)
        after = {rel: (safe_path(contract['root'], rel).stat().st_mtime_ns,
                       safe_path(contract['root'], rel).stat().st_size) for rel in contract['files']}
        check_equal('frozen whitelist size/mtime unchanged', before, after)
        run.update(status='collected', inputs_unchanged=True, collected_utc=time.time())
        write_json(output/'run.json', run)
    except BaseException as exc:
        run.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(output/'run.json', run)
        raise
    return run


def verify_collected(output):
    run = read_json(output/'run.json')
    if run['status'] not in ('collected', 'analyzed', 'traced', 'awaiting_review', 'complete'):
        raise ContractError('Collection is incomplete/failed')
    protocol = read_json(output/'protocol.json')
    check_equal('protocol SHA', run['protocol_sha256'], sha256(output/'protocol.json'))
    for purpose, stem in zip(PURPOSES, ('train', 'dev')):
        check_equal('objects SHA', run['splits'][purpose]['objects_sha256'], sha256(output/stem/'objects.jsonl.gz'))
        check_equal('native SHA', run['splits'][purpose]['native_sha256'], sha256(output/stem/'native.json'))
    return run, protocol


def bind_stage(output, run, stage):
    """Bind actual postprocessing code separately without rewriting collection snapshots."""
    references = {}
    sources = dict(run['identity']['code'])
    sources.setdefault('scripts/diagnostic_review.py', None)
    sources.setdefault('scripts/source_policy.py', None)
    sources.setdefault('tests/train/diagnose/test_review.py', None)
    for relative, original_hash in sources.items():
        source = ROOT/relative
        digest = sha256(source)
        if digest == original_hash:
            snapshot = 'code/'+relative
        else:
            snapshot = 'code/versions/'+digest[:12]+'/'+relative
            path = output/snapshot
            if path.exists():
                check_equal('existing code version', digest, sha256(path))
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, path)
        references[relative] = {'sha256': digest, 'snapshot': snapshot}
    run.setdefault('stage_code', {})[stage] = references
    write_json(output/'run.json', run)
    return references


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


@contextlib.contextmanager
def model_log(path, mode='w'):
    # LOGGER handlers keep their original stream across redirect_stdout/stderr.
    # Attach explicitly so native warnings and summaries enter retained evidence.
    from ultralytics.utils import LOGGER
    with path.open(mode, encoding='utf-8') as stream:
        handler = logging.StreamHandler(stream)
        LOGGER.addHandler(handler)
        try:
            yield stream
        finally:
            LOGGER.removeHandler(handler)


def analyze(args):
    output = args.output.resolve()
    isolate(args.scratch)
    run, protocol = verify_collected(output)
    bind_stage(output, run, 'analysis')
    started = time.perf_counter()
    run['analysis_started_utc'] = time.time()
    if (output/'cases.jsonl').is_file() and any(c['status'] == 'reviewed' for c in read_cases(output)):
        raise ContractError('Refusing to overwrite existing human judgments during reanalysis')
    write_json(output/'run.json', run)
    datasets = {stem: dm.read_rows(output/stem/'objects.jsonl.gz') for stem in ('train', 'dev')}
    rules = protocol['rules']['classes']
    summaries, curves = {}, {}
    for stem, rows in datasets.items():
        summaries[stem], curves[stem] = dm.summarize_ap(rows)
    points = {}
    for c, name in enumerate(NAMES):
        curve = dm.work_curve(datasets['dev'], c)
        points[name] = dm.choose_point(curve, rules[name]['precision'], rules[name]['recall'])
        for key in ('threshold', 'tp', 'fp', 'fn', 'precision', 'recall'):
            curves['dev'][name+'_work_'+key] = curve[key]
        print('DEV_POINT', name, json.dumps(points[name]), flush=True)
    if (output/'trace.jsonl.gz').is_file():
        check_equal('unchanged workpoints for retained trace', canonical_hash(read_json(output/'workpoints.json')),
                    canonical_hash(points))
    matches = {}
    slices = []
    comparison = []
    for stem, rows in datasets.items():
        matches[stem] = [dm.match_record(r, points) for r in rows]
        dm.write_rows(output/stem/'matches.jsonl.gz', matches[stem])
        for c, name in enumerate(NAMES):
            value = dm.aggregate_point(rows, c, points[name]['threshold'])
            value.update(origin=points[name]['origin'], t_rule=points[name]['t_rule'])
            summaries[stem]['per_class'][name]['workpoint'] = value
            if stem == 'train':
                # Full train PR is retained, but thresholds are never fitted on train.
                curve = dm.work_curve(rows, c)
                for key in ('threshold', 'tp', 'fp', 'fn', 'precision', 'recall'):
                    curves[stem][name+'_work_'+key] = curve[key]
        slices.extend(dm.slice_statistics(rows, matches[stem], stem))
        write_json(output/stem/'metrics.json', summaries[stem])
        np.savez_compressed(output/stem/'pr.npz', **curves[stem])
    write_json(output/'workpoints.json', points)
    write_csv(output/'slices.csv', slices)
    for name in NAMES:
        tr, dv = summaries['train']['per_class'][name], summaries['dev']['per_class'][name]
        item = {'class': name, 'threshold': points[name]['threshold'], 'point_origin': points[name]['origin']}
        for stem, stats in [('train', tr), ('dev', dv)]:
            for field in ('ap50', 'ap75', 'ap50_95'):
                item[stem+'_'+field] = stats[field]
            for field in ('tp', 'fp', 'fn', 'precision', 'recall'):
                item[stem+'_'+field] = stats['workpoint'][field]
        comparison.append(item)
    write_csv(output/'comparison.csv', comparison)
    intervals = dm.bootstrap(datasets['dev'], matches['dev'], repeats=protocol['bootstrap']['repeats'],
                            seed=protocol['bootstrap']['seed'],
                            progress=lambda i, n: print(f'bootstrap {i}/{n}', flush=True))
    write_json(output/'intervals.json', intervals)
    baseline_path = ROOT/'runs/a_001/train/best_dev_metrics.json'
    baseline = read_json(baseline_path)
    check_equal('quality baseline checkpoint', run['checkpoint_sha256'], baseline['checkpoint_sha256'])
    quality = {'scope': 'source_dev diagnostic assessment, not A freeze',
               'rules_sha256': protocol['rules']['sha256'], 'historical_reference_sha256': sha256(baseline_path),
               'classes': {}, 'human_review_complete': False, 'training_repeats_available': 1}
    for name in NAMES:
        metric = summaries['dev']['per_class'][name]
        rule = rules[name]
        minimum = rule['ap50_95']
        original = baseline['metrics'][f'metrics/{name}_AP50-95']
        if name in ('cyclist', 'car', 'truck', 'tram'):
            minimum = max(minimum, original-protocol['rules']['protection_drop'])
        ci = intervals['intervals'][name+'/recall']
        width = ci['upper']-ci['lower'] if ci['lower'] is not None else None
        wp = metric['workpoint']
        quality['classes'][name] = {
            'required_ap50_95_effective': minimum, 'historical_ap50_95': original,
            'new_minus_historical_ap_pp': (metric['ap50_95']-original)*100,
            'ap50_95_pass': metric['ap50_95'] >= minimum, 'ap50_pass': metric['ap50'] >= rule['ap50'],
            'workpoint_pass': points[name]['status'] == 'passed',
            'recall_ci_width': width, 'recall_precision_target_met': width is not None and
            width <= protocol['rules']['recall_ci_width'] and not ci['exploratory'],
            'gt': metric['instances'], 'positive_images': metric['images'], 'positive_risk_groups': metric['groups']}
    write_json(output/'quality.json', quality)
    errors = {}
    for stem, rows in matches.items():
        errors[stem] = {name: dict(Counter(e['type'] for r in rows for e in r['classes'][name]['fp_items'])) for name in NAMES}
    write_json(output/'errors.json', errors)
    cases = build_cases(datasets, matches, points)
    with (output/'cases.jsonl').open('w', encoding='utf-8') as stream:
        for case in cases:
            stream.write(json.dumps(case, ensure_ascii=False, allow_nan=False)+'\n')
    write_json(output/'review.json', {'status': 'pending_human_review', 'items': len(cases),
                                     'layers': dict(Counter(layer for c in cases for layer in c['layers'])),
                                     'completed': 0})
    run.update(status='traced' if (output/'trace.jsonl.gz').is_file() else 'analyzed', analysis_utc=time.time(),
               analysis_seconds=time.perf_counter()-started,
               analysis_numerics={'slice_width_height_round_decimals': 6, 'reason': 'remove sub-micropixel boundary noise'},
               human_review_complete=False)
    write_json(output/'run.json', run)
    return {'points': points, 'review': read_json(output/'review.json')}


def build_cases(datasets, matches, points):
    cases = {}
    rng = np.random.default_rng(20261026)

    def add(stem, row, kind, index, layer, sampling=None):
        cid = f'{stem}/{row["key"]}/{kind}:{index}'
        if cid not in cases:
            c = row['gt' if kind == 'gt' else 'pred']['cls'][index]
            cases[cid] = {'case_id': cid, 'purpose': row['purpose'], 'key': row['key'],
                          'group_id': row['group_id'], 'image': row['image'], 'image_sha256': row['image_sha256'],
                          'object': kind, 'index': int(index), 'class': NAMES[c],
                          'threshold': points[NAMES[c]]['threshold'], 'layers': [], 'sampling': {},
                          'status': 'pending', 'decision': None, 'reviewer': None, 'note': ''}
        case = cases[cid]
        if layer not in case['layers']:
            case['layers'].append(layer)
        if sampling:
            case['sampling'][layer] = sampling

    for stem, rows in datasets.items():
        pools = defaultdict(list)
        for row, match in zip(rows, matches[stem]):
            tags = dm.slice_tags(row)
            if stem == 'dev':
                for g, c in enumerate(row['gt']['cls']):
                    if c == 5:
                        add(stem, row, 'gt', g, 'tricycle_all_gt')
                floor_tp = {p for g, p in row['ap_pairs'][0] if row['gt']['cls'][g] == 5}
                for pid, c in enumerate(row['pred']['cls']):
                    if c == 5 and pid not in floor_tp:
                        add(stem, row, 'pred', pid, 'tricycle_floor_fp')
                for error in match['classes']['tricycle']['fp_items']:
                    add(stem, row, 'pred', error['pred_id'], 'tricycle_workpoint_fp')
            for name in NAMES[:-1]:
                c = NAMES.index(name)
                m = match['classes'][name]
                for g in m['fn_ids']:
                    pools[name+'/fn'].append((row, 'gt', g))
                    if c == 0:
                        for tag in tags[g]:
                            if tag in ('area_S', 'width_le8', 'crowded', 'edge'):
                                pools[name+'/fn_'+tag].append((row, 'gt', g))
                for error in m['fp_items']:
                    pools[name+'/'+error['type']].append((row, 'pred', error['pred_id']))
                for g, pid in m['pairs']:
                    pools[name+'/tp'].append((row, 'gt', g))
                    if c == 0:
                        for tag in tags[g]:
                            if tag in ('area_S', 'width_le8'):
                                pools[name+'/tp_'+tag].append((row, 'gt', g))
        for layer, pool in sorted(pools.items()):
            name = layer.split('/')[0]
            # Train is an explanatory counterpart only. Auxiliary protection checks are dev-only.
            if stem == 'train' and name != 'pedestrian':
                continue
            limit = 4 if stem == 'train' else 8 if name == 'pedestrian' else 5
            order = rng.permutation(len(pool))
            unique, extra, seen_groups = [], [], set()
            for i in order:
                group = pool[i][0]['group_id']
                if group in seen_groups:
                    extra.append(i)
                else:
                    unique.append(i); seen_groups.add(group)
            selected = (unique+extra)[:limit]
            for i in selected:
                row, kind, index = pool[i]
                add(stem, row, kind, index, layer,
                    {'population': len(pool), 'selected': len(selected), 'seed': 20261026})
    return list(cases.values())


def trace_image(raw, row, baselines, points):
    """Replay native NMS; preserve compact exhaustive scan evidence, not dense tensors."""
    import torch
    from ultralytics.utils import nms, ops
    from ultralytics.utils.metrics import box_iou
    raw = raw.detach()
    if raw.ndim != 2 or raw.shape[0] != 10:
        raise ContractError(f'Expected decoded 4+6 channels, got {tuple(raw.shape)}')
    base = baselines[row['key']]
    bpc, rpc = np.asarray(base['pred']['cls']), np.asarray(row['pred']['cls'])
    cf = np.asarray(base['pred']['conf']); current_cf = np.asarray(row['pred']['conf'])
    # Strict cross-pass identity: uncertain replays are never attached as causes of old FNs.
    aligned = (np.array_equal(bpc, rpc) and np.array_equal(cf, current_cf) and
               np.array_equal(np.asarray(base['pred']['boxes']), np.asarray(row['pred']['boxes'])))
    native, anchor = nms.non_max_suppression(raw[None].clone(), .001, .7, nc=6, multi_label=True,
                                            max_det=300, return_idxs=True)
    expected = torch.tensor([b + [s, float(c)] for b, s, c in
                             zip(row['pred']['boxes'], row['pred']['conf'], row['pred']['cls'])],
                            dtype=torch.float32, device=raw.device).reshape(-1, 6)
    if not torch.equal(native[0], expected):
        raise ContractError('Same-tensor native NMS replay differs')
    # Preserve FP16 xywh conversion before promotion, as the locked NMS does.
    xyxy = ops.xywh2xyxy(raw[:4].T)
    scores = raw[4:10].T
    aids = torch.arange(len(xyxy), device=raw.device)
    mask = scores.amax(1) > .001
    b, s, ids = xyxy[mask], scores[mask], aids[mask]
    ii, cc = torch.where(s > .001)
    expanded = torch.cat((b[ii], s[ii, cc, None], cc[:, None].float()), 1)
    source_ids = ids[ii]
    truncated = len(expanded) > 30000
    if truncated:
        order = expanded[:, 4].argsort(descending=True)[:30000]
        expanded, source_ids = expanded[order], source_ids[order]
    all_keep, all_ids = nms.non_max_suppression(raw[None].clone(), .001, .7, nc=6, multi_label=True,
                                              max_det=100000, return_idxs=True)
    keep = all_keep[0]
    keep_ids = all_ids[0].cpu().tolist()
    # Counterfactuals retain the confidence floor, multi-label policy and max_nms.
    without_nms = expanded[expanded[:, 4].argsort(descending=True)[:300]]
    without_cap = keep
    gt_boxes = torch.tensor(row['gt']['boxes'], dtype=torch.float32, device=raw.device).reshape(-1, 4)
    finite = torch.isfinite(xyxy).all(1) & torch.isfinite(scores).all(1) & (xyxy[:, 2:] > xyxy[:, :2]).all(1)
    raw_iou = box_iou(gt_boxes, xyxy.float()).cpu().numpy()
    class_scores = scores.float().cpu().numpy()
    valid = finite.cpu().numpy()
    gc = np.asarray(row['gt']['cls'])
    cf_recovery = {'nms_removed': {}, 'cap_removed': {}}
    for label, predictions in [('nms_removed', without_nms), ('cap_removed', without_cap)]:
        pred_cls = predictions[:, 5].long().cpu().numpy()
        conf = predictions[:, 4].cpu().numpy()
        matrix = box_iou(gt_boxes, predictions[:, :4]).cpu().numpy()
        for c, name in enumerate(NAMES):
            chosen = np.flatnonzero((pred_cls == c) & (conf >= points[name]['threshold']))
            pairs = dm.native_pairs(pred_cls[chosen], gc, matrix[:, chosen], .5)
            cf_recovery[label][name] = set(pairs[:, 0].tolist())
    working = dm.match_record(row, points)
    kept_pairs = {(int(a), int(c)): i for i, (a, c) in enumerate(zip(keep_ids, keep[:, 5].int().cpu().tolist()))}
    pre_pairs = {(int(a), int(c)): i for i, (a, c) in enumerate(zip(source_ids.cpu().tolist(), expanded[:, 5].int().cpu().tolist()))}
    evidence = []
    for g, c in enumerate(row['gt']['cls']):
        name = NAMES[c]; threshold = points[name]['threshold']
        geom = np.flatnonzero((raw_iou[g] >= .5) & valid)
        best = int(geom[np.argmax(class_scores[geom, c])]) if len(geom) else None
        high = [int(a) for a in geom if class_scores[a, c] >= threshold and class_scores[a, c] > .001]
        chains = []
        stages = Counter()
        for a in high:
            pair = (a, c)
            if pair not in pre_pairs:
                stages['max_nms'] += 1
                continue
            if pair in kept_pairs:
                stages['kept' if kept_pairs[pair] < 300 else 'max_det'] += 1
                continue
            stages['nms'] += 1
            candidate = expanded[pre_pairs[pair]]
            # NMS uses class-offset coordinates; keep its actual numerical geometry.
            same = torch.nonzero((keep[:, 5] == c) & (keep[:, 4] >= candidate[4]), as_tuple=False).flatten()
            if len(same):
                overlap = box_iou((candidate[:4]+c*7680)[None], keep[same, :4]+c*7680).flatten()
                suppressors = same[overlap > .7]
                if len(suppressors):
                    j = int(suppressors[0]); tied = bool(keep[j, 4] == candidate[4])
                    if len(chains) < 8:
                        candidate_orig = ops.scale_boxes(row['input_shape'], candidate[None, :4].clone(),
                                                        (row['height'], row['width']), row['ratio_pad'])[0]
                        suppressor_orig = ops.scale_boxes(row['input_shape'], keep[j:j+1, :4].clone(),
                                                         (row['height'], row['width']), row['ratio_pad'])[0]
                        chains.append({'candidate_anchor': a, 'class': c, 'score': float(candidate[4]),
                                       'candidate_box_input': candidate[:4].float().cpu().tolist(),
                                       'candidate_box_original': candidate_orig.float().cpu().tolist(),
                                       'gt_iou': float(raw_iou[g, a]), 'suppressor_anchor': keep_ids[j],
                                       'suppressor_box_input': keep[j, :4].float().cpu().tolist(),
                                       'suppressor_box_original': suppressor_orig.float().cpu().tolist(),
                                       'suppressor_score': float(keep[j, 4]),
                                       'suppression_iou': float(box_iou((candidate[:4]+c*7680)[None],
                                                                 (keep[j, :4]+c*7680)[None])[0, 0]),
                                       'same_score_order_ambiguous': tied})
        is_fn = g in working['classes'][name]['fn_ids']
        _, final_cls, final_conf, final_iou = dm.arrays(row)
        lower = np.flatnonzero((final_cls == c) & (final_iou[g] >= .5))
        low_recovered = False
        if len(lower):
            low_threshold = float(final_conf[lower].max())
            _, low_pairs, _ = dm.image_point(row, c, low_threshold)
            low_recovered = g in set(low_pairs[:, 0]) and low_threshold < threshold
        labels = []
        if not aligned:
            labels.append('cross_pass_drift_unconfirmed')
        elif is_fn:
            if not len(geom): labels.append('no_geometric_decoded_candidate')
            elif class_scores[best, c] <= .001: labels.append('class_score_below_output_floor')
            elif class_scores[best, c] < threshold: labels.append('class_score_below_workpoint')
            if low_recovered: labels.append('lower_workpoint_recovers_local_gt')
            if (chains and any(not chain['same_score_order_ambiguous'] for chain in chains)
                    and g in cf_recovery['nms_removed'][name]):
                labels.append('nms_removal_recovers_local_gt')
            if stages['max_det'] and g in cf_recovery['cap_removed'][name]:
                labels.append('max_det_removal_recovers_local_gt')
            if not labels: labels.append('assignment_or_mixed_unresolved')
        else:
            labels.append('working_point_tp')
        closest = int(np.flatnonzero(valid)[np.argmax(raw_iou[g, valid])]) if valid.any() else None

        def candidate_record(a):
            if a is None: return None
            original = ops.scale_boxes(row['input_shape'], xyxy[a:a+1].float().clone(),
                                       (row['height'], row['width']), row['ratio_pad'])[0]
            return {'anchor': a, 'box_input': xyxy[a].float().cpu().tolist(),
                    'box_original': original.cpu().tolist(), 'class_scores': class_scores[a].tolist(),
                    'gt_iou': float(raw_iou[g, a])}

        evidence.append({'gt_id': g, 'class': name, 'is_fn': is_fn, 'labels': labels,
                         'decoded_finite': int(valid.sum()), 'decoded_geometric_count': len(geom),
                         'max_decoded_iou': float(raw_iou[g, valid].max()) if valid.any() else None,
                         'max_class_score_at_iou50': float(class_scores[best, c]) if best is not None else None,
                         'best_anchor': best, 'high_score_geometric_count': len(high), 'stage_counts': dict(stages),
                         'best_geometric_class_score_candidate': candidate_record(best),
                         'closest_geometry_candidate': candidate_record(closest),
                         'suppression_chain_example_limit': 8,
                         'suppression_chain_examples': chains,
                         'nms_counterfactual_tp': g in cf_recovery['nms_removed'][name],
                         'cap_counterfactual_tp': g in cf_recovery['cap_removed'][name]})
    return {'key': row['key'], 'purpose': row['purpose'], 'baseline_exact': aligned,
            'same_tensor_native_nms_equal': True, 'decoded_total': len(xyxy),
            'pre_nms_truncated': truncated, 'post_nms_before_cap': len(keep), 'gt_evidence': evidence}


def trace(args):
    scratch_root = isolate(args.scratch)
    output = args.output.resolve()
    run, protocol = verify_collected(output)
    bind_stage(output, run, 'trace')
    if run['status'] != 'analyzed':
        raise ContractError('Trace requires analyzed evidence')
    check_equal('trace checkpoint', run['checkpoint_sha256'], sha256(args.weights))
    check_equal('trace READY', run['ready_sha256'], args.ready_sha256)
    contract = verify_source(args.data_root, args.ready_sha256)
    current_env = environment()
    check_equal('trace library identity', run['environment']['library_code'], current_env['library_code'])
    points = read_json(output/'workpoints.json')
    datasets = {stem: dm.read_rows(output/stem/'objects.jsonl.gz') for stem in ('train', 'dev')}
    matched = {stem: dm.read_rows(output/stem/'matches.jsonl.gz') for stem in ('train', 'dev')}
    selected, reasons = {}, {}
    dev_keys = set()
    for row, match in zip(datasets['dev'], matched['dev']):
        why = []
        if match['classes']['pedestrian']['fn_ids']: why.append('dev_pedestrian_fn')
        if 5 in row['gt']['cls']: why.append('dev_tricycle_gt')
        floor_tp = {p for g, p in row['ap_pairs'][0]}
        if any(c == 5 and p not in floor_tp for p, c in enumerate(row['pred']['cls'])):
            why.append('dev_tricycle_floor_fp')
        if match['classes']['tricycle']['fp_items']: why.append('dev_tricycle_workpoint_fp')
        if why:
            dev_keys.add(row['key']); reasons[row['key']] = why
    selected['source_dev'] = dev_keys
    pools = defaultdict(list)
    for row, match in zip(datasets['train'], matched['train']):
        tags = dm.slice_tags(row)
        for g in match['classes']['pedestrian']['fn_ids']:
            for tag in ('area_S', 'width_le8', 'crowded', 'edge', 'all'):
                if tag in tags[g]: pools['fn_'+tag].append(row)
        for g, _ in match['classes']['pedestrian']['pairs']:
            pools['tp_small' if 'width_le8' in tags[g] else 'tp_other'].append(row)
    rng = np.random.default_rng(20261026)
    train_keys, groups = set(), set()
    for layer, pool in sorted(pools.items()):
        n = 0
        for i in rng.permutation(len(pool)):
            row = pool[i]
            if row['group_id'] in groups or row['key'] in train_keys: continue
            train_keys.add(row['key']); groups.add(row['group_id']); reasons[row['key']] = [layer]
            n += 1
            if n >= 11 or len(train_keys) >= 80: break
        if len(train_keys) >= 80: break
    selected['source_train'] = train_keys
    all_traces = []
    with model_log(output/'evaluation.log', 'a') as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            for purpose, stem in zip(PURPOSES, ('train', 'dev')):
                if not selected[purpose]: continue
                records = [r for r in contract['manifests'][purpose] if r['key'] in selected[purpose]]
                baselines = {r['key']: r for r in datasets[stem]}
                with tempfile.TemporaryDirectory(prefix='trace_', dir=scratch_root) as temp:
                    val, seconds = run_validator(subset(contract, purpose, records), args.weights, Path(temp),
                                                 trace_context=(baselines, points))
                    for evidence in val.traces:
                        evidence['selection_reasons'] = reasons[evidence['key']]
                    all_traces.extend(val.traces)
                    run.setdefault('trace', {})[purpose] = {'images': len(records), 'seconds': seconds,
                                                          'exact_reproduced': sum(t['baseline_exact'] for t in val.traces)}
                    del val
    dm.write_rows(output/'trace.jsonl.gz', all_traces)
    check_equal('trace checkpoint after', run['checkpoint_sha256'], sha256(args.weights))
    check_equal('trace READY after', run['ready_sha256'], sha256(args.data_root/'READY.json'))
    run.update(status='traced', trace_utc=time.time(), trace_sha256=sha256(output/'trace.jsonl.gz'))
    write_json(output/'run.json', run)
    return run['trace']


def read_cases(output):
    return [json.loads(line) for line in (output/'cases.jsonl').read_text(encoding='utf-8').splitlines() if line]


def review_identity(output, run, cases):
    return canonical_hash({'checkpoint': run['checkpoint_sha256'], 'ready': run['ready_sha256'],
                           'objects': {p: run['splits'][p]['objects_sha256'] for p in PURPOSES},
                           'workpoints': sha256(output/'workpoints.json'),
                           'case_ids': [c['case_id'] for c in cases]})


def import_review(args):
    output = args.output.resolve()
    run, _ = verify_collected(output)
    submitted = read_json(args.review_file)
    check_equal('human review phase', 'human_review', submitted.get('phase'))
    check_equal('review checkpoint', run['checkpoint_sha256'], submitted['checkpoint_sha256'])
    check_equal('review READY', run['ready_sha256'], submitted['ready_sha256'])
    cases = read_cases(output); lookup = {c['case_id']: c for c in cases}
    check_equal('review queue identity', review_identity(output, run, cases), submitted['review_identity'])
    dr.validate_bundle(submitted, {'checkpoint_sha256': run['checkpoint_sha256'],
                       'ready_sha256': run['ready_sha256'],
                       'review_identity': review_identity(output, run, cases)}, set(lookup), allow_pending=True)
    seen = set()
    for entry in submitted['judgments']:
        cid = entry['case_id']
        if cid in seen or cid not in lookup: raise ContractError('Duplicate/unknown review case')
        seen.add(cid)
        if not entry.get('reviewer', '').strip(): raise ContractError('Reviewer is required')
        target = lookup[cid]
        if target['status'] == 'reviewed' and entry.get('decision') in dr.DECISIONS and target['decision'] != entry['decision']:
            raise ContractError('Conflicting existing review; explicit reconciliation required')
        previous = {k: target.get(k) for k in ('case_id', 'decision', 'reviewer', 'note')}
        if target.get('note_history'): previous['note_history'] = target['note_history']
        merged = dr.merge_feedback(previous, entry)
        target.update(merged, status='reviewed' if merged.get('decision') in dr.DECISIONS else 'pending')
    with (output/'cases.jsonl').open('w', encoding='utf-8') as stream:
        for case in cases: stream.write(json.dumps(case, ensure_ascii=False)+'\n')
    complete = sum(c['status'] == 'reviewed' for c in cases)
    previous = read_json(output/'review.json') if (output/'review.json').exists() else {}
    review_status = dict(previous, status='reviewed' if complete == len(cases) else 'pending_human_review',
                         items=len(cases), completed=complete,
                         feedback_items=sum(bool(c.get('note') or c.get('note_history')) for c in cases),
                         decisions=dict(Counter(c['decision'] for c in cases if c['status'] == 'reviewed')),
                         layers=dict(Counter(layer for c in cases for layer in c['layers'])))
    if previous.get('audit'):
        review_status['audit_summary'] = dr.audit_summary(cases, previous['audit'])
        review_status['status'] = ('audit_ready_for_acceptance' if review_status['audit_summary']['coverage_complete']
                                  else 'pending_human_audit')
    if previous.get('user_assessment', {}).get('agent_review_usable'):
        review_status['status'] = 'agent_review_adopted'
    write_json(output/'review.json', review_status)
    return review_status


def import_initial(args):
    """Import explicitly labeled Agent observations without promoting human review state."""
    output = args.output.resolve()
    run, _ = verify_collected(output)
    incoming = read_json(args.review_file)
    check_equal('Agent initial phase', 'agent_initial', incoming.get('phase'))
    check_equal('Agent checkpoint', run['checkpoint_sha256'], incoming['checkpoint_sha256'])
    check_equal('Agent READY', run['ready_sha256'], incoming['ready_sha256'])
    cases = read_cases(output)
    check_equal('Agent queue identity', review_identity(output, run, cases), incoming['review_identity'])
    lookup, seen = {c['case_id']: c for c in cases}, set()
    for judgment in incoming['judgments']:
        cid = judgment['case_id']
        if cid not in lookup or cid in seen: raise ContractError('Unknown/duplicate Agent review case')
        seen.add(cid)
        if judgment['decision'] not in ('correct_detection', 'true_error', 'annotation_suspect', 'uncertain'):
            raise ContractError('Invalid Agent decision')
        if judgment.get('confidence') not in ('high', 'medium', 'low') or not judgment.get('note', '').strip():
            raise ContractError('Agent review requires confidence and a visible-evidence note')
        lookup[cid]['initial_review'] = dict(judgment, phase='agent_initial', model=incoming['model'])
    with (output/'cases.jsonl').open('w', encoding='utf-8') as stream:
        for case in cases: stream.write(json.dumps(case, ensure_ascii=False)+'\n')
    status = read_json(output/'review.json')
    initial = [c['initial_review'] for c in cases if c.get('initial_review')]
    status['initial'] = {'phase': 'agent_initial', 'model': incoming['model'], 'items': len(initial),
                         'complete': len(initial) == len(cases),
                         'decisions': dict(Counter(c['decision'] for c in initial)),
                         'confidence': dict(Counter(c['confidence'] for c in initial)),
                         'method': incoming.get('method', {})}
    write_json(output/'review.json', status)
    return status


def audit(args):
    """Register the user-requested Agent-led review protocol; never mark cases human-reviewed."""
    output = args.output.resolve()
    run, _ = verify_collected(output)
    cases = read_cases(output)
    plan = dr.audit_plan(cases, review_identity(output, run, cases))
    status = read_json(output/'review.json')
    if status.get('audit'):
        check_equal('existing audit selection', plan, status['audit'])
    status.update(audit=plan, audit_summary=dr.audit_summary(cases, plan),
                  status='agent_review_adopted' if status.get('user_assessment', {}).get('agent_review_usable')
                  else 'pending_human_audit')
    write_json(output/'review.json', status)
    run.update(visual_review_mode='agent_primary_selective_human_review'
               if status.get('user_assessment', {}).get('agent_review_usable') else 'agent_primary_human_audit',
               human_review_complete=False,
               human_audit_complete=False, status='awaiting_review')
    write_json(output/'run.json', run)
    bind_stage(output, run, 'review')
    return {'audit_cases': len(plan['case_ids']), 'mandatory_cases': plan['mandatory_count'],
            'status': status['status']}


def review(args):
    """Simple localhost viewer. Browser judgments export to JSON, never silently label data."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import parse_qs, urlparse
    from threading import Lock
    scratch = isolate(args.scratch)
    output = args.output.resolve()
    run, _ = verify_collected(output)
    rows = {r['key']: r for stem in ('train', 'dev') for r in dm.read_rows(output/stem/'objects.jsonl.gz')}
    cases = read_cases(output)
    status = read_json(output/'review.json')
    if not status.get('audit'): raise ContractError('Register the human audit with mode audit first')
    expected_plan = dr.audit_plan(cases, review_identity(output, run, cases))
    check_equal('current audit identity', expected_plan, status['audit'])
    case_keys = {c['key'] for c in cases}
    rows = {key: row for key, row in rows.items() if key in case_keys}
    traces = {r['key']: r for r in dm.read_rows(output/'trace.jsonl.gz')} if (output/'trace.jsonl.gz').exists() else {}
    gt_ids = defaultdict(set)
    for c in cases:
        if c['object'] == 'gt': gt_ids[c['key']].add(c['index'])
    traces = {key: {'baseline_exact': t['baseline_exact'],
                    'gt_evidence': [g for g in t['gt_evidence'] if g['gt_id'] in gt_ids[key]]}
              for key, t in traces.items() if key in gt_ids}
    matches = {m['key']: m for stem in ('train', 'dev')
               for m in dm.read_rows(output/stem/'matches.jsonl.gz') if m['key'] in case_keys}
    facts = {}
    for c in cases:
        if c['object'] == 'gt':
            matched = any(g == c['index'] for g, _ in matches[c['key']]['classes'][c['class']]['pairs'])
            fact = '工作点记录：已正确匹配该标注目标' if matched else '工作点记录：未正确匹配该标注目标（漏检）'
        elif rows[c['key']]['pred']['conf'][c['index']] < c['threshold']:
            fact = '低于使用阈值：仅作诊断候选，不是实际工作点误报'
        else:
            fact = '达到使用阈值：按冻结标注记为工作点误报候选'
        facts[c['case_id']] = fact
    payload = {'checkpoint_sha256': run['checkpoint_sha256'], 'ready_sha256': run['ready_sha256'],
               'review_identity': review_identity(output, run, cases),
               'cases': cases, 'rows': rows, 'traces': traces,
               'workpoints': read_json(output/'workpoints.json'), 'names': list(NAMES),
               'audit': status['audit'], 'facts': facts,
               'user_assessment': status.get('user_assessment')}
    identity = {k: payload[k] for k in ('checkpoint_sha256', 'ready_sha256', 'review_identity')}
    draft = dr.DraftStore(scratch/'review_draft.json', identity, cases)
    draft_lock = Lock()
    page = dr.PAGE
    page_path = scratch/'review.html'
    page_path.write_text(page, encoding='utf-8')
    image_root = Path(run['data_root'])
    # No arbitrary file endpoint: only images referenced by verified exported rows.
    for row in rows.values():
        if row['purpose'] not in PURPOSES: raise ContractError('Forbidden review purpose')
    payload_bytes = lambda: json.dumps(dict(payload, draft=draft.bundle()), ensure_ascii=False,
                                      separators=(',', ':')).encode('utf-8')

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == '/': body, kind = page.encode('utf-8'), 'text/html; charset=utf-8'
            elif parsed.path == '/data':
                with draft_lock: body = payload_bytes()
                kind = 'application/json; charset=utf-8'
            elif parsed.path == '/draft':
                with draft_lock:
                    body = json.dumps(draft.bundle(), ensure_ascii=False).encode('utf-8')
                kind = 'application/json; charset=utf-8'
            elif parsed.path == '/image':
                key = parse_qs(parsed.query).get('key', [''])[0]
                if key not in rows: self.send_error(404); return
                path = safe_path(image_root, rows[key]['image'])
                if sha256(path) != rows[key]['image_sha256']: self.send_error(409); return
                body, kind = path.read_bytes(), 'image/jpeg'
            else: self.send_error(404); return
            self.send_response(200); self.send_header('Content-Type', kind)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_POST(self):
            if self.path != '/draft': self.send_error(404); return
            if self.headers.get('Origin') != f'http://127.0.0.1:{args.port}':
                self.send_error(403); return
            if self.headers.get('Content-Type') != 'application/json': self.send_error(415); return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 2_000_000: raise ContractError('Invalid draft size')
                submitted = json.loads(self.rfile.read(size).decode('utf-8'))
                with draft_lock: count = draft.save(submitted)
            except (ContractError, ValueError, TypeError, KeyError) as exc:
                self.send_error(400, str(exc)); return
            body = json.dumps({'saved': count}).encode('utf-8')
            self.send_response(200); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)

        def log_message(self, *_): pass

    class ReviewServer(ThreadingHTTPServer):
        allow_reuse_address = False

    server = ReviewServer(('127.0.0.1', args.port), Handler)
    print(f'REVIEW http://127.0.0.1:{args.port}/ ; external page: {page_path}', flush=True)
    server.serve_forever()


def seal(args):
    output = args.output.resolve()
    run, _ = verify_collected(output)
    bind_stage(output, run, 'delivery')
    status = read_json(output/'review.json')
    cases = read_cases(output)
    summary = dr.audit_summary(cases, status['audit']) if status.get('audit') else None
    complete = bool(cases) and all(c['status'] == 'reviewed' and c.get('reviewer') and
                                   c.get('decision') in ('correct_detection', 'true_error', 'annotation_suspect', 'uncertain')
                                   for c in cases) and (output/'trace.jsonl.gz').is_file()
    run.update(status='complete' if complete else 'awaiting_review', human_review_complete=complete)
    if summary is not None:
        status.update(audit_summary=summary, status='audit_ready_for_acceptance' if summary['coverage_complete']
                      else 'pending_human_audit')
        if status.get('user_assessment', {}).get('agent_review_usable'):
            status['status'] = 'agent_review_adopted'
        write_json(output/'review.json', status)
        run.update(human_audit_complete=summary['coverage_complete'],
                   human_audit_needs_followup=summary['needs_followup'])
    write_json(output/'run.json', run)
    files = {}
    for path in sorted(output.rglob('*')):
        if path.is_file() and path.name != 'manifest.json':
            files[path.relative_to(output).as_posix()] = {'sha256': sha256(path), 'bytes': path.stat().st_size}
    manifest = {'schema_version': 1, 'checkpoint_sha256': run['checkpoint_sha256'],
                'ready_sha256': run['ready_sha256'], 'status': run['status'], 'files': files}
    write_json(output/'manifest.json', manifest)
    return {'status': run['status'], 'files': len(files), 'review': status}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['collect', 'analyze', 'trace', 'audit', 'review', 'import-initial', 'import-review', 'seal'])
    p.add_argument('--output', type=Path, default=ROOT/'runs/a_001_diag')
    p.add_argument('--scratch', type=Path, required=True, help='External task-owned scratch root')
    p.add_argument('--data-root', type=Path, default=ROOT/'data/soda10m/splits/source_v1')
    p.add_argument('--ready-sha256', default=READY_SHA)
    p.add_argument('--weights', type=Path, default=ROOT/'runs/a_001/weights/best.pt')
    p.add_argument('--weights-sha256', default=BEST_SHA)
    p.add_argument('--review-file', type=Path)
    p.add_argument('--port', type=int, default=8766)
    args = p.parse_args()
    result = globals()[args.mode.replace('-', '_')](args)
    print(json.dumps(result, ensure_ascii=False, default=str, allow_nan=False, indent=2))


if __name__ == '__main__':
    main()
