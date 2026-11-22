"""Fixed R/17 pedestrian candidate preparation, fitting and offline comparison."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
import time

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import detection_metrics as dm
from evaluate import original_gt, verify_subset
from source_contract import ContractError, NAMES, check_equal, read_json, sha256, verified_file, write_json
from source_policy import load_rules, paired_effects


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/soda10m/splits/source_v1'
R_RUN = ROOT / 'runs/a_002/train'
RUNS = ROOT / 'runs'
RULES = ROOT / 'RULES.md'
EXCLUSIONS = ROOT / 'configs/rerank.json'
READY_SHA = '72e75ebd99014c830e0227061e390e6aed768ae724b532a5c2c90fd8dec93e85'
R_SHA = 'e4c873eda51d1a2b9ab722a9cb8f8bc0de9091a320184a8dc6f49fb84421c6ab'
PREDICTION_SHA = {
    'source_train': '1fadd6202f4c011df5377f40fa8e50aafacfd2d490323d84a908802e6406e1dd',
    'source_dev': 'ad15355b4aedcb1285146e8e88d3b2eb868f3b54089be231ddde490bc0af765f',
}
RECIPE = {'seed': 17, 'floor': .02, 'expansion': 1.2, 'size': 96, 'batch': 64,
          'optimizer': 'AdamW', 'lr': .001, 'weight_decay': .0001, 'epochs': 20,
          'loss': 'BCEWithLogitsLoss', 'channels': [16, 32, 64], 'inner_group_fraction': .2}


def sealed_file(manifest, relative):
    entry = manifest['files'][relative]
    path = verified_file(R_RUN, dict(entry, path=relative))
    check_equal(f'{relative} size', entry['bytes'], path.stat().st_size)
    return path


def verify_r():
    manifest = read_json(R_RUN / 'manifest.json')
    check_equal('R READY', READY_SHA, manifest['identity']['ready_sha256'])
    check_equal('R scope', 'formal', manifest['scope'])
    run = read_json(sealed_file(manifest, 'final_eval/run.json'))
    check_equal('R evaluation status', 'complete_local', run['status'])
    check_equal('R evaluation role', 'diagnostic', run['selected_role'])
    role = run['roles']['diagnostic']
    check_equal('R epoch', 80, role['epoch'])
    check_equal('R checkpoint', R_SHA, role['sha256'])
    check_equal('R checkpoint file', R_SHA, manifest['files'][role['path']]['sha256'])
    sealed_file(manifest, role['path'])
    return manifest, run


@dataclass(frozen=True)
class Sample:
    key: str
    index: int
    image: str
    size: tuple[int, int]
    box: tuple
    kind: str
    target: float | None


def candidate_row(row):
    """Select using only saved predictions, then remap frozen native IoU edges."""
    pred = row['pred']
    keep = [i for i, (c, score) in enumerate(zip(pred['cls'], pred['conf'], strict=True))
            if c == 0 and score >= RECIPE['floor']]
    remap = {old: new for new, old in enumerate(keep)}
    filtered = {k: [v[i] for i in keep] for k, v in pred.items()}
    # Recompute matches after filtering. The archived AP assignment used more candidates.
    return {**row, 'pred': filtered,
            'iou_edges': [[g, remap[p], v] for g, p, v in row['iou_edges'] if p in remap]}


def add_ap(row):
    flags, pairs = dm.ap_flags(row)
    return dict(row, ap_tp=flags.tolist(), ap_pairs=pairs)


def label_samples(row):
    gt, pred, _, iou = dm.arrays(row)
    hits = {int(p): float(iou[g, p]) for g, p in dm.native_pairs(pred, gt, iou)}
    overlaps = iou[gt == 0].max(axis=0, initial=0)
    for i, (original, box) in enumerate(zip(row['pred']['ids'], row['pred']['original_boxes'], strict=True)):
        if i in hits:
            kind, target = 'positive', hits[i]
        elif overlaps[i] < np.float32(.30):
            kind, target = 'negative', 0.
        else:
            kind, target = ('gray' if overlaps[i] < .5 else 'duplicate'), None
        yield make_sample(row, original, box, kind, target)


def make_sample(row, index, box, kind, target=None):
    return Sample(row['key'], index, row['image'], (row['width'], row['height']), tuple(box), kind, target)


@dataclass
class Pool:
    rows: list
    samples: list[Sample]

    def subset(self, groups):
        rows = [r for r in self.rows if r['group_id'] in groups]
        keys = {r['key'] for r in rows}
        return Pool(rows, [s for s in self.samples if s.key in keys])


def build_pool(purpose, identity):
    if purpose not in PREDICTION_SHA:
        raise ContractError(f'Forbidden reranker purpose: {purpose}')
    manifest, run = identity
    relative = f'final_eval/{purpose}/objects.jsonl.gz'
    check_equal('sealed predictions', PREDICTION_SHA[purpose], manifest['files'][relative]['sha256'])
    path = sealed_file(manifest, relative)
    contract = verify_subset(SOURCE, READY_SHA, purpose)
    records = {r['key']: r for r in contract['records']}
    check_equal('R evaluation subset', {'images': len(records), 'checkpoint_sha256': R_SHA},
                run['results'][purpose])
    rows, samples = [], []
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            original = json.loads(line)
            key = original['key']
            record = records.pop(key, None)
            if record is None:
                raise ContractError(f'Duplicate or unapproved R image: {purpose}/{key}')
            fields = ('purpose', 'group_id', 'image', 'image_sha256', 'label', 'label_sha256', 'width', 'height')
            check_equal(key, [record[k] for k in fields], [original[k] for k in fields])
            gt = original_gt(record, contract['root'])
            check_equal(f'{key} GT', gt, list(zip(original['gt']['cls'], original['gt']['original_boxes'], strict=True)))
            pred = original['pred']
            check_equal(f'{key} prediction order', list(range(len(pred['cls']))), pred['ids'])
            audit = original['nms_audit']
            check_equal(f'{key} NMS counts', len(pred['cls']), audit['final_count'])
            check_equal(f'{key} truncated before NMS', False, audit['pre_nms_truncated'])
            row = candidate_row(original)
            # Preserve the NMS audit in the sealed artifact; these are derived metric rows.
            del row['nms_audit']
            rows.append(add_ap(row))
            if purpose == 'source_train':
                samples.extend(label_samples(row))
            else:
                samples.extend(make_sample(row, i, box, 'evaluation')
                               for i, box in zip(row['pred']['ids'], row['pred']['original_boxes'], strict=True))
    check_equal(f'unconsumed {purpose} whitelist keys', 0, len(records))
    return Pool(rows, samples)


def split_train(pool):
    check_equal('training is only from source_train', {'source_train'}, {r['purpose'] for r in pool.rows})
    groups = {r['group_id'] for r in pool.rows}
    ordered = sorted(groups, key=lambda g: hashlib.sha256(f"{RECIPE['seed']}:{g}".encode('utf-8')).hexdigest())
    count = math.ceil(len(groups) * RECIPE['inner_group_fraction'])
    if count >= len(groups):
        raise ContractError('Fitting and inner validation require separate risk groups')
    return pool.subset(set(ordered[count:])), pool.subset(set(ordered[:count]))


def isolate(*pools):
    for field in ('key', 'image_sha256', 'group_id'):
        seen = set()
        for pool in pools:
            values = {r[field] for r in pool.rows}
            check_equal(f'overlapping split {field}', set(), seen & values)
            seen.update(values)


def exclusions(pool):
    config = read_json(EXCLUSIONS)
    check_equal('exclusion purpose', 'source_train', config['purpose'])
    check_equal('exclusion source candidates', PREDICTION_SHA['source_train'], config['predictions_sha256'])
    rows = {r['key']: r for r in pool.rows}
    result = set()
    for e in config['candidates']:
        row = rows[e['key']]
        check_equal('exclusion image identity', e['image_sha256'], row['image_sha256'])
        candidate = e['key'], e['index']
        if candidate in result or e['index'] not in row['pred']['ids'] or not e['reason']:
            raise ContractError(f'Invalid or duplicate exclusion: {candidate}')
        result.add(candidate)
    return result


def fitting_samples(pool, blocked):
    return [s for s in pool.samples if s.target is not None and (s.key, s.index) not in blocked]


def pool_summary(pool):
    return {'images': len(pool.rows), 'groups': len({r['group_id'] for r in pool.rows}),
            'gt': sum(r['gt']['cls'].count(0) for r in pool.rows), 'candidates': len(pool.samples),
            'labels': dict(Counter(s.kind for s in pool.samples)),
            'matches50': sum(len(r['ap_pairs'][0]) for r in pool.rows),
            'matches75': sum(len(r['ap_pairs'][5]) for r in pool.rows)}


def crop_rgb(image, box):
    x0, y0, x1, y1 = box
    if not all(math.isfinite(x) for x in box) or x1 < x0 or y1 < y0:
        raise ContractError(f'Invalid candidate box: {box}')
    extra = (RECIPE['expansion'] - 1) / 2
    dx, dy = (x1-x0) * extra, (y1-y0) * extra
    left, top = math.floor(x0-dx), math.floor(y0-dy)
    right, bottom = max(left+1, math.ceil(x1+dx)), max(top+1, math.ceil(y1+dy))
    color = (114, 114, 114)
    patch = Image.new('RGB', (right-left, bottom-top), color)
    clip = max(left, 0), max(top, 0), min(right, image.width), min(bottom, image.height)
    if clip[0] < clip[2] and clip[1] < clip[3]:
        patch.paste(image.crop(clip), (clip[0]-left, clip[1]-top))
    size = RECIPE['size']
    scale = size / max(patch.size)
    w, h = [max(1, round(x * scale)) for x in patch.size]
    out = Image.new('RGB', (size, size), color)
    out.paste(patch.resize((w, h), Image.Resampling.BILINEAR), ((size-w)//2, (size-h)//2))
    return np.array(out).transpose(2, 0, 1)


class Crops(Dataset):
    def __init__(self, root, samples, cache=False):
        self.root, self.samples = root, samples
        self.last_path, self.last_image = None, None
        self.cache = None
        if cache:
            size = RECIPE['size']
            self.cache = np.empty((len(samples), 3, size, size), dtype=np.uint8)
            for i, sample in enumerate(samples):
                self.cache[i] = self.read_crop(sample)
            self.last_path, self.last_image = None, None

    def __len__(self):
        return len(self.samples)

    def read_crop(self, sample):
        if sample.image != self.last_path:
            with Image.open(self.root / sample.image) as image:
                self.last_image = ImageOps.exif_transpose(image).convert('RGB')
            check_equal(f'{sample.key} decoded image size', sample.size, self.last_image.size)
            self.last_path = sample.image
        return crop_rgb(self.last_image, sample.box)

    def __getitem__(self, i):
        sample = self.samples[i]
        pixels = self.cache[i] if self.cache is not None else self.read_crop(sample)
        tensor = torch.from_numpy(pixels).float().div_(255)
        return tensor, np.float32(sample.target if sample.target is not None else np.nan)


def model_for(device):
    layers = []
    channels = [3, *RECIPE['channels']]
    for a, b in zip(channels[:-1], channels[1:], strict=True):
        layers.extend((nn.Conv2d(a, b, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU()))
    return nn.Sequential(*layers, nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(channels[-1]*16, 1)).to(device)


def optimizer_for(model):
    return torch.optim.AdamW(model.parameters(), lr=RECIPE['lr'], weight_decay=RECIPE['weight_decay'])


def fit_step(model, optimizer, loss_fn, image, target, device):
    optimizer.zero_grad(set_to_none=True)
    loss = loss_fn(model(image.to(device)).squeeze(1), target.to(device))
    if not torch.isfinite(loss).item():
        raise ContractError('Nonfinite reranker training loss')
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    return [score for images, _ in loader for score in model(images.to(device)).sigmoid().flatten().cpu().tolist()]


def with_scores(pool, scores):
    scores = np.asarray(scores, dtype=np.float32)
    check_equal('one score per candidate', (len(pool.samples),), scores.shape)
    if not np.all(np.isfinite(scores) & (scores >= 0) & (scores <= 1)):
        raise ContractError('Candidate scores must be finite probabilities')
    result, offset = [], 0
    for row in pool.rows:
        ids = row['pred']['ids']
        end = offset + len(ids)
        check_equal('score identities', [(row['key'], i) for i in ids],
                    [(s.key, s.index) for s in pool.samples[offset:end]])
        result.append(dict(row, pred=dict(row['pred'], conf=scores[offset:end].tolist())))
        offset = end
    check_equal('consumed scores', len(scores), offset)
    return result


def pedestrian_metrics(rows, rules):
    rule = rules['classes']['pedestrian']
    ap, pr = dm.summarize_ap(rows)
    curve = dm.work_curve(rows, 0)
    point = dm.choose_point(curve, rule['precision'], rule['recall'])
    pr = {k: v for k, v in pr.items() if k.startswith('pedestrian_')}
    pr.update({f'pedestrian_work_{k}': v for k, v in curve.items()})
    return {'ap': ap['per_class']['pedestrian'], 'point': point}, pr


def assert_r_baseline(pool, rules):
    result, _ = pedestrian_metrics(pool.rows, rules)
    ap, point = result['ap'], result['point']
    check_equal('R dev support', (484, 1880), (ap['instances'], ap['predictions']))
    check_equal('R dev workpoint', (232, 76, 252, .375732421875),
                (point['tp'], point['fp'], point['fn'], point['t_rule']))
    expected = (.5653402269574705, .3406715600640458, .3235566444557635)
    actual = (ap['ap50'], ap['ap75'], ap['ap50_95'])
    if not np.allclose(expected, actual, rtol=0, atol=1e-10):
        raise ContractError(f'R dev AP drift: expected {expected}, got {actual}')
    return result


def match_details(rows, point):
    details, strata = {}, Counter()
    for row in rows:
        _, pairs, selected = dm.image_point(row, 0, point['threshold'])
        _, strict, _ = dm.image_point(row, 0, point['threshold'], .75)
        ids = row['pred']['ids']
        hits = {int(g): ids[p] for g, p in pairs}
        strict_hits = {int(g) for g, _ in strict}
        matched = set(pairs[:, 1])
        fp = {ids[p]: dm.error_type(row, p, pairs)['type'] for p in selected if p not in matched}
        details[row['key']] = hits, fp
        for g, (c, tags) in enumerate(zip(row['gt']['cls'], dm.slice_tags(row), strict=True)):
            if c == 0:
                small = 'area_S' in tags or 'width_le8' in tags
                name = 'small_thin_union' if small else 'outside_union'
                strata[name, 'gt'] += 1
                strata[name, 'tp50'] += g in hits
                strata[name, 'tp75'] += g in strict_hits
    slices = {}
    for name in ('small_thin_union', 'outside_union'):
        gt, tp50, tp75 = [strata[name, k] for k in ('gt', 'tp50', 'tp75')]
        slices[name] = dict(gt=gt, tp50=tp50, tp75=tp75, recall50=tp50/gt if gt else None)
    return details, slices


def transitions(before, after):
    changes = []
    for key, (r_hits, r_fp) in before.items():
        hits, fp = after[key]
        gt_changes = [{'gt': g, 'R': r_hits.get(g), 'rerank': hits.get(g)}
                      for g in sorted(r_hits.keys() | hits.keys()) if r_hits.get(g) != hits.get(g)]
        added = {i: fp[i] for i in sorted(fp.keys() - r_fp.keys())}
        removed = {i: r_fp[i] for i in sorted(r_fp.keys() - fp.keys())}
        if gt_changes or added or removed:
            changes.append({'key': key, 'gt_changes': gt_changes, 'added_fp': added, 'removed_fp': removed})
    return changes


def compare(pool, scores, rules, repeats=2000):
    arms = {'R': pool.rows, 'rerank': with_scores(pool, scores)}
    reports, curves, matches = {}, {}, {}
    for name, rows in arms.items():
        reports[name], pr = pedestrian_metrics(rows, rules)
        matches[name], reports[name]['slices'] = match_details(rows, reports[name]['point'])
        curves.update({f'{name}_{k}': v for k, v in pr.items()})
    stats = paired_effects([(RECIPE['seed'], arms['R'], arms['rerank'])], rules, repeats=repeats)
    stats['effects'] = {k: v for k, v in stats['effects'].items() if k.startswith('pedestrian/')}
    del stats['per_seed']  # The two point estimates are already in reports.
    r, candidate = reports['R'], reports['rerank']
    effects = stats['effects']
    recall = effects['pedestrian/recall']
    quality = all(candidate['ap'][key] >= r['ap'][key] for key in ('ap50', 'ap50_95'))
    review = [key for key, effect in effects.items() if effect['mean_delta'] is not None and
              ((key.endswith('/ap75') and effect['mean_delta'] < -rules['ap75_drop']) or
               (key.count('/') == 2 and effect['mean_delta'] < -rules['slice_drop']))]
    improved = recall['mean_delta'] is not None and recall['mean_delta'] >= rules['recall_gain']
    if not (improved and quality):
        decision = 'pause'
    elif review:
        decision = 'review_protection'
    elif recall['exploratory'] or recall['lower'] is None or recall['lower'] <= 0:
        decision = 'repeat_for_evidence'
    else:
        decision = 'worth_continuing'
    return {'arms': reports, 'intervals': stats, 'decision': decision, 'protection_review': review,
            'transitions': transitions(matches['R'], matches['rerank'])}, curves


def split_record(pool, samples=None):
    return dict(pool_summary(pool), groups_used=sorted({r['group_id'] for r in pool.rows}),
                supervised=dict(Counter(s.kind for s in samples)) if samples is not None else {})


def train_inputs(identity):
    pool = build_pool('source_train', identity)
    fit, inner = split_train(pool)
    isolate(fit, inner)
    blocked = exclusions(pool)
    supervised = fitting_samples(fit, blocked)
    if {s.kind for s in supervised} != {'positive', 'negative'}:
        raise ContractError('Fitting requires both positive and negative candidates')
    return fit, inner, supervised, blocked


def loader(samples, shuffle=False, cache=False):
    rng = torch.Generator().manual_seed(RECIPE['seed'])
    return DataLoader(Crops(SOURCE, samples, cache=cache), batch_size=RECIPE['batch'], shuffle=shuffle, generator=rng)


def preflight(identity, rules):
    fit, inner, supervised, blocked = train_inputs(identity)
    dev = build_pool('source_dev', identity)
    isolate(fit, inner, dev)
    baseline = assert_r_baseline(dev, rules)
    reference, curves = compare(dev, [score for row in dev.rows for score in row['pred']['conf']], rules, repeats=20)
    check_equal('identity arm metrics', reference['arms']['R'], reference['arms']['rerank'])
    check_equal('identity arm changes', [], reference['transitions'])
    for name, values in curves.items():
        if name.startswith('R_') and not np.array_equal(values, curves['rerank_'+name[2:]]):
            raise ContractError(f'R versus identity arm differs: {name}')
    model = model_for('cpu')
    n = 8
    probes = [*supervised[:n], *inner.samples[:n], *dev.samples[:n]]
    scores = predict(model, loader(probes), 'cpu')
    check_equal('real crops inference count', 3*n, len(scores))
    if not np.isfinite(scores).all():
        raise ContractError('Nonfinite preflight scores')
    return {'status': 'preflight_passed', 'ready_sha256': READY_SHA, 'r_checkpoint_sha256': R_SHA,
            'prediction_sha256': PREDICTION_SHA, 'fit': pool_summary(fit), 'inner': pool_summary(inner),
            'dev': pool_summary(dev), 'supervised': dict(Counter(s.kind for s in supervised)),
            'exclusions_mapped': len(blocked),
            'exclusions_applied': sorted([s.key, s.index] for s in fit.samples
                                         if s.target is not None and (s.key, s.index) in blocked),
            'R_same_pool': baseline, 'overlap': {'key': 0, 'image_sha256': 0, 'group_id': 0},
            'identity_comparison': 'passed', 'real_crop_forward': len(scores), 'optimizer_steps': 0}


def epoch_key(result, epoch):
    point = result['point']
    if point['t_rule'] is None:
        return None
    return point['recall'], result['ap']['ap50_95'], -epoch


def environment(device):
    return {'python': platform.python_version(), 'torch': str(torch.__version__),
            'numpy': np.__version__, 'pillow': Image.__version__, 'device': str(device),
            'cuda': torch.version.cuda if device.type == 'cuda' else None,
            'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None}


def protocol(rules):
    paths = [Path(__file__), ROOT/'scripts/evaluate.py', ROOT/'scripts/detection_metrics.py',
             ROOT/'scripts/source_policy.py', ROOT/'scripts/source_contract.py']
    return {'ready_sha256': READY_SHA, 'r_checkpoint_sha256': R_SHA, 'predictions': PREDICTION_SHA,
            'rules_sha256': rules['sha256'], 'code': {p.name: sha256(p) for p in paths}, 'recipe': RECIPE,
            'exclusions_sha256': sha256(EXCLUSIONS)}


def create_run(path):
    path = path.resolve()
    if path.parent != RUNS.resolve():
        raise ContractError('Choose a new experiment directory directly under runs/')
    path.mkdir(exist_ok=False)
    return path


def train(output, device, identity, rules):
    fit, inner, supervised, blocked = train_inputs(identity)
    torch.manual_seed(RECIPE['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = model_for(device)
    optimizer, loss_fn = optimizer_for(model), nn.BCEWithLogitsLoss()
    record = {'identity': protocol(rules), 'environment': environment(device), 'status': 'running',
              'fit': split_record(fit, supervised), 'inner': split_record(inner),
              'exclusions': sorted(blocked), 'best': None, 'epochs': []}
    output = create_run(output)
    history_path = output / 'run.json'
    write_json(history_path, record)
    start = time.perf_counter()
    batches = loader(supervised, shuffle=True, cache=True)
    validation = loader(inner.samples, cache=True)
    record['crop_cache'] = {'bytes': batches.dataset.cache.nbytes + validation.dataset.cache.nbytes,
                            'seconds': time.perf_counter()-start}
    write_json(output / 'run.tmp', record)
    (output / 'run.tmp').replace(history_path)
    best_key = None
    for epoch in range(1, RECIPE['epochs']+1):
        start = time.perf_counter()
        model.train()
        total = 0.
        for image, target in batches:
            total += len(image) * fit_step(model, optimizer, loss_fn, image, target, device)
        scores = predict(model, validation, device)
        result, _ = pedestrian_metrics(with_scores(inner, scores), rules)
        rank = epoch_key(result, epoch)
        weights = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
        # Stage each replacement inside the new run so an interrupted write preserves the previous file.
        scratch = output / 'checkpoint.tmp'
        torch.save(weights, scratch)
        scratch.replace(output / 'last.pt')
        if rank is not None and (best_key is None or rank > best_key):
            best_key = rank
            torch.save({'epoch': epoch, 'model': weights['model']}, scratch)
            scratch.replace(output / 'best.pt')
            record['best'] = {'epoch': epoch, 'sha256': sha256(output / 'best.pt')}
        entry = {'epoch': epoch, 'loss': total/len(supervised), 'inner': result,
                 'seconds': time.perf_counter()-start}
        record['epochs'].append(entry)
        write_json(output / 'run.tmp', record)
        (output / 'run.tmp').replace(history_path)
        print(json.dumps(entry, allow_nan=False), flush=True)
    record['last'] = {'epoch': RECIPE['epochs'], 'sha256': sha256(output / 'last.pt')}
    record['status'] = 'complete' if best_key is not None else 'no_feasible_inner_point'
    write_json(output / 'run.tmp', record)
    (output / 'run.tmp').replace(history_path)


def load_selected(run, device, rules):
    run = run.resolve()
    if run.parent != RUNS.resolve():
        raise ContractError('Expected a reranker experiment under runs/')
    record = read_json(run/'run.json')
    check_equal('training run status', 'complete', record['status'])
    check_equal('training protocol', protocol(rules), record['identity'])
    best = record['best']
    path = verified_file(run, dict(best, path='best.pt'))
    bundle = torch.load(path, weights_only=True, map_location=device)
    check_equal('selected epoch', best['epoch'], bundle['epoch'])
    model = model_for(device)
    model.load_state_dict(bundle['model'])
    return model, best


def evaluate_run(run, device, identity, rules):
    model, best = load_selected(run, device, rules)
    pool = build_pool('source_dev', identity)
    assert_r_baseline(pool, rules)
    relative = 'final_eval/source_dev/metrics.json'
    manifest, _ = identity
    archive = read_json(sealed_file(manifest, relative))
    destination = run.resolve()/'source_dev'
    if destination.exists():
        raise ContractError(f'Development comparison already exists: {destination}')
    start = time.perf_counter()
    scores = predict(model, loader(pool.samples), device)
    seconds = time.perf_counter()-start
    report, curves = compare(pool, scores, rules)
    # Only pedestrian scores are emitted as an overlay. All other detections
    # come byte-for-byte from the same original R prediction file in both arms.
    report.update({'selected': best, 'environment': environment(device), 'identity': protocol(rules),
                   'pool': pool_summary(pool), 'crop_and_inference_seconds': seconds,
                   'R_full': {'file': relative, 'sha256': manifest['files'][relative]['sha256'],
                              'classes': archive['metrics']['per_class'],
                              'local_workpoints': archive['local_reselected_workpoints']},
                   'unchanged_classes': NAMES[1:]})
    scored = []
    for row, reranked in zip(pool.rows, with_scores(pool, scores), strict=True):
        pred = row['pred']
        for index, box, original, score in zip(pred['ids'], pred['original_boxes'], pred['conf'],
                                                reranked['pred']['conf'], strict=True):
            scored.append({'purpose': 'source_dev', 'key': row['key'], 'image_sha256': row['image_sha256'],
                           'index': index, 'box': box, 'R': original, 'rerank': score})
    with TemporaryDirectory(prefix='.eval-', dir=destination.parent) as staging:
        result = Path(staging)/'source_dev'
        result.mkdir()
        dm.write_rows(result/'scores.jsonl.gz', scored)
        np.savez_compressed(result/'pr.npz', **curves)
        write_json(result/'metrics.json', report)
        result.rename(destination)
    print(json.dumps({'decision': report['decision'], 'output': str(destination)}, ensure_ascii=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    modes.add_parser('preflight', help='Verify frozen input, split and same-pool R; forward-only crop probes')
    fit = modes.add_parser('train', help='Run the registered fit when authorized')
    fit.add_argument('--output', required=True, type=Path)
    fit.add_argument('--device', default='cpu')
    evaluation = modes.add_parser('evaluate', help='Evaluate a previously locked best on source_dev')
    evaluation.add_argument('--run', required=True, type=Path)
    evaluation.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    device = torch.device(getattr(args, 'device', 'cpu'))
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    rules = load_rules(RULES)
    try:
        identity = verify_r()
        if args.mode == 'preflight':
            print(json.dumps(preflight(identity, rules), ensure_ascii=False, indent=2, allow_nan=False))
        elif args.mode == 'train':
            train(args.output, device, identity, rules)
        else:
            evaluate_run(args.run, device, identity, rules)
    except (ContractError, OSError) as exc:
        parser.exit(2, f'{exc}\n')


if __name__ == '__main__':
    main()
