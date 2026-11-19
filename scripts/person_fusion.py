"""Fuse only pedestrian predictions, preserving the same checkpoint's other classes."""
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

import detection_metrics as dm
from source_contract import ContractError, NAMES, check_equal, sha256, write_json

FIELDS = ('cls', 'conf', 'original_boxes')
CLUSTER_IOU = .55
NMS_IOU = .7
FLOOR = .001
MAX_DET = 300
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'data/soda10m/splits/source_v1'
READY_SHA = '72e75ebd99014c830e0227061e390e6aed768ae724b532a5c2c90fd8dec93e85'
PROFILES = {
    'a1': ('runs/a_001/weights/best.pt', '9db1ad388d28ecf3d912feb8e686056d497acde705197a5156a320579dc18bd7',
           (640, False), ((960, False), (1280, False), (960, True), (1280, True))),
    'r80': ('runs/a_002/train/weights/diagnostic.pt', 'e4c873eda51d1a2b9ab722a9cb8f8bc0de9091a320184a8dc6f49fb84421c6ab',
            (960, False), ((960, False), (1280, False), (1600, False))),
}


def _iou(one, many):
    intersection = np.maximum(np.minimum(one[2:], many[:, 2:]) - np.maximum(one[:2], many[:, :2]), 0).prod(axis=1)
    a = np.maximum(one[2:] - one[:2], 0).prod()
    b = np.maximum(many[:, 2:] - many[:, :2], 0).prod(axis=1)
    return intersection / np.maximum(a+b-intersection, np.float32(1e-7))


def _detections(pred):
    return zip(*(pred[k] for k in FIELDS), strict=True)


def _clusters(boxes, scores, votes, num_views):
    sums, means = np.zeros_like(boxes), np.zeros_like(boxes)
    used = np.zeros((len(boxes), num_views), dtype=bool)
    masses = np.zeros_like(scores)
    size = 0
    for i in np.argsort(-scores, kind='stable'):
        overlaps = _iou(boxes[i], means[:size])
        overlaps[used[:size, votes[i]]] = -1
        j = int(np.argmax(overlaps)) if size else -1
        if j < 0 or overlaps[j] < CLUSTER_IOU:
            j = size
            size += 1
        masses[j] += scores[i]
        sums[j] += boxes[i]*scores[i]
        means[j] = sums[j]/masses[j]
        used[j, votes[i]] = True
    return means[:size], masses[:size]/num_views


def _people(views):
    candidates = [(box, score, view) for view, pred in enumerate(views) for c, score, box in _detections(pred)
                  if c == 0 and score > FLOOR]
    if not candidates:
        return []
    boxes, scores, votes = zip(*candidates, strict=True)
    boxes, scores = _clusters(np.asarray(boxes, dtype=np.float32), np.asarray(scores, dtype=np.float32),
                              votes, len(views))
    active = np.argsort(-scores, kind='stable')
    active = active[scores[active] > FLOOR]
    result = []
    while len(active):
        i, remaining = active[0], active[1:]
        result.append((0, float(scores[i]), boxes[i].tolist()))
        active = remaining[_iou(boxes[i], boxes[remaining]) <= NMS_IOU]
    return result


def fuse_predictions(anchor, views):
    """Take box/score/class only, with no labels. Reserve the final budget for the anchor's non-person boxes first."""
    if not views:
        raise ValueError('At least one pedestrian view is required')
    others = [d for d in _detections(anchor) if d[0] != 0]
    if len(others) > MAX_DET:
        raise ValueError('Non-pedestrian anchor exceeds the fixed output budget')
    people = _people(views)
    remaining = MAX_DET - len(others)
    result = {k: [] for k in FIELDS}
    for detection in others+people[:remaining]:
        for name, value in zip(FIELDS, detection, strict=True):
            result[name].append(value)
    return result, max(0, len(people)-remaining)


def unmirror_boxes(boxes, width):
    """Map x corners of horizontal-flip predictions back to the padded input."""
    if len(boxes):
        boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
    return boxes


def _dev_index(rows):
    by_key = {r['key']: r for r in rows}
    if not by_key or len(by_key) != len(rows) or any(r['purpose'] != 'source_dev' for r in rows):
        raise ContractError('Fusion evaluation requires unique, nonempty source_dev rows')
    return by_key


def _identity(row):
    return ([row[k] for k in ('group_id','image_sha256','label_sha256','width','height')],
            row['gt']['cls'], row['gt']['original_boxes'])


def _original(row):
    return dict(row, gt=dict(row['gt'], boxes=row['gt']['original_boxes']))


def evaluate_rows(anchor, views):
    """Recompute native-style matches; anchor edges remain exact for its untouched classes."""
    index = _dev_index(anchor)
    others = [_dev_index(v) for v in views]
    for view in others:
        check_equal('fusion image key identity', set(index), set(view))
    output, caps = [], []
    for row in anchor:
        key = row['key']
        candidates = [v[key] for v in others]
        for candidate in candidates:
            check_equal('fusion data identity '+key, _identity(row), _identity(candidate))
        pred, dropped = fuse_predictions(row['pred'], tuple(v['pred'] for v in candidates))
        original_indices = [i for i, c in enumerate(row['pred']['cls']) if c != 0]
        keep = {p: i for i, p in enumerate(original_indices)}
        prefix = len(keep)
        iou = dm.box_iou(row['gt']['original_boxes'], pred['original_boxes'][prefix:])
        gt_idx, pred_idx = np.nonzero(iou)
        edges = [[g, keep[p], v] for g, p, v in row['iou_edges'] if p in keep]
        edges.extend([int(g), int(p)+prefix, float(iou[g, p])] for g,p in zip(gt_idx, pred_idx, strict=True))
        joint = _original({k: v for k, v in row.items() if k not in ('nms_audit','input_shape','ratio_pad')})
        joint.update(pred=dict(pred, ids=list(range(len(pred['cls']))), boxes=pred['original_boxes']), iou_edges=edges)
        flags, pairs = dm.ap_flags(joint)
        joint.update(ap_tp=flags.tolist(), ap_pairs=pairs)
        output.append(joint)
        if dropped:
            caps.append({'key':key, 'person_dropped':dropped})
    return output, caps


def _capture(torch, contract, data, checkpoint, scratch, size, mirrored):
    from evaluate import evidence_validator_class, validator_args
    from source_policy import summarize

    class MirrorValidator(evidence_validator_class()):
        def preprocess(self, batch):
            batch = super().preprocess(batch)
            self.input_width = batch['img'].shape[-1]
            batch['img'] = batch['img'].flip(-1)
            return batch

        def postprocess(self, predictions):
            output = super().postprocess(predictions)
            for item in output:
                unmirror_boxes(item['bboxes'], self.input_width)
            return output

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    validator = MirrorValidator if mirrored else evidence_validator_class()
    engine = validator(args=validator_args(data, imgsz=size), save_dir=scratch/f'{size}-{mirrored}',
                       records=contract['records'], source_root=contract['root'], stream=None)
    engine(model=str(checkpoint))
    check_equal('complete dev keys', {r['key'] for r in contract['records']}, engine.keys)
    check_equal('explicit size', size, engine.dataloader.dataset.imgsz)
    check_equal('no validator augmentation', False, engine.dataloader.dataset.augment)
    rules = contract['rules']
    record, _, _ = summarize(engine.rows, rules)
    for cid, native in zip(engine.metrics.box.ap_class_index, engine.metrics.box.all_ap, strict=True):
        value = record['metrics']['per_class'][NAMES[int(cid)]]['ap_iou']
        if not np.allclose(native, value, atol=1e-12, rtol=0):
            raise ContractError('Native view AP disagrees with the offline evaluator')
    info = {'imgsz':size, 'mirror':mirrored, 'settings':{k:getattr(engine.args,k) for k in validator_args(data) if k!='data'},
            'seconds':time.perf_counter()-started, 'peak_torch_bytes':torch.cuda.max_memory_allocated(),
            'speed_ms':engine.speed, 'nms':record['nms']}
    rows = engine.rows
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return rows, info


def run(profile, scratch, output):
    from shutil import copyfile
    from evaluate import stage_subset, verify_subset
    from source_policy import load_rules, paired_effects, recall_intervals, reference_at, SelectionPolicy, summarize

    scratch, output = Path(scratch).resolve(), Path(output).resolve()
    if scratch.is_relative_to(ROOT) or (output.is_relative_to(ROOT) and not output.is_relative_to(ROOT/'runs')):
        raise ContractError('Scratch must be external; in-project evidence must stay in runs/')
    if output.exists():
        raise ContractError('Will not overwrite an existing experiment')
    relative, expected, anchor_view, person_views = PROFILES[profile]
    checkpoint = ROOT/relative
    check_equal('checkpoint before inference', expected, sha256(checkpoint))
    contract = verify_subset(SOURCE, READY_SHA, 'source_dev')
    check_equal('full source_dev count', 500, len(contract['records']))
    rules_path = ROOT/'RULES.md'
    rules = contract['rules'] = load_rules(rules_path)
    ref = reference_at(ROOT)
    with tempfile.TemporaryDirectory(prefix='person-', dir=scratch) as path:
        temp = Path(path)
        for name in ('YOLO_CONFIG_DIR','TORCH_HOME','MPLCONFIGDIR','TEMP','TMP','TMPDIR','XDG_CONFIG_HOME'):
            folder = temp/name.lower()
            folder.mkdir()
            os.environ[name] = str(folder)
        os.environ.update(YOLO_AUTOINSTALL='false', WANDB_MODE='disabled')
        import torch
        import torchvision
        import ultralytics
        torch.set_num_threads(4)
        data = stage_subset(contract, temp/'data')
        completed, observations = {}, []
        for view in dict.fromkeys((anchor_view, *person_views)):
            completed[view], info = _capture(torch,contract,data,checkpoint,temp,*view)
            observations.append(info)
            print(f'Native view {view}: {len(completed[view])} images, {info["seconds"]:.1f}s',flush=True)
        source = completed[anchor_view]
        rows, caps = evaluate_rows(source, tuple(completed[v] for v in person_views))
        baseline, old_match, _ = summarize(source,rules)
        report, matches, curves = summarize(rows,rules)
        policy = SelectionPolicy(rules,ref)
        checks, guards = policy.assess(report)
        retained = sum(len([d for d in _detections(r['pred']) if d[0] != 0]) for r in source)
        for a,b in zip(source, rows, strict=True):
            check_equal('nonperson boxes/scores '+a['key'], [d for d in _detections(a['pred']) if d[0] != 0],
                        [d for d in _detections(b['pred']) if d[0] != 0])
        for name in NAMES[1:]:
            check_equal('nonperson operating point '+name, baseline['workpoints'][name], report['workpoints'][name])
        baseline_intervals = recall_intervals(source,old_match)
        candidate_intervals = recall_intervals(rows,matches)
        changes = paired_effects([(17, [_original(r) for r in source], rows)], rules)
        check_equal('checkpoint after inference', expected, sha256(checkpoint))
        check_equal('READY after inference', READY_SHA, sha256(SOURCE/'READY.json'))

        output.mkdir(parents=True, exist_ok=False)
        code = output/'code'
        code.mkdir()
        copyfile(__file__,code/'person_fusion.py')
        sub = output/'source_dev'
        sub.mkdir()
        dm.write_rows(sub/'objects.jsonl.gz',rows)
        dm.write_rows(sub/'matches.jsonl.gz',matches)
        np.savez_compressed(sub/'pr.npz',**curves)
        report.pop('nms')  # Fusion has its own final cap accounting below.
        compare = {'profile':profile, 'baseline':baseline, 'candidate':report, 'checks':checks, 'guards':guards,
                   'nonperson_boxes_scores_and_workpoints_exact':True, 'nonperson_predictions':retained,
                   'fusion_output_cap_images':caps, 'baseline_intervals':baseline_intervals,
                   'candidate_intervals':candidate_intervals, 'paired':changes,
                   'metric_scope':'source_dev development evaluation, with native per-view metrics and fused original-coordinate pedestrian matches',
                   'cloud':{'gpu_hours_budget':0,'cost_cny_budget':0,'gpu_hours_used':0,'cost_cny_used':0}}
        write_json(output/'comparison.json',compare)
        files = {p.relative_to(output).as_posix(): {'bytes':p.stat().st_size,'sha256':sha256(p)}
                 for p in output.rglob('*') if p.is_file()}
        scripts = ('person_fusion.py','evaluate.py','detection_metrics.py','source_contract.py','source_policy.py')
        write_json(output/'manifest.json', {'status':'complete','scope':'source_dev','profile':profile,
                                           'checkpoint':{'path':relative,'sha256':expected},'ready':READY_SHA,
                                           'dev_manifest':contract['manifest'],'rules':rules['sha256'],
                                           'anchor_view':anchor_view,'person_views':person_views,'observations':observations,
                                           'fusion':{'cluster_iou':CLUSTER_IOU,'nms_iou':NMS_IOU,'floor':FLOOR,'max_det':MAX_DET},
                                           'environment':{'python':sys.version,'torch':torch.__version__,
                                                          'torchvision':torchvision.__version__,
                                                          'ultralytics':ultralytics.__version__,
                                                          'gpu':torch.cuda.get_device_name(0)},
                                           'code_sha256':{f'scripts/{s}':sha256(ROOT/'scripts'/s) for s in scripts},
                                           'training_performed':False,'final_test_executed':False, 'files':files})
    p = report['workpoints']['pedestrian']
    aps = report['metrics']['per_class']['pedestrian']
    print(f'AP50={aps["ap50"]:.6f} AP50:95={aps["ap50_95"]:.6f} AP75={aps["ap75"]:.6f} '
          f'P={p["precision"]:.6f} R={p["recall"]:.6f} TP/FP/FN={p["tp"]}/{p["fp"]}/{p["fn"]}',flush=True)
    print('Checks:',checks,'\nUntouched nonperson predictions:',retained,'\nSealed:',output,flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--profile', choices=PROFILES, default='a1')
    cli.add_argument('--scratch', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    opts = cli.parse_args()
    run(opts.profile, opts.scratch, opts.output)
