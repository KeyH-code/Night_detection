"""Evaluate a frozen checkpoint on one explicit source purpose, without training."""
from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np
import detection_metrics as dm

from source_contract import (ContractError, NAMES, NORMALIZATION, check_equal, primary_fitness,
                             read_json, safe_path, sha256, verified_file, write_json)

ROOT = Path(__file__).resolve().parents[1]


def verify_subset(root, ready_hash, purpose):
    if purpose not in ('source_train', 'source_dev'):
        raise ContractError(f'Forbidden evaluation purpose: {purpose}')
    root = root.resolve()
    check_equal('READY SHA', ready_hash, sha256(root / 'READY.json'))
    ready = read_json(root / 'READY.json')
    check_equal('schema', 1, ready['schema_version'])
    check_equal('version', 'source_v1', ready['version'])
    check_equal('status', 'frozen', ready['status'])
    check_equal('normalization', NORMALIZATION, ready['normalization_version'])
    check_equal('names', NAMES, ready['categories']['names'])
    verified_file(root, ready['categories'])
    verified_file(root, ready['grouping_protocol'])
    entry = ready['manifests'][purpose]
    manifest = read_json(verified_file(root, entry))
    check_equal('manifest purpose', purpose, manifest['purpose'])
    check_equal('manifest count', ready['counts'][purpose], len(manifest['records']))
    whitelist = {e['path']: e for e in ready['files'] if e['purpose'] == purpose}
    used, keys = set(), set()
    for r in manifest['records']:
        check_equal('record purpose', purpose, r['purpose'])
        if r['key'] in keys:
            raise ContractError('Duplicate image key')
        keys.add(r['key'])
        for kind in ('image', 'label'):
            rel = r[kind]
            prefix = ('images/' if kind == 'image' else 'labels/') + purpose + '/'
            if rel not in whitelist or not rel.startswith(prefix) or rel in used:
                raise ContractError(f'Unapproved/duplicate evaluation path: {rel}')
            e = whitelist[rel]
            check_equal('file kind', kind, e['kind'])
            check_equal('record hash', r[kind + '_sha256'], e['sha256'])
            verified_file(root, e)
            used.add(rel)
    check_equal('subset whitelist', used, set(whitelist))
    return {'root': root, 'ready': ready, 'ready_hash': ready_hash,
            'purpose': purpose, 'records': manifest['records'], 'manifest': entry}


def stage_subset(contract, destination):
    """Only the selected verified images/labels enter the writable validator sandbox."""
    purpose = contract['purpose']
    for r in contract['records']:
        for kind in ('image', 'label'):
            source = safe_path(contract['root'], r[kind])
            target = destination / r[kind]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            check_equal('copied ' + r[kind], r[kind + '_sha256'], sha256(target))
    config = destination / 'dataset.yaml'
    # Both schema fields point at the same selected split; DetectionValidator alone runs, never Trainer.
    write_json(config, {'path': destination.as_posix(), 'train': 'images/' + purpose,
                       'val': 'images/' + purpose, 'names': NAMES, 'nc': 6})
    return config


def validator_args(data, batch=1, device='0', workers=0, imgsz=640):
    return {'data': str(data), 'imgsz': imgsz, 'batch': batch, 'device': device,
            'workers': workers, 'split': 'val', 'task': 'detect', 'mode': 'val',
            'conf': .001, 'iou': .7, 'max_det': 300, 'augment': False,
            'agnostic_nms': False, 'rect': True, 'quantize': None if device == 'cpu' else 16,
            'plots': False, 'save_json': False, 'save_txt': False, 'verbose': False}


def evaluation_ema(ema, device):
    """Match saved fp16 EMA -> fp32 -> device before backend fusion, without mutating training."""
    from copy import deepcopy
    import torch
    model = deepcopy(ema).half().to(memory_format=torch.contiguous_format).float().eval().to(device)
    if hasattr(model, 'criterion'):
        model.criterion = None
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ContractError('Nonfinite/overflowed evaluation EMA; no clamping')
    return model


def metric_record(metrics):
    check_equal('metric names', NAMES, [metrics.names[i] for i in range(6)])
    box = metrics.box
    score, aps = primary_fitness(box.ap_class_index, box.all_ap)
    rows = {int(c): i for i, c in enumerate(box.ap_class_index)}
    per_class = {}
    for c, name in enumerate(NAMES):
        support = {'images': int(metrics.nt_per_image[c]), 'instances': int(metrics.nt_per_class[c])}
        if c not in rows:
            per_class[name] = dict(support, precision=None, recall=None, ap50=None, ap50_95=None)
        else:
            p, r, ap50, ap = box.class_result(rows[c])
            per_class[name] = dict(support, precision=float(p), recall=float(r), ap50=float(ap50), ap50_95=float(ap))
    from ultralytics.utils.metrics import smooth
    index = smooth(box.f1_curve.mean(0), .1).argmax()
    return {'primary_ap50_95': score, 'map50': float(box.map50), 'map50_95': float(box.map),
            'precision': float(box.mp), 'recall': float(box.mr), 'per_class': per_class,
            'pr_confidence': float(box.px[index]),
            'pr_policy': 'shared smoothed six-class mean-F1 maximum at IoU0.5; not FPPI operating points'}


def original_gt(record, root):
    rows = []
    for line in safe_path(root, record['label']).read_text(encoding='utf-8').splitlines():
        c, x, y, w, h = map(float, line.split())
        rows.append((int(c), [(x-w/2)*record['width'], (y-h/2)*record['height'],
                              (x+w/2)*record['width'], (y+h/2)*record['height']]))
    return rows


def pack_image(pred, pbatch, record, root, iou_thresholds):
    import torch
    from ultralytics.utils import ops
    from ultralytics.utils.metrics import box_iou
    gt = original_gt(record, root)
    check_equal('GT row count ' + record['key'], len(gt), len(pbatch['cls']))
    check_equal('GT row classes ' + record['key'], [x[0] for x in gt], pbatch['cls'].int().cpu().tolist())
    gt_orig = ops.scale_boxes(pbatch['imgsz'], pbatch['bboxes'].clone(), pbatch['ori_shape'], pbatch['ratio_pad'])
    if len(gt) and np.max(np.abs(gt_orig.cpu().numpy()-np.array([v[1] for v in gt]))) > .01:
        raise ContractError('GT coordinate/label row mismatch: ' + record['key'])
    pred_orig = ops.scale_boxes(pbatch['imgsz'], pred['bboxes'].clone(), pbatch['ori_shape'], pbatch['ratio_pad'])
    iou = box_iou(pbatch['bboxes'], pred['bboxes']).cpu().numpy()
    g, p = np.nonzero(iou)
    row = {k: record[k] for k in ('key', 'purpose', 'group_id', 'image', 'label', 'image_sha256', 'label_sha256', 'width', 'height')}
    row.update(input_shape=list(pbatch['imgsz']), ratio_pad=pbatch['ratio_pad'],
               iou_thresholds=list(iou_thresholds),
               gt={'ids': list(range(len(gt))), 'cls': [x[0] for x in gt],
                   'boxes': pbatch['bboxes'].float().cpu().tolist(), 'original_boxes': [x[1] for x in gt]},
               pred={'ids': list(range(len(pred['cls']))), 'cls': pred['cls'].int().cpu().tolist(),
                     'conf': pred['conf'].float().cpu().tolist(), 'boxes': pred['bboxes'].float().cpu().tolist(),
                     'original_boxes': pred_orig.float().cpu().tolist()},
               iou_edges=[[int(a), int(b), float(iou[a, b])] for a, b in zip(g, p)])
    flags, pairs = dm.ap_flags(row)
    row.update(ap_tp=flags.tolist(), ap_pairs=pairs)
    return row


def evidence_validator_class(trace_fn=None):
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.utils import nms

    class EvidenceValidator(DetectionValidator):
        def __init__(self, *args, records, source_root, stream, trace_context=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.records_by_name = {Path(r['image']).name: r for r in records}
            check_equal('unique image stems', len(records), len(self.records_by_name))
            self.source_root, self.stream = source_root, stream
            self.rows, self.keys, self.capture = [], set(), None
            self.trace_context, self.traces = trace_context, []

        def postprocess(self, preds):
            raw = preds[0] if isinstance(preds, (tuple, list)) else preds
            if not raw.isfinite().all():
                raise ContractError('Nonfinite raw predictions; refusing silent NMS filtering')
            self.capture = raw.clone() if self.trace_context else None
            self.candidate_counts = (raw[:, 4:4+self.nc] > self.args.conf).sum((1, 2)).cpu().tolist()
            class TimeoutGuard(logging.Handler):
                def emit(self, record):
                    if 'NMS time limit' in record.getMessage():
                        raise ContractError('NMS timed out; refusing incomplete evaluation')
            guard = TimeoutGuard()
            nms.LOGGER.addHandler(guard)
            try:
                output = super().postprocess(preds)
                self.post_nms_counts = []
                for i, pred in enumerate(output):
                    count = len(pred['cls'])
                    if count == self.args.max_det:
                        # Equality with the cap alone does not prove truncation.
                        # Replay only NMS, on identical raw predictions, retaining
                        # all survivors of its unchanged max_nms candidate limit.
                        uncapped = nms.non_max_suppression(
                            raw[i:i+1].clone(), self.args.conf, self.args.iou,
                            nc=0 if self.args.task == 'detect' else self.nc,
                            multi_label=True,
                            agnostic=self.args.single_cls or self.args.agnostic_nms,
                            max_det=30000, end2end=self.end2end,
                            rotated=self.args.task == 'obb')[0]
                        import torch
                        prefix = uncapped[:self.args.max_det]
                        if not (torch.equal(prefix[:, :4], pred['bboxes']) and
                                torch.equal(prefix[:, 4], pred['conf']) and
                                torch.equal(prefix[:, 5], pred['cls'])):
                            raise ContractError('NMS cap replay changed the retained prefix')
                        count = len(uncapped)
                    self.post_nms_counts.append(count)
                return output
            finally:
                nms.LOGGER.removeHandler(guard)

        def update_metrics(self, preds, batch):
            for i, pred in enumerate(preds):
                pbatch = self._prepare_batch(i, batch)
                rec = self.records_by_name[Path(pbatch['im_file']).name]
                if rec['key'] in self.keys:
                    raise ContractError('Duplicate emitted image key')
                self.keys.add(rec['key'])
                row = pack_image(pred, pbatch, rec, self.source_root, self.iouv.cpu().tolist())
                native = self._process_batch(pred, pbatch)['tp']
                if not np.array_equal(native, np.asarray(row['ap_tp'], dtype=bool).reshape(-1, 10)):
                    raise ContractError('Offline/native per-image TP mismatch: ' + rec['key'])
                row['nms_audit'] = {'expanded_above_floor': self.candidate_counts[i],
                                    'max_nms': 30000, 'pre_nms_truncated': self.candidate_counts[i] > 30000,
                                    'final_count': len(pred['cls']), 'at_max_det': len(pred['cls']) == self.args.max_det,
                                    'max_det': self.args.max_det,
                                    'post_nms_count': self.post_nms_counts[i],
                                    'max_det_truncated': self.post_nms_counts[i] > self.args.max_det,
                                    'backend': 'torchvision' if 'torchvision' in sys.modules else 'TorchNMS'}
                if self.trace_context:
                    self.traces.append(trace_fn(self.capture[i], row, *self.trace_context))
                self.rows.append(row)
                if self.stream:
                    self.stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n')
            self.capture = None
            super().update_metrics(preds, batch)

    return EvidenceValidator



def evaluate(weights, expected_hash, contract, output, batch=1, device='0', workers=0):
    check_equal('checkpoint SHA before', expected_hash, sha256(weights))
    if output.resolve().is_relative_to(contract['root']):
        raise ContractError('Evaluation output must stay outside frozen data')
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    meta = {'status': 'running', 'purpose': contract['purpose'], 'images': len(contract['records']),
            'checkpoint_sha256': expected_hash, 'ready_sha256': contract['ready_hash'],
            'data_fingerprint': contract['ready']['data_fingerprint'], 'manifest': contract['manifest'],
            'code_sha256': sha256(Path(__file__)), 'training_performed': False,
            'dev_rerun': contract['purpose'] == 'source_dev',
            'final_test_executed': False}
    write_json(output / 'run.json', meta)
    try:
        with tempfile.TemporaryDirectory(prefix='night_eval_') as temp:
            scratch = Path(temp)
            for key in ('YOLO_CONFIG_DIR', 'MPLCONFIGDIR', 'TORCH_HOME', 'TEMP', 'TMP', 'TMPDIR'):
                os.environ[key] = str(scratch / key.lower())
                Path(os.environ[key]).mkdir()
            os.environ['WANDB_MODE'] = 'disabled'
            os.environ['YOLO_AUTOINSTALL'] = 'false'
            data = stage_subset(contract, scratch / 'data')
            staged = time.perf_counter()
            with (output / 'evaluation.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                import torch
                from ultralytics.models.yolo.detect import DetectionValidator
                torch.set_num_threads(min(4, os.cpu_count() or 1))
                args = validator_args(data, batch, device, workers)
                validator = DetectionValidator(args=args, save_dir=scratch / 'validator')
                model_start = time.perf_counter()
                validator(model=str(weights.resolve()))
                evaluated = time.perf_counter()
                check_equal('evaluated count', len(contract['records']), validator.seen)
                check_equal('no eval augmentation', False, validator.dataloader.dataset.augment)
                record = metric_record(validator.metrics)
                meta.update(parameters={k: getattr(validator.args, k) for k in args if k != 'data'},
                            validation_seconds=evaluated-model_start,
                            staging_seconds=staged-started, speed_ms_per_image=validator.speed,
                            environment={'python': sys.version, 'torch': torch.__version__,
                                         'ultralytics': importlib.metadata.version('ultralytics'),
                                         'gpu': torch.cuda.get_device_name() if device != 'cpu' else None})
                del validator
                if device != 'cpu':
                    torch.cuda.empty_cache()
        check_equal('checkpoint SHA after', expected_hash, sha256(weights))
        check_equal('READY SHA after', contract['ready_hash'], sha256(contract['root'] / 'READY.json'))
        write_json(output / 'metrics.json', record)
        meta.update(status='complete', total_seconds=time.perf_counter()-started,
                    metrics_sha256=sha256(output / 'metrics.json'))
        code = output / 'code'
        code.mkdir()
        shutil.copyfile(Path(__file__), code / 'evaluate.py')
        write_json(output / 'run.json', meta)
        return record, meta
    except BaseException as exc:
        meta.update(status='failed', error=f'{type(exc).__name__}: {exc}', elapsed_seconds=time.perf_counter()-started)
        write_json(output / 'run.json', meta)
        raise


def compare_dev(record, reference_path, checkpoint_hash, output):
    reference = read_json(reference_path)
    check_equal('reference scope', 'source_dev only', reference['scope'])
    check_equal('same comparison checkpoint', checkpoint_hash, reference['checkpoint_sha256'])
    path = output / 'comparison.csv'
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['class', 'train_ap50_95', 'dev_ap50_95',
                                                   'train_minus_dev_pp', 'train_precision', 'train_recall'])
        writer.writeheader()
        for name in NAMES:
            train = record['per_class'][name]
            dev = reference['metrics'].get(f'metrics/{name}_AP50-95')
            writer.writerow({'class': name, 'train_ap50_95': train['ap50_95'], 'dev_ap50_95': dev,
                             'train_minus_dev_pp': (train['ap50_95']-dev)*100 if train['ap50_95'] is not None and dev is not None else None,
                             'train_precision': train['precision'], 'train_recall': train['recall']})
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--weights-sha256')
    parser.add_argument('--finalize-run', type=Path, help='Complete deferred S3 train/dev exports on this local machine')
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data/soda10m/splits/source_v1')
    parser.add_argument('--ready-sha256', required=True)
    parser.add_argument('--purpose', choices=('source_train', 'source_dev'), default='source_train')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--device', default='0')
    parser.add_argument('--reference', type=Path, help='Existing same-checkpoint best_dev_metrics.json; no dev rerun')
    args = parser.parse_args()
    if args.finalize_run:
        if args.weights or args.weights_sha256 or args.output or args.reference or args.batch != 1:
            parser.error('--finalize-run uses sealed role weights and its run directory; no alternate weights/output/reference')
        from train_source import configure_paths
        configure_paths()
        from train_monitor import finalize_s3_local
        result = finalize_s3_local(args.finalize_run, args.data_root, args.ready_sha256, args.device)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if not args.weights or not args.weights_sha256 or not args.output:
        parser.error('--weights, --weights-sha256 and --output are required for single-purpose evaluation')
    contract = verify_subset(args.data_root, args.ready_sha256, args.purpose)
    result, meta = evaluate(args.weights, args.weights_sha256, contract, args.output, args.batch, args.device)
    if args.reference:
        path = compare_dev(result, args.reference, args.weights_sha256, args.output)
        meta.update(dev_reference_sha256=sha256(args.reference), comparison_sha256=sha256(path))
        write_json(args.output / 'run.json', meta)
    print(json.dumps({'metrics': result, 'seconds': meta['total_seconds']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
