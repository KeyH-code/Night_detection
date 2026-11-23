"""Capture and seal the frozen-A, five-view night_dev zero-shot run."""
from __future__ import annotations

import gc
from importlib.metadata import version
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time

import numpy as np

import detection_metrics as dm
from evaluate import evidence_validator_class, stage_subset, validator_args
from night_contract import DEV, ROOT, read_sealed_json, verify_dev
from night_eval import RESEARCH, evaluate_rows, fuse_dev_rows, recovery
from person_fusion import PROFILES, unmirror_boxes
from seed_eval import probe_gpu
from source_contract import ContractError, NAMES, canonical_hash, check_equal, read_json, sha256, write_json

LOCKED_VIEWS = (PROFILES['a1'][2], *PROFILES['a1'][3])
CODE = ('night_run.py', 'night_eval.py', 'night_contract.py', 'evaluate.py', 'person_fusion.py',
        'seed_eval.py', 'detection_metrics.py', 'source_contract.py')
PACKAGE_NAMES = ('torch', 'torchvision', 'ultralytics')


def preflight_records(records):
    ordered = sorted(records, key=lambda r: (r['group_id'], r['key']))
    if (len(ordered) < 8 or len({r['key'] for r in ordered}) != len(ordered)
            or any(r['purpose'] != DEV for r in ordered)):
        raise ContractError('Expected the verified night_dev list for preflight')
    # Sample across the stable risk-group order; labels do not enter the choice.
    return [ordered[i * (len(ordered) - 1) // 7] for i in range(8)]


def _capture(torch, contract, data, checkpoint, temp, size, mirror):
    base = evidence_validator_class()

    class MirrorValidator(base):
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
    start = time.perf_counter()
    args = validator_args(data, imgsz=size)
    engine = (MirrorValidator if mirror else base)(args=args, save_dir=temp / f'{size}-{mirror}',
                                                  records=contract['records'], source_root=data.parent, stream=None)
    engine(model=str(checkpoint))
    rows = engine.rows
    check_equal('captured development keys', {r['key'] for r in contract['records']}, engine.keys)
    check_equal('eval settings', (size, False, 1, 0, True, 16, 'val', 300),
                (engine.dataloader.dataset.imgsz, engine.dataloader.dataset.augment,
                 engine.args.batch, engine.args.workers, engine.args.rect, engine.args.quantize,
                 engine.args.mode, engine.args.max_det))
    offline, _ = dm.summarize_ap(rows)
    for cid, native in zip(engine.metrics.box.ap_class_index, engine.metrics.box.all_ap, strict=True):
        computed = offline['per_class'][NAMES[int(cid)]]['ap_iou']
        if not np.allclose(native, computed, atol=1e-12, rtol=0):
            raise ContractError('Native view AP disagrees with the offline matcher')
    elapsed = time.perf_counter() - start
    audit = {flag: [r['key'] for r in rows if r['nms_audit'][flag]]
             for flag in ('pre_nms_truncated', 'at_max_det', 'max_det_truncated')}
    obs = {'imgsz': size, 'mirror': mirror, 'images': len(rows), 'seconds': elapsed,
           'speed_ms': engine.speed, 'peak_torch_bytes': torch.cuda.max_memory_allocated(),
           'nms': audit, 'settings': {k: getattr(engine.args, k) for k in args if k != 'data'}}
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return rows, obs


def _collect(torch, contract, checkpoint, tmp, role):
    tmp.mkdir()
    data = stage_subset(contract, tmp / 'data')
    captures, observations = [], []
    for size, mirror in LOCKED_VIEWS:
        rows, obs = _capture(torch, contract, data, checkpoint, tmp, size, mirror)
        captures.append(rows)
        observations.append(obs)
        print(f'{role} view {size} mirror={mirror}: {len(rows)} images in {obs["seconds"]:.1f}s', flush=True)
    fused, caps = fuse_dev_rows(captures[0], captures[1:])
    return fused, observations, caps


def _matches(rows, results):
    for row in rows:
        record = {}
        for name, workpoint in results['workpoints'].items():
            t = workpoint['threshold']
            classes = {}
            for c in RESEARCH:
                stats, pairs, _ = dm.image_point(row, c, t)
                classes[NAMES[c]] = dict(stats, pairs=pairs.tolist())
            record[name] = classes
        yield {'key': row['key'], 'group_id': row['group_id'], 'workpoints': record}


def save_predictions(folder, rows, expected, protocol, reference):
    check_equal('complete night_dev coverage', (len(expected), expected, {DEV}),
                (len(rows), {r['key'] for r in rows}, {r['purpose'] for r in rows}))
    metrics, curves = evaluate_rows(rows, protocol)
    path = folder / 'objects.jsonl.gz'
    dm.write_rows(path, rows)
    replay = dm.read_rows(path)
    check_equal('persisted objects', canonical_hash(rows), canonical_hash(replay))
    replayed_metrics, replayed_curves = evaluate_rows(replay, protocol)
    check_equal('offline metrics replay', metrics, replayed_metrics)
    check_equal('offline curve keys', set(curves), set(replayed_curves))
    for k, values in curves.items():
        if not np.array_equal(values, replayed_curves[k], equal_nan=True):
            raise ContractError(f'Offline curve replay disagrees: {k}')
    np.savez_compressed(folder / 'pr.npz', **curves)
    match_path = folder / 'matches.jsonl.gz'
    dm.write_rows(match_path, _matches(rows, metrics))
    check_equal('offline workpoint matches replay', list(_matches(replay, replayed_metrics)), dm.read_rows(match_path))
    day, night = (m['workpoints']['primary']['per_class'] for m in (reference, metrics))
    metrics['zero_shot_comparison'] = recovery(day, night, protocol['recovery_ratio'])
    write_json(folder / 'metrics.json', metrics)
    return metrics


def run(protocol, scratch, output, reference_path, reference_sha):
    scratch, output, reference_path = (Path(p).absolute() for p in (scratch, output, reference_path))
    if (not scratch.is_dir() or scratch.resolve() != scratch or scratch.is_relative_to(ROOT)
            or scratch.drive != output.drive or output.parent.resolve() != ROOT / 'runs'
            or output.parent != ROOT / 'runs' or not re.fullmatch(r'night_\d{3}', output.name) or output.exists()
            or output.is_symlink() or not reference_path.is_relative_to(scratch)):
        raise ContractError('Use an existing external scratch with the reference and a new runs/night_NNN on the same volume')
    source = protocol['source_a']
    reference = read_sealed_json(scratch, reference_path.relative_to(scratch).as_posix(), reference_sha, [])
    check_equal('reference identity and gate', ('source_dev', 500, protocol['sha256'], source['checkpoint_sha256'],
                                               source['manifest_sha256'], []),
                (reference['purpose'], reference['images'], reference['protocol_sha256'],
                 reference['source']['checkpoint_sha256'], reference['source']['run_sha256'], reference['review_reasons']))
    if not all(v is not None and v > 0 for v in reference['target_recall'].values()):
        raise ContractError('White-day reference has no usable denominator; stop for review')
    manifest = read_sealed_json(ROOT / 'runs/align_001', 'manifest.json', source['manifest_sha256'], [])
    versions = {name: version(name) for name in PACKAGE_NAMES}
    check_equal('dependencies', manifest['versions'], list(versions.values()))
    checkpoint = ROOT / 'runs/align_001/17/last.pt'
    check_equal('checkpoint before inference', source['checkpoint_sha256'], sha256(checkpoint))
    print('Scope: frozen A, night_dev only; preflight, then full zero-shot development evaluation.', flush=True)
    contract = verify_dev(ROOT / 'data/soda10m/splits/night_v1', protocol['data']['night_ready_sha256'])
    payload_size = sum((contract['root'] / row[k]).stat().st_size for row in contract['records'] for k in ('image', 'label'))
    if shutil.disk_usage(scratch).free < 2 * payload_size + 512 * 1024**2:
        raise ContractError('Insufficient space for a clean night_dev evaluation sandbox')
    gpu = probe_gpu()
    hashes = {f'scripts/{name}': sha256(ROOT / 'scripts' / name) for name in CODE}
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='night-dev-', dir=scratch) as tmpdir:
        temp = Path(tmpdir)
        for var in ('YOLO_CONFIG_DIR', 'TORCH_HOME', 'MPLCONFIGDIR', 'TEMP', 'TMP', 'TMPDIR', 'XDG_CONFIG_HOME'):
            path = temp / var.lower()
            path.mkdir()
            os.environ[var] = str(path)
        os.environ.update(YOLO_AUTOINSTALL='false', WANDB_MODE='disabled')
        import torch
        import torchvision  # Use the same NMS backend as the sealed daytime views.
        import ultralytics
        check_equal('runtime packages', list(versions.values()), [torch.__version__, torchvision.__version__, ultralytics.__version__])
        if not torch.cuda.is_available():
            raise ContractError('The locked FP16 GPU evaluation is not available')
        torch.set_num_threads(4)
        small = preflight_records(contract['records'])
        _, preflight, _ = _collect(torch, dict(contract, records=small), checkpoint, temp / 'preflight', 'Preflight')
        pre_peak = max(v['peak_torch_bytes'] for v in preflight)
        full_gpu = probe_gpu()
        if full_gpu['free_mib'] * 1024**2 < pre_peak + 512 * 1024**2:
            raise ContractError('Insufficient GPU headroom after night_dev preflight')
        rows, observations, fusion_caps = _collect(torch, contract, checkpoint, temp / 'full', 'Full')
        folder = temp / 'evidence'
        folder.mkdir()
        metrics = save_predictions(folder, rows, {r['key'] for r in contract['records']}, protocol, reference)
        check_equal('checkpoint after inference', source['checkpoint_sha256'], sha256(checkpoint))
        check_equal('READY after inference', contract['ready_hash'], sha256(contract['root'] / 'READY.json'))
        check_equal('protocol after inference', protocol['sha256'], sha256(ROOT / 'configs/night.json'))
        check_equal('execution code unchanged', hashes, {name: sha256(ROOT / name) for name in hashes})
        shutil.copyfile(reference_path, folder / 'reference.json')
        check_equal('reference copy SHA-256', reference_sha, sha256(folder / 'reference.json'))
        shutil.copyfile(ROOT / 'configs/night.json', folder / 'protocol.json')
        for name, digest in hashes.items():
            target = folder / 'code' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
            check_equal('archived code SHA-256', digest, sha256(target))
        truncated = any(v['nms']['pre_nms_truncated'] or v['nms']['max_det_truncated'] for v in observations)
        reasons = [*metrics['review_reasons'], *(['output_truncation_review'] if truncated or fusion_caps else [])]
        write_json(folder / 'run.json', {
            'status': 'complete_needs_review' if reasons else 'complete', 'purpose': DEV, 'images': len(rows),
            'review_reasons': reasons, 'training_performed': False, 'final_test_executed': False, 'cloud_gpu_used': False,
            'checkpoint_sha256': source['checkpoint_sha256'], 'protocol_sha256': protocol['sha256'],
            'reference_sha256': reference_sha, 'ready_sha256': contract['ready_hash'], 'input': contract['manifest'],
            'preflight': {'keys': [r['key'] for r in small], 'views': preflight, 'metric_used': False},
            'views': observations, 'fused_person_caps': fusion_caps, 'code_sha256': hashes,
            'environment': {'python': sys.version, 'packages': {**versions, 'numpy': np.__version__},
                            'gpu': torch.cuda.get_device_name(), 'initial_gpu': gpu, 'post_preflight_gpu': full_gpu},
            'evaluation_and_sealing_seconds': time.perf_counter() - started, 'offline_replay_exact': True,
        })
        sealed = {f.relative_to(folder).as_posix(): {'sha256': sha256(f), 'bytes': f.stat().st_size}
                  for f in sorted(folder.rglob('*')) if f.is_file()}
        write_json(folder / 'manifest.json', {'status': 'sealed', 'scope': 'night_dev zero-shot, white-day reference only',
                                               'checkpoint_sha256': source['checkpoint_sha256'], 'files': sealed})
        if output.exists():
            raise ContractError('Refusing to overwrite a night evaluation run')
        folder.rename(output)
        check_equal('persisted manifest', sealed, read_json(output / 'manifest.json')['files'])
        for name, seal in sealed.items():
            path = output / name
            check_equal(f'published {name}', (seal['bytes'], seal['sha256']), (path.stat().st_size, sha256(path)))
        print(f'Sealed {output.name}: {len(rows)} night_dev images; review flags={reasons}', flush=True)
