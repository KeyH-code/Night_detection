"""Replay the reported five-class development evaluation from official SODA10M pixels."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import tempfile

import night_accept as accept
from make_splits import label_bytes
from night_contract import _path, read_sealed_json
from normalize import clip_box
from source_contract import ContractError, NAMES, check_equal, read_json, sha256, write_json
from verify_results import EVIDENCE, night_results, percent, verify

ROOT = Path(__file__).resolve().parents[1]
INDEX = 'configs/replay_dev.json'
INDEX_SHA256 = '23f15adfaa535ff58ea154eab41a5028852a94fd776140486a2e6689fae7d02d'
DEV_SPLITS = {'source_dev': ('train', 500), 'night_dev': ('val', 331)}


def stage_dev(raw, destination, purpose, split, mapping):
    """Recreate the frozen labels from selected official annotations; hash every copied image."""
    official, count = DEV_SPLITS[purpose]
    check_equal('development split and size', (official, count),
                (split['official_split'], len(split['samples'])))
    doc = read_sealed_json(raw, f'annotations/instance_{official}.json',
                           split['annotations_sha256'], [])
    check_equal('official class mapping', {c['official_id']: c['official_name'] for c in mapping},
                {c['id']: c['name'] for c in doc['categories']})
    chosen = {}
    for image in doc['images']:
        fingerprint = hashlib.sha256(image['file_name'].encode('utf-8')).hexdigest()
        if fingerprint in split['samples']:
            check_equal('official image basename', Path(image['file_name']).name, image['file_name'])
            if image['id'] in chosen:
                raise ContractError('Duplicate approved image ID')
            chosen[image['id']] = image, fingerprint
    check_equal('approved development membership', set(split['samples']), {f for _, f in chosen.values()})
    check_equal('unique development image count', count, len(chosen))

    labels = defaultdict(list)
    for annotation in doc['annotations']:
        if annotation['image_id'] in chosen:
            labels[annotation['image_id']].append(annotation)

    images_dir, labels_dir = destination / 'images' / purpose, destination / 'labels' / purpose
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    records = []
    for image_id, (image, fingerprint) in chosen.items():
        name = image['file_name']
        image_hash, label_hash, group = split['samples'][fingerprint]
        clipped = [dict(a, bbox=clip_box(a['bbox'], image['width'], image['height']))
                   for a in sorted(labels[image_id], key=lambda a: a['id'])]
        payload = label_bytes(image, clipped, mapping)
        check_equal(f'{purpose} label SHA-256', label_hash, hashlib.sha256(payload).hexdigest())
        label_name = Path(name).with_suffix('.txt').name
        (labels_dir / label_name).write_bytes(payload)
        target = images_dir / name
        shutil.copyfile(_path(raw, f'{official}/{name}'), target)
        check_equal(f'{purpose} image SHA-256', image_hash, sha256(target))
        records.append({'key': fingerprint, 'purpose': purpose, 'group_id': f'{group:04d}',
                        'width': image['width'], 'height': image['height'],
                        'image': f'images/{purpose}/{name}', 'label': f'labels/{purpose}/{label_name}',
                        'image_sha256': image_hash, 'label_sha256': label_hash})

    data = destination / 'dataset.yaml'
    selected = f'images/{purpose}'
    write_json(data, {'path': destination.as_posix(), 'train': selected, 'val': selected,
                      'names': NAMES, 'nc': len(NAMES)})
    return {'root': destination, 'purpose': purpose, 'records': records}, data


def capture_metrics(torch, weight, staged, policy, scratch):
    from night_eval import fuse_dev_rows
    from night_run import LOCKED_VIEWS, _capture
    from person_fusion import evaluate_rows

    contract, data = staged
    purpose = contract['purpose']
    scratch.mkdir()
    captures, observations = [], []
    for size, mirror in LOCKED_VIEWS:
        rows, observed = _capture(torch, contract, data, weight, scratch, size, mirror)
        captures.append(rows)
        observations.append(observed)
        print(f'{purpose} {size} mirror={mirror}: {len(rows)} images, {observed["seconds"]:.1f}s', flush=True)
    fuse = evaluate_rows if purpose == 'source_dev' else fuse_dev_rows
    fused, caps = fuse(captures[0], captures[1:])
    metrics = accept.evaluate_dev(fused, policy)
    metrics['capture'] = accept._audit_views(observations, caps)
    return metrics


def headline(reference, day, night, policy, supplement=None):
    protection = accept.day_protection(reference['per_class'], day['per_class'], policy)
    gaps = {name: percent(point['delta']) for name, point in protection['per_class'].items()}
    day_summary = {'macro_pp': percent(protection['macro_delta']), 'worst_pp': min(gaps.values()),
                   'per_class_pp': gaps, 'passed': protection['met']}
    base = {name: point_counts(ref) for name, ref in reference['workpoints'].items()}
    summary = {'day': day_summary}
    for name, metrics in (('night_300', night), ('night_340_targeted', supplement)):
        if metrics is not None:
            points = {cls: point_counts(row) for cls, row in metrics['workpoints'].items()}
            summary[name] = night_results(base, points, policy['recovery_ratio'])
    return summary


def point_counts(row):
    return {'tp': row['tp'], 'fp': row['fp'], 'gt': row['tp'] + row['fn']}


def headline_differences(actual, expected, path=''):
    """Match reported percentage points at display precision and counts exactly."""
    if isinstance(actual, dict):
        return [issue for key in actual
                for issue in headline_differences(actual[key], expected[key], f'{path}/{key}')]
    if type(actual) is float:
        actual, expected = round(actual, 2), round(expected, 2)
    return [] if actual == expected else [{'field': path, 'actual': actual, 'expected': expected}]


def replay(weights, staged, config, scratch):
    from seed_eval import probe_gpu
    gpu = probe_gpu()
    versions = {name: version(name) for name in config['versions']}
    check_equal('evaluation package versions', config['versions'], versions)
    import cv2
    import torch

    if not torch.cuda.is_available():
        raise ContractError('The locked FP16 development evaluation requires CUDA')
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    _, expected = verify(EVIDENCE)
    policy = config['policy']
    torch.use_deterministic_algorithms(True, warn_only=True)
    reference = capture_metrics(torch, weights['source'], staged['source_dev'], policy, scratch/'source')
    torch.use_deterministic_algorithms(False)
    day = capture_metrics(torch, weights['adapted'], staged['source_dev'], policy, scratch/'adapted-day')
    night = capture_metrics(torch, weights['adapted'], staged['night_dev'], policy, scratch/'adapted-night')
    extra = None
    if 'supplement' in weights:
        extra = capture_metrics(torch, weights['supplement'], staged['night_dev'], policy, scratch/'supplement')
    metrics = {'source_day': reference, 'adapted_day': day, 'adapted_night': night}
    if extra is not None:
        metrics['supplement_night'] = extra
    actual = headline(reference, day, night, policy, extra)
    differences = headline_differences(actual, expected)
    return {'scope': 'locked five-class development evaluation', 'index_sha256': INDEX_SHA256,
            'weights_sha256': {name: config['weights'][name] for name in weights},
            'versions': versions, 'gpu_at_admission': gpu, 'policy': policy,
            'metrics': metrics, 'summary': actual,
            'archived_headline_match': not differences, 'differences': differences}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--soda-root', required=True, type=Path, help='Official labeled directory with annotations, train and val')
    parser.add_argument('--scratch', required=True, type=Path, help='Existing temporary directory outside the repository and data')
    parser.add_argument('--source-weights', type=Path)
    parser.add_argument('--adapted-weights', type=Path)
    parser.add_argument('--supplement-weights', type=Path)
    parser.add_argument('--output', type=Path, help='New JSON file for development evaluation')
    parser.add_argument('--check-inputs', action='store_true', help='Validate the approved data and any supplied weights on CPU')
    args = parser.parse_args()
    if args.check_inputs and args.output is not None:
        parser.error('--check-inputs does not write a report')
    if not args.check_inputs and not (args.source_weights and args.adapted_weights and args.output):
        parser.error('evaluation requires --source-weights, --adapted-weights and --output')
    raw, scratch = args.soda_root.resolve(), args.scratch.resolve()
    if not scratch.is_dir() or scratch.is_relative_to(ROOT) or scratch.is_relative_to(raw):
        parser.error('--scratch must exist outside the repository and data')
    if args.output is not None:
        output = args.output.resolve()
        if (output.suffix.lower() != '.json' or output.exists() or not output.parent.is_dir()
                or output.is_relative_to(raw)):
            parser.error('--output must be a new JSON file in an existing directory outside the data')

    try:
        config = read_sealed_json(ROOT, INDEX, INDEX_SHA256, [])
        check_equal('index schema', 1, config['schema'])
        mapping = read_json(ROOT/'configs/soda10m_categories.json')['category_mapping']
        check_equal('training class order', NAMES, [entry['name'] for entry in mapping])
        with tempfile.TemporaryDirectory(dir=scratch, prefix='replay-dev-') as temporary:
            tmp = Path(temporary)
            weights = {}
            for name in ('source', 'adapted', 'supplement'):
                weight = getattr(args, f'{name}_weights')
                if weight is None:
                    continue
                if not args.check_inputs:
                    weight = Path(shutil.copyfile(weight, tmp/f'{name}.pt'))
                check_equal(f'{name} checkpoint SHA-256', config['weights'][name], sha256(weight))
                weights[name] = weight
            staged = {purpose: stage_dev(raw, tmp/purpose, purpose, config['splits'][purpose], mapping)
                      for purpose in DEV_SPLITS}
            print('PASS: verified 500 daytime and 331 nighttime development images and labels.', flush=True)
            if args.check_inputs:
                return 0
            for name in ('YOLO_CONFIG_DIR', 'MPLCONFIGDIR', 'TORCH_HOME', 'XDG_CACHE_HOME'):
                folder = tmp/name.lower()
                folder.mkdir()
                os.environ[name] = str(folder)
            (tmp/'yolo_config_dir'/'Ultralytics').mkdir()
            os.environ.update(YOLO_AUTOINSTALL='false', WANDB_MODE='disabled')
            result = replay(weights, staged, config, tmp)
        with output.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        if result['archived_headline_match']:
            print(f'PASS: archived headline metrics reproduced. Report: {output}')
            return 0
        print(f'Evaluation completed with metric differences. Report: {output}')
        return 2
    except (ContractError, OSError, ValueError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
