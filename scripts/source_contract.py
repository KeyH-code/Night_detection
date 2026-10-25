"""Read-only source_v1 consumer. No torch dependency; never opens held-out data."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path, PurePosixPath

NAMES = ['pedestrian', 'cyclist', 'car', 'truck', 'tram', 'tricycle']
PURPOSES = ('source_train', 'source_dev')
NORMALIZATION = 'clip_xyxy_to_image_bounds_v1'


class ContractError(ValueError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def check_equal(path, expected, actual):
    if expected != actual:
        raise ContractError(f'{path}: expected={expected!r}; actual={actual!r}')


def safe_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or '\\' in relative or ':' in relative:
        raise ContractError(f'Invalid relative POSIX path: {relative!r}')
    p = PurePosixPath(relative)
    if p.is_absolute() or any(x in ('..', '.') for x in relative.split('/')):
        raise ContractError(f'Path traversal/absolute path: {relative!r}')
    target = root.joinpath(*p.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ContractError(f'Path escapes root: {relative!r}')
    # Reject symlinks/junctions even when their destination happens to remain in root.
    for parent in (target, *target.parents):
        if parent == root.parent:
            break
        if parent.is_symlink() or (hasattr(parent, 'is_junction') and parent.is_junction()):
            raise ContractError(f'Symlink/junction in data path: {relative!r}')
    return target


def verified_file(root, entry):
    path = safe_path(root, entry['path'])
    if not path.is_file():
        raise ContractError(f'{path}: expected=file; actual=missing')
    check_equal(path, entry['sha256'], sha256(path))
    return path


def verify_source(root: Path, expected_ready_sha256: str) -> dict:
    root = root.resolve()
    ready_path = root / 'READY.json'
    if not ready_path.is_file():
        raise ContractError(f'WAITING_FOR_SOURCE: {ready_path}: expected=frozen READY; actual=missing')
    check_equal(ready_path, expected_ready_sha256, sha256(ready_path))
    ready = read_json(ready_path)
    check_equal('schema_version', 1, ready['schema_version'])
    check_equal('version', 'source_v1', ready['version'])
    check_equal('status', 'frozen', ready['status'])
    check_equal('normalization_version', NORMALIZATION, ready['normalization_version'])
    if not re.fullmatch('[0-9a-f]{64}', ready['data_fingerprint']):
        raise ContractError('data_fingerprint: expected=sha256 hex')
    protocol = read_json(verified_file(root, ready['grouping_protocol']))
    check_equal('data_fingerprint', canonical_hash(protocol['fingerprint_components']), ready['data_fingerprint'])
    check_equal('protocol.normalization_version', NORMALIZATION, protocol['normalization_version'])
    check_equal('protocol.development_gradient_training', False, protocol['development_gradient_training'])
    check_equal('final_test_executed', False, ready['final_test_executed'])
    category_path = verified_file(root, ready['categories'])
    check_equal('categories.names', NAMES, ready['categories']['names'])
    mapping = ready['category_mapping']
    check_equal('category_mapping', [(i, i+1, n) for i, n in enumerate(NAMES)],
                [(x['training_id'], x['official_id'], x['name']) for x in mapping])
    snapshot = read_json(category_path)
    snapshot_mapping = snapshot['category_mapping'] if isinstance(snapshot, dict) else snapshot
    check_equal('category snapshot', mapping, snapshot_mapping)
    counts = ready['counts']
    check_equal('counts total', 5000, sum(counts[p] for p in (*PURPOSES, 'source_test')))
    for p, target in zip((*PURPOSES, 'source_test'), (4000, 500, 500)):
        if abs(counts[p] - target) > 5:
            raise ContractError(f'counts.{p}: expected={target} +/-5; actual={counts[p]}')
    files = {}
    for item in ready['files']:
        if item['purpose'] not in PURPOSES or item['kind'] not in ('image', 'label'):
            raise ContractError(f'Forbidden purpose/kind: {item}')
        rel = item['path']
        expected_prefix = ('images' if item['kind'] == 'image' else 'labels') + '/' + item['purpose'] + '/'
        if not rel.startswith(expected_prefix) or rel in files:
            raise ContractError(f'Invalid/duplicate whitelist path: {rel}')
        verified_file(root, item)
        files[rel] = item
    used, keys, groups, manifests = set(), set(), {}, {}
    for purpose in PURPOSES:
        entry = ready['manifests'][purpose]
        manifest = read_json(verified_file(root, entry))
        check_equal(f'{purpose}.purpose', purpose, manifest['purpose'])
        records = manifest['records']
        check_equal(f'{purpose}.count', counts[purpose], len(records))
        check_equal(f'{purpose}.manifest count', counts[purpose], entry['count'])
        for rec in records:
            check_equal(rec['key'] + '.purpose', purpose, rec['purpose'])
            check_equal(rec['key'] + '.official_split', 'train', rec['official_split'])
            check_equal(rec['key'] + '.official_period', 'Daytime', rec['official_period'])
            components = protocol['fingerprint_components']
            for field in ('raw_annotation_sha256', 'normalized_annotation_sha256'):
                check_equal(rec['key'] + '.' + field, components[field]['train'], rec[field])
            if rec['key'] in keys:
                raise ContractError(f'Duplicate image key: {rec["key"]}')
            keys.add(rec['key'])
            group = rec['group_id']
            if group in groups and groups[group] != purpose:
                raise ContractError(f'Group crosses train/dev: {group}')
            groups[group] = purpose
            for kind in ('image', 'label'):
                rel = rec[kind]
                if rel not in files:
                    raise ContractError(f'Not whitelisted: {rel}')
                check_equal(rel + '.purpose', purpose, files[rel]['purpose'])
                check_equal(rel + '.kind', kind, files[rel]['kind'])
                check_equal(rel + '.record hash', rec[kind + '_sha256'], files[rel]['sha256'])
                if rel in used:
                    raise ContractError(f'Duplicate manifest path: {rel}')
                used.add(rel)
            # YOLO resolves labels from image stem; reject a mismatch before training.
            expected_label = str(PurePosixPath(rec['image']).with_suffix('.txt')).replace('images/', 'labels/', 1)
            check_equal(rec['image'] + '.label pairing', expected_label, rec['label'])
            actual_classes = set()
            for line in safe_path(root, rec['label']).read_text(encoding='utf-8').splitlines():
                fields = line.split()
                if len(fields) != 5:
                    raise ContractError(f'{rec["label"]}: invalid YOLO row {line!r}')
                cls = int(fields[0])
                coords = [float(x) for x in fields[1:]]
                if cls not in range(6) or not all(0 <= x <= 1 for x in coords) or min(coords[2:]) <= 0:
                    raise ContractError(f'{rec["label"]}: invalid class/coordinates {line!r}')
                actual_classes.add(cls)
            check_equal(rec['label'] + '.classes', sorted(set(rec['classes'])), sorted(actual_classes))
        manifests[purpose] = records
    check_equal('whitelist exactly equals manifest image+label paths', used, set(files))
    return {'ready': ready, 'root': root, 'ready_sha256': expected_ready_sha256,
            'manifests': manifests, 'files': files}


def select_records(records, count, seed=17):
    """Deterministic coverage first; all available classes included, blanks preserved when possible."""
    order = sorted(records, key=lambda r: canonical_hash([seed, r['key']]))
    if count >= len(order):
        return order
    selected, covered = [], set()
    available = set(c for r in records for c in r['classes'])
    while covered != available and len(selected) < count:
        choices = [r for r in order if r not in selected]
        best = max(choices, key=lambda r: len(set(r['classes']) - covered))
        selected.append(best)
        covered.update(best['classes'])
    blank = next((r for r in order if not r['classes'] and r not in selected), None)
    if blank is not None and len(selected) < count:
        selected.append(blank)
    selected.extend(r for r in order if r not in selected)
    result = selected[:count]
    check_equal('subset class coverage', available, {c for r in result for c in r['classes']})
    return result


def stage_dataset(contract, destination: Path, limits=None):
    """Copy verified whitelist only. Caller owns this writable cache sandbox."""
    if destination.exists():
        raise ContractError(f'Run sandbox already exists: {destination}')
    if destination.resolve().is_relative_to(contract['root']):
        raise ContractError('Run sandbox must be outside frozen source root')
    destination.mkdir(parents=True)
    selections = {}
    for purpose in PURPOSES:
        records = contract['manifests'][purpose]
        chosen = select_records(records, limits[purpose]) if limits else records
        selections[purpose] = chosen
        for record in chosen:
            for kind in ('image', 'label'):
                rel = record[kind]
                target = safe_path(destination, rel)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(safe_path(contract['root'], rel), target)
                check_equal(target, contract['files'][rel]['sha256'], sha256(target))
    write_json(destination / 'selection.json', {p: [r['key'] for r in v] for p, v in selections.items()})
    # JSON is also valid YAML; avoid an optional YAML dependency in contract tests.
    data = {'path': destination.resolve().as_posix(), 'train': 'images/source_train',
            'val': 'images/source_dev', 'names': NAMES, 'nc': 6}
    write_json(destination / 'dataset.yaml', data)
    return destination / 'dataset.yaml', selections


def primary_fitness(ap_class_indices, all_ap, target=(0, 2)):
    """Use AP rows indexed by actual class IDs, never Metric.maps' missing-class mean fallback."""
    rows = {int(c): [float(x) for x in row] for c, row in zip(ap_class_indices, all_ap)}
    if any(c not in rows for c in target):
        raise ContractError(f'Main validation classes missing: expected={target}; actual={list(rows)}')
    per_class = {c: sum(row) / len(row) for c, row in rows.items()}
    import math
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in per_class.values()):
        raise ContractError(f'Invalid AP values: {per_class}')
    return sum(per_class[c] for c in target) / len(target), per_class
