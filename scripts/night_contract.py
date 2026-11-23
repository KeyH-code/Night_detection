"""S5's separate night-dev and image-only consumers. Never opens other split receipts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat

from source_contract import ContractError, NAMES, check_equal, sha256

ROOT = Path(__file__).resolve().parents[1]
DEV = 'night_dev'
DEV_MANIFEST = 'manifests/night_dev.json'
POOL = 'night_pool_selection_images_only'
POOL_MANIFEST = 'selection/night_pool_images.json'
POOL_FIELDS = {'key', 'group_id', 'image', 'image_sha256'}
INVALID_PART = re.compile(r'[<>:"\\|?*\x00-\x1f]')
DEVICES = frozenset({'con', 'prn', 'aux', 'nul', 'conin$', 'conout$'} |
                    {prefix + digit for prefix in ('com', 'lpt') for digit in '123456789¹²³'})


def _no_reparse(path):
    for part in reversed((path, *path.parents)):
        info = part.lstat()
        if (stat.S_ISLNK(info.st_mode)
                or getattr(info, 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
            raise ContractError(f'link/junction/reparse point: {part}')


def _parts(rel):
    parts = rel.split('/') if isinstance(rel, str) else []
    if not parts or any(not p or p in ('.', '..') or p.endswith(('.', ' ')) or INVALID_PART.search(p)
                        or p.split('.', 1)[0].rstrip().casefold() in DEVICES for p in parts):
        raise ContractError(f'Invalid relative POSIX path: {rel!r}')
    return parts


def _path(root, rel):
    file = root.joinpath(*_parts(rel))
    try:
        _no_reparse(file)
        if not file.is_file():
            raise ContractError(f'Expected file at path: {rel}')
    except OSError as exc:
        raise ContractError(f'Cannot access required path: {rel}') from exc
    return file


def _digest(value):
    if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value):
        raise ContractError('Invalid SHA-256 in approved input')
    return value


def read_sealed_bytes(root, rel, expected, reads):
    expected = _digest(expected)
    path = _path(root, rel)
    data = path.read_bytes()
    reads.append(rel)
    check_equal(f'{rel} SHA-256', expected, hashlib.sha256(data).hexdigest())
    return data


def read_sealed_json(root, rel, expected, reads):
    # Decode the same bytes that were hashed instead of reopening the manifest.
    data = read_sealed_bytes(root, rel, expected, reads)
    try:
        value = json.loads(data.decode('utf-8-sig'))
    except (UnicodeError, ValueError) as exc:
        raise ContractError(f'Invalid JSON: {rel}') from exc
    if not isinstance(value, dict):
        raise ContractError(f'Expected JSON object: {rel}')
    return value


def _payload_rel(value, prefix, suffixes):
    parts = _parts(value)
    if parts[:-1] != prefix or PurePosixPath(parts[-1]).suffix.lower() not in suffixes:
        raise ContractError(f'Wrong payload path for {"/".join(prefix)}: {value!r}')
    return PurePosixPath(parts[-1]).stem


def _records(manifest, purpose, count, *, images_only):
    check_equal('manifest purpose', purpose, manifest.get('purpose'))
    if images_only:
        check_equal('selection manifest fields', {'purpose', 'records'}, set(manifest))
    records = manifest.get('records')
    if not isinstance(records, list) or type(count) is not int or count <= 0 or len(records) != count:
        raise ContractError(f'Invalid records or count for {purpose}')
    keys, stems = set(), set()
    for row in records:
        if not isinstance(row, dict):
            raise ContractError('Expected record object')
        if images_only:
            check_equal('selection record fields', POOL_FIELDS, set(row))
        else:
            check_equal('record purpose', DEV, row.get('purpose'))
            if any(type(row.get(k)) is not int or row[k] <= 0 for k in ('width', 'height')):
                raise ContractError('Invalid image dimensions')
        key, group = row.get('key'), row.get('group_id')
        if not isinstance(key, str) or not key.strip() or not isinstance(group, str) or not group.strip():
            raise ContractError('Invalid record key or risk group')
        stem = _payload_rel(row.get('image'), ['images', 'night_pool' if images_only else DEV],
                            {'.jpg', '.jpeg', '.png'})
        if key in keys or stem.casefold() in stems:
            raise ContractError('Duplicate key or image stem')
        keys.add(key)
        stems.add(stem.casefold())
        _digest(row.get('image_sha256'))
        if not images_only:
            check_equal('image/label stem', stem, _payload_rel(row.get('label'), ['labels', DEV], {'.txt'}))
            _digest(row.get('label_sha256'))
    return records


def verify_dev(root, expected_ready, *, count=331):
    """Verify exactly the approved READY, night-dev manifest and its complete payload."""
    root, reads = Path(root).absolute(), []
    ready = read_sealed_json(root, 'READY.json', expected_ready, reads)
    check_equal('night READY version', (1, 'night_v1', 'frozen'),
                (ready.get('schema_version'), ready.get('version'), ready.get('status')))
    try:
        check_equal('original category mapping', NAMES, ready['categories']['names'])
        entry = ready['held_out'][DEV]
        check_equal('night-dev receipt', (DEV_MANIFEST, count, count),
                    (entry['path'], entry['count'], ready['counts'][DEV]))
        manifest = read_sealed_json(root, DEV_MANIFEST, entry['sha256'], reads)
    except (KeyError, TypeError) as exc:
        raise ContractError('Invalid night READY or night-dev receipt') from exc
    records = _records(manifest, DEV, count, images_only=False)
    for row in records:
        for kind in ('image', 'label'):
            rel = row[kind]
            path = _path(root, rel)
            reads.append(rel)
            check_equal(f'{rel} SHA-256', row[kind + '_sha256'], sha256(path))
    return {'root': root, 'purpose': DEV, 'records': records, 'manifest': entry,
            'ready_hash': expected_ready, 'reads': reads}


def stage_selection(root, expected_manifest, destination, *, count=994):
    """Copy just the individually verified images and sealed narrow list into a new sandbox."""
    root, dest = Path(root).absolute(), Path(destination).absolute()
    if '..' in Path(destination).parts or '..' in Path(root).parts:
        raise ContractError('Selection root and destination paths must not contain parent traversals')
    if dest.is_relative_to(root) or dest.is_relative_to(ROOT) or dest.exists() or dest.is_symlink():
        raise ContractError('Selection destination must be new and external to the project and dataset')
    try:
        _no_reparse(dest.parent)
    except OSError as exc:
        raise ContractError('Selection destination must have an existing external parent') from exc
    reads = []
    manifest = read_sealed_json(root, POOL_MANIFEST, expected_manifest, reads)
    records = _records(manifest, POOL, count, images_only=True)
    sources = []
    for row in records:
        rel = row['image']
        file = _path(root, rel)
        reads.append(rel)
        check_equal(f'{rel} SHA-256', row['image_sha256'], sha256(file))
        sources.append((file, rel, row['image_sha256']))
    dest.mkdir(exist_ok=False)
    for file, rel, digest in [(_path(root, POOL_MANIFEST), POOL_MANIFEST, expected_manifest), *sources]:
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, target)
        check_equal(f'copied {rel} SHA-256', digest, sha256(target))
    return {'root': dest, 'purpose': POOL, 'records': records, 'manifest_sha256': expected_manifest, 'reads': reads}
