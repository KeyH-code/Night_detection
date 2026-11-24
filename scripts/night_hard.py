"""Lock hard-tail selections from sealed image-only scores, then audit only those labels."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

import night_strategy as strategy
import night_train as train
from night_accept import _checked_file
from night_contract import POOL, POOL_MANIFEST, ROOT, _no_reparse, _records, read_sealed_bytes, read_sealed_json
from source_contract import ContractError, NAMES, canonical_hash, check_equal, sha256, write_json

CONFIG = 'configs/hard.json'
INPUTS = {'pool_001': ('random.json',), 'pool_002': ('features.npz', 'scores.npz'),
          'presence_001': ('selection.json',), 'width_001': ('selection.json',)}
LOCK = 'pool_007'
AUDIT = 'hard_001'
SEED = 43


def rank(keys, features, presence, width, uncertainty, groups, anchors, sequence):
    check_equal('matching image and group inventories', set(keys), set(groups))
    order, ordered, cos, selected = strategy._geometry(keys, features, anchors)
    p, w, u = (strategy._scores(keys, q)[order] for q in (presence, width, uncertainty))
    base = np.sqrt(p * w)
    values = {'hard': base * (.5 + .5 * u) * (u > 0), 'support': base, 'general': p}
    used = {groups[ordered[i]] for i in selected}
    near = 1 - cos[:, selected].max(axis=1)
    steps = []
    for kind, cls in sequence:
        scores = values[kind][:, cls]
        eligible = [i for i, key in enumerate(ordered) if scores[i] > 0 and groups[key] not in used]
        if not eligible:
            raise ContractError(f'Insufficient positive independent candidates for quota {kind}/{cls}')
        priority = scores * (.5 + .5 * np.clip(near, 0., 1.))
        picked = min(eligible, key=lambda i: (-round(float(priority[i]), 12), -round(float(scores[i]), 12), ordered[i]))
        key = ordered[picked]
        steps.append({'kind': kind, 'class_id': cls, 'key': key, 'group_id': groups[key],
                      'base': float(scores[picked]), 'priority': float(priority[picked])})
        used.add(groups[key])
        selected.append(picked)
        np.minimum(near, 1 - cos[:, picked], out=near)
    return {'keys': [ordered[i] for i in selected], 'steps': steps, 'groups': len(used)}


def support(records, labels):
    result = {}
    for cid, name in enumerate(NAMES[:5]):
        groups, medium, cooccurrence, negatives = set(), set(), set(), set()
        images = instances = 0
        for row, coords in zip(records, labels, strict=True):
            group = row['group_id']
            examples = [w for c, _x, _y, w, _h in coords if c == cid]
            other = any(c in (2, 3, 4) and c != cid for c, *_ in coords)
            if not examples:
                if other:
                    negatives.add(group)
                continue
            groups.add(group)
            images += 1
            instances += len(examples)
            ref = row['width'] * 640 / max(row['width'], row['height'])
            if any(8 < w * ref <= 64 for w in examples):
                medium.add(group)
            if other:
                cooccurrence.add(group)
        result[name] = {'instances': instances, 'images': images, 'groups': len(groups),
                        'medium_groups': len(medium), 'vehicle_cooccurrence_groups': len(cooccurrence),
                        'other_vehicle_negative_groups': len(negatives - groups)}
    return result


def configuration(digest):
    cfg = read_sealed_json(ROOT, CONFIG, digest, [])
    check_equal('registered hard-tail config', (1, SEED, set(INPUTS), {'anchors': 20, 'hard': 16, 'support': 15, 'general': 6},
                {'groups': 20, 'medium_groups': 18, 'vehicle_cooccurrence_groups': 8,
                 'other_vehicle_negative_groups': 20, 'general_groups': 8}),
                (cfg['schema_version'], cfg['seed'], set(cfg['seals']), cfg['quota'], cfg['gate']))
    return cfg


def _selection(digest):
    # Initialize NumPy's archive reader and Python's codec using no project files.
    sample = io.BytesIO()
    np.savez_compressed(sample, keys=['warmup'], vectors=np.ones((1, 1)), scores=np.zeros((1, 5)))
    sample.seek(0)
    with np.load(sample, allow_pickle=False) as archive:
        for field in archive.files:
            archive[field]
    approved = [ROOT/CONFIG, train.NIGHT/POOL_MANIFEST]
    for folder, args in INPUTS.items():
        approved.extend(ROOT/'runs'/folder/x for x in ('manifest.json', *args))
    opens = strategy.worker_files(readable=approved)
    cfg = configuration(digest)
    pool = read_sealed_json(train.NIGHT, POOL_MANIFEST, cfg['images_sha256'], [])
    inventory = {r['key']: r for r in _records(pool, POOL, 994, images_only=True)}
    parents = {name: read_sealed_json(ROOT/'runs'/name, 'manifest.json', dg, []) for name, dg in cfg['seals'].items()}
    check_equal('sealed approved parents', {'sealed'}, {p['status'] for p in parents.values()})

    def read(parent, name):
        return _checked_file(ROOT/'runs'/parent, parents[parent], name, [])

    random = json.loads(read('pool_001', 'random.json'))
    d = random['arms'][str(SEED)]
    dkeys = [r['key'] for r in d['records']]
    check_equal('frozen random draw', (cfg['images_sha256'], 100, 100, d['record_sha256'], d['records']),
                (random['image_manifest_sha256'], len(dkeys), len(set(dkeys)), canonical_hash(d['records']),
                 [inventory[k] for k in dkeys]))
    keys, z = strategy.load_features(io.BytesIO(read('pool_002', 'features.npz')))
    check_equal('complete image feature scope', set(inventory), set(keys))
    u = strategy._load_e1_scores(io.BytesIO(read('pool_002', 'scores.npz')), keys)
    features_sha = parents['pool_002']['files']['features.npz']['sha256']

    def archived(folder, mode):
        doc = json.loads(read(folder, 'selection.json'))
        cap = doc['presence']
        check_equal('same frozen A capture identity', (False, False, cfg['images_sha256'], features_sha, False, cfg['a_sha256'],
                    mode, keys, ['original', 'bright', 'flip'], 994),
                    (doc['selection_labels_read'], doc['selection_development_read'], doc['images_sha256'], doc['feature_sha256'],
                     cap['labels_read'], cap['a_sha256'], cap.get('score_mode', 'presence'), cap['keys'], cap['views'], cap['images_read']))
        return strategy._scores(keys, cap['scores'])

    p, w = archived('presence_001', 'presence'), archived('width_001', 'medium_width')
    quota = cfg['quota']
    sequence = [('hard', k) for _ in range(quota['hard']) for k in (3, 4)]
    sequence += [('support', k) for _ in range(quota['support']) for k in (3, 4)]
    sequence += [('general', k) for _ in range(quota['general']) for k in (0, 1, 2)]
    groups = {k: v['group_id'] for k, v in inventory.items()}
    anchors = dkeys[:quota['anchors']]
    args = [groups, anchors, sequence]
    result = rank(keys, z, p, w, u, *args)
    check_equal('independent reversed-input ranking', result, rank(keys[::-1], z[::-1], p[::-1], w[::-1], u[::-1], *args))
    picked = result['keys']
    check_equal('100 distinct keys, fixed ordered random anchors, new groups only', (100, 100, 100, anchors),
                (len(picked), len(set(picked)), result['groups'], picked[:len(anchors)]))
    check_equal('only registered sealed files opened once', (set(approved), len(approved)), (set(opens), len(opens)))
    return {**result, 'records': [inventory[k] for k in picked], 'd_overlap': len(set(dkeys)&set(picked)),
            'reads': [p.relative_to(ROOT).as_posix() for p in opens], 'protocol_sha256': digest,
            'scope': 'image_only', 'candidate_labels_read': False, 'development_read': False, 'holdout_read': False}


def _support(digest, selection_sha):
    cfg = configuration(digest)
    folder = ROOT/'runs'/LOCK
    seal = read_sealed_json(folder, 'manifest.json', selection_sha, [])
    selected = json.loads(_checked_file(folder, seal, 'selection.json', []))
    check_equal('label verification follows sealed image-only selection', ('sealed', digest, 'image_only', False, False, False),
                (seal['status'], selected['protocol_sha256'], selected['scope'], selected['candidate_labels_read'],
                 selected['development_read'], selected['holdout_read']))
    locked = _records({'purpose': POOL, 'records': selected['records']}, POOL, 100, images_only=True)
    check_equal('exact immutable label grant', selected['keys'], [r['key'] for r in locked])
    ready = read_sealed_json(train.NIGHT, 'READY.json', cfg['ready_sha256'], [])
    entry = ready['held_out']['night_pool']
    check_equal('fixed frozen management hash-index path', ('manifests/night_pool.json', 994), (entry['path'], entry['count']))
    management = read_sealed_json(train.NIGHT, entry['path'], entry['sha256'], [])
    check_equal('no other data purpose fetched', ('night_pool', 994), (management['purpose'], len(management['records'])))
    rows = train.selected_records(locked, management['records'])
    del management
    approved = [train.NIGHT/r['label'] for r in rows]
    opens = strategy.worker_files(readable=approved)
    reads, parsed = [], []
    for row in rows:
        label = train._labels(read_sealed_bytes(train.NIGHT, row['label'], row['label_sha256'], reads))
        check_equal('frozen class set of selected label', sorted(row['classes']), sorted({c for c, *_ in label}))
        parsed.append(label)
    check_equal('only 100 locked label files opened', approved, opens)
    result = support(rows, parsed)
    gate = cfg['gate']
    checks = {n: {k: result[n][k] >= gate[k] for k in gate if k != 'general_groups'} for n in NAMES[3:5]}
    checks.update({n: {'groups': result[n]['groups'] >= gate['general_groups']} for n in NAMES[:3]})
    roles = {'anchors': list(range(cfg['quota']['anchors']))}
    for i, step in enumerate(selected['steps'], cfg['quota']['anchors']):
        roles.setdefault(step['kind'], []).append(i)
    return {'protocol_sha256': digest, 'selection_manifest_sha256': selection_sha, 'management_sha256': entry['sha256'],
            'labels_read': len(reads), 'selected_label_receipt_sha256': canonical_hash([(r['label'], r['label_sha256']) for r in rows]),
            'no_image_reads': True, 'development_read': False, 'holdout_read': False, 'support': result,
            'by_role': {k: support([rows[i] for i in ix], [parsed[i] for i in ix]) for k, ix in roles.items()},
            'checks': checks, 'support_met': all(p for v in checks.values() for p in v.values())}


def _call(scratch, mode, digest, *extra):
    command = [sys.executable, '-B', str(ROOT/'scripts/night_hard.py'), mode, digest, *extra]
    task = subprocess.run(command, cwd=scratch, timeout=120, capture_output=True, text=True, encoding='utf-8')
    if task.returncode:
        raise ContractError(task.stderr.strip())
    return json.loads(task.stdout)


def _publish(scratch, out, name, data):
    _no_reparse(out.parent)
    if out.exists() or out.is_symlink():
        raise ContractError(f'Refusing to overwrite {out.name}')
    with tempfile.TemporaryDirectory(dir=scratch, prefix='hard-tail-') as tmp:
        folder = Path(tmp)/'evidence'
        folder.mkdir()
        write_json(folder/name, data)
        (folder/'code').mkdir()
        for f in ('night_hard.py', 'night_strategy.py', 'night_train.py', 'night_contract.py', 'night_accept.py', 'source_contract.py'):
            shutil.copyfile(ROOT/'scripts'/f, folder/'code'/f)
        files = train._receipt(folder)
        write_json(folder/'manifest.json', {'status': 'sealed', 'files': files})
        if out.exists() or out.is_symlink():
            raise ContractError(f'Refusing to overwrite {out.name}')
        folder.rename(out)
    check_equal('published artifacts', files, train._receipt(out))


def main():
    if len(sys.argv) == 3 and sys.argv[1] == '_select':
        print(json.dumps(_selection(sys.argv[2]), allow_nan=False))
        return
    if len(sys.argv) == 4 and sys.argv[1] == '_support':
        print(json.dumps(_support(*sys.argv[2:]), allow_nan=False))
        return
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('select', 'support', 'verify'))
    p.add_argument('--config-sha256', required=True)
    p.add_argument('--selection-sha256')
    p.add_argument('--support-sha256')
    p.add_argument('--scratch', required=True, type=Path)
    a = p.parse_args()
    scratch = a.scratch.absolute()
    if (not scratch.is_dir() or scratch.resolve() != scratch or scratch.is_relative_to(ROOT) or scratch.drive != ROOT.drive):
        p.error('Use an existing external scratch directory on the same volume')
    _no_reparse(scratch)
    if (bool(a.selection_sha256) != (a.mode != 'select') or bool(a.support_sha256) != (a.mode == 'verify')):
        p.error('support and verify require --selection-sha256; verify also requires --support-sha256')
    if a.mode == 'select':
        out = ROOT/'runs'/LOCK
        data = _call(scratch, '_select', a.config_sha256)
        _publish(scratch, out, 'selection.json', data)
        print(json.dumps({'keys': len(data['keys']), 'groups': data['groups'], 'overlap_D': data['d_overlap'],
                          'read_files': len(data['reads']), 'manifest_sha256': sha256(out/'manifest.json')}))
    elif a.mode == 'support':
        out = ROOT/'runs'/AUDIT
        data = _call(scratch, '_support', a.config_sha256, a.selection_sha256)
        _publish(scratch, out, 'support.json', data)
        print(json.dumps({**data, 'manifest_sha256': sha256(out/'manifest.json')}, allow_nan=False))
    else:
        for mode, name, doc, dg, args in (
                ('_select', LOCK, 'selection.json', a.selection_sha256, ()),
                ('_support', AUDIT, 'support.json', a.support_sha256, (a.selection_sha256,))):
            path = ROOT/'runs'/name
            seal = read_sealed_json(path, 'manifest.json', dg, [])
            check_equal('sealed artifact inventory', ('sealed', seal['files']), (seal['status'], train._receipt(path)))
            data = _call(scratch, mode, a.config_sha256, *args)
            check_equal('repeatable locked receipt', json.loads(_checked_file(path, seal, doc, [])), data)
            print(f'{name}: verified {len(seal["files"])} files and reproduced {doc}', flush=True)
        print(f'Support passed: {data["support_met"]}')


if __name__ == '__main__':
    main()
