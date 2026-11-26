"""Blind S5 selection: score detections for E1, or rank image embeddings only for E2."""
from __future__ import annotations

import argparse
from collections import Counter
import io
from itertools import combinations
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

from night_contract import POOL, POOL_MANIFEST, ROOT, _no_reparse, _records, read_sealed_bytes, read_sealed_json, stage_selection
from night_select import _expect_inventory
from source_contract import ContractError, canonical_hash, check_equal, read_json, sha256, write_json

CLASS_COUNT = 5


def worker_files(*, readable, writable=()):
    """Guard Python file opens in a disposable, already initialized worker."""
    inputs = frozenset(Path(p).absolute() for p in readable)
    outputs = frozenset(Path(p).absolute() for p in writable)
    if inputs & outputs:
        raise ContractError('Worker input and output permissions overlap')
    opened = []
    write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC

    def hook(event, args):
        if event != 'open':
            return
        filename, _, flags = args
        if not isinstance(filename, (str, bytes)):
            raise ContractError('Worker refuses file descriptor opens')
        path = Path(os.fsdecode(filename)).absolute()
        if flags & write_flags:
            if path not in outputs or not flags & os.O_EXCL or flags & os.O_RDWR:
                raise ContractError('Worker refuses unapproved writes or overwrite')
        elif path not in inputs:
            raise ContractError('Worker refuses read outside the approved file whitelist')
        else:
            opened.append(path)

    sys.addaudithook(hook)
    return opened


class BlindPool:
    """An already-staged image-only namespace; bytes can be requested by key only."""

    def __init__(self, root, manifest_sha, count):
        self.root = Path(root).absolute()
        _no_reparse(self.root)
        _expect_inventory(self.root, {'images', 'selection'}, dirs=True)
        _expect_inventory(self.root / 'images', {'night_pool'}, dirs=True)
        _expect_inventory(self.root / 'selection', {'night_pool_images.json'}, dirs=False)
        manifest = read_sealed_json(self.root, POOL_MANIFEST, manifest_sha, [])
        approved = _records(manifest, POOL, count, images_only=True)
        _expect_inventory(self.root / 'images/night_pool', {Path(r['image']).name for r in approved}, dirs=False)
        self.records = {r['key']: r for r in sorted(approved, key=lambda r: r['key'])}

    def read(self, key):
        if key not in self.records:
            raise ContractError('Image key not in blind pool')
        row = self.records[key]
        return read_sealed_bytes(self.root, row['image'], row['image_sha256'], [])


def load_features(path):
    """E2's feature data is intentionally separate from the E1 scoring payload."""
    with np.load(path, allow_pickle=False) as data:
        check_equal('features fields', ['keys', 'vectors'], sorted(data.files))
        keys, vectors = data['keys'].tolist(), data['vectors']
    _validate_vectors(keys, vectors)
    return keys, vectors


def _scores(keys, values):
    u = np.asarray(values, dtype=np.float64)
    if u.shape != (len(keys), CLASS_COUNT) or not np.isfinite(u).all() or np.any((u < 0) | (u > 1)):
        raise ContractError('Invalid E1 scores')
    return u


def _load_e1_scores(path, keys):
    with np.load(path, allow_pickle=False) as payload:
        check_equal('E1 scores fields', ['keys', 'scores'], sorted(payload.files))
        check_equal('exact E1 score keys', keys, payload['keys'].tolist())
        return _scores(keys, payload['scores'])


def _boxes(rows):
    a = np.asarray(rows, dtype=np.float64)
    if a.shape == (0,):
        return np.empty((0, 6), dtype=np.float64)
    if (a.ndim != 2 or a.shape[1] != 6 or not np.isfinite(a).all()
            or np.any(a[:, 2:4] <= a[:, :2]) or not np.isin(a[:, 5], np.arange(6)).all()
            or np.any((a[:, 4] < 0) | (a[:, 4] > 1))):
        raise ContractError('Invalid prediction geometry, class, or confidence')
    kept = a[(a[:, 4] > .05) & (a[:, 5] < CLASS_COUNT)]
    return np.asarray(sorted(kept, key=lambda r: (-round(float(r[4]), 12), r[5], *np.round(r[:4], 12))),
                      dtype=np.float64).reshape(-1, 6)


def view_scores(original, variant):
    """Greedy geometry matching (not the class-aware evaluation matcher)."""
    a, b = _boxes(original), _boxes(variant)
    inter = np.maximum(np.minimum(a[:, None, 2:4], b[None, :, 2:4])
                       - np.maximum(a[:, None, :2], b[None, :, :2]), 0).prod(axis=2)
    union = np.prod(a[:, 2:4]-a[:, :2], axis=1)[:, None] + np.prod(b[:, 2:4]-b[:, :2], axis=1)[None, :] - inter
    iou = inter / union
    edges = sorted(zip(*np.nonzero(iou >= .5), strict=True), key=lambda ij: (-round(float(iou[ij]), 12), *ij))
    left, right, scores = set(), set(), np.zeros(CLASS_COUNT)
    for i, j in edges:
        if i in left or j in right:
            continue
        left.add(i)
        right.add(j)
        ca, cb = int(a[i, 5]), int(b[j, 5])
        v = (1 - iou[i, j] + (ca != cb) + abs(a[i, 4] - b[j, 4])) / 3
        scores[ca] = max(scores[ca], v)
        scores[cb] = max(scores[cb], v)
    for boxes, used in ((a, left), (b, right)):
        for i, box in enumerate(boxes):
            if i not in used:
                c = int(box[5])
                scores[c] = max(scores[c], box[4])
    return scores


def score_e1(views):
    if len(views) != 4:
        raise ContractError('E1 requires original plus three transformed predictions')
    return np.mean([view_scores(views[0], variant) for variant in views[1:]], axis=0)


def _validate_vectors(keys, vectors):
    if not isinstance(keys, (list, tuple)) or not keys or any(not isinstance(k, str) or not k for k in keys):
        raise ContractError('Feature keys must be nonempty strings')
    if len(keys) != len(set(keys)):
        raise ContractError('Duplicate feature key')
    z = np.asarray(vectors, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] != len(keys) or not z.shape[1] or not np.isfinite(z).all():
        raise ContractError('Feature shape or finite check failed')
    n = np.linalg.norm(z, axis=1)
    if not np.isfinite(n).all() or np.any(n == 0) or not np.allclose(n, 1, atol=1e-10, rtol=0):
        raise ContractError('Features must be non-zero unit vectors')
    return z


def _geometry(keys, vectors, anchors):
    z = _validate_vectors(keys, vectors)
    order = sorted(range(len(keys)), key=lambda i: keys[i])
    ordered = [keys[i] for i in order]
    index = {k: i for i, k in enumerate(ordered)}
    if not anchors or len(set(anchors)) != len(anchors) or any(k not in index for k in anchors):
        raise ContractError('Invalid fixed anchors')
    z = z[order]
    cos = np.clip(z @ z.T, -1, 1)
    np.fill_diagonal(cos, 1.)
    selected = [index[k] for k in anchors]
    return order, ordered, cos, selected


class MechanismQuotaError(ContractError):
    """No fallback is allowed in a mechanism comparison."""


def rank_e1(keys, vectors, scores, anchors, *, shortlist=64, quota=16, score_quota=None):
    order, ordered, cos, selected = _geometry(keys, vectors, anchors)
    if type(shortlist) is not int or type(quota) is not int or min(shortlist, quota) <= 0:
        raise ContractError('Invalid E1 shortlist settings')
    if score_quota is not None and (type(score_quota) is not int or not 0 <= score_quota <= quota):
        raise ContractError('Invalid E1 score stage quota')
    u = _scores(keys, scores)[order]
    chosen = set(selected)
    pool = [i for i in range(len(keys)) if i not in chosen]
    tie_keys = np.round(u, 12)
    if len(pool) < CLASS_COUNT*quota or np.all(tie_keys[pool] == tie_keys[pool[0]]):
        raise ContractError('E1 shortlist lacks budget or distinct scores')
    lists = [sorted((i for i in pool if u[i, c] > 0), key=lambda i: (-tie_keys[i,c], ordered[i]))[:shortlist]
             for c in range(CLASS_COUNT)]
    union = set().union(*lists)
    if len(union) < CLASS_COUNT * quota:
        raise ContractError('E1 shortlist union cannot fill the fixed budget')
    distance = 1-cos[:, selected].max(axis=1)
    counts, steps = [0]*CLASS_COUNT, []

    def pick(candidates, values, c, stage='diversity'):
        def order_key(j):
            score_key = (-round(float(values[j]), 12), ordered[j])
            return score_key if stage == 'score' else (-round(float(distance[j]), 12), *score_key)

        i = min(candidates, key=order_key)
        step = {'key': ordered[i], 'class_id': c, 'distance': float(distance[i]), 'score': float(values[i])}
        if score_quota is not None:
            step['stage'] = stage
        steps.append(step)
        selected.append(i)
        chosen.add(i)
        np.minimum(distance, 1-cos[:, i], out=distance)

    for round_index in range(quota):
        stage = 'score' if score_quota is not None and round_index < score_quota else 'diversity'
        for c, subset in enumerate(lists):
            candidates = set(subset) - chosen
            if candidates:
                pick(candidates, u[:, c], c, stage)
                counts[c] += 1
            elif score_quota is not None:
                raise MechanismQuotaError(f'E1 {stage} phase class {c} exhausted in round {round_index + 1}; '
                                          f'counts={counts}, shortlist_union={len(union)}; no fallback used')
    largest = np.zeros(len(keys))
    for c, subset in enumerate(lists):
        largest[subset] = np.maximum(largest[subset], u[subset, c])
    shortfalls = [quota - v for v in counts]
    for _ in range(sum(shortfalls)):
        pick(union - chosen, largest, None)
    return {'keys': [ordered[i] for i in selected], 'steps': steps, 'per_class': counts,
            'shortfalls': shortfalls, 'shortlist_union': len(union)}


def _percentiles(values):
    keys = np.round(values, 12)
    order = np.argsort(keys, kind='stable')
    ranks = np.empty(len(order))
    left = 0
    while left < len(order):
        right = left + 1
        while right < len(order) and keys[order[right]] == keys[order[left]]:
            right += 1
        ranks[order[left:right]] = (left+right-1)/(2*(len(order)-1))
        left = right
    return ranks


def rank_e2(keys, groups, vectors, anchors, *, representatives=40, diverse=40, neighbors=5):
    order, ordered, cos, selected = _geometry(keys, vectors, anchors)
    if (len(groups) != len(keys) or any(not isinstance(g, str) or not g for g in groups)
            or any(type(n) is not int or n <= 0 for n in (representatives, diverse, neighbors))):
        raise ContractError('Invalid E2 risk groups or budget')
    group = np.array([groups[i] for i in order])
    sizes = Counter(group)
    covered = {group[i] for i in selected}
    if len(sizes)-len(covered) < representatives + diverse or len(sizes)-1 < neighbors:
        raise ContractError('Not enough uncovered or neighboring risk groups for E2')
    similar = (1+cos)/2
    weights = np.array([1/sizes[g] for g in group])
    m = similar[:, selected].max(axis=1)
    steps = []

    def choose(values, secondary=None):
        eligible = [i for i in range(len(keys)) if group[i] not in covered]
        return min(eligible, key=lambda i: (-round(float(values[i]), 12),
                                             -round(float(secondary[i]), 12) if secondary is not None else 0,
                                             ordered[i]))

    def add(i, step):
        steps.append({'key': ordered[i], 'group_id': str(group[i]), **step})
        selected.append(i)
        covered.add(group[i])
        np.maximum(m, similar[:, i], out=m)

    for _ in range(representatives):
        gain = weights @ np.maximum(similar-m[:, None], 0)
        i = choose(gain)
        if round(float(gain[i]), 12) <= 0:
            raise ContractError('No positive E2 representative coverage gain')
        add(i, {'kind': 'representative', 'gain': float(gain[i])})

    unique = sorted(sizes)
    per_group = np.column_stack([similar[:, group == g].max(axis=1) for g in unique])
    per_group[np.arange(len(keys)), np.searchsorted(unique, group)] = -1
    support = np.sort(per_group, axis=1)[:, -neighbors:].mean(axis=1)
    ranks = _percentiles(support)
    for _ in range(diverse):
        distance = 2*(1-m)
        value = distance*(.5+.5*ranks)
        i = choose(value, distance)
        if round(float(distance[i]), 12) <= 0:
            raise ContractError('No positive E2 diversity distance')
        add(i, {'kind': 'diverse', 'distance': float(distance[i]), 'support': float(support[i]),
                'support_percentile': float(ranks[i]), 'score': float(value[i])})
    return {'keys': [ordered[i] for i in selected], 'steps': steps}


def run_rank_worker(arm, features, metadata, *, scores=None):
    if arm not in ('e1', 'e1-mechanism', 'e2') or (arm != 'e2' and scores is None):
        raise ContractError('Expected E1 with scores or E2 with features only')
    if arm == 'e2' and scores is not None:
        raise ContractError('E2 cannot accept scores')
    f = Path(features).absolute()
    p = Path(scores).absolute() if scores is not None else None
    command = [sys.executable, '-B', str(Path(__file__).resolve()), '_rank', arm, str(f)]
    if p is not None:
        command.append(str(p))
    env = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1'}
    if arm == 'e2':
        env['OPENBLAS_NUM_THREADS'] = '1'
    try:
        result = subprocess.run(command, cwd=f.parent, input=json.dumps(metadata, ensure_ascii=False),
                                check=True, capture_output=True, encoding='utf-8', timeout=120,
                                env=env)
    except subprocess.CalledProcessError as exc:
        raise ContractError(f'{arm} ranking failed: {exc.stderr.strip()}') from exc
    return json.loads(result.stdout)


def _rank_payload(arm, features, scores, metadata):
    feature_path = Path(features).absolute()
    e1 = arm in ('e1', 'e1-mechanism')
    if feature_path.name != 'features.npz':
        raise ContractError('Expected separate feature-only file')
    if e1 and (scores is None or Path(scores).absolute() != feature_path.with_name('scores.npz')):
        raise ContractError('E1 requires a separate E1 score file')
    if arm == 'e2' and scores is not None:
        raise ContractError('E2 cannot accept scores')
    if not e1 and arm != 'e2':
        raise ContractError('Unknown strategy arm')
    fields = {'keys', 'anchors', 'params'} | ({'groups'} if arm == 'e2' else set())
    check_equal('ranking metadata fields', fields, set(metadata) if isinstance(metadata, dict) else None)
    params = metadata['params']
    expected_params = ({'shortlist', 'quota', 'score_quota'} if arm == 'e1-mechanism' else
                       {'shortlist', 'quota'} if e1 else {'representatives', 'diverse', 'neighbors'})
    check_equal('ranking parameter fields', expected_params, set(params) if isinstance(params, dict) else None)
    if arm == 'e1-mechanism' and type(params['score_quota']) is not int:
        raise ContractError('Mechanism worker requires an integer score quota')
    allowed = [feature_path] + ([Path(scores).absolute()] if scores is not None else [])
    for path in allowed:
        _no_reparse(path)
        if not path.is_file():
            raise ContractError('Missing selection artifact')
    # Warm NumPy's ZIP header readers before installing the strict stage guard.
    buffer = io.BytesIO()
    np.savez(buffer, warm=np.ones(1))
    buffer.seek(0)
    with np.load(buffer, allow_pickle=False) as data:
        data['warm']
    reads = worker_files(readable=allowed)
    keys, vectors = load_features(feature_path)
    check_equal('exact feature keys', metadata['keys'], keys)
    u = _load_e1_scores(allowed[1], keys) if e1 else None
    results = {}
    for seed, anchors in metadata['anchors'].items():
        try:
            results[seed] = (rank_e1(keys, vectors, u, anchors, **params) if e1 else
                             rank_e2(keys, metadata['groups'], vectors, anchors, **params))
        except MechanismQuotaError as exc:
            if arm != 'e1-mechanism':
                raise
            results[seed] = {'quota_failure': str(exc)}
    check_equal('ranking worker allowed reads', allowed, reads)
    return {'arms': results, 'reads': [p.name for p in reads]}


def _strategy_config(expected, *, reads=None):
    raw = read_sealed_bytes(ROOT, 'configs/strategy.json', expected, [] if reads is None else reads)
    cfg = json.loads(raw.decode('utf-8'))
    check_equal('approved E1/E2 budgets', (1, 994, 20, [17, 29, 43], {'shortlist': 64, 'quota': 16},
                                          {'representatives': 40, 'diverse': 40, 'neighbors': 5}),
                (cfg['schema_version'], cfg['pool_count'], cfg['anchor_count'], cfg['seeds'], cfg['e1'], cfg['e2']))
    return raw, cfg


def _stage_preflight(root, cfg, sandbox):
    reads = []
    manifest = read_sealed_json(root, POOL_MANIFEST, cfg['images_sha256'], reads)
    pool = sorted(_records(manifest, POOL, cfg['pool_count'], images_only=True), key=lambda r: r['key'])
    small = [pool[i*(len(pool)-1)//7] for i in range(8)]
    for row in small:
        relative = row['image']
        path = sandbox / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(read_sealed_bytes(root, relative, row['image_sha256'], reads))
    write_json(sandbox / POOL_MANIFEST, {'purpose': POOL, 'records': small})
    return {'root': sandbox, 'manifest_sha256': sha256(sandbox / POOL_MANIFEST), 'records': small, 'reads': reads}


def _anchor_lists(cfg, records, *, reads=None):
    draw = read_sealed_json(ROOT, 'runs/pool_001/random.json', cfg['random_sha256'], [] if reads is None else reads)
    check_equal('sealed D source', ('night_pool_random_control', cfg['images_sha256'], 100, {str(s) for s in cfg['seeds']}),
                (draw['purpose'], draw['image_manifest_sha256'], draw['budget_per_seed'], set(draw['arms'])))
    lookup = {r['key']: r for r in records}
    anchors, d_keys = {}, {}
    for s in cfg['seeds']:
        seed = str(s)
        arm = draw['arms'][seed]
        claimed = _records({'purpose': POOL, 'records': arm['records']}, POOL, 100, images_only=True)
        check_equal('D draw record hash', arm['record_sha256'], canonical_hash(claimed))
        check_equal('D identities within approved pool', claimed, [lookup.get(r['key']) for r in claimed])
        d_keys[seed] = [r['key'] for r in claimed]
        anchors[seed] = d_keys[seed][:cfg['anchor_count']]
    return anchors, d_keys


def _checked_capture(folder, records):
    from night_capture import FEATURE_LAYER, INFERENCE, VIEWS

    capture = read_json(folder / 'capture.json')
    check_equal('capture protocol', ('blind E1 capture', POOL_MANIFEST, list(VIEWS), FEATURE_LAYER, INFERENCE),
                tuple(capture.get(k) for k in ('purpose', 'manifest_read', 'views', 'feature_layer', 'inference')))
    check_equal('only staged image opens', [r['image'] for r in records], capture['reads'])
    shapes = capture['input_shapes']
    if (not isinstance(shapes, dict) or not shapes or any(type(n) is not int or n <= 0 for n in shapes.values())
            or sum(shapes.values()) != len(records)):
        raise ContractError('Invalid capture shape coverage')
    keys, vectors = load_features(folder / 'features.npz')
    check_equal('captured feature coverage', ([r['key'] for r in records], FEATURE_LAYER['channels']),
                (keys, vectors.shape[1]))
    _load_e1_scores(folder / 'scores.npz', keys)
    return capture


def _strategy_paths(scratch, output):
    scratch = Path(scratch).absolute()
    output = Path(output).absolute() if output is not None else None
    if (not scratch.is_dir() or scratch.resolve() != scratch or scratch.is_relative_to(ROOT)
            or (output is not None and
            (scratch.drive != output.drive or output.parent != ROOT / 'runs' or output.parent.resolve() != output.parent
             or output.exists() or output.is_symlink() or not re.fullmatch(r'pool_\d{3}', output.name)))):
        raise ContractError('Use an external scratch and (except for preflight) a new same-volume runs/pool_NNN')
    _no_reparse(scratch)
    if output is not None:
        _no_reparse(output.parent)
    return scratch, output


def run_e1(protocol_sha, scratch, output=None, *, preflight=False):
    from night_capture import A_SHA, MODEL, VERSIONS
    from seed_eval import probe_gpu
    from importlib.metadata import version

    if preflight != (output is None):
        raise ContractError('Expected output only for full E1')
    scratch, output = _strategy_paths(scratch, output)
    raw, cfg = _strategy_config(protocol_sha)
    check_equal('fixed A strategy identity', A_SHA, cfg['checkpoint_sha256'])
    check_equal('frozen weight SHA-256', A_SHA, sha256(ROOT / MODEL))
    check_equal('capture environment', VERSIONS, tuple(version(k) for k in ('torch', 'torchvision', 'ultralytics')))
    initial_gpu = probe_gpu()
    source = ROOT / 'data/soda10m/splits/night_v1'
    names = ('night_strategy.py', 'night_capture.py', 'night_contract.py', 'night_select.py', 'source_contract.py', 'seed_eval.py')
    code = {name: sha256(ROOT / 'scripts' / name) for name in names}
    print('Scope: eight-image E1 preflight only.' if preflight else 'Scope: E1-only full pool; no label or training reads.', flush=True)
    with tempfile.TemporaryDirectory(prefix='strategy-', dir=scratch) as tmp:
        base = Path(tmp)
        sandbox = base / 'images-only'
        stage = (_stage_preflight(source, cfg, sandbox) if preflight else
                 stage_selection(source, cfg['images_sha256'], sandbox, count=cfg['pool_count']))
        records = sorted(stage['records'], key=lambda r: r['key'])
        check_equal('staged pool reads', [POOL_MANIFEST, *[r['image'] for r in stage['records']]], stage['reads'])
        env = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1', 'YOLO_AUTOINSTALL': 'false', 'WANDB_MODE': 'disabled'}
        for name in ('YOLO_CONFIG_DIR', 'TORCH_HOME', 'MPLCONFIGDIR', 'TEMP', 'TMP', 'TMPDIR', 'XDG_CONFIG_HOME'):
            path = base / name.lower()
            path.mkdir()
            env[name] = str(path)
        folder = base / 'evidence'
        folder.mkdir()
        subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/night_capture.py'), str(sandbox), stage['manifest_sha256'],
                        str(len(records)), str(folder)], cwd=sandbox, env=env, check=True)
        capture = _checked_capture(folder, records)
        if preflight:
            print(json.dumps({'preflight_only': True, 'image_count': len(records), 'gpu_gate': initial_gpu,
                              'checkpoint_sha256': A_SHA, 'config_sha256': protocol_sha, 'code_sha256': code,
                              'images_sha256': cfg['images_sha256'], 'staged_manifest_sha256': stage['manifest_sha256'],
                              'staging_verified_paths': stage['reads'], **capture},
                             ensure_ascii=False, allow_nan=False))
            return
        anchors, d_keys = _anchor_lists(cfg, records)
        metadata = {'keys': [r['key'] for r in records], 'anchors': anchors, 'params': cfg['e1']}
        def replay():
            return run_rank_worker('e1', folder / 'features.npz', metadata, scores=folder / 'scores.npz')
        result = replay()
        check_equal('independently replayed E1 ranks', result, replay())
        check_equal('E1 ranked seeds', set(anchors), set(result['arms']))
        by_key = {r['key']: r for r in records}
        arms, memberships = {}, {}
        for seed in anchors:
            pick = result['arms'][seed]
            keys = pick['keys']
            chosen_keys = set(keys)
            check_equal('E1 budget and anchor ordering', (100, 100, anchors[seed]), (len(keys), len(chosen_keys), keys[:20]))
            if not chosen_keys <= by_key.keys():
                raise ContractError('E1 selected a key outside the approved pool')
            chosen = [by_key[k] for k in keys]
            memberships[seed] = chosen_keys
            arms[seed] = {**pick, 'records': chosen, 'record_sha256': canonical_hash(chosen),
                          'groups': len({r['group_id'] for r in chosen}), 'overlap_with_D': len(chosen_keys & set(d_keys[seed]))}
        overlaps = {f'{a}/{b}': len(memberships[a] & memberships[b]) for a, b in combinations(memberships, 2)}
        (folder / 'strategy.json').write_bytes(raw)
        write_json(folder / 'e1.json', {'purpose': 'night_pool_strategy_e1', 'arms': arms, 'images_sha256': cfg['images_sha256']})
        write_json(folder / 'run.json', {'status': 'sealed_e1_only', 'trial': 'E1', 'e2_executed': False,
                   'labels_read': False, 'training_performed': False, 'checkpoint_sha256': A_SHA, 'config_sha256': protocol_sha,
                   'images_sha256': cfg['images_sha256'], 'random_sha256': cfg['random_sha256'], 'gpu_gate': initial_gpu,
                   'staging_verified_paths': stage['reads'], 'overlap_images': overlaps,
                   'unique_images_across_seeds': len(set().union(*memberships.values())),
                   'ranking_worker_reads': result['reads'], 'replay_exact': True, 'code_sha256': code,
                   'environment': {'python': sys.version, 'numpy': np.__version__}})
        for name, digest in code.items():
            out = folder / 'code' / 'scripts' / name
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / 'scripts' / name, out)
            check_equal('recorded E1 source', digest, sha256(out))
        seal = {p.relative_to(folder).as_posix(): {'sha256': sha256(p), 'bytes': p.stat().st_size}
                for p in sorted(folder.rglob('*')) if p.is_file()}
        write_json(folder / 'manifest.json', {'status': 'sealed', 'scope': 'E1 and shared feature capture only', 'files': seal})
        check_equal('A unchanged during E1', A_SHA, sha256(ROOT / MODEL))
        check_equal('strategy protocol unchanged during E1', protocol_sha, sha256(ROOT / 'configs/strategy.json'))
        if output.exists():
            raise ContractError('Refusing to overwrite a strategy run')
        folder.rename(output)
        for relative, item in seal.items():
            file = output / relative
            check_equal('published E1 evidence', (item['bytes'], item['sha256']), (file.stat().st_size, sha256(file)))
        print(f'Sealed E1 {output.name}; 3 × 100, no E2 list generated', flush=True)


def run_e2(protocol_sha, scratch, output, e1_run, e1_manifest_sha):
    """Seal a separate CPU-only E2 run, consuming only the verified feature handoff."""
    scratch, output = _strategy_paths(scratch, output)
    parent = Path(e1_run).absolute()
    if parent.parent != ROOT / 'runs' or not re.fullmatch(r'pool_\d{3}', parent.name):
        raise ContractError('E2 requires an E1 source under runs/pool_NNN')
    parent_rel = parent.relative_to(ROOT).as_posix()
    reads = []
    raw, cfg = _strategy_config(protocol_sha, reads=reads)
    seal = read_sealed_json(ROOT, f'{parent_rel}/manifest.json', e1_manifest_sha, reads)
    check_equal('E1 parent scope', ('sealed', 'E1 and shared feature capture only'),
                (seal.get('status'), seal.get('scope')))
    try:
        files = seal['files']
        feature = files['features.npz']
        feature_sha, feature_size = feature['sha256'], feature['bytes']
        check_equal('E1 feature protocol SHA-256', protocol_sha, files['strategy.json']['sha256'])
    except (KeyError, TypeError) as exc:
        raise ContractError('E1 seal lacks the feature or strategy identity') from exc
    if type(feature_size) is not int or feature_size <= 0:
        raise ContractError('Invalid sealed feature size')
    print('Scope: E2 only; sealed original features, image-only manifest and D anchors. CPU ranking, no images or E1 scores.', flush=True)
    started = time.perf_counter()
    blob = read_sealed_bytes(ROOT, f'{parent_rel}/features.npz', feature_sha, reads)
    check_equal('sealed feature size', feature_size, len(blob))
    image_rel = f'data/soda10m/splits/night_v1/{POOL_MANIFEST}'
    manifest = read_sealed_json(ROOT, image_rel, cfg['images_sha256'], reads)
    records = sorted(_records(manifest, POOL, cfg['pool_count'], images_only=True), key=lambda r: r['key'])
    anchors, d_keys = _anchor_lists(cfg, records, reads=reads)
    metadata = {'keys': [r['key'] for r in records], 'groups': [r['group_id'] for r in records],
                'anchors': anchors, 'params': cfg['e2']}
    by_key = {r['key']: r for r in records}
    kinds = ['representative']*cfg['e2']['representatives'] + ['diverse']*cfg['e2']['diverse']
    code = {name: sha256(ROOT / 'scripts' / name) for name in
            ('night_strategy.py', 'night_contract.py', 'night_select.py', 'source_contract.py')}
    with tempfile.TemporaryDirectory(prefix='strategy-', dir=scratch) as tmp:
        base = Path(tmp)
        folder = base / 'evidence'
        folder.mkdir()
        features = base / 'input/features.npz'
        features.parent.mkdir()
        features.write_bytes(blob)
        del blob
        keys, z = load_features(features)
        check_equal('E2 staged feature coverage', (metadata['keys'], (cfg['pool_count'], 512)), (keys, z.shape))
        result = run_rank_worker('e2', features, metadata)
        check_equal('independently replayed E2 ranks', result, run_rank_worker('e2', features, metadata))
        check_equal('E2 allowed rank reads', ['features.npz'], result['reads'])
        check_equal('E2 ranked seeds', set(anchors), set(result['arms']))
        check_equal('unchanged staged features', feature_sha, sha256(features))
        arms, memberships = {}, {}
        head = cfg['anchor_count']
        for seed in anchors:
            arm = result['arms'][seed]
            keys = arm['keys']
            unique_keys = set(keys)
            check_equal('E2 budget and anchor ordering', (100, 100, anchors[seed]), (len(keys), len(unique_keys), keys[:head]))
            if not unique_keys <= by_key.keys():
                raise ContractError('E2 selected a key outside the approved pool')
            chosen = [by_key[k] for k in keys]
            anchor_groups = {r['group_id'] for r in chosen[:head]}
            extra_groups = {r['group_id'] for r in chosen[head:]}
            if len(extra_groups) != len(kinds) or anchor_groups & extra_groups:
                raise ContractError('E2 new picks must cover distinct groups absent from anchors')
            check_equal('E2 representative and diversity steps',
                        [(k, by_key[k]['group_id'], kind) for k, kind in zip(keys[head:], kinds, strict=True)],
                        [(step['key'], step['group_id'], step['kind']) for step in arm['steps']])
            arms[seed] = {**arm, 'records': chosen, 'record_sha256': canonical_hash(chosen),
                          'groups': len(anchor_groups | extra_groups), 'new_groups': len(extra_groups),
                          'overlap_with_D': len(unique_keys & set(d_keys[seed]))}
            memberships[seed] = unique_keys

        source = {'run': parent_rel, 'manifest_sha256': e1_manifest_sha,
                  'features_sha256': feature_sha, 'features_bytes': feature_size}
        (folder / 'strategy.json').write_bytes(raw)
        write_json(folder / 'e2.json', {'purpose': 'night_pool_strategy_e2', 'arms': arms, 'images_sha256': cfg['images_sha256']})
        write_json(folder / 'run.json', {'status': 'sealed_e2_only', 'trial': 'E2', 'feature_source': source,
                   'config_sha256': protocol_sha, 'images_sha256': cfg['images_sha256'], 'random_sha256': cfg['random_sha256'],
                   'source_reads': reads, 'ranking_worker_reads': result['reads'], 'replay_exact': True,
                   'labels_read': False, 'candidate_images_read': False, 'e1_scores_read': False,
                   'e1_list_read': False, 'prediction_performed': False, 'training_performed': False,
                   'pool_images': len(records), 'pool_risk_groups': len(set(metadata['groups'])),
                   'overlap_images': {f'{a}/{b}': len(memberships[a] & memberships[b]) for a, b in combinations(memberships, 2)},
                   'unique_images_across_seeds': len(set().union(*memberships.values())), 'code_sha256': code,
                   'environment': {'device': 'cpu', 'python': sys.version, 'numpy': np.__version__, 'openblas_threads': 1},
                   'seconds': time.perf_counter()-started})
        for name, digest in code.items():
            target = folder / 'code/scripts' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / 'scripts' / name, target)
            check_equal('recorded E2 source', digest, sha256(target))
        sealed = {p.relative_to(folder).as_posix(): {'sha256': sha256(p), 'bytes': p.stat().st_size}
                  for p in sorted(folder.rglob('*')) if p.is_file()}
        write_json(folder / 'manifest.json', {'status': 'sealed', 'scope': 'E2 feature-only ranking', 'files': sealed})
        if output.exists() or output.is_symlink():
            raise ContractError('Refusing to overwrite a strategy run')
        folder.rename(output)
        for rel, file in sealed.items():
            p = output / rel
            check_equal('published E2 evidence', (file['bytes'], file['sha256']), (p.stat().st_size, sha256(p)))
        print(f'Sealed E2 {output.name}; 3 × 100, CPU-only feature ranking', flush=True)


def ratio_parameters(arm, random_count):
    if arm not in ('e1', 'e2') or type(random_count) is not int or random_count not in (20, 40, 60, 100):
        raise ContractError('Invalid registered ratio')
    if random_count == 100:
        return None  # pure D endpoint; no strategy ranking
    count = 100-random_count
    if arm == 'e1':
        return {'shortlist': 64, 'quota': count//CLASS_COUNT}
    return {'representatives': count//2, 'diverse': count//2, 'neighbors': 5}


def ratio_config(protocol_sha, reads):
    raw = read_sealed_bytes(ROOT, 'configs/ratio.json', protocol_sha, reads)
    cfg = json.loads(raw)
    check_equal('approved single-seed ratio sweep', (1, 43, [40, 60], {'e1', 'e2'}),
                (cfg['schema_version'], cfg['seed'], cfg['random_counts'], set(cfg['first_runs_sha256'])))
    return raw, cfg


def ratio_pick_rows(arm, random_count, d_keys, pick, by_key):
    params = ratio_parameters(arm, random_count)
    keys = pick['keys']
    check_equal('ratio selection budget', (100, 100), (len(keys), len(set(keys))))
    check_equal('ratio ordered D prefix', d_keys[:random_count], keys[:random_count])
    if any(k not in by_key for k in keys):
        raise ContractError('Ratio selected a key outside the image-only pool')
    rows = [by_key[k] for k in keys]
    if params is None:
        return rows
    check_equal('ratio new step order', keys[random_count:], [s['key'] for s in pick['steps']])
    if arm == 'e2':
        anchor_groups = {r['group_id'] for r in rows[:random_count]}
        tail_groups = [r['group_id'] for r in rows[random_count:]]
        if len(set(tail_groups)) != 100-random_count or anchor_groups.intersection(tail_groups):
            raise ContractError('E2 ratio risk groups must be distinct and absent from the D prefix')
        check_equal('E2 ratio step groups', tail_groups, [s['group_id'] for s in pick['steps']])
        check_equal('E2 ratio 1:1 labels', ['representative']*params['representatives'] + ['diverse']*params['diverse'],
                    [s['kind'] for s in pick['steps']])
    else:
        counts = Counter(s['class_id'] for s in pick['steps'])
        check_equal('E1 ratio class selection counts', [counts[i] for i in range(CLASS_COUNT)], pick['per_class'])
        check_equal('E1 ratio class shortfalls', [params['quota']-counts[i] for i in range(CLASS_COUNT)], pick['shortfalls'])
        check_equal('E1 ratio fallback budget', sum(pick['shortfalls']), counts[None])
        if any(n < 0 for n in pick['shortfalls']):
            raise ContractError('E1 ratio class quota exceeded')
    return rows


def _ratio_file(parent, seal, name, reads):
    entry = seal['files'][name]
    rel = (parent / name).relative_to(ROOT).as_posix()
    blob = read_sealed_bytes(ROOT, rel, entry['sha256'], reads)
    check_equal(f'{rel} bytes', entry['bytes'], len(blob))
    return blob


def run_ratio_selection(protocol_sha, scratch, output):
    scratch, output = _strategy_paths(scratch, output)
    common = []
    raw, cfg = ratio_config(protocol_sha, common)
    _, strategy = _strategy_config(cfg['strategy_sha256'], reads=common)
    pair = read_sealed_json(ROOT, 'configs/pair.json', cfg['pair_sha256'], common)
    manifest = read_sealed_json(ROOT, f'data/soda10m/splits/night_v1/{POOL_MANIFEST}', strategy['images_sha256'], common)
    rows = sorted(_records(manifest, POOL, strategy['pool_count'], images_only=True), key=lambda r: r['key'])
    by_key = {r['key']: r for r in rows}
    seed = str(cfg['seed'])
    _, d = _anchor_lists(strategy, rows, reads=common)
    d_keys = d[seed]
    check_equal('pure random endpoint', d_keys, [r['key'] for r in ratio_pick_rows('e1', 100, d_keys, {'keys': d_keys}, by_key)])
    parents = {a: ROOT/'runs'/name for a, name in (('e1', 'pool_002'), ('e2', 'pool_003'))}
    seals = {a: read_sealed_json(ROOT, (p/'manifest.json').relative_to(ROOT).as_posix(), pair['pool_manifest_sha256'][a], common)
             for a,p in parents.items()}
    check_equal('sealed parent scopes', [('sealed', 'E1 and shared feature capture only'), ('sealed', 'E2 feature-only ranking')],
                [(m['status'], m['scope']) for m in seals.values()])
    check_equal('original feature protocol', cfg['strategy_sha256'], seals['e1']['files']['strategy.json']['sha256'])
    code = {name: sha256(ROOT/'scripts'/name) for name in ('night_strategy.py', 'night_contract.py', 'night_select.py', 'source_contract.py')}
    print(f'Ratio selection seed {seed}: sealed features (E2), sealed features + scores (E1), no raw payload.', flush=True)
    with tempfile.TemporaryDirectory(prefix='ratio-', dir=scratch) as td:
        base = Path(td)
        box = base/'input'
        box.mkdir()
        features = box/'features.npz'
        features.write_bytes(_ratio_file(parents['e1'], seals['e1'], 'features.npz', common))
        keys, z = load_features(features)
        check_equal('ratio feature coverage', (list(by_key), (len(by_key), 512)), (keys, z.shape))
        evidence = base/'evidence'
        evidence.mkdir()
        results, logs = {}, {}
        for arm in ('e2', 'e1'):  # E2 finishes before any score file is staged.
            reads, opened = [], None
            if arm == 'e1':
                score = box/'scores.npz'
                score.write_bytes(_ratio_file(parents[arm], seals[arm], 'scores.npz', reads))
            else:
                score = None
            first = json.loads(_ratio_file(parents[arm], seals[arm], f'{arm}.json', reads))
            check_equal('same first-arm inputs', (f'night_pool_strategy_{arm}', strategy['images_sha256']),
                        (first['purpose'], first['images_sha256']))
            first_keys = first['arms'][seed]['keys']
            result = {}
            for r in (20, *cfg['random_counts']):
                params = ratio_parameters(arm, r)
                metadata = {'keys': keys, 'anchors': {seed: d_keys[:r]}, 'params': params}
                if arm == 'e2':
                    metadata['groups'] = [x['group_id'] for x in rows]
                replay = run_rank_worker(arm, features, metadata, scores=score)
                check_equal('exact ratio ranking replay', replay, run_rank_worker(arm, features, metadata, scores=score))
                opened = replay['reads']
                check_equal('ratio ranking read boundary', ['features.npz']+(['scores.npz'] if score else []), opened)
                found = replay['arms'][seed]
                picked = ratio_pick_rows(arm, r, d_keys, found, by_key)
                if r == 20:
                    if found['keys'] != first_keys:
                        raise ContractError(f'{arm} 20/80 replay differs from the sealed first list')
                    continue
                selected = set(found['keys'])
                result[str(r)] = {**found, 'records': picked, 'record_sha256': canonical_hash(picked), 'params': params,
                                  'groups': len({x['group_id'] for x in picked}),
                                  'overlap_with_D': len(selected.intersection(d_keys)),
                                  'overlap_with_first': len(selected.intersection(first_keys))}
                print(f'{arm} seed {seed} {r}/{100-r}: 100 unique; D overlap={result[str(r)]["overlap_with_D"]}', flush=True)
            results[arm] = result
            logs[arm] = {'source_reads': reads, 'worker_reads': opened, 'first_replay_exact': True}
        write_json(evidence/'lists.json', {'purpose': 'night_pool_ratio', 'seed': cfg['seed'], 'images_sha256': strategy['images_sha256'],
                                           'arms': results})
        (evidence/'ratio.json').write_bytes(raw)
        write_json(evidence/'run.json', {'status': 'sealed_ratio_selection', 'protocol_sha256': protocol_sha, 'reads': common,
                                        'arms': logs, 'random_endpoint_exact': True, 'pool_manifest_sha256': pair['pool_manifest_sha256'],
                                        'feature_sha256': seals['e1']['files']['features.npz']['sha256'], 'code_sha256': code,
                                        'numpy': np.__version__, 'python': sys.version, 'labels_read': False,
                                        'prediction_performed': False, 'training_performed': False})
        for name, digest in code.items():
            target = evidence/'code/scripts'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/'scripts'/name, target)
            check_equal('archived ratio code', digest, sha256(target))
        files = {p.relative_to(evidence).as_posix(): {'sha256': sha256(p), 'bytes': p.stat().st_size}
                 for p in sorted(evidence.rglob('*')) if p.is_file()}
        write_json(evidence/'manifest.json', {'status': 'sealed', 'scope': 'single-seed ratio selection only', 'files': files})
        if output.exists() or output.is_symlink():
            raise ContractError('Refusing to overwrite a ratio run')
        evidence.rename(output)
        for rel, entry in files.items():
            p = output/rel
            check_equal(f'published ratio selection {rel}', (entry['bytes'], entry['sha256']), (p.stat().st_size, sha256(p)))
    print(f'Sealed {output.name}; all four ratio manifests, no training', flush=True)


def mechanism_config(protocol_sha, reads):
    raw = read_sealed_bytes(ROOT, 'configs/e1.json', protocol_sha, reads)
    cfg = json.loads(raw)
    check_equal('approved E1 mechanism sweep', (1, 43, 20, [16, 8], {'e1'}),
                (cfg['schema_version'], cfg['seed'], cfg['random_count'], cfg['score_quotas'], set(cfg['first_runs_sha256'])))
    return raw, cfg


def mechanism_rows(score_quota, anchors, pick, by_key):
    if type(score_quota) is not int or score_quota not in (0, 8, 16):
        raise ContractError('Unknown registered E1 mechanism quota')
    keys = pick['keys']
    check_equal('E1 mechanism count and prefix', (20, 100, 100, anchors), (len(anchors), len(keys), len(set(keys)), keys[:20]))
    if any(k not in by_key for k in keys):
        raise ContractError('E1 mechanism picked a key outside the image-only pool')
    check_equal('E1 mechanism quota and fallback', ([16]*CLASS_COUNT, [0]*CLASS_COUNT), (pick['per_class'], pick['shortfalls']))
    expected = [(k, c, 'score' if i//CLASS_COUNT < score_quota else 'diversity')
                for i,k in enumerate(keys[20:]) for c in [i % CLASS_COUNT]]
    check_equal('E1 mechanism phase and category order', expected, [(s['key'], s['class_id'], s['stage']) for s in pick['steps']])
    return [by_key[k] for k in keys]


def _publish_mechanism(evidence, output, code, scope):
    for name, digest in code.items():
        to = evidence/'code/scripts'/name
        to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/'scripts'/name, to)
        check_equal('archived mechanism selection code', digest, sha256(to))
    files = {p.relative_to(evidence).as_posix(): {'sha256': sha256(p), 'bytes': p.stat().st_size}
             for p in sorted(evidence.rglob('*')) if p.is_file()}
    write_json(evidence/'manifest.json', {'status': 'sealed', 'scope': scope, 'files': files})
    if output.exists() or output.is_symlink():
        raise ContractError('Refusing to overwrite a mechanism run')
    evidence.rename(output)
    for rel, item in files.items():
        p = output/rel
        check_equal('published mechanism file', (item['bytes'], item['sha256']), (p.stat().st_size, sha256(p)))


def run_mechanism_selection(protocol_sha, scratch, output):
    scratch, output = _strategy_paths(scratch, output)
    reads = []
    raw, cfg = mechanism_config(protocol_sha, reads)
    _, strategy = _strategy_config(cfg['strategy_sha256'], reads=reads)
    pair = read_sealed_json(ROOT, 'configs/pair.json', cfg['pair_sha256'], reads)
    parent = ROOT/'runs/pool_002'
    seal = read_sealed_json(ROOT, 'runs/pool_002/manifest.json', pair['pool_manifest_sha256']['e1'], reads)
    check_equal('E1 mechanism parent', ('sealed', 'E1 and shared feature capture only', cfg['strategy_sha256']),
                (seal['status'], seal['scope'], seal['files']['strategy.json']['sha256']))
    manifest = read_sealed_json(ROOT, f'data/soda10m/splits/night_v1/{POOL_MANIFEST}', strategy['images_sha256'], reads)
    rows = sorted(_records(manifest, POOL, strategy['pool_count'], images_only=True), key=lambda r: r['key'])
    by_key = {r['key']: r for r in rows}
    seed = str(cfg['seed'])
    anchors, random = _anchor_lists(strategy, rows, reads=reads)
    first = json.loads(_ratio_file(parent, seal, 'e1.json', reads))
    check_equal('E1 original ranking receipt', ('night_pool_strategy_e1', strategy['images_sha256']), (first['purpose'], first['images_sha256']))
    first = first['arms'][seed]
    code = {name: sha256(ROOT/'scripts'/name) for name in ('night_strategy.py', 'night_contract.py', 'night_select.py', 'source_contract.py')}
    print('Scope: E1 mechanism, seed 43, verified image-only list and sealed E1 scores/features; no raw payload.', flush=True)
    with tempfile.TemporaryDirectory(prefix='mechanism-', dir=scratch) as td:
        base = Path(td)
        box = base/'input'
        box.mkdir()
        feature_path, score_path = box/'features.npz', box/'scores.npz'
        for path in (feature_path, score_path):
            path.write_bytes(_ratio_file(parent, seal, path.name, reads))
        keys, z = load_features(feature_path)
        check_equal('mechanism feature identity', (list(by_key), (994, 512)), (keys, z.shape))
        vector_ids = {key: i for i,key in enumerate(keys)}
        d_keys = set(random[seed])
        known = {'D': d_keys, '0/16': set(first['keys'])}
        results = {}
        for q in (0, *cfg['score_quotas']):
            params = dict(strategy['e1'], score_quota=q)
            metadata = {'keys': keys, 'anchors': {seed: anchors[seed]}, 'params': params}
            result = run_rank_worker('e1-mechanism', feature_path, metadata, scores=score_path)
            check_equal('mechanism worker reads', ['features.npz', 'scores.npz'], result['reads'])
            check_equal('mechanism deterministic replay', result, run_rank_worker('e1-mechanism', feature_path, metadata, scores=score_path))
            pick = result['arms'][seed]
            name = f'{q}/{16-q}'
            if q and 'quota_failure' in pick:
                results[str(q)] = {'status': 'quota_failure', 'params': params, 'reason': pick['quota_failure']}
                print(f'{name}: {pick["quota_failure"]}', flush=True)
                continue
            picked = mechanism_rows(q, anchors[seed], pick, by_key)
            if q == 0:
                def step_keys(steps):
                    return [(s['key'], s['class_id'], round(s['distance'], 12), round(s['score'], 12)) for s in steps]
                check_equal('exact 0/16 endpoint ordered replay', (first['keys'], first['per_class'], first['shortfalls'], step_keys(first['steps'])),
                            (pick['keys'], pick['per_class'], pick['shortfalls'], step_keys(pick['steps'])))
                print('0/16 ordered first-arm ranking and step evidence reproduced', flush=True)
            selected = set(pick['keys'])
            duplicate_of = next((k for k,v in known.items() if v == selected), None) if q else None
            ids = [vector_ids[k] for k in pick['keys'][20:]]
            v = z[ids]
            similarities = np.clip(v @ v.T, -1, 1)[np.triu_indices(len(ids), k=1)]
            results[str(q)] = {**pick, 'params': params, 'status': 'reference' if not q else 'duplicate' if duplicate_of else 'ready',
                               'duplicate_of': duplicate_of, 'records': picked, 'record_sha256': canonical_hash(picked),
                               'groups': len({r['group_id'] for r in picked}),
                               'new_groups': len({r['group_id'] for r in picked[20:]}),
                               'mean_added_pair_cosine': float(similarities.mean()),
                               'mean_added_score': float(np.mean([s['score'] for s in pick['steps']])),
                               'overlap_with_D': len(selected & d_keys), 'overlap_with_first': len(selected & known['0/16'])}
            known[name] = selected
            if q:
                print(f'{name}: {results[str(q)]["status"]}; 100 unique, D overlap={len(selected & d_keys)}', flush=True)
        evidence = base/'evidence'
        evidence.mkdir()
        write_json(evidence/'lists.json', {'purpose': 'night_pool_e1_mechanism', 'seed': cfg['seed'], 'images_sha256': strategy['images_sha256'],
                                           'arms': results})
        (evidence/'e1.json').write_bytes(raw)
        write_json(evidence/'run.json', {'status': 'sealed_e1_mechanism_selection', 'protocol_sha256': protocol_sha, 'reads': reads,
                                        'worker_reads': ['features.npz', 'scores.npz'], 'first_replay_exact': True,
                                        'pool_manifest_sha256': pair['pool_manifest_sha256']['e1'], 'code_sha256': code,
                                        'numpy': np.__version__, 'python': sys.version, 'labels_read': False,
                                        'prediction_performed': False, 'training_performed': False})
        _publish_mechanism(evidence, output, code, 'E1 mechanism selection only')
    print(f'Sealed {output.name}; selection only', flush=True)


def e2_mechanism_config(protocol_sha, reads):
    raw = read_sealed_bytes(ROOT, 'configs/e2.json', protocol_sha, reads)
    cfg = json.loads(raw)
    check_equal('approved E2 mechanism sweep', (1, 43, 40, [40, 20]),
                (cfg['schema_version'], cfg['seed'], cfg['random_count'], cfg['representatives']))
    return raw, cfg


def e2_mechanism_rows(representatives, d_keys, pick, by_key):
    if type(representatives) is not int or representatives not in (20, 30, 40):
        raise ContractError('Unknown registered E2 mechanism quota')
    keys = pick['keys']
    check_equal('E2 mechanism count and D prefix', (100, 100, d_keys[:40]), (len(keys), len(set(keys)), keys[:40]))
    if any(k not in by_key for k in keys):
        raise ContractError('E2 mechanism picked a key outside the image-only pool')
    rows = [by_key[k] for k in keys]
    anchor_groups = {r['group_id'] for r in rows[:40]}
    tail_groups = [r['group_id'] for r in rows[40:]]
    if len(set(tail_groups)) != 60 or anchor_groups.intersection(tail_groups):
        raise ContractError('E2 mechanism risk groups must be distinct and absent from the D prefix')
    kinds = ['representative']*representatives + ['diverse']*(60-representatives)
    check_equal('E2 mechanism phase and group order', list(zip(keys[40:], tail_groups, kinds, strict=True)),
                [(s['key'], s['group_id'], s['kind']) for s in pick['steps']])
    return rows


def e2_representative_prefix(first, new):
    common = min(len(first), len(new))
    check_equal('E2 representative prefix', first[:common], new[:common])


def e2_reference_receipt(cfg, reads):
    """Validate the sealed E2-only schedule receipt, never opening mixed-arm lists.json."""
    parent = read_sealed_json(ROOT, 'runs/pool_004/manifest.json', cfg['ratio_pool_sha256'], reads)
    check_equal('E2 30/30 parent identity', ('sealed', 'single-seed ratio selection only', cfg['ratio_sha256']),
                (parent['status'], parent['scope'], parent['files']['ratio.json']['sha256']))
    folder = ROOT/'runs/e2_002'
    seal = read_sealed_json(ROOT, 'runs/e2_002/manifest.json', cfg['first_run_sha256'], reads)
    run = json.loads(_ratio_file(folder, seal, 'run.json', reads))
    check_equal('E2 30/30 reference run', ('sealed', 'complete', [43], 'e2', 'S6 single-seed ratio exploration', cfg['ratio_sha256']),
                (seal['status'], run['status'], run['completed_seeds'], run['arm'], run['scope'], run['protocol_sha256']))
    ratio = run['ratio']
    check_equal('E2 30/30 reference provenance', (43, 40, 'runs/pool_004', cfg['ratio_pool_sha256'], cfg['strategy_sha256'], cfg['pair_sha256']),
                tuple(ratio[k] for k in ('seed', 'random_count', 'pool_run', 'pool_sha256', 'strategy_sha256', 'pair_sha256')))
    slots = json.loads(_ratio_file(folder, seal, 'slots.json', reads))
    check_equal('E2 reference slot receipt', ('e2', True, {'43'}), (slots['arm'], slots['slots_verified'], set(slots['seeds'])))
    return folder, seal, run, slots['seeds']['43']


def run_e2_mechanism_selection(protocol_sha, scratch, output):
    scratch, output = _strategy_paths(scratch, output)
    reads = []
    raw, cfg = e2_mechanism_config(protocol_sha, reads)
    _, strategy = _strategy_config(cfg['strategy_sha256'], reads=reads)
    pair = read_sealed_json(ROOT, 'configs/pair.json', cfg['pair_sha256'], reads)
    parent = ROOT/'runs/pool_002'
    seal = read_sealed_json(ROOT, 'runs/pool_002/manifest.json', pair['pool_manifest_sha256']['e1'], reads)
    check_equal('E2 mechanism feature parent', ('sealed', 'E1 and shared feature capture only', cfg['strategy_sha256']),
                (seal['status'], seal['scope'], seal['files']['strategy.json']['sha256']))
    manifest = read_sealed_json(ROOT, f'data/soda10m/splits/night_v1/{POOL_MANIFEST}', strategy['images_sha256'], reads)
    rows = sorted(_records(manifest, POOL, strategy['pool_count'], images_only=True), key=lambda r: r['key'])
    by_key = {r['key']: r for r in rows}
    seed = str(cfg['seed'])
    _, random = _anchor_lists(strategy, rows, reads=reads)
    d_keys = random[seed]
    _, _, _, prior = e2_reference_receipt(cfg, reads)
    check_equal('E2 30/30 D anchor identities', (40, d_keys, d_keys[:40]),
                (prior['random_count'], prior['d_input_slots'], prior['anchors']))
    code = {name: sha256(ROOT/'scripts'/name) for name in ('night_strategy.py', 'night_contract.py', 'night_select.py', 'source_contract.py')}
    print('Scope: E2 mechanism seed 43; feature-only ranking, no E1 scores or model replay.', flush=True)
    with tempfile.TemporaryDirectory(prefix='mechanism-', dir=scratch) as td:
        base = Path(td)
        box = base/'input'
        box.mkdir()
        feature_path = box/'features.npz'
        feature_path.write_bytes(_ratio_file(parent, seal, feature_path.name, reads))
        keys, z = load_features(feature_path)
        check_equal('E2 mechanism feature identity', (list(by_key), (994, 512)), (keys, z.shape))
        vector_ids = {k: i for i,k in enumerate(keys)}
        groups = [r['group_id'] for r in rows]
        group_sizes = Counter(groups)
        weights = np.asarray([1/group_sizes[g] for g in groups], dtype=np.float64)
        known = {'D': set(d_keys), '30/30': set(prior['selection_order'])}
        results, first_reps = {}, None
        for n in (30, *cfg['representatives']):
            params = {'representatives': n, 'diverse': 60-n, 'neighbors': strategy['e2']['neighbors']}
            metadata = {'keys': keys, 'groups': groups, 'anchors': {seed: d_keys[:40]}, 'params': params}
            ranked = run_rank_worker('e2', feature_path, metadata)
            check_equal('E2 mechanism worker reads', ['features.npz'], ranked['reads'])
            check_equal('E2 mechanism replay', ranked, run_rank_worker('e2', feature_path, metadata))
            pick = ranked['arms'][seed]
            picked = e2_mechanism_rows(n, d_keys, pick, by_key)
            reps = pick['keys'][40:40+n]
            if n == 30:
                check_equal('exact E2 30/30 ordered replay', (prior['selection_order'], prior['selection_sha256']),
                            (pick['keys'], canonical_hash(picked)))
                first_reps = reps
                print('30/30 ordered E2-only slot receipt reproduced; exact phases and groups verified', flush=True)
            else:
                e2_representative_prefix(first_reps, reps)
            selected = set(pick['keys'])
            duplicate_of = next((k for k,v in known.items() if v == selected), None) if n != 30 else None
            suffix = f'{n}/{60-n}'
            zs = z[[vector_ids[k] for k in pick['keys']]]
            similarities = np.clip(zs[40:] @ zs[40:].T, -1, 1)[np.triu_indices(60, k=1)]
            m = (1+np.clip(z @ zs.T, -1, 1).max(axis=1))/2
            results[str(n)] = {**pick, 'params': params, 'status': 'reference' if n == 30 else 'duplicate' if duplicate_of else 'ready',
                               'duplicate_of': duplicate_of, 'records': picked, 'record_sha256': canonical_hash(picked),
                               'new_groups': len({r['group_id'] for r in picked[40:]}),
                               'weighted_pool_coverage': float(np.average(m, weights=weights)),
                               'mean_added_pair_cosine': float(similarities.mean()),
                               'overlap_with_D': len(selected & known['D']), 'overlap_with_first': len(selected & known['30/30'])}
            known[suffix] = selected
            if n != 30:
                print(f'{suffix}: {results[str(n)]["status"]}; 100 unique, D overlap={len(selected & known["D"])}', flush=True)
        evidence = base/'evidence'
        evidence.mkdir()
        write_json(evidence/'lists.json', {'purpose': 'night_pool_e2_mechanism', 'seed': cfg['seed'], 'images_sha256': strategy['images_sha256'],
                                           'arms': results, 'new_arm_overlap': len(known['40/20'] & known['20/40'])})
        (evidence/'e2.json').write_bytes(raw)
        write_json(evidence/'run.json', {'status': 'sealed_e2_mechanism_selection', 'protocol_sha256': protocol_sha, 'reads': reads,
                                        'worker_reads': ['features.npz'], 'first_replay_exact': True,
                                        'pool_manifest_sha256': pair['pool_manifest_sha256']['e1'], 'code_sha256': code,
                                        'numpy': np.__version__, 'python': sys.version, 'labels_read': False, 'e1_scores_read': False,
                                        'prediction_performed': False, 'training_performed': False})
        _publish_mechanism(evidence, output, code, 'E2 mechanism selection only')
    print(f'Sealed {output.name}; feature-only E2 selection', flush=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '_rank':
        _, _, arm, features, *tail = sys.argv
        if len(tail) > 1:
            raise ContractError('Only a single E1 score file may be passed')
        result = _rank_payload(arm, features, tail[0] if tail else None, json.load(sys.stdin))
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('e1-preflight', 'e1', 'e2', 'ratio', 'mechanism', 'e2-mechanism'))
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--e1-run', type=Path)
    parser.add_argument('--e1-manifest-sha256')
    args = parser.parse_args()
    if (args.mode != 'e1-preflight') != (args.output is not None):
        parser.error('--output required except for e1-preflight')
    if args.mode == 'e2':
        if args.e1_run is None or args.e1_manifest_sha256 is None:
            parser.error('e2 requires --e1-run and --e1-manifest-sha256')
        run_e2(args.protocol_sha256, args.scratch, args.output, args.e1_run, args.e1_manifest_sha256)
        return
    if args.e1_run is not None or args.e1_manifest_sha256 is not None:
        parser.error('--e1-run and --e1-manifest-sha256 may only be used for e2')
    if args.mode == 'ratio':
        run_ratio_selection(args.protocol_sha256, args.scratch, args.output)
        return
    if args.mode == 'mechanism':
        run_mechanism_selection(args.protocol_sha256, args.scratch, args.output)
        return
    if args.mode == 'e2-mechanism':
        run_e2_mechanism_selection(args.protocol_sha256, args.scratch, args.output)
        return
    run_e1(args.protocol_sha256, args.scratch, args.output, preflight=args.mode == 'e1-preflight')


if __name__ == '__main__':
    main()
