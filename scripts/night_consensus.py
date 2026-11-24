"""Score fixed tail boxes using two extra views; develop and replay, never train."""
from __future__ import annotations

import argparse
from fractions import Fraction
import gzip
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time

import numpy as np

import detection_metrics as dm
import night_accept as accept
from night_contract import ROOT, _no_reparse, read_sealed_json, verify_dev
from source_contract import ContractError, check_equal, sha256, write_json

VIEWS = ((640, True), (960, False))
VEHICLES = (2, 3, 4)
TAILS = (3, 4)
ARMS = ('consensus', 'competition')
DOMAINS = {'day': 'source_dev', 'night': 'night_dev'}
PARENT = 'expand_003'
SEED = 43
PARENT_SHA = '71c2942af5343795066793488d661b68e4e6d5701027fd0f6f19c8e298722d35'
INNER_SHA = '902f891efea87cd242ff4f0e383055cc5ef828607c01e88f4069a21baaa3de9d'
ACCEPT_SHA = 'fbaf414b42e750260f8aa552f50da0853f69028ac4821d366891a6ea08676403'
POLICY_SHA = '47469459997e2d2439c91ad884ea90b081a11387d45515968fbf00e6afeb678f'
READY = {'source_dev': '72e75ebd99014c830e0227061e390e6aed768ae724b532a5c2c90fd8dec93e85',
         'night_dev': 'fc9e71fdfd7384e54ce33811844fef7f6e364a44c2d5c97230acfad199fabd5d'}
CODE = ('night_consensus.py', 'night_run.py', 'night_accept.py', 'night_contract.py', 'evaluate.py',
        'detection_metrics.py', 'person_fusion.py', 'seed_eval.py', 'source_contract.py')
LIMIT = 600


def _columns(pred):
    classes, conf, boxes = (pred[k] for k in ('cls', 'conf', 'original_boxes'))
    return np.asarray(classes, dtype=int), np.asarray(conf, dtype=np.float64), np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def score_rows(rows, auxiliary):
    """Derive float32 scores from predictions only; leave the anchor's geometry intact."""
    keys = {r['key'] for r in rows}
    check_equal('complete and unique view keys', (len(rows), [keys, keys]),
                (len(keys), [set(view) for view in auxiliary]))
    result = {arm: [] for arm in ARMS}
    for row in rows:
        pred = row['pred']
        cls, conf, boxes = _columns(pred)
        targets = np.isin(cls, TAILS)
        base_boxes, target_cls = boxes[targets], cls[targets, None]
        plain, penalized = np.zeros(sum(targets)), np.zeros(sum(targets))
        for i, view in enumerate([pred, *(a[row['key']] for a in auxiliary)]):
            c, s, xy = _columns(view)
            vehicles = np.isin(c, VEHICLES)
            c, s, xy = c[vehicles], s[vehicles], xy[vehicles]
            covers = dm.box_iou(base_boxes, xy) >= .5
            support = (np.where(covers & (c == target_cls), s, 0).max(axis=1, initial=0)
                       if i else conf[targets])
            rival = np.where(covers & (c != target_cls), s, 0).max(axis=1, initial=0)
            plain += support
            penalized += support * (1-rival)
        for arm, scores in zip(ARMS, (plain, penalized), strict=True):
            new = list(pred['conf'])
            for p, score in zip(np.flatnonzero(targets), (scores/3).astype(np.float32).tolist(), strict=True):
                new[p] = score
            result[arm].append({**row, 'pred': {**pred, 'conf': new}})
    return result


def parent_inputs():
    reads = []
    _, policy = accept._read_policy(ROOT, POLICY_SHA, reads)
    ref_dir = ROOT/'runs/accept_001'
    ref_seal = read_sealed_json(ref_dir, 'manifest.json', ACCEPT_SHA, [])
    reference = json.loads(accept._checked_file(ref_dir, ref_seal, 'reference.json', []))
    base_dir = ROOT/'runs'/PARENT
    seal = read_sealed_json(base_dir, 'manifest.json', PARENT_SHA, [])

    def receipt(rel):
        return json.loads(accept._checked_file(base_dir, seal, rel, []))

    state = receipt('run.json')
    check_equal('fixed balanced E2', ('sealed', 'e2', SEED, 'complete', False, False, None, policy['sealed']['checkpoint_a']),
                (seal['status'], state['arm'], state['seed'], state['status'], state['final_test_executed'],
                 state['cloud_used'], state.get('e2_mix'), state['a_sha256']))
    dev_rel = f'{SEED}/dev'
    captures = receipt(f'{dev_rel}/evaluation.json')
    rows, metrics = {}, {}
    for domain, purpose in DOMAINS.items():
        dataset, count = accept.DEV[purpose]
        rel = f'manifests/{purpose}.json'
        expected = policy['development'][purpose]
        check_equal('registered development purpose', {'count': count, 'path': rel, 'sha256': expected}, state[purpose])
        doc = read_sealed_json(ROOT/'data/soda10m/splits'/dataset, rel, expected, [])
        check_equal('complete development manifest', (purpose, count), (doc['purpose'], len(doc['records'])))
        rows[domain] = accept.sealed_objects(base_dir, seal, f'{dev_rel}/{domain}/objects.jsonl.gz', purpose,
                                             accept._index(doc['records'], purpose), [])
        baseline = accept.evaluate_dev(rows[domain], policy)
        baseline['capture'] = accept._audit_views(captures[f'{domain}_views'], captures[f'{domain}_fusion_caps'])
        check_equal(f'{domain} exact replay of E2 baseline', receipt(f'{dev_rel}/{domain}/metrics.json'), baseline)
        metrics[domain] = baseline
    weight = seal['files'][f'{SEED}/last.pt']['sha256']
    return policy, reference, rows, metrics, state, weight


def check_nested(outer, inner):
    check_equal('ordered parent prefix and unique label budget', (300, 300, 100, inner),
                (len(outer), len(set(outer)), len(inner), outer[:100]))


def sealed_parents(policy, reference, rows):
    inner = ROOT/'runs/e2_001'
    seal = read_sealed_json(inner, 'manifest.json', INNER_SHA, [])
    state = json.loads(accept._checked_file(inner, seal, 'run.json', []))
    check_equal('approved inner E2', ('sealed', 'e2', 'complete', False, False, policy['sealed']['checkpoint_a'], True),
                (seal['status'], state['arm'], state['status'], state['cloud_used'], state['final_test_executed'],
                 state['a_sha256'], SEED in state['completed_seeds']))
    inner_slots = json.loads(accept._checked_file(inner, seal, 'slots.json', []))['seeds'][str(SEED)]
    outer = ROOT/'runs'/PARENT
    outer_seal = read_sealed_json(outer, 'manifest.json', PARENT_SHA, [])
    selection = json.loads(accept._checked_file(outer, outer_seal, 'selection.json', []))
    check_nested(selection['keys'], inner_slots['selection_order'])
    check_equal('inner ordered grant has the same set as its gradient slots', set(inner_slots['selection_order']),
                set(inner_slots['input_slots']))
    check_equal('outer ordered grant has the same set as its gradient slots', set(selection['keys']), set(selection['slots']))
    views, info = {}, {}
    for domain, purpose in DOMAINS.items():
        a_label, a_path = ('align_001', '17/dev/objects.jsonl.gz') if domain == 'day' else ('night_001', 'objects.jsonl.gz')
        a_dir = ROOT/'runs'/a_label
        a_seal = read_sealed_json(a_dir, 'manifest.json', policy['sealed'][a_label], [])
        a_weight = (a_seal['files']['17/last.pt']['sha256'] if domain == 'day' else a_seal['checkpoint_sha256'])
        check_equal('frozen A support checkpoint', policy['sealed']['checkpoint_a'], a_weight)
        rel = f'{SEED}/dev/{domain}/objects.jsonl.gz'
        scope = accept._index(rows[domain], purpose)
        parent_a = accept.sealed_objects(a_dir, a_seal, a_path, purpose, scope, [])
        parent_e2 = accept.sealed_objects(inner, seal, rel, purpose, scope, [])
        for parent in (parent_a, parent_e2):
            check_equal('identical five-class development truth', accept._pair_identity(rows[domain]), accept._pair_identity(parent))
        views[domain] = [{r['key']: r['pred'] for r in parent} for parent in (parent_a, parent_e2)]
        info[domain] = [{'run': a_label, 'manifest_sha256': policy['sealed'][a_label], 'objects': a_path,
                         'objects_sha256': a_seal['files'][a_path]['sha256']},
                        {'run': inner.name, 'manifest_sha256': INNER_SHA, 'objects': rel,
                         'objects_sha256': seal['files'][rel]['sha256']}]
    check_equal('white-day object identity of frozen A', reference['source']['objects_sha256'], info['day'][0]['objects_sha256'])
    return views, info


def report(policy, reference, rows, base, views):
    def diagnostic(objects, recovery):
        out = {}
        for cls in TAILS:
            name = accept.CLASSES[cls]
            goal = Fraction(policy['recovery_ratio'])*accept._recall(reference['workpoints'][name])
            curve = dm.work_curve(objects, cls)
            needed = int(np.ceil(goal * curve['gt']))
            good = curve['fp'][curve['tp'] >= needed]
            out[name] = {'target_tp': needed, 'pool_max_tp': int(curve['tp'].max(initial=0)),
                         'min_fp_for_target': int(good.min()) if len(good) else None,
                         'recovery_met': recovery[name]['met']}
        return out

    base_rec = accept.recovery(reference['workpoints'], base['night']['workpoints'], policy)
    results = {'baseline': {**base, 'tail_diagnostic': diagnostic(rows['night'], base_rec)}}
    scored = {domain: score_rows(rows[domain], inputs) for domain, inputs in views.items()}
    for arm in ARMS:
        day, night = (accept.evaluate_dev(scored[d][arm], policy) for d in DOMAINS)
        for domain, candidate in (('day', day), ('night', night)):
            for name in accept.CLASSES[:3]:
                check_equal(f'{domain} {name} workpoint isolation', base[domain]['workpoints'][name],
                            candidate['workpoints'][name])
        protection = accept.day_protection(reference['per_class'], day['per_class'], policy)
        rec = accept.recovery(reference['workpoints'], night['workpoints'], policy)
        pair = accept.paired_recall({SEED: (rows['night'], base['night']['workpoints'], scored['night'][arm],
                                            night['workpoints'])}, policy)['mean']
        gains = {n: night['workpoints'][n]['tp']-base['night']['workpoints'][n]['tp']
                 for n in map(accept.CLASSES.__getitem__, TAILS)}
        results[arm] = {'day': day, 'night': night, 'day_protection': protection, 'night_recovery': rec,
                        'night_classes_met': sum(x['met'] is True for x in rec.values()),
                        'tail_diagnostic': diagnostic(scored['night'][arm], rec), 'night_pair': pair,
                        'continue_gate': protection['met'] and min(gains.values()) >= 3 and
                        any(pair[n]['ci95'][0] > 0 for n in gains)}
    return results


def _capture(rows, state, weight, scratch, folder, started):
    from evaluate import stage_subset, verify_subset
    from night_run import _capture as collect
    from seed_eval import probe_gpu

    check_equal('locked capture package versions', state['versions'], {n: version(n) for n in state['versions']})
    checkpoint = ROOT/'runs'/PARENT/str(SEED)/'last.pt'
    check_equal('fixed EMA weight', weight, sha256(checkpoint))
    gpu = probe_gpu()
    for key in ('YOLO_CONFIG_DIR', 'TORCH_HOME', 'MPLCONFIGDIR', 'TEMP', 'TMP', 'TMPDIR', 'XDG_CONFIG_HOME'):
        path = scratch/key.lower()
        path.mkdir()
        os.environ[key] = str(path)
    os.environ.update(YOLO_AUTOINSTALL='false', WANDB_MODE='disabled')
    import torch
    import torchvision  # Register the backend used by the sealed native capture.

    if not torch.cuda.is_available():
        raise ContractError('The locked FP16 GPU capture is not available')
    torch.set_num_threads(4)
    observations = {}
    for domain, purpose in DOMAINS.items():
        dataset, _ = accept.DEV[purpose]
        root = ROOT/'data/soda10m/splits'/dataset
        contract = (verify_dev(root, READY[purpose]) if domain == 'night'
                    else verify_subset(root, READY[purpose], purpose))
        check_equal('capture uses the registered development manifest', state[purpose], contract['manifest'])
        work = scratch/domain
        work.mkdir()
        staged = stage_subset(contract, work/'data')
        observations[domain] = []
        for size, mirror in VIEWS:
            if time.monotonic()-started >= LIMIT:
                raise ContractError('Local diagnostic time limit reached')
            capture, audit = collect(torch, contract, staged, checkpoint, work, size, mirror)
            check_equal('auxiliary view and anchor identity', accept._pair_identity(rows[domain]),
                        accept._pair_identity(capture))
            obs = []
            for r in capture:
                classes, conf, boxes = _columns(r['pred'])
                use = np.isin(classes, VEHICLES)
                obs.append({'key': r['key'], 'cls': classes[use].tolist(), 'conf': conf[use].tolist(),
                            'original_boxes': boxes[use].tolist()})
            dm.write_rows(folder/f'{domain}/{size}-{int(mirror)}.jsonl.gz', obs)
            observations[domain].append(audit)
    check_equal('unchanged weight after capture', weight, sha256(checkpoint))
    return gpu, observations


def read_views(folder, seal=None):
    views = {}
    for domain in DOMAINS:
        views[domain] = []
        for size, mirror in VIEWS:
            rel = f'{domain}/{size}-{int(mirror)}.jsonl.gz'
            if seal:
                accept._checked_file(folder, seal, rel, [])
            with gzip.open(folder/rel, 'rt', encoding='utf-8') as files:
                values = [json.loads(line) for line in files]
            keys = {v['key']: v for v in values}
            check_equal('unique auxiliary view keys', len(values), len(keys))
            views[domain].append(keys)
    return views


def _inventory(folder):
    _no_reparse(folder)
    return {p.relative_to(folder).as_posix(): {'sha256': sha256(p), 'bytes': p.stat().st_size}
            for p in sorted(folder.rglob('*')) if p.is_file() and p != folder/'manifest.json'}


def _output(path, exists):
    folder = path.absolute()
    if (folder.parent != ROOT/'runs' or not re.fullmatch(r'consensus_\d{3}', folder.name)
            or folder.exists() != exists or folder.is_symlink()):
        raise ContractError('Expected a consensus_NNN run in runs/')
    _no_reparse(folder if exists else folder.parent)
    return folder


def run(output, scratch, *, capture=True):
    started = time.monotonic()
    target = _output(output, False)
    scratch = scratch.absolute()
    if (not scratch.is_dir() or scratch.resolve() != scratch or scratch.is_relative_to(ROOT) or scratch.drive != target.drive):
        raise ContractError('Expected existing external, same-volume scratch')
    _no_reparse(scratch)
    policy, reference, rows, base, state, weight = parent_inputs()
    with tempfile.TemporaryDirectory(dir=scratch, prefix='consensus-') as temp:
        path = Path(temp)
        folder = path/'evidence'
        folder.mkdir()
        if capture:
            for domain in DOMAINS:
                (folder/domain).mkdir()
            gpu, audit = _capture(rows, state, weight, path, folder, started)
            views = read_views(folder)
        else:
            gpu = None
            views, audit = sealed_parents(policy, reference, rows)
        values = report(policy, reference, rows, base, views)
        if time.monotonic()-started >= LIMIT:
            raise ContractError('Local diagnostic time limit reached before sealing')
        write_json(folder/'metrics.json', values)
        write_json(folder/'run.json', {'status': 'complete', 'parent': PARENT, 'parent_sha256': PARENT_SHA,
                   'reference_manifest_sha256': ACCEPT_SHA, 'policy_sha256': POLICY_SHA, 'seed': SEED,
                   'checkpoint_sha256': weight, 'auxiliary': audit, 'gpu': gpu,
                   'mode': 'views' if capture else 'parents', 'inference_performed': capture,
                   'raw_payload_read': capture, 'capture_versions': state['versions'] if capture else None,
                   'python': sys.version, 'numpy': np.__version__, 'seconds': time.monotonic()-started,
                   'purpose': 'development_only', 'training_performed': False, 'final_test_executed': False,
                   'cloud_used': False, 'candidate_labels_read': False})
        (folder/'code').mkdir()
        code = CODE if capture else tuple(n for n in CODE if n not in {'night_run.py', 'evaluate.py', 'person_fusion.py', 'seed_eval.py'})
        for name in code:
            shutil.copyfile(ROOT/'scripts'/name, folder/'code'/name)
        write_json(folder/'manifest.json', {'status': 'sealed', 'files': _inventory(folder)})
        _output(target, False)
        folder.rename(target)
    print(f'Sealed {target.name}.', flush=True)


def verify(output):
    folder = _output(output, True)
    seal = json.loads((folder/'manifest.json').read_bytes())
    check_equal('sealed view experiment files', ('sealed', seal['files']), (seal['status'], _inventory(folder)))
    state = json.loads(accept._checked_file(folder, seal, 'run.json', []))
    policy, reference, rows, base, _, weight = parent_inputs()
    keys = ('parent_sha256', 'reference_manifest_sha256', 'policy_sha256', 'seed', 'checkpoint_sha256', 'purpose',
            'training_performed', 'final_test_executed', 'cloud_used', 'candidate_labels_read')
    check_equal('pinned experiment scope and identity', (PARENT_SHA, ACCEPT_SHA, POLICY_SHA, SEED, weight,
                'development_only', False, False, False, False), tuple(state[k] for k in keys))
    if state['mode'] == 'views':
        check_equal('fresh capture scope', (True, True), (state['inference_performed'], state['raw_payload_read']))
        check_equal('capture domains', set(DOMAINS), set(state['auxiliary']))
        for domain, audits in state['auxiliary'].items():
            check_equal('registered auxiliary view sequence', VIEWS, tuple((a['imgsz'], a['mirror']) for a in audits))
            for audit in audits:
                check_equal('complete auxiliary capture', len(rows[domain]), audit['images'])
        views = read_views(folder, seal)
    elif state['mode'] == 'parents':
        check_equal('sealed-only scope', (False, False, None), (state['inference_performed'], state['raw_payload_read'], state['gpu']))
        views, info = sealed_parents(policy, reference, rows)
        check_equal('pinned sealed parent sources', info, state['auxiliary'])
    else:
        raise ContractError('Unknown experiment mode')
    values = report(policy, reference, rows, base, views)
    check_equal('independent exact computation from sealed view scores',
                json.loads(accept._checked_file(folder, seal, 'metrics.json', [])), values)
    print(f'Verified {len(seal["files"])} files and both development domains without loading a model.')
    for arm, v in values.items():
        print(arm, {n: (p['tp'], p['fp']) for n, p in v['night']['workpoints'].items()},
              v['tail_diagnostic'], {k: v[k] for k in ('night_classes_met', 'continue_gate') if k in v})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('run', 'parents', 'verify'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--scratch', type=Path)
    args = parser.parse_args()
    if (args.mode != 'verify') != (args.scratch is not None):
        parser.error('run and parents require --scratch')
    if args.mode == 'verify':
        verify(args.output)
    else:
        run(args.output, args.scratch, capture=args.mode == 'run')


if __name__ == '__main__':
    main()
