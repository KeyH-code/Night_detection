"""S6 paired strategy adaptation, preserving D's released gradient schedule."""
from __future__ import annotations

import argparse
from collections import Counter
from importlib.metadata import version
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

import detection_metrics as dm
import night_accept as accept
import night_train as train
from night_contract import POOL, POOL_MANIFEST, ROOT, _no_reparse, _records, read_sealed_bytes, read_sealed_json, verify_dev
from source_contract import ContractError, canonical_hash, check_equal, read_json, sha256, write_json

POOLS = {'d': ('pool_001', 'random.json'), 'e1': ('pool_002', 'e1.json'), 'e2': ('pool_003', 'e2.json')}
SEEDS = (17, 29, 43)
ARMS = ('e1', 'e2')


def aligned_slots(d, e):
    if len(d) != 100 or len(e) != 100 or len(set(d)) != 100 or len(set(e)) != 100:
        raise ContractError('Slot alignment needs 100 unique images on each arm')
    check_equal('ordered common anchors', d[:20], e[:20])
    d_set, e_set = set(d), set(e)
    new = iter(k for k in e if k not in d_set)
    return [k if k in e_set else next(new) for k in d]


def _schedule(seed, days, slots, cfg):
    return train.schedule(seed, days, slots, salt=cfg['schedule_salt'], flip=cfg['flip_probability'], brightness=cfg['brightness'])


def paired_schedule(seed, days, d, e_slots, cfg):
    old, new = (_schedule(seed, days, slots, cfg) for slots in (d, e_slots))
    sub = dict(zip(d, e_slots, strict=True))
    shared = set(d) & set(e_slots)
    for a, b in zip(old, new, strict=True):
        if (a[0] != b[0] or a[1][1:] != b[1][1:] or sub[a[1][0]] != b[1][0]
                or (a[1][0] in shared and a[1][0] != b[1][0])):
            raise ContractError('Slot permutation, common image or augmentation differs from D')
    return new


def draw_records(arm, seed, draw, images_sha, inventory):
    purpose = 'night_pool_random_control' if arm == 'd' else f'night_pool_strategy_{arm}'
    field = 'image_manifest_sha256' if arm == 'd' else 'images_sha256'
    check_equal('sealed selection identity', (purpose, images_sha, {str(s) for s in SEEDS}),
                (draw['purpose'], draw[field], set(draw['arms'])))
    entry = draw['arms'][str(seed)]
    rows = _records({'purpose': POOL, 'records': entry['records']}, POOL, 100, images_only=True)
    check_equal('selected record hash', entry['record_sha256'], canonical_hash(rows))
    check_equal('selected records in approved image list', rows, [inventory.get(r['key']) for r in rows])
    if arm != 'd':
        check_equal('selected key order', [r['key'] for r in rows], entry['keys'])
    return rows


def _json(folder, seal, rel):
    reads = []
    return json.loads(accept._checked_file(folder, seal, rel, reads))


def _config(protocol_sha):
    raw = read_sealed_bytes(ROOT, 'configs/pair.json', protocol_sha, [])
    cfg = json.loads(raw)
    check_equal('approved pair protocol', (1, 60, set(POOLS)),
                (cfg['schema_version'], cfg['max_minutes_per_arm'], set(cfg['pool_manifest_sha256'])))
    old = json.loads(read_sealed_bytes(ROOT, 'configs/adapt.json', cfg['adapt_sha256'], []))
    check_equal('fixed D training budget', (list(SEEDS), 640, 10, 200, 10),
                tuple(old[k] for k in ('seeds', 'size', 'accumulate', 'updates', 'preflight_updates')))
    return raw, cfg, old


def _day_keys(entry):
    check_equal('day replay manifest name and count', ('manifests/source_train.json', 4000), (entry['path'], entry['count']))
    day = read_sealed_json(train.SOURCE, 'manifests/source_train.json', entry['sha256'], [])
    check_equal('day replay manifest purpose', 'source_train', day['purpose'])
    days = [r['key'] for r in day['records'] if r['purpose'] == 'source_train']
    check_equal('unique day replay keys', (4000, 4000), (len(days), len(set(days))))
    return days


def alignment(cfg, old, *, arms=ARMS, seeds=SEEDS):
    """Check requested schedules against observed D before touching a label."""
    folder = ROOT/'runs/d_001'
    seal = read_sealed_json(folder, 'manifest.json', cfg['d_manifest_sha256'], [])
    identity = _json(folder, seal, 'run.json')
    check_equal('sealed D training identity', ('sealed', 'complete', list(SEEDS), cfg['adapt_sha256']),
                (seal['status'], identity['status'], identity['completed_seeds'], identity['config_sha256']))
    check_equal('active and original trainer', seal['files']['code/night_train.py']['sha256'], sha256(ROOT/'scripts/night_train.py'))
    protocol = train.load_protocol()
    check_equal('legacy identity for staging', (old['evaluation_sha256'], identity['A_sha256']),
                (protocol['sha256'], protocol['source_a']['checkpoint_sha256']))
    days = _day_keys(identity['source_train'])
    images_sha = protocol['data']['pool_images_sha256']
    pool = read_sealed_json(train.NIGHT, POOL_MANIFEST, images_sha, [])
    inventory = {r['key']: r for r in _records(pool, POOL, 994, images_only=True)}
    draws = {}
    for arm in ('d', *arms):
        name, rel = POOLS[arm]
        parent = ROOT/'runs'/name
        m = read_sealed_json(parent, 'manifest.json', cfg['pool_manifest_sha256'][arm], [])
        check_equal('pool sealed', 'sealed', m['status'])
        if arm == 'd':
            check_equal('frozen D list hash', old['random_sha256'], m['files'][rel]['sha256'])
        draws[arm] = _json(parent, m, rel)
    tasks = {arm: {} for arm in arms}
    for seed in seeds:
        d_rows = draw_records('d', seed, draws['d'], images_sha, inventory)
        d = [r['key'] for r in d_rows]
        d_plan = _schedule(seed, days, d, old)
        rec = _json(folder, seal, f'{seed}/train.json')
        inputs = _json(folder, seal, f'{seed}/inputs.json')
        check_equal('D actual updates', (200, 0, True), (rec['updates'], rec['skipped_updates'], rec['ema_reload_verified']))
        check_equal('D exact augmentation schedule hash', rec['augment_schedule_sha256'], canonical_hash(d_plan))
        check_equal('D observed microbatch order', rec['microbatches'], [[a[0], b[0]] for a, b in d_plan])
        check_equal('D observed day replay and night inventory', (inputs['source_keys'], [r['key'] for r in inputs['night']]),
                    ([a[0] for a, _ in d_plan], d))
        for arm in arms:
            rows = draw_records(arm, seed, draws[arm], images_sha, inventory)
            picked = [r['key'] for r in rows]
            slots = aligned_slots(d, picked)
            by_key = {r['key']: r for r in rows}
            plan = paired_schedule(seed, days, d, slots, old)
            check_equal('night exposure counts', {k: 20 for k in picked}, Counter(b[0] for _, b in plan))
            tasks[arm][seed] = {'rows': [by_key[k] for k in slots], 'plan': plan,
                                'slots': {'selection_order': picked, 'input_slots': slots, 'd_input_slots': d,
                                          'common_images': len(set(d)&set(picked)), 'anchors': d[:20],
                                          'selection_sha256': canonical_hash(rows), 'd_schedule_sha256': canonical_hash(d_plan),
                                          'schedule_sha256': canonical_hash(plan)}}
    return tasks, protocol, identity


def ratio_slot_task(seed, random_count, days, rows, old, previous):
    d = previous['d_input_slots']
    picked = [r['key'] for r in rows]
    check_equal('ratio full random prefix', d[:random_count], picked[:random_count])
    slots = aligned_slots(d, picked)
    check_equal('ratio D slot prefix', d[:random_count], slots[:random_count])
    d_sha = canonical_hash(_schedule(seed, days, d, old))
    check_equal('ratio frozen D schedule hash', previous['d_schedule_sha256'], d_sha)
    plan = paired_schedule(seed, days, d, slots, old)
    check_equal('ratio night exposure counts', {k: 20 for k in picked}, Counter(b[0] for _, b in plan))
    by_key = {r['key']: r for r in rows}
    return {'rows': [by_key[k] for k in slots], 'plan': plan,
            'slots': {'selection_order': picked, 'input_slots': slots, 'd_input_slots': d,
                      'common_images': len(set(d)&set(picked)), 'anchors': d[:random_count], 'random_count': random_count,
                      'selection_sha256': canonical_hash(rows), 'd_schedule_sha256': d_sha,
                      'schedule_sha256': canonical_hash(plan)}}


def ratio_alignment(cfg, old, ratio, ratio_sha, pool, pool_sha):
    from night_strategy import ratio_parameters, ratio_pick_rows

    pool = Path(pool).absolute()
    if pool.parent != ROOT/'runs' or not re.fullmatch(r'pool_\d{3}', pool.name):
        raise ContractError('Expected ratio list seal in runs/pool_NNN')
    seal = read_sealed_json(pool, 'manifest.json', pool_sha, [])
    check_equal('ratio list seal', ('sealed', 'single-seed ratio selection only', ratio_sha),
                (seal['status'], seal['scope'], seal['files']['ratio.json']['sha256']))
    receipt, picked = (_json(pool, seal, f'{k}.json') for k in ('run', 'lists'))
    check_equal('ratio input receipts', ('sealed_ratio_selection', ratio_sha, True, cfg['pool_manifest_sha256']),
                (receipt['status'], receipt['protocol_sha256'], receipt['random_endpoint_exact'], receipt['pool_manifest_sha256']))
    if any(receipt[k] for k in ('labels_read', 'prediction_performed', 'training_performed')):
        raise ContractError('Ratio selection receipt violated payload isolation')
    previous, protocol, identity = alignment(cfg, old)
    seed = ratio['seed']
    digest = protocol['data']['pool_images_sha256']
    check_equal('ratio list source', ('night_pool_ratio', seed, digest, set(ARMS)),
                (picked['purpose'], picked['seed'], picked['images_sha256'], set(picked['arms'])))
    manifest = read_sealed_json(train.NIGHT, POOL_MANIFEST, digest, [])
    by_key = {r['key']: r for r in _records(manifest, POOL, 994, images_only=True)}
    days = _day_keys(identity['source_train'])
    tasks = {}
    for arm in ARMS:
        check_equal('all registered ratio counts', {str(r) for r in ratio['random_counts']}, set(picked['arms'][arm]))
        check_equal('arm ranking replay boundary', (True, ['features.npz']+(['scores.npz'] if arm == 'e1' else [])),
                    (receipt['arms'][arm]['first_replay_exact'], receipt['arms'][arm]['worker_reads']))
        prior = previous[arm][seed]['slots']
        for r in ratio['random_counts']:
            pick = picked['arms'][arm][str(r)]
            check_equal('ratio selector settings', ratio_parameters(arm, r), pick['params'])
            rows = ratio_pick_rows(arm, r, prior['d_input_slots'], pick, by_key)
            check_equal('ratio image-only identities and digest', (rows, canonical_hash(rows)), (pick['records'], pick['record_sha256']))
            task = ratio_slot_task(seed, r, days, rows, old, prior)
            check_equal('ratio actual D overlap', task['slots']['common_images'], pick['overlap_with_D'])
            tasks[arm, r] = task
    return tasks, protocol, identity


def _mechanism_slots(seed, random_count, rows, entry, days, old, previous):
    check_equal('mechanism image-only identities and digest', (rows, canonical_hash(rows)), (entry['records'], entry['record_sha256']))
    task = ratio_slot_task(seed, random_count, days, rows, old, previous)
    check_equal('mechanism actual D overlap', task['slots']['common_images'], entry['overlap_with_D'])
    first = {key: i for i,key in enumerate(previous['input_slots'])}
    shared = {key for key in task['slots']['input_slots'] if key in first}
    d = set(previous['d_input_slots'])
    check_equal('mechanism overlap with reference', len(shared), entry['overlap_with_first'])
    task['slots']['common_non_d_slot_changes_vs_first'] = sum(key in shared and key not in d and first[key] != i
                                                             for i,key in enumerate(task['slots']['input_slots']))
    return task


def mechanism_slot_task(seed, score_quota, entry, inventory, days, old, previous):
    from night_strategy import mechanism_rows

    if seed != 43 or type(score_quota) is not int or score_quota not in (16, 8):
        raise ContractError('Expected registered E1 mechanism seed and new quota')
    check_equal('ready E1 mechanism arm', 'ready', entry['status'])
    check_equal('E1 mechanism rank parameters', {'shortlist': 64, 'quota': 16, 'score_quota': score_quota}, entry['params'])
    rows = mechanism_rows(score_quota, previous['d_input_slots'][:20], entry, inventory)
    task = _mechanism_slots(seed, 20, rows, entry, days, old, previous)
    task['slots']['score_quota'] = score_quota
    return task


def mechanism_alignment(cfg, old, mechanism, protocol_sha, pool, pool_sha):
    from night_strategy import mechanism_rows

    pool = Path(pool).absolute()
    if pool.parent != ROOT/'runs' or not re.fullmatch(r'pool_\d{3}', pool.name):
        raise ContractError('Expected mechanism selection seal in runs/pool_NNN')
    seal = read_sealed_json(pool, 'manifest.json', pool_sha, [])
    check_equal('mechanism list seal', ('sealed', 'E1 mechanism selection only', protocol_sha),
                (seal['status'], seal['scope'], seal['files']['e1.json']['sha256']))
    receipt, picked = (_json(pool, seal, f'{k}.json') for k in ('run', 'lists'))
    check_equal('mechanism input receipts', ('sealed_e1_mechanism_selection', protocol_sha, True, ['features.npz', 'scores.npz'],
                                             cfg['pool_manifest_sha256']['e1']),
                (receipt['status'], receipt['protocol_sha256'], receipt['first_replay_exact'], receipt['worker_reads'],
                 receipt['pool_manifest_sha256']))
    if any(receipt[k] for k in ('labels_read', 'prediction_performed', 'training_performed')):
        raise ContractError('Mechanism selection violated payload isolation')
    seed = mechanism['seed']
    previous, protocol, identity = alignment(cfg, old, arms=('e1',), seeds=(seed,))
    prior = previous['e1'][seed]['slots']
    digest = protocol['data']['pool_images_sha256']
    arms = picked['arms']
    check_equal('mechanism list source', ('night_pool_e1_mechanism', seed, digest, {'0', '16', '8'}),
                (picked['purpose'], picked['seed'], picked['images_sha256'], set(arms)))
    manifest = read_sealed_json(train.NIGHT, POOL_MANIFEST, digest, [])
    by_key = {r['key']: r for r in _records(manifest, POOL, 994, images_only=True)}
    original = mechanism_rows(0, prior['d_input_slots'][:20], arms['0'], by_key)
    check_equal('mechanism first endpoint', ('reference', prior['selection_order'], prior['selection_sha256']),
                (arms['0']['status'], [r['key'] for r in original], canonical_hash(original)))
    days = _day_keys(identity['source_train'])
    tasks, statuses = {}, {}
    for q in mechanism['score_quotas']:
        entry = arms[str(q)]
        statuses[q] = {k: entry[k] for k in ('status', 'reason', 'duplicate_of') if k in entry}
        if entry['status'] == 'ready':
            tasks[q] = mechanism_slot_task(seed, q, entry, by_key, days, old, prior)
            statuses[q].update(schedule_sha256=tasks[q]['slots']['schedule_sha256'])
        elif entry['status'] not in ('duplicate', 'quota_failure'):
            raise ContractError('Unknown mechanism selection status')
    return tasks, statuses, protocol, identity


def e2_mechanism_slot_task(seed, representatives, entry, inventory, days, old, previous):
    from night_strategy import e2_mechanism_rows

    if seed != 43 or type(representatives) is not int or representatives not in (40, 20):
        raise ContractError('Expected registered E2 mechanism seed and new quota')
    check_equal('ready E2 mechanism arm', 'ready', entry['status'])
    check_equal('E2 mechanism rank parameters', {'representatives': representatives, 'diverse': 60-representatives, 'neighbors': 5}, entry['params'])
    rows = e2_mechanism_rows(representatives, previous['d_input_slots'], entry, inventory)
    task = _mechanism_slots(seed, 40, rows, entry, days, old, previous)
    task['slots']['representatives'] = representatives
    return task


def e2_mechanism_alignment(cfg, old, mechanism, protocol_sha, pool, pool_sha):
    from night_strategy import e2_mechanism_rows, e2_reference_receipt, e2_representative_prefix

    pool = Path(pool).absolute()
    if pool.parent != ROOT/'runs' or not re.fullmatch(r'pool_\d{3}', pool.name):
        raise ContractError('Expected E2 mechanism selection seal in runs/pool_NNN')
    seal = read_sealed_json(pool, 'manifest.json', pool_sha, [])
    check_equal('E2 mechanism list seal', ('sealed', 'E2 mechanism selection only', protocol_sha),
                (seal['status'], seal['scope'], seal['files']['e2.json']['sha256']))
    receipt, picked = (_json(pool, seal, f'{k}.json') for k in ('run', 'lists'))
    check_equal('E2 mechanism input receipt', ('sealed_e2_mechanism_selection', protocol_sha, True, ['features.npz'],
                                               cfg['pool_manifest_sha256']['e1']),
                (receipt['status'], receipt['protocol_sha256'], receipt['first_replay_exact'], receipt['worker_reads'],
                 receipt['pool_manifest_sha256']))
    if any(receipt[k] for k in ('labels_read', 'e1_scores_read', 'prediction_performed', 'training_performed')):
        raise ContractError('E2 mechanism selection violated feature-only payload isolation')
    seed = mechanism['seed']
    previous, protocol, identity = alignment(cfg, old, arms=('e2',), seeds=(seed,))
    _, _, _, prior = e2_reference_receipt(mechanism, [])
    frozen_d = previous['e2'][seed]['slots']
    check_equal('E2 reference exact observed D anchors and schedule', (frozen_d['d_input_slots'], frozen_d['d_schedule_sha256'],
                                                                    frozen_d['d_input_slots'][:40]),
                (prior['d_input_slots'], prior['d_schedule_sha256'], prior['anchors']))
    digest = protocol['data']['pool_images_sha256']
    arms = picked['arms']
    check_equal('E2 mechanism list source', ('night_pool_e2_mechanism', seed, digest, {'30', '40', '20'}),
                (picked['purpose'], picked['seed'], picked['images_sha256'], set(arms)))
    manifest = read_sealed_json(train.NIGHT, POOL_MANIFEST, digest, [])
    by_key = {r['key']: r for r in _records(manifest, POOL, 994, images_only=True)}
    original = e2_mechanism_rows(30, prior['d_input_slots'], arms['30'], by_key)
    check_equal('E2 mechanism 30/30 reference', ('reference', prior['selection_order'], prior['selection_sha256']),
                (arms['30']['status'], [r['key'] for r in original], canonical_hash(original)))
    days = _day_keys(identity['source_train'])
    tasks, statuses = {}, {}
    for n in mechanism['representatives']:
        entry = arms[str(n)]
        statuses[n] = {k: entry[k] for k in ('status', 'duplicate_of')}
        e2_representative_prefix(prior['selection_order'][40:70], entry['keys'][40:40+n])
        if entry['status'] == 'ready':
            tasks[n] = e2_mechanism_slot_task(seed, n, entry, by_key, days, old, prior)
            statuses[n].update(schedule_sha256=tasks[n]['slots']['schedule_sha256'])
        elif entry['status'] != 'duplicate':
            raise ContractError('Unknown E2 mechanism selection status')
    return tasks, statuses, protocol, identity


def granted_rows(box, arm, seed, digest):
    grant = read_sealed_json(box, 'grant.json', digest, [])
    check_equal('label grant fields', {'arm', 'seed', 'records'}, set(grant))
    if arm not in ARMS or seed not in SEEDS:
        raise ContractError('Unknown strategy arm or seed in label grant')
    check_equal('label grant identity', (arm, seed), (grant['arm'], grant['seed']))
    return _records({'purpose': POOL, 'records': grant['records']}, POOL, 100, images_only=True)


def _prepare(box, arm, seed, grant_sha, protocol):
    _no_reparse(box)
    check_equal('empty preparation sandbox except grant', ['grant.json'], sorted(p.name for p in box.iterdir()))
    locked = granted_rows(box, arm, seed, grant_sha)
    ready = read_sealed_json(train.NIGHT, 'READY.json', protocol['data']['night_ready_sha256'], [])
    entry = ready['held_out']['night_pool']
    check_equal('frozen night preparation entry', ('manifests/night_pool.json', 994), (entry['path'], entry['count']))
    managed = read_sealed_json(train.NIGHT, entry['path'], entry['sha256'], [])
    check_equal('managed night preparation namespace', ('night_pool', 994), (managed['purpose'], len(managed['records'])))
    rows = train.selected_records(locked, managed['records'])
    staged, reads = train._stage(train.NIGHT, box, rows)
    check_equal('granted image and label opens', [r[k] for r in rows for k in ('image', 'label')], reads)
    write_json(box/'inputs.json', {'arm': arm, 'seed': seed, 'records': staged, 'source_reads': reads,
                                  'grant_sha256': grant_sha, 'management_sha256': entry['sha256']})


def _reference(cfg, protocol, dev, night, seeds=SEEDS):
    folder = ROOT/'runs/accept_001'
    seal = read_sealed_json(folder, 'manifest.json', cfg['accept_manifest_sha256'], [])
    check_equal('five-class development seal', ('sealed', 'A/D five-class development replay'), (seal['status'], seal['scope']))
    policy, ref, metrics = (_json(folder, seal, f'{k}.json') for k in ('protocol', 'reference', 'metrics'))
    check_equal('fixed evaluation source identities', (protocol['source_a']['checkpoint_sha256'], cfg['d_manifest_sha256'],
                                                       dev['manifest']['sha256'], night['manifest']['sha256']),
                (policy['sealed']['checkpoint_a'], policy['sealed']['d_001'], policy['development']['source_dev'],
                 policy['development']['night_dev']))
    approved_night = {r['key']: r for r in night['records']}
    approved_day = {r['key']: r for r in dev['records']}
    base_d = ROOT/'runs/d_001'
    d_manifest = read_sealed_json(base_d, 'manifest.json', cfg['d_manifest_sha256'], [])
    anchors = {}
    for seed in seeds:
        frozen = metrics['D'][str(seed)]['night']
        rel = f'{seed}/dev/night/objects.jsonl.gz'
        check_equal('D paired prediction source', frozen['source']['objects_sha256'], d_manifest['files'][rel]['sha256'])
        rows = accept.sealed_objects(base_d, d_manifest, rel, 'night_dev', approved_night, [])
        anchors[seed] = rows, frozen['workpoints']
    base_a = ROOT/'runs/align_001'
    a_manifest = read_sealed_json(base_a, 'manifest.json', policy['sealed']['align_001'], [])
    a_day = accept.sealed_objects(base_a, a_manifest, '17/dev/objects.jsonl.gz', 'source_dev', approved_day, [])
    check_equal('A reference objects', ref['source']['objects_sha256'], a_manifest['files']['17/dev/objects.jsonl.gz']['sha256'])
    return policy, ref, metrics, anchors, a_day


def _evaluate(torch, weight, out, tmp, night, dev, policy, reference, d_pair, a_day, until):
    from evaluate import stage_subset
    from night_eval import _five
    from night_run import LOCKED_VIEWS, _capture, _collect
    from person_fusion import evaluate_rows

    out.mkdir()
    tmp.mkdir()
    with train.telemetry() as night_gpu:
        n_rows, night_views, night_caps = _collect(torch, night, weight, tmp/'night', 'night_dev')
    if time.monotonic() >= until:
        raise ContractError('Local arm exceeded its time budget after night evaluation')
    with train.telemetry() as day_gpu:
        box = tmp/'day'
        box.mkdir()
        data = stage_subset(dev, box/'data')
        captures, day_views = [], []
        for side, mirror in LOCKED_VIEWS:
            rows, view = _capture(torch, dev, data, weight, box, side, mirror)
            captures.append(rows)
            day_views.append(view)
            print(f'source_dev view {side} mirror={mirror}: {len(rows)} images in {view["seconds"]:.1f}s', flush=True)
            if time.monotonic() >= until:
                raise ContractError('Local arm exceeded its time budget during day evaluation')
        fused, day_caps = evaluate_rows(captures[0], captures[1:])
        d_rows = [_five(r) for r in fused]
    if accept._pair_identity(d_rows) != accept._pair_identity(a_day):
        raise ContractError('Day evaluation identity differs from fixed A')
    if accept._pair_identity(n_rows) != accept._pair_identity(d_pair[0]):
        raise ContractError('Night evaluation identity differs from paired D')
    results = {}
    for domain, rows, views, caps in (('day', d_rows, day_views, day_caps), ('night', n_rows, night_views, night_caps)):
        dest = out/domain
        dest.mkdir()
        objects = dest/'objects.jsonl.gz'
        dm.write_rows(objects, rows)
        check_equal('reloaded strategy prediction identity', canonical_hash(rows), canonical_hash(dm.read_rows(objects)))
        metrics = accept.evaluate_dev(rows, policy)
        metrics['capture'] = accept._audit_views(views, caps)
        write_json(dest/'metrics.json', metrics)
        results[domain] = metrics
    assessment = accept._assessment(reference, results['day'], results['night'], policy)
    write_json(out/'evaluation.json', {'night_views': night_views, 'night_fusion_caps': night_caps, 'night_gpu': night_gpu,
                                        'day_views': day_views, 'day_fusion_caps': day_caps, 'day_gpu': day_gpu,
                                        'assessment': assessment})
    return (d_pair[0], d_pair[1], n_rows, results['night']['workpoints']), assessment


def _comparison(pairs, assessments, d_metrics, policy):
    effects = accept.paired_recall(pairs, policy)
    directions = {}
    for name in (*accept.CLASSES, 'macro'):
        entries = [r[name]['delta'] for r in effects['per_seed'].values()]
        directions[name] = {'positive_seeds': sum(x is not None and x > 0 for x in entries),
                            'negative_seeds': sum(x is not None and x < 0 for x in entries)}
    class_passes = {n: sum(v['recovery'][n]['met'] is True for v in assessments.values()) for n in accept.CLASSES}
    comparison_review = any(x['review'] or d_metrics['D'][str(s)]['assessment']['review'] for s, x in assessments.items())
    macro_ci = effects['mean']['macro']['ci95']
    stable = (not comparison_review and directions['macro']['positive_seeds'] == len(SEEDS) and macro_ci is not None
              and macro_ci[0] > 0 and all(x['day_protection']['met'] is True for x in assessments.values()))
    return {'candidate_minus_d_night_recall': effects, 'directions': directions, 'recovery_points_met': class_passes,
            'all_stable_recovered': all(a['point_met'] is True and not a['review'] for a in assessments.values()),
            'comparison_needs_review': comparison_review, 'stable_overall_benefit': stable}


def _first_ratio_reference(arm, seed, ratio, a_sha, night):
    folder = ROOT/'runs'/f'{arm}_001'
    seal = read_sealed_json(folder, 'manifest.json', ratio['first_runs_sha256'][arm], [])
    receipt = _json(folder, seal, 'run.json')
    check_equal('frozen first-ratio model identity', ('sealed', 'complete', list(SEEDS), arm, a_sha, ratio['pair_sha256'], night['manifest']),
                (seal['status'], receipt['status'], receipt['completed_seeds'], receipt['arm'], receipt['a_sha256'],
                 receipt['protocol_sha256'], receipt['night_dev']))
    rel = f'{seed}/dev'
    night_rows = accept.sealed_objects(folder, seal, f'{rel}/night/objects.jsonl.gz', 'night_dev',
                                       {r['key']: r for r in night['records']}, [])
    night_metrics = _json(folder, seal, f'{rel}/night/metrics.json')
    assessment = _json(folder, seal, f'{rel}/evaluation.json')['assessment']
    return night_rows, night_metrics['workpoints'], assessment


def _first_e2_mechanism_reference(mechanism, a_sha, night):
    from night_strategy import e2_reference_receipt

    folder, seal, receipt, _ = e2_reference_receipt(mechanism, [])
    check_equal('E2 30/30 reference evaluation identity', (a_sha, night['manifest'], False),
                (receipt['a_sha256'], receipt['night_dev'], receipt['final_test_executed']))
    dev = '43/dev'
    rows = accept.sealed_objects(folder, seal, f'{dev}/night/objects.jsonl.gz', 'night_dev',
                                 {r['key']: r for r in night['records']}, [])
    points = _json(folder, seal, f'{dev}/night/metrics.json')['workpoints']
    return rows, points, _json(folder, seal, f'{dev}/evaluation.json')['assessment']


def _ratio_comparison(seed, pairs, assessments, baseline, first, policy):
    result = _comparison(pairs, assessments, baseline, policy)
    n_rows, n_points = pairs[seed][2:]
    first_rows, first_points, old = first
    result['candidate_minus_first_night_recall'] = accept.paired_recall({seed: (first_rows, first_points, n_rows, n_points)}, policy)
    current = assessments[seed]
    new, start, random = (a['day_protection'] for a in (current, old, baseline['D'][str(seed)]['assessment']))
    result['day_protection'] = {'candidate': new, 'first': start, 'd': random,
                                'candidate_minus_first': {'macro': new['macro_delta'] - start['macro_delta'],
                                    'per_class': {k: new['per_class'][k]['delta'] - start['per_class'][k]['delta']
                                                  for k in new['per_class']}}}
    result['night_recovery'] = current['recovery']
    result['night_false_positives'] = {name: {n: p[n]['fp'] for n in accept.CLASSES} for name, p in
                                       (('candidate', n_points), ('first', first_points), ('d', pairs[seed][1]))}
    gain = result['candidate_minus_d_night_recall']['mean']['macro']['delta']
    result['screen_point_eligible'] = new['met'] is True and gain is not None and gain > 0
    result['comparison_needs_review'] |= bool(old['review'])
    result['exploration_only'] = True
    return result


def run_arm(arm, out, scratch, raw, cfg, old, tasks, protocol, d_identity, protocol_sha, *, ratio=None, mechanism=None):
    from seed_eval import probe_gpu
    from night_run import CODE

    started = time.monotonic()
    until = started + 60*cfg['max_minutes_per_arm']
    if mechanism is not None and (ratio is not None or arm not in ARMS or mechanism['seed'] != 43):
        raise ContractError('Expected only an approved seed-43 mechanism study')
    study = mechanism if mechanism is not None else ratio
    check_equal('frozen A training bytes', d_identity['A_sha256'], sha256(train.CHECKPOINT))
    env = {k: version(k) for k in train.LIBS}
    check_equal('locked environment', d_identity['versions'], env)
    seeds = list(SEEDS) if study is None else [study['seed']]
    check_equal('locked NumPy schedule already replayed', seeds, list(tasks))
    gpu = probe_gpu()
    day = train.verify_subset(train.SOURCE, protocol['data']['source_ready_sha256'], 'source_train')
    dev = train.verify_subset(train.SOURCE, protocol['data']['source_ready_sha256'], 'source_dev')
    night = verify_dev(train.NIGHT, protocol['data']['night_ready_sha256'])
    for split, result in (('source_train', day), ('source_dev', dev), ('night_dev', night)):
        check_equal(f'D {split} identity', d_identity[split], result['manifest'])
    policy, reference, baseline, d_pairs, a_day = _reference(cfg, protocol, dev, night, seeds)
    first = None if study is None else (_first_e2_mechanism_reference(mechanism, d_identity['A_sha256'], night)
                                      if mechanism is not None and arm == 'e2' else
                                      _first_ratio_reference(arm, seeds[0], study, d_identity['A_sha256'], night))
    day_by = {r['key']: r for r in day['records']}
    dev_groups = {r['group_id'] for r in dev['records']+night['records']}
    for task in tasks.values():
        gradient_groups = {day_by[k]['group_id'] for (k, _, _), _ in task['plan']} | {r['group_id'] for r in task['rows']}
        check_equal('gradient and dev risk group separation', set(), gradient_groups & dev_groups)
    out.mkdir(exist_ok=False)
    code_names = ('night_adapt.py', 'night_accept.py', 'night_train.py', *CODE, 'source_policy.py', 'tricycle_fusion.py')
    if study is not None:
        code_names += ('night_strategy.py', 'night_select.py')
    code = {name: sha256(ROOT/'scripts'/name) for name in code_names}
    done, pairs, assessments = [], {}, {}
    failure = None
    identity = {'scope': 'S6 paired strategy', 'arm': arm, 'protocol_sha256': protocol_sha,
                'a_sha256': d_identity['A_sha256'], 'inputs': cfg, 'source_train': day['manifest'],
                'source_dev': dev['manifest'], 'night_dev': night['manifest'], 'python': sys.version,
                'numpy': np.__version__, 'versions': env, 'initial_gpu': gpu, 'cloud_used': False,
                'final_test_executed': False, 'code_sha256': code}
    if ratio is not None:
        identity.update(scope='S6 single-seed ratio exploration', ratio=ratio)
    if mechanism is not None:
        identity.update(scope=f'S6 {arm.upper()} mechanism exploration', mechanism=mechanism)
    prep_sha = protocol_sha if study is None else study['pair_sha256']
    try:
        (out/'protocol.json').write_bytes(raw)
        (out/'code/scripts').mkdir(parents=True)
        for name, digest in code.items():
            dest = out/'code/scripts'/name
            shutil.copyfile(ROOT/'scripts'/name, dest)
            check_equal('archived strategy code', digest, sha256(dest))
        write_json(out/'slots.json', {'arm': arm, 'slots_verified': True, 'seeds': {str(s): t['slots'] for s,t in tasks.items()}})
        with tempfile.TemporaryDirectory(prefix=arm+'-', dir=scratch) as td:
            tmp = Path(td)
            for key in ('YOLO_CONFIG_DIR', 'TORCH_HOME', 'MPLCONFIGDIR', 'TEMP', 'TMP', 'TMPDIR', 'XDG_CONFIG_HOME'):
                path = tmp/key.lower()
                path.mkdir()
                os.environ[key] = str(path)
            os.environ.update(YOLO_AUTOINSTALL='false', WANDB_MODE='disabled', CUBLAS_WORKSPACE_CONFIG=':4096:8')
            import cv2
            import torch
            import torchvision

            if not torch.cuda.is_available():
                raise ContractError('CUDA required; no CPU fallback')
            torch.set_num_threads(4)
            cv2.setNumThreads(1)
            identity['gpu'] = torch.cuda.get_device_name()
            for seed, task in tasks.items():
                if time.monotonic() >= until:
                    raise ContractError('Local arm exceeded its time budget before starting a seed')
                print(f'{arm} seed {seed}: staging only its 100 approved night image/label pairs', flush=True)
                plan = task['plan']
                box = tmp/str(seed)
                nt = box/'night-input'
                nt.mkdir(parents=True)
                grant = nt/'grant.json'
                write_json(grant, {'arm': arm, 'seed': seed, 'records': task['rows']})
                grant_sha = sha256(grant)
                subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), 'prepare', '--arm', arm,
                                '--seed', str(seed), '--scratch', str(nt), '--protocol-sha256', prep_sha,
                                '--grant-sha256', grant_sha], check=True, timeout=min(180, max(1, until-time.monotonic())))
                received = read_json(nt/'inputs.json')
                check_equal('narrow training receipt identity', (arm, seed, grant_sha, task['slots']['input_slots']),
                            (received['arm'], received['seed'], received['grant_sha256'], [r['key'] for r in received['records']]))
                phase = out/str(seed)
                phase.mkdir()
                stage, _ = train._stage(train.SOURCE, box/'day-input', [day_by[a[0]] for a, _ in plan])
                payload = {**train.training_rows(box/'day-input', stage, 'source_train'),
                           **train.training_rows(nt, received['records'], 'night_pool')}
                check_equal('exact gradient scope', (2100, {a[0] for a, _ in plan} | set(task['slots']['input_slots'])),
                            (len(payload), set(payload)))
                write_json(phase/'inputs.json', {**received, 'source_keys': [a[0] for a, _ in plan],
                                                   'schedule_sha256': task['slots']['schedule_sha256']})
                if seed == seeds[0]:
                    preview = box/'preview'
                    preview.mkdir()
                    with train.telemetry() as pre_gpu:
                        preview_weight, pre = train._fit(torch, cv2, payload, plan, seed, old, preview,
                                                         old['preflight_updates'], until, preview=True)
                    preview_weight.unlink()
                    check_equal('preflight aligned schedule prefix', canonical_hash(plan[:old['preflight_updates']*old['accumulate']]),
                                pre['augment_schedule_sha256'])
                    estimate = pre['seconds']*old['updates']/old['preflight_updates']*len(seeds) + 240*len(seeds)
                    write_json(out/'preflight.json', {'training': pre, 'gpu': pre_gpu, 'rough_total_seconds': estimate,
                                                     'status': 'passed', 'used_in_formal': False})
                    if estimate > until-time.monotonic():
                        raise ContractError('Measured local preflight projects past the 60-minute arm ceiling')
                    print(f'{arm} preflight passed: {pre["seconds"]:.1f}s; next: formal', flush=True)
                torch.cuda.empty_cache()
                probe_gpu()
                with train.telemetry() as training_gpu:
                    weight, receipt = train._fit(torch, cv2, payload, plan, seed, old, phase, old['updates'], until)
                check_equal('actual aligned schedule SHA-256', task['slots']['schedule_sha256'], receipt['augment_schedule_sha256'])
                write_json(phase/'train.json', dict(receipt, gpu=training_gpu, seed=seed, arm=arm,
                                                   checkpoint_roles={'last': 'last.pt', 'best': {'path': 'last.pt', 'selected_on_dev': False}}))
                del payload
                torch.cuda.empty_cache()
                torch.use_deterministic_algorithms(False)
                pairs[seed], assessments[seed] = _evaluate(torch, weight, phase/'dev', box/'eval', night, dev,
                                                          policy, reference, d_pairs[seed], a_day, until)
                done.append(seed)
                print(f'{arm} seed {seed} completed: {receipt["seconds"]:.1f}s training + development evaluation', flush=True)
            result = (_comparison(pairs, assessments, baseline, policy) if study is None else
                      _ratio_comparison(seeds[0], pairs, assessments, baseline, first, policy))
            write_json(out/'comparison.json', result)
            check_equal('frozen A unchanged', d_identity['A_sha256'], sha256(train.CHECKPOINT))
            check_equal('sealed A/D reference unchanged', cfg['accept_manifest_sha256'], sha256(ROOT/'runs/accept_001/manifest.json'))
    except BaseException as exc:
        failure = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        write_json(out/'run.json', dict(identity, status='complete' if not failure and done == seeds else 'incomplete',
                                       completed_seeds=done, error=failure, seconds=time.monotonic()-started))
        write_json(out/'manifest.json', {'status': 'sealed', 'files': train._receipt(out)})
    print(f'Sealed {out.name}; seeds {done} completed', flush=True)


def _runner_output(arm, arg):
    out = arg.absolute()
    if (out.parent != ROOT/'runs' or out.parent.resolve() != out.parent or out.exists() or out.is_symlink()
            or not re.fullmatch(rf'{arm}_\d{{3}}', out.name)):
        raise ContractError('Output must be a new arm-specific runs/e1_NNN or runs/e2_NNN')
    _no_reparse(out.parent)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('align', 'run', 'prepare', 'ratio-align', 'ratio', 'mechanism-align', 'mechanism',
                                         'e2-mechanism-align', 'e2-mechanism'))
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--scratch', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--grant-sha256')
    parser.add_argument('--pool-run', type=Path)
    parser.add_argument('--pool-sha256')
    parser.add_argument('--random-count', type=int, choices=(40, 60))
    parser.add_argument('--score-quota', type=int, choices=(16, 8))
    parser.add_argument('--representatives', type=int, choices=(40, 20))
    a = parser.parse_args(argv)
    if a.mode in ('mechanism-align', 'mechanism', 'e2-mechanism-align', 'e2-mechanism'):
        from night_strategy import e2_mechanism_config, mechanism_config

        e2 = a.mode.startswith('e2-')
        arm = 'e2' if e2 else 'e1'
        quota, other = (a.representatives, a.score_quota) if e2 else (a.score_quota, a.representatives)
        if a.pool_run is None or not a.pool_sha256 or any(x is not None for x in (a.arm, a.seed, a.grant_sha256, a.random_count, other)):
            parser.error(f'{arm} mechanism modes require the sealed selection and use only seed 43')
        aligning = a.mode.endswith('-align')
        if aligning and any(x is not None for x in (quota, a.output, a.scratch)):
            parser.error('mechanism alignment verifies the approved schedules without staging')
        if not aligning:
            if any(x is None for x in (quota, a.output, a.scratch)):
                parser.error(f'{arm} mechanism requires its quota, new output and external scratch')
            out, scratch = _runner_output(arm, a.output), a.scratch.absolute()
            if not scratch.is_dir() or scratch.resolve() != scratch or scratch.is_relative_to(ROOT):
                parser.error('scratch must be an existing clean external path')
            _no_reparse(scratch)
        raw, mechanism = (e2_mechanism_config if e2 else mechanism_config)(a.protocol_sha256, [])
        _, cfg, old = _config(mechanism['pair_sha256'])
        align = e2_mechanism_alignment if e2 else mechanism_alignment
        tasks, statuses, protocol, d_identity = align(cfg, old, mechanism, a.protocol_sha256, a.pool_run, a.pool_sha256)
        if aligning:
            print(json.dumps(statuses, indent=2), flush=True)
            return
        if quota not in tasks:
            raise ContractError(f'{arm} mechanism selection was not ready: {statuses[quota]}')
        quota_field = 'representative_count' if e2 else 'score_quota'
        mechanism = dict(mechanism, pool_run=a.pool_run.absolute().relative_to(ROOT).as_posix(),
                         pool_sha256=a.pool_sha256)
        mechanism[quota_field] = quota
        total, units = (60, 'representative/diverse') if e2 else (16, 'score/diversity')
        print(f'{arm.upper()} mechanism slots verified; start seed 43, {quota}/{total-quota} {units}.', flush=True)
        run_arm(arm, out, scratch, raw, cfg, old, {43: tasks[quota]}, protocol, d_identity, a.protocol_sha256, mechanism=mechanism)
        return
    if a.score_quota is not None or a.representatives is not None:
        parser.error('mechanism quotas apply only to their corresponding mechanism mode')
    if a.mode in ('ratio-align', 'ratio'):
        from night_strategy import ratio_config

        if a.pool_run is None or not a.pool_sha256 or a.seed is not None or a.grant_sha256:
            parser.error('ratio modes require the sealed pool and use only the registered seed')
        if a.mode == 'ratio-align' and any((a.arm, a.random_count, a.output, a.scratch)):
            parser.error('ratio-align verifies all four schedules without staging')
        if a.mode == 'ratio':
            if any(x is None for x in (a.arm, a.random_count, a.output, a.scratch)):
                parser.error('ratio requires an arm, random count, new output and external scratch')
            out, scratch = _runner_output(a.arm, a.output), a.scratch.absolute()
            if not scratch.is_dir() or scratch.resolve() != scratch or scratch.is_relative_to(ROOT):
                parser.error('scratch must be an existing clean external path')
            _no_reparse(scratch)
        raw, ratio = ratio_config(a.protocol_sha256, [])
        _, cfg, old = _config(ratio['pair_sha256'])
        tasks, protocol, d_identity = ratio_alignment(cfg, old, ratio, a.protocol_sha256, a.pool_run, a.pool_sha256)
        if a.mode == 'ratio-align':
            print(json.dumps({f'{arm}/{r}': {k: t['slots'][k] for k in ('common_images', 'schedule_sha256')}
                              for (arm,r), t in tasks.items()}, indent=2), flush=True)
            return
        ratio = dict(ratio, random_count=a.random_count, pool_run=a.pool_run.absolute().relative_to(ROOT).as_posix(),
                     pool_sha256=a.pool_sha256)
        print(f'All four ratio schedules verified; start {a.arm} seed {ratio["seed"]}, {a.random_count} random images.', flush=True)
        run_arm(a.arm, out, scratch, raw, cfg, old, {ratio['seed']: tasks[a.arm, a.random_count]},
                protocol, d_identity, a.protocol_sha256, ratio=ratio)
        return
    if any(x is not None for x in (a.pool_run, a.pool_sha256, a.random_count)):
        parser.error('pool and random-count options are for ratio modes only')
    raw, cfg, old = _config(a.protocol_sha256)
    if a.mode != 'align':
        if a.scratch is None or not a.scratch.is_dir():
            parser.error('expected existing external --scratch')
        a.scratch = a.scratch.absolute()
        if a.scratch.resolve() != a.scratch or a.scratch.is_relative_to(ROOT):
            parser.error('scratch must be clean and external')
        _no_reparse(a.scratch)
    if a.mode == 'prepare':
        if a.arm is None or a.seed is None or not a.grant_sha256 or a.output is not None:
            parser.error('prepare requires a single arm, seed and grant')
        protocol = train.load_protocol()
        check_equal('preparation legacy protocol SHA-256', old['evaluation_sha256'], protocol['sha256'])
        _prepare(a.scratch, a.arm, a.seed, a.grant_sha256, protocol)
        return
    if a.grant_sha256 or a.seed is not None or (a.mode == 'align' and any((a.arm, a.scratch, a.output))):
        parser.error('align accepts only the pair protocol SHA-256; grants are preparation-only')
    if a.mode == 'run':
        if a.arm is None or a.output is None:
            parser.error('run requires a single arm and new --output')
        out = _runner_output(a.arm, a.output)
    tasks, protocol, d_identity = alignment(cfg, old)
    if a.mode == 'align':
        print(json.dumps({arm: {s: {'common_images': t['slots']['common_images'],
                                    'schedule_sha256': t['slots']['schedule_sha256']} for s,t in values.items()}
                          for arm, values in tasks.items()}, indent=2), flush=True)
        return
    print(f'All six slot/schedule alignments verified against the actual D receipts. Starting {a.arm}.', flush=True)
    run_arm(a.arm, out, a.scratch, raw, cfg, old, tasks[a.arm], protocol, d_identity, a.protocol_sha256)


if __name__ == '__main__':
    main()
