"""Single-factor E1 quota contrast; blind selection precedes all label staging."""
from __future__ import annotations

from collections import Counter
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

import night_adapt as adapt
import night_rank
import night_strategy as strategy
import night_train as train
from night_contract import POOL, POOL_MANIFEST, ROOT, _records, read_sealed_json
from source_contract import ContractError, canonical_hash, check_equal, read_json, sha256, write_json


def rank_quota(keys, vectors, scores, anchors, shortlist, quotas):
    order, ordered, cos, selected = strategy._geometry(keys, vectors, anchors)
    if (len(quotas) != 5 or type(shortlist) is not int or not all(type(x) is int and 0 < x <= shortlist for x in quotas)
            or len(keys)-len(anchors) < sum(quotas)):
        raise ContractError('Invalid complete E1 quota budget')
    u = strategy._scores(keys, scores)[order]
    chosen = set(selected)
    pool = set(range(len(ordered))) - chosen
    rounded = np.round(u, 12)
    lists = [set(sorted((i for i in pool if u[i, c] > 0), key=lambda i: (-rounded[i,c], ordered[i]))[:shortlist])
             for c in range(5)]
    distance = 1-cos[:, selected].max(axis=1)
    steps, counts = [], [0]*5
    for round_index in range(max(quotas)):
        for c, n in enumerate(quotas):
            if round_index >= n:
                continue
            candidates = lists[c] - chosen
            if not candidates:
                raise ContractError(f'Tail class {c} shortage; no cross-class fallback')
            i = min(candidates, key=lambda j: (-round(float(distance[j]), 12), -rounded[j,c], ordered[j]))
            steps.append({'key': ordered[i], 'class_id': c, 'score': float(u[i,c]), 'distance': float(distance[i])})
            counts[c] += 1
            selected.append(i)
            chosen.add(i)
            np.minimum(distance, 1-cos[:,i], out=distance)
    return {'keys': [ordered[i] for i in selected], 'steps': steps, 'per_class': counts}


def selection_worker(box):
    features, scores, metadata = (box/f for f in ('features.npz', 'scores.npz', 'metadata.json'))
    check_equal('only blind rank inputs', {features, scores, metadata}, set(box.iterdir()))
    warm = io.BytesIO()
    np.savez_compressed(warm, value=np.ones(1))
    warm.seek(0)
    with np.load(warm, allow_pickle=False) as data:
        check_equal('synthetic ZIP warmup', [1.], data['value'].tolist())
    reads = strategy.worker_files(readable=[features, scores, metadata])
    keys, z = strategy.load_features(features)
    u = strategy._load_e1_scores(scores, keys)
    q = read_json(metadata)
    check_equal('blind worker payload fields', {'keys', 'anchors', 'quota', 'shortlist'}, set(q))
    check_equal('full image/feature key alignment', keys, q['keys'])
    check_equal('worker only read approved payloads', [features, scores, metadata], reads)
    base = rank_quota(keys, z, u, q['anchors'], q['shortlist'], [16]*5)
    new = rank_quota(keys, z, u, q['anchors'], q['shortlist'], q['quota'])
    print(json.dumps({'reference': base['keys'], 'new': new, 'reads': [p.name for p in reads]}), flush=True)


def validate(cfg):
    check_equal('registered tail treatment', (2000, 200, [12, 8, 8, 22, 30], 64, 40),
                tuple(cfg[k] for k in ('day_images', 'updates', 'quotas', 'shortlist', 'max_minutes')))


def select(cfg, scratch):
    _, pair, old = adapt._config(cfg['pair_sha256'])
    protocol = train.load_protocol()
    digest = protocol['data']['pool_images_sha256']
    inventory = read_sealed_json(train.NIGHT, POOL_MANIFEST, digest, [])
    by_key = {r['key']: r for r in _records(inventory, POOL, 994, images_only=True)}
    b = ROOT/'runs/pool_002'; d = ROOT/'runs/pool_001'
    b_seal, d_seal = (read_sealed_json(p, 'manifest.json', pair['pool_manifest_sha256'][k], [])
                      for p,k in ((b,'e1'), (d,'d')))
    check_equal('A parent capture and random lists sealed', ('sealed', 'sealed'), (b_seal['status'], d_seal['status']))
    check_equal('original random endpoint', old['random_sha256'], d_seal['files']['random.json']['sha256'])
    e1, random = (adapt._json(p, m, name) for p,m,name in ((b,b_seal,'e1.json'), (d,d_seal,'random.json')))
    result = {}
    with tempfile.TemporaryDirectory(dir=scratch, prefix='tail-blind-') as td:
        box = Path(td)
        for name in ('features.npz','scores.npz'):
            raw = night_rank.accept._checked_file(b, b_seal, name, [])
            (box/name).write_bytes(raw)
        for seed in cfg['seeds']:
            d_keys, first = ([q['key'] for q in adapt.draw_records(k, seed, draw, digest, by_key)]
                             for k,draw in (('d', random), ('e1',e1)))
            write_json(box/'metadata.json', {'keys': sorted(by_key), 'anchors': d_keys[:20],
                      'quota': cfg['quotas'], 'shortlist': cfg['shortlist']})
            command = [sys.executable, '-B', str(Path(__file__).resolve()), 'worker', str(box)]
            a, c = [json.loads(subprocess.check_output(command, timeout=180)) for _ in range(2)]
            check_equal('two blind replays', a, c)
            check_equal('ordered E1 equality', first, a['reference'])
            roll = a['new']; keys = roll['keys']
            check_equal('tail locked budget and anchors', (100, 100, d_keys[:20], cfg['quotas']),
                        (len(keys), len(set(keys)), keys[:20], roll['per_class']))
            if set(keys) in (set(first), set(d_keys)):
                raise ContractError('Tail quota reproduces an existing image set; no need to train')
            records = [by_key[k] for k in keys]
            result[seed] = {**roll, 'records': records, 'records_sha256': canonical_hash(records), 'worker_reads': a['reads'],
                            'original_e1_replayed': True, 'original_e1_overlap': len(set(first)&set(keys)),
                            'd_overlap':len(set(d_keys)&set(keys)), 'risk_groups':len({r['group_id'] for r in records})}
            print(f'Tail quota seed {seed}: exact original replay, new/E1 overlap={result[seed]["original_e1_overlap"]}, '
                  f'groups={result[seed]["risk_groups"]}, lists use no labels.', flush=True)
    return {'lists': result, 'images_sha256': digest, 'selection_labels_read': False,
            'selection_development_read': False, 'parent_manifest_sha256': pair['pool_manifest_sha256']['e1']}


def fit(torch, cv2, payloads, plan, seed, original, cfg, until, phase):
    check_equal('unchanged update budget and day order', (200, 2000, list(payloads['day'])),
                (original['updates'], len(plan), [a[0] for a,_ in plan]))
    weight, result = train._fit(torch, cv2, payloads['day'] | payloads['night'], plan, seed,
                                original, phase, cfg['updates'], until)
    check_equal('exact original augmentation schedule', canonical_hash(plan), result['augment_schedule_sha256'])
    images, boxes = Counter(), Counter()
    for _,_,_,labels in payloads['night'].values():
        c = [x[0] for x in labels]
        images.update(set(c)); boxes.update(c)
    return weight, dict(result, initializer_sha256=sha256(train.CHECKPOINT), night_images_by_class=images,
                        night_instances_by_class=boxes,
                        checkpoint_roles={'last':'last.pt', 'best':{'path':'last.pt','selected_on_dev':False}})


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == 'worker':
        selection_worker(Path(sys.argv[2]).absolute())
    else:
        night_rank.main(kind='tail', fit=fit, validate=validate, select=select, description=__doc__)
