"""Verify sealed S6 development evidence and aggregate only identical recipes."""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import json
import math
from pathlib import Path
import re

import detection_metrics as dm
import night_accept as accept
import night_adapt as adapt
import night_train as train
from night_contract import ROOT, _no_reparse, read_sealed_json
from source_contract import ContractError, canonical_hash, check_equal, read_json, sha256


def recipe_identity(cfg):
    return canonical_hash({k: v for k, v in cfg.items() if k not in ('seeds', 'max_minutes')})


def seed_assessments(runs):
    combined = {}
    for run in runs:
        if set(run) & set(combined):
            raise ContractError('Duplicate seed cannot count as independent recipe evidence')
        combined.update(run)
    if not combined or not set(combined) <= set(accept.SEEDS):
        raise ContractError('Unexpected S6 seeds')
    all_three = set(combined) == set(accept.SEEDS)
    return {'seeds': sorted(combined), 'all_three_point_pass': all_three and all(a['point_met'] is True for a in combined.values()),
            'all_stable_recovered': all_three and all(a['point_met'] is True and not a['review'] for a in combined.values())}


def verify(folder):
    folder = Path(folder).absolute()
    if folder.parent != ROOT/'runs' or not re.fullmatch(r'(rank|replay|tail|repeat)_\d{3}', folder.name):
        raise ContractError('Expected a released runs/rank_NNN, replay_NNN, tail_NNN or repeat_NNN folder')
    _no_reparse(folder)
    seal_path = folder/'manifest.json'
    seal_sha = sha256(seal_path)
    seal = read_sealed_json(folder, 'manifest.json', seal_sha, [])
    check_equal('published exact files', ('sealed', seal['files']), (seal['status'], train._receipt(folder)))
    cfg = read_json(folder/'protocol.json')
    run = read_json(folder/'run.json')
    kind = folder.name.split('_')[0]
    source = f'night_{kind}.py'
    check_equal('complete local run', ('complete', cfg['seeds'], False, False, source),
                (run['status'], run['completed_seeds'], run['cloud_used'], run['final_test_executed'], run.get('entry', 'night_rank.py')))
    if 'config_sha256' in run:
        check_equal('actual recipe bytes', run['config_sha256'], sha256(folder/'protocol.json'))
    _, pair, old = adapt._config(cfg['pair_sha256'])
    initial, protocol, identity = adapt.alignment(pair, old, arms=('e1',), seeds=cfg['seeds'])
    check_equal('frozen shared A', identity['A_sha256'], run['a_sha256'])
    a_root = ROOT/'runs/accept_001'
    common = read_sealed_json(a_root, 'manifest.json', pair['accept_manifest_sha256'], [])
    policy, reference, d_ref = (adapt._json(a_root, common, n+'.json') for n in ('protocol','reference','metrics'))
    paired_root = {'parent': ROOT/'runs/e1_001', 'd': ROOT/'runs/d_001'}
    paired_seal = {key: read_sealed_json(p, 'manifest.json', sha, []) for key,p,sha in
                   [(k, paired_root[k], cfg['parent_manifest_sha256'] if k=='parent' else pair['d_manifest_sha256'])
                    for k in paired_root]}
    approved = {}
    for purpose, (dv, n) in accept.DEV.items():
        receipt = run[purpose]
        digest = policy['development'][purpose]
        rel = f'manifests/{purpose}.json'
        check_equal('sealed only-development receipt', (n, rel, digest), (receipt['count'], receipt['path'], receipt['sha256']))
        records = read_sealed_json(ROOT/'data/soda10m/splits'/dv, rel, digest, [])
        check_equal('development coverage', (purpose, n), (records['purpose'], len(records['records'])))
        approved[purpose] = accept._index(records['records'], purpose)

    summaries, assessments = {}, {}
    comparisons = {k: {} for k in paired_root}
    saved = read_json(folder/'comparison.json')
    check_equal('random control attribution', kind=='tail', saved['D_is_same_training_mechanism'])
    for seed in cfg['seeds']:
        phase = folder/str(seed)
        task = initial['e1'][seed]
        original = task['plan']
        if kind == 'tail':
            selected = read_json(folder/'selection.json')
            new = selected['lists'][str(seed)]
            check_equal('blind selection worker', (False, False, True, ['features.npz','scores.npz','metadata.json']),
                        (selected['selection_labels_read'], selected['selection_development_read'],
                         new['original_e1_replayed'], new['worker_reads']))
            check_equal('selected image-only record digest', new['records_sha256'], canonical_hash(new['records']))
            task = adapt.ratio_slot_task(seed, 20, adapt._day_keys(identity['source_train']), new['records'], old, task['slots'])
            check_equal('actual tail slots', selected['slots'][str(seed)], task['slots'])
        inputs = read_json(phase/'inputs.json')
        verified_night = [r['key'] for r in inputs['records']]
        check_equal('locked source and night gradient inputs',
                    ([x[0][0] for x in original[:cfg['day_images']]], task['slots']['input_slots']),
                    (inputs['source_keys'], verified_night))
        fitted = read_json(phase/'train.json')
        check_equal('actual and published fixed checkpoint', (cfg['updates'], seed, seal['files'][f'{seed}/last.pt']['sha256'], False),
                    (fitted['updates'], fitted['seed'], fitted['last_sha256'], fitted['checkpoint_roles']['best']['selected_on_dev']))
        if kind != 'rank':
            plan = task['plan']
            if kind == 'replay':
                from night_replay import day_replay_plan
                check_equal('unmodified parent schedule', canonical_hash(plan), fitted['original_schedule_sha256'])
                plan = day_replay_plan(plan, old['accumulate'], cfg['dark_multiplier'])
                check_equal('registered low light update accounting', (1000, 5),
                            (fitted['synthetic_source_exposures'], fitted['synthetic_source_per_update']))
            elif kind == 'repeat':
                from night_repeat import reweight_schedule
                check_equal('original repeat plan', canonical_hash(plan), fitted['original_schedule_sha256'])
                plan, weights = reweight_schedule(plan,{r['key']:r['classes'] for r in inputs['records']},seed,
                                                  cfg['night_weights'],cfg['salt'])
                check_equal('verified selected-label weights', weights, fitted['night_sampling_weights'])
            check_equal('exact gradients, fixed A and no skipped updates',
                        (canonical_hash(plan), [[d[0], n[0]] for d,n in plan], identity['A_sha256'], 0,
                         dict(Counter(n[0] for _,n in plan))),
                        (fitted['augment_schedule_sha256'], fitted['microbatches'], fitted['initializer_sha256'],
                         fitted['skipped_updates'], fitted['counts']['night']))
        metrics, eval_rows = {}, {}
        receipt = read_json(phase/'dev/evaluation.json')
        for domain, purpose in (('day','source_dev'), ('night','night_dev')):
            key = f'{seed}/dev/{domain}'
            rows = accept.sealed_objects(folder, seal, f'{key}/objects.jsonl.gz', purpose, approved[purpose], [])
            metric = accept.evaluate_dev(rows, policy)
            metric['capture'] = accept._audit_views(receipt[f'{domain}_views'], receipt[f'{domain}_fusion_caps'])
            check_equal('replayed complete development metric', read_json(folder/f'{key}/metrics.json'), metric)
            eval_rows[domain], metrics[domain] = rows, metric
        assessment = accept._assessment(reference, metrics['day'], metrics['night'], policy)
        check_equal('replayed five-class protection/recovery', (assessment, assessment),
                    (receipt['assessment'], saved['assessments'][str(seed)]))
        assessments[seed] = assessment
        for name, p in paired_root.items():
            rel = f'{seed}/dev/night'
            rows = accept.sealed_objects(p, paired_seal[name], f'{rel}/objects.jsonl.gz', 'night_dev', approved['night_dev'], [])
            points = (adapt._json(p, paired_seal[name], f'{rel}/metrics.json')['workpoints'] if name=='parent'
                      else d_ref['D'][str(seed)]['night']['workpoints'])
            comparisons[name][seed] = (rows, points, eval_rows['night'], metrics['night']['workpoints'])
        points = metrics['night']['workpoints']
        fppi = Fraction(policy['recovery_ratio'])
        summaries[seed] = {'classes_met': assessment['classes_met'], 'point_met':assessment['point_met'],
                           'day_protection': assessment['day_protection'], 'review':assessment['review'],
                           'night': {k:{'actual_tp': p['tp'], 'actual_fp': p['fp'], 'recall':p['recall'],
                                       'ci95':p['interval']['ci95'], 'target_tp':math.ceil(fppi*accept._recall(reference['workpoints'][k])*(p['tp']+p['fn'])),
                                       'review':assessment['recovery'][k]['review']}
                                     for k,p in points.items()},
                           'truncated': {d:len(set().union(set(receipt[f'{d}_fusion_caps']),
                                         *(set(v['nms'][k]) for v in receipt[f'{d}_views']
                                           for k in ('pre_nms_truncated','max_det_truncated')))) for d in metrics}}
    final = {name:accept.paired_recall(p, policy) for name,p in comparisons.items()}
    check_equal('replayed risk group comparisons', (saved['candidate_minus_parent'], saved['candidate_minus_historical_d']),
                (final['parent'],final['d']))
    code = {k:v['sha256'] for k,v in seal['files'].items() if k.startswith('code/') and k.endswith('.py')}
    run_summary = {'run':folder.name, 'manifest_sha256':seal_sha, 'files_verified':len(seal['files']),
                   'training_recipe_sha256':recipe_identity(cfg), 'seconds':run['seconds'], 'per_seed':summaries}
    return run_summary, assessments, comparisons, policy, (source, recipe_identity(cfg), code, run['versions'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path, nargs='+')
    args = p.parse_args()
    results = [verify(folder) for folder in args.run]
    head = results[0]
    for result in results[1:]:
        check_equal('distinct seeds need identical recipe, code, environment and policy', (head[4],head[3]), (result[4],result[3]))
    aggregate = seed_assessments([result[1] for result in results])
    effects = {k: accept.paired_recall({s:pair for result in results for s,pair in result[2][k].items()}, head[3])
               for k in ('parent','d')}
    print(json.dumps({'runs': [r[0] for r in results], **aggregate, 'paired_effects': effects}, allow_nan=False))


if __name__ == '__main__':
    main()
