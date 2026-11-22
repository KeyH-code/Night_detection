"""Strict S3 U-control compatibility and offline comparisons; never trains or infers."""
from __future__ import annotations

import argparse
import ast
import copy
import csv
import difflib
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from source_contract import (ContractError, canonical_hash, check_equal, read_json,
                             safe_path, sha256, verify_source, write_json)

ROOT = Path(__file__).resolve().parents[1]
PROFILE = 'configs/source_s3_control.json'
BASELINES = {
    'R': ('a_002', '6256164eb144127028b714d5a43fef6e818f75cf4553f24b72e7f6712cdd960d'),
    'S': ('a_003', 'a2487cb58717d7ef7390d27092e2bf7ee72f0d5e863cd69fe406e6cf08e920b5'),
}
REFERENCE_CONFIG = 'bbbc53bd7835ed69a1f66d401f555815fff667636dfe9061d3c35829136dd2c9'
BASELINE_FIELDS = {
    'old_files': 'd1f12a98a094b8f4fdcd1563b6bdcbe4f9d444dbf92a2cd572c54bf1a00cf7bb',
    'common_identity': '4efb8ac879a5551b8060fe7a7bd0c40e6f1d5dbc10d853ad4f0459b62a4b6dfd',
    'runtime_environment': 'd2890c6809fb8123e4eaf53be7a1ba3a905d5029baf423195594afa6a2d5fba9',
    'core_files': '05297d01adf51cdeab6a342b7ba010fb42195367a09df182ef544756c6cc4485',
}
# These are the only legacy files permitted to differ. All exact new hashes are
# also pinned in PROFILE; this list is not permission to ignore arbitrary bytes.
ADDITIVE_FILES = {'scripts/source_sampling.py', 'scripts/train_source.py',
                  'scripts/source_cloud_gate.py', 'scripts/run_source_cloud_job.py',
                  'scripts/bundle_source.py', 'scripts/verify_source_bundle.py', 'specs/s3/plan.md'}


def normalized_config(config):
    value = copy.deepcopy(config)
    arm = value['sampling']['arm']
    if arm not in ('R', 'S', 'U') or value['training']['seed'] != 17:
        raise ContractError('Control comparison is registered for R/S/U seed17 only')
    if arm == 'U':
        check_equal('U effective probabilities', (1, 1, 'S'),
                    tuple(value['sampling'].get(k) for k in ('target_weight', 'other_weight', 'indices_rng_scope')))
        value['sampling'].pop('indices_rng_scope')
        value['sampling']['target_weight'] = 2
    elif 'indices_rng_scope' in value['sampling']:
        raise ContractError('Legacy R/S RNG namespace changed')
    value['sampling']['arm'] = 'R'
    return value


def validate_build(root=ROOT):
    """An exact, reviewed old/new build pair, not an allow-code-mismatch switch."""
    root = Path(root)
    p = read_json(root/PROFILE)
    check_equal('control profile version', 1, p['schema_version'])
    check_equal('registered baseline manifests', {a: h for a, (_, h) in BASELINES.items()}, p['baseline_manifests'])
    check_equal('registered reference config', REFERENCE_CONFIG, p['reference_config_sha256'])
    for key, expected in BASELINE_FIELDS.items():
        check_equal('immutable baseline '+key, expected, canonical_hash(p[key]))
    check_equal('control semantic differences', sorted(ADDITIVE_FILES), p['allowed_legacy_changes'])
    check_equal('complete new file set', set(p['old_files']) | {'scripts/source_control.py', 'configs/source_s3.yaml'}, set(p['new_files']))
    for rel, old in p['old_files'].items():
        if rel not in ADDITIVE_FILES:
            check_equal('unchanged common implementation '+rel, old, p['new_files'][rel])
    for rel, expected in p['new_files'].items():
        check_equal('exact control build '+rel, expected, sha256(safe_path(root, rel)))
    from train_source import config_at
    r = config_at(root/'configs/source_s3.yaml', 'R', 17)
    u = config_at(root/'configs/source_s3.yaml', 'U', 17)
    check_equal('original resolved R recipe', REFERENCE_CONFIG, canonical_hash(r))
    check_equal('U changes sampling only', r, normalized_config(u))
    check_equal('exact resolved U recipe', p['U_config_sha256'], canonical_hash(u))
    return p


def validate_u_runtime(runtime, profile):
    check_equal('U runtime schema', (3, 'source_s3', True, False),
                tuple(runtime[k] for k in ('schema_version', 'experiment', 'training_recipe_changed', 'frozen_data_changed')))
    check_equal('U runtime data', profile['source_ready_sha256'], runtime['source_ready_sha256'])
    check_equal('U runtime pretrained', profile['pretrained_sha256'], runtime['pretrained_sha256'])
    for key in ('platform', 'machine', 'packages', 'library_code'):
        check_equal('control runtime '+key, profile['runtime_environment'][key], runtime[key])
    check_equal('U runtime arm/seed', ('U', 17), (runtime['arm'], runtime['seed']))
    check_equal('U runtime resolved config', profile['U_config_sha256'], runtime['resolved_config_sha256'])
    expected = {k: profile['new_files'][k] for k in profile['old_files']}
    expected['scripts/source_control.py'] = profile['new_files']['scripts/source_control.py']
    # The profile is separately bound in the runtime identity (no self-hash cycle).
    expected[PROFILE] = runtime['control_profile_sha256']
    check_equal('U full runtime code', expected, runtime['code_identity'])
    check_equal('U config file bytes', profile['new_files']['configs/source_s3.yaml'], runtime['config_file_sha256'])


def verify_manifest(folder, expected_hash=None):
    folder = Path(folder)
    if expected_hash is not None:
        check_equal('original manifest bytes', expected_hash, sha256(folder/'manifest.json'))
    m = read_json(folder/'manifest.json')
    check_equal('control requires completed formal run', 'completed_formal', m['status'])
    for rel, item in m['files'].items():
        path = safe_path(folder, rel)
        check_equal('sealed evidence '+rel, item['sha256'], sha256(path))
        check_equal('sealed evidence size '+rel, item['bytes'], path.stat().st_size)
    return m


def legacy_ast_check(root, folder):
    """All pre-existing functions/classes outside the additive edits are identical."""
    report = {}
    for name, allowed in [('source_sampling.py', {'indices_for', 'FiniteEpochLoader.finish'}),
                          ('train_source.py', {'config_at', 'run_segment', '_main'})]:
        old = folder/'code/scripts'/name
        new = root/'scripts'/name
        a, b = [ast.parse(p.read_text(encoding='utf-8')) for p in (old, new)]
        def definitions(tree):
            result = {}
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    result[node.name] = ast.dump(node, include_attributes=False)
                elif isinstance(node, ast.ClassDef):
                    for child in node.body:
                        if isinstance(child, ast.FunctionDef):
                            result[node.name+'.'+child.name] = ast.dump(child, include_attributes=False)
            return result
        before, after = definitions(a), definitions(b)
        check_equal('public definitions '+name, set(before), set(after))
        changed = [k for k in before if before[k] != after[k]]
        if set(changed)-allowed:
            raise ContractError('Unregistered common function changes: '+str(changed))
        report[name] = {'changed_definitions': changed, 'unchanged_definitions': sorted(set(before)-set(changed)),
                        'old_sha256': sha256(old), 'new_sha256': sha256(new),
                        'diff': ''.join(difflib.unified_diff(old.read_text(encoding='utf-8').splitlines(True),
                            new.read_text(encoding='utf-8').splitlines(True), fromfile='sealed/'+name, tofile='active/'+name))}
    return report


def verify_local(root=ROOT):
    """Rebuild train membership and compare every saved R/S consumed index, read-only."""
    import numpy as np
    from source_sampling import training_index, indices_for
    root = Path(root); profile = validate_build(root)
    data = root/'data/soda10m/splits/source_v1'
    contract = verify_source(data, profile['source_ready_sha256'])
    index = training_index(contract)
    check_equal('full train membership', (4000, 922, 1617),
                (len(index), sum(r['target'] for r in index), sum(r['target_gt'] for r in index)))
    arms, implementation = {}, None
    for arm, (name, h) in BASELINES.items():
        folder = root/'runs'/name/'train'
        m = verify_manifest(folder, h)
        check_equal('baseline resolved recipe', REFERENCE_CONFIG, canonical_hash(normalized_config(read_json(folder/'audit_initial/config_resolved.json'))))
        checked = []
        for epoch in range(100):
            rel = f'sampling/epoch{epoch+1:03d}'
            meta = read_json(folder/(rel+'.json'))
            check_equal('source train index identity', meta['index_sha256'], canonical_hash(index))
            with np.load(folder/(rel+'.npz'), allow_pickle=False) as stored:
                if not np.array_equal(indices_for(index, 17, arm, epoch), stored['consumed']):
                    raise ContractError(f'Legacy {arm} epoch{epoch+1} index drift')
            checked.append(m['files'][rel+'.npz']['sha256'])
        review = legacy_ast_check(root, folder)
        if implementation is None:
            implementation = review
        else:
            check_equal('same sealed R/S common implementation', implementation, review)
        arms[arm] = {'run': name, 'manifest_sha256': h, 'files_verified': len(m['files']),
                     'epochs_exact': 100, 'consumed_npz_sha256': checked,
                     'code_review_sha256': canonical_hash(review)}
    # Gate/packaging snapshots are not in the legacy core training snapshot.
    # Read their exact, hash-checked Git version without executing historical code.
    import subprocess
    diffs = {}
    for rel in sorted(ADDITIVE_FILES):
        snapshot = root/'runs/a_002/train/code'/rel
        before = snapshot.read_bytes() if snapshot.exists() else subprocess.check_output(
            ['git', 'show', profile['baseline_git_commit']+':'+rel], cwd=root)
        # Git stores LF while one sealed Windows-origin helper used CRLF. Only
        # accept a reconstructed byte stream if it equals the exact sealed SHA.
        if hashlib.sha256(before).hexdigest() != profile['old_files'][rel]:
            crlf = before.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
            if hashlib.sha256(crlf).hexdigest() == profile['old_files'][rel]:
                before = crlf
        check_equal('original diff source '+rel, profile['old_files'][rel], hashlib.sha256(before).hexdigest())
        after = (root/rel).read_text(encoding='utf-8')
        diffs[rel] = ''.join(difflib.unified_diff([s+'\n' for s in before.decode('utf-8').splitlines()],
                            [s+'\n' for s in after.splitlines()], fromfile='sealed/'+rel, tofile='active/'+rel))
    return {'schema_version': 1, 'status': 'local_compatibility_verified_cloud_pending',
            'checked_utc': datetime.now(timezone.utc).isoformat(), 'control_profile_sha256': sha256(root/PROFILE),
            'source_ready_sha256': contract['ready_sha256'], 'training_index_sha256': canonical_hash(index),
            'baselines': arms, 'implementation_review': implementation, 'exact_source_diffs': diffs,
            'new_files': profile['new_files'], 'U_run_exists': (root/'runs/a_008').exists(),
            'cloud_verified': False, 'new_training': False, 'new_inference': False,
            'limitations': ['No new U model or Linux/4090 preflight exists.',
                            'Legacy records do not contain driver/cuDNN/backend-flag snapshots; exact numerical environment equivalence is not established.']}


def validate_run_identity(manifest, config, runtime, profile, arm):
    identity = manifest['identity']
    check_equal('control arm/seed', (arm, 17), (config['sampling']['arm'], config['training']['seed']))
    check_equal('control effective common recipe', REFERENCE_CONFIG, canonical_hash(normalized_config(config)))
    check_equal('control config hash', canonical_hash(config), identity['config_sha256'])
    body = {k: v for k, v in runtime.items() if k != 'identity_sha256'}
    check_equal('runtime self identity', canonical_hash(body), runtime['identity_sha256'])
    check_equal('embedded runtime identity', runtime['identity_sha256'], identity['runtime_identity_sha256'])
    check_equal('control READY', profile['source_ready_sha256'], identity['ready_sha256'])
    check_equal('control initialization', profile['pretrained_sha256'], identity['pretrained_sha256'])
    files = profile['new_files'] if arm == 'U' else profile['old_files']
    check_equal('exact core code keys', set(profile['core_files']), set(identity['code']))
    check_equal('exact core code', {k: files[k] for k in profile['core_files']}, identity['code'])
    if arm == 'U':
        validate_u_runtime(runtime, profile)
    else:
        check_equal('original runtime hash', profile['baseline_runtime'][arm], runtime['identity_sha256'])
    policy = copy.deepcopy(identity)
    for k in ('code', 'config_sha256', 'runtime_identity_sha256'):
        policy.pop(k)
    if arm == 'U':
        check_equal('U approved plan snapshot', profile['new_files']['specs/s3/plan.md'], policy['s3_policy']['plan_sha256'])
        policy['s3_policy']['plan_sha256'] = profile['old_files']['specs/s3/plan.md']
    check_equal('complete data/evaluation/policy identity', profile['common_identity'], policy)


def _load_arm(root, folder, profile, arm):
    from source_release import audit_training_roles
    from detection_metrics import read_rows
    folder = Path(folder)
    m = verify_manifest(folder, BASELINES[arm][1] if arm in BASELINES else None)
    cfg = read_json(folder/'audit_initial/config_resolved.json')
    gate = read_json(folder/'audit_initial/resource_gate.json')
    check_equal('passed cloud resource gate', 'passed', gate['status'])
    runtime = gate['runtime_identity']
    check_equal('source/manifest identity', m['identity'], read_json(folder/'source_run.json')['identity'])
    validate_run_identity(m, cfg, runtime, profile, arm)
    if arm == 'U':
        check_equal('U reviewed compatibility profile', sha256(root/PROFILE), runtime['control_profile_sha256'])
    roles = audit_training_roles(folder)
    check_equal('complete local final evaluation', 'complete_local', roles['evaluation_status'])
    final = read_json(folder/'final_eval/run.json')
    monitor = read_json(folder/'train_monitor.json')
    check_equal('monitor identity', m['identity'], monitor['source_identity'])
    check_equal('control full update budget', (100, 25000, 6427, 0),
                tuple(monitor[k] for k in ('epoch', 'gradient_batches', 'optimizer_steps', 'amp_skipped_steps')))
    dev = read_json(folder/'final_eval/source_dev/metrics.json')
    rows = read_rows(folder/'final_eval/source_dev/objects.jsonl.gz')
    check_equal('full development rows', 500, len(rows))
    role = roles['selected_role']; selected = roles['roles'][role]
    check_equal('selected metrics weight', selected['sha256'], final['results']['source_dev']['checkpoint_sha256'])
    return {'manifest': m, 'config': cfg, 'rows': rows, 'dev': dev, 'roles': roles,
            'environment': final['local_environment'], 'folder': folder,
            'row': {'arm': arm, 'seed': 17, 'run': folder.parent.name, 'role': role,
                    'epoch': selected['epoch'], 'qualification': roles['qualification'],
                    'checkpoint_sha256': selected['sha256'], 'metrics': dev['metrics'],
                    'workpoints': dev['workpoints'], 'slices': dev['slices'],
                    'train': read_json(folder/'final_eval/source_train/metrics.json')}}


def trajectory_summary(arms, rules):
    """Cloud epoch summaries and local last are separate sources; no epoch bootstrap."""
    import numpy as np
    from source_contract import NAMES
    tracks, last, flat_last = {}, {}, {}
    for arm, data in arms.items():
        folder = data['folder']
        with (folder/'dev_metrics.csv').open(encoding='utf-8', newline='') as stream:
            history = list(csv.DictReader(stream))
        check_equal('complete ordered dev history', list(range(1, 101)), [int(r['epoch']) for r in history])
        slices, feasible = [], []
        for epoch in range(1, 101):
            entry = read_json(folder/f'observations/epoch{epoch:03d}.json')
            record = entry['dev']
            # Observations are the original per-epoch native summaries.
            target = next(s for s in record['slices'] if s['class'] == 'pedestrian' and s['slice'] == 'small_thin_union')
            wp = record['workpoints']['pedestrian']
            ok = wp['origin'] == 'rule' and wp['precision'] is not None and wp['precision'] >= rules['classes']['pedestrian']['precision']
            feasible.append(ok)
            slices.append(target['recall50'] if ok else None)
        tracks[arm] = [{'epoch': int(r['epoch']), 'pedestrian/ap50_95': float(r['pedestrian_ap50_95']),
                       'pedestrian/ap75': float(r['pedestrian_ap75']), 'pedestrian/recall': float(r['pedestrian_recall']) if feasible[i] else None,
                       'pedestrian/small_thin_union/recall': slices[i]} for i, r in enumerate(history)]
        last[arm] = read_json(folder/'final_eval/last_dev.json')
        values = {n+'/'+key: last[arm]['metrics']['per_class'][n][key]
                  for n in NAMES for key in ('ap50', 'ap75', 'ap50_95')}
        valid = {n: last[arm]['workpoints'][n]['origin'] == 'rule' and last[arm]['workpoints'][n]['precision'] is not None
                  and last[arm]['workpoints'][n]['precision'] >= rules['classes'][n]['precision'] for n in NAMES}
        values.update({n+'/recall': last[arm]['workpoints'][n]['recall'] if valid[n] else None for n in NAMES})
        values.update({'pedestrian/'+s['slice']+'/recall': s['recall50'] if valid['pedestrian'] else None for s in last[arm]['slices']
                       if s['class'] == 'pedestrian' and s['slice'] in ('area_S', 'width_le8', 'small_thin_union')})
        flat_last[arm] = values
    windows, deltas = {}, {}
    for label, ref, candidate in [('S-U', 'U', 'S'), ('U-R', 'R', 'U')]:
        deltas[label] = {k: flat_last[candidate][k]-v if v is not None and flat_last[candidate][k] is not None else None
                         for k, v in flat_last[ref].items()}
        windows[label] = {}
        for lo, hi in ((61, 80), (81, 90), (91, 100)):
            window = {}
            for k in tracks[ref][0]:
                if k == 'epoch': continue
                values = [tracks[candidate][i][k]-tracks[ref][i][k] for i in range(lo-1, hi)
                          if tracks[candidate][i][k] is not None and tracks[ref][i][k] is not None]
                window[k] = {'mean_delta': float(np.mean(values)) if len(values) == hi-lo+1 else None,
                             'valid': len(values), 'expected': hi-lo+1}
            windows[label][f'{lo}-{hi}'] = window
    return {'source': 'original cloud dev histories; local last100 separate', 'tracks': tracks,
            'window_differences': windows, 'last100': last, 'last100_differences': deltas}


def control_effects(rows, points, rules, repeats=2000):
    """Each contrast uses the same deterministic group draws in paired_effects."""
    from source_policy import paired_effects
    results = {}
    for label, reference, candidate in [('S-U', 'U', 'S'), ('U-R', 'R', 'U'), ('S-R_check', 'R', 'S')]:
        results[label] = paired_effects([(17, rows[reference], rows[candidate])], rules,
                    repeats=repeats, seed=20261026, points_by_seed={17: (points[reference], points[candidate])})
    for metric, original in results['S-R_check']['effects'].items():
        a, b, c = (results[k]['effects'][metric]['mean_delta'] for k in ('S-U', 'U-R', 'S-R_check'))
        if all(v is not None for v in (a, b, c)) and abs(a+b-c) > 1e-12:
            raise ContractError('Control contrast sign/identity failed: '+metric)
    return results


def compare_control(root, u_folder, output):
    """Requires a real sealed U; validation failures cannot create result directories."""
    from source_policy import load_rules
    root, output = Path(root), Path(output)
    profile = validate_build(root)
    rules = load_rules(root/'RULES.md')
    prior = read_json(output/'run.json')
    for rel, entry in prior['files'].items():
        check_equal('original pair evidence '+rel, entry['sha256'], sha256(safe_path(output, rel)))
    local = verify_local(root)
    arms = {a: _load_arm(root, root/'runs'/name/'train', profile, a) for a, (name, _) in BASELINES.items()}
    arms['U'] = _load_arm(root, u_folder, profile, 'U')
    for a in ('S', 'U'):
        check_equal('common local evaluation environment', arms['R']['environment'], arms[a]['environment'])
        check_equal('same images/GT/groups', [(r['key'], r['image_sha256'], r['group_id'], r['gt']) for r in arms['R']['rows']],
                    [(r['key'], r['image_sha256'], r['group_id'], r['gt']) for r in arms[a]['rows']])
    result = {'schema_version': 1, 'status': 'complete_exploratory_seed17_control',
              'arms': [arms[a]['row'] for a in ('R', 'U', 'S')],
              'contrasts': control_effects({a: v['rows'] for a, v in arms.items()},
                        {a: v['dev']['workpoints'] for a, v in arms.items()}, rules),
              'trajectory': trajectory_summary(arms, rules),
              'limitations': ['Post-hoc mechanism study, single training seed and reused source_dev.',
                              'Conditional 95% intervals have no familywise multiplicity adjustment.',
                              'Last100 metrics are auxiliary, not per-GT predictions or separately selected weights.']}
    # Refuse to silently replace a previous completed control experiment.
    if (output/'control.json').exists():
        raise ContractError('Control output already exists')
    write_json(output/'control.json', result)
    with (output/'control.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['contrast', 'metric', 'mean_delta', 'lower', 'upper', 'valid', 'undefined'])
        writer.writeheader()
        for contrast, data in result['contrasts'].items():
            for metric, value in data['effects'].items():
                writer.writerow({'contrast': contrast, 'metric': metric, **{k: value[k] for k in writer.fieldnames[2:]}})
    local.update(status='control_runs_verified', cloud_verified=True, numerical_equivalence_verified=False,
                 U_manifest_sha256=sha256(Path(u_folder)/'manifest.json'))
    write_json(output/'compatibility.json', local)
    for rel in ('control.json', 'control.csv', 'compatibility.json'):
        path = output/rel
        prior['files'][rel] = {'sha256': sha256(path), 'bytes': path.stat().st_size}
    prior['control_analysis'] = {'status': result['status'], 'control_profile_sha256': sha256(root/PROFILE),
                                 'U_manifest_sha256': local['U_manifest_sha256']}
    write_json(output/'run.json', prior)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['verify', 'compare'])
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--u-run', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    if args.mode == 'verify':
        result = verify_local(args.root)
        if args.output:
            write_json(args.output, result)
        print(result['status'])
    else:
        if not args.u_run or not args.output:
            p.error('compare requires --u-run and existing --output pair directory')
        print(compare_control(args.root, args.u_run, args.output)['status'])


if __name__ == '__main__':
    main()
