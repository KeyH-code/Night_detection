"""U compatibility identities and contrasts; synthetic records are never formal results."""
import copy
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from source_contract import ContractError, canonical_hash, read_json, sha256, write_json
import source_control as control
from train_source import config_at

ROOT = Path(__file__).resolve().parents[2]


def profile():
    return read_json(ROOT/control.PROFILE)


@pytest.fixture
def build(tmp_path):
    p = profile()
    for rel in [*p['new_files'], control.PROFILE]:
        dest = tmp_path/rel; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/rel, dest)
    return tmp_path


def u_identity():
    """Fabricated identity only; no U training/evaluation or checkpoint is produced."""
    p = profile()
    config = config_at(ROOT/'configs/source_s3.yaml', 'U', 17)
    runtime = dict(schema_version=3, experiment='source_s3', source_ready_sha256=p['source_ready_sha256'],
        pretrained_sha256=p['pretrained_sha256'], arm='U', seed=17, training_recipe_changed=True, frozen_data_changed=False,
        config_file_sha256=p['new_files']['configs/source_s3.yaml'], resolved_config_sha256=p['U_config_sha256'],
        code_identity={k: p['new_files'][k] for k in p['old_files']}, **copy.deepcopy(p['runtime_environment']))
    runtime['control_profile_sha256'] = sha256(ROOT/control.PROFILE)
    runtime['code_identity'].update({'scripts/source_control.py': p['new_files']['scripts/source_control.py'],
                                    control.PROFILE: runtime['control_profile_sha256']})
    runtime['identity_sha256'] = canonical_hash(runtime)
    ident = copy.deepcopy(p['common_identity'])
    ident['s3_policy']['plan_sha256'] = p['new_files']['specs/s3/plan.md']
    ident.update(code={k: p['new_files'][k] for k in p['core_files']}, config_sha256=canonical_hash(config),
                 runtime_identity_sha256=runtime['identity_sha256'])
    return {'identity': ident}, config, runtime, p


def test_u_effective_config_is_only_sampling_change_and_rejects_unregistered_seed():
    r = config_at(ROOT/'configs/source_s3.yaml', 'R', 17)
    u = config_at(ROOT/'configs/source_s3.yaml', 'U', 17)
    assert control.normalized_config(u) == r
    assert canonical_hash(r) == control.REFERENCE_CONFIG
    assert u['sampling']['indices_rng_scope'] == 'S'
    with pytest.raises(ContractError): config_at(ROOT/'configs/source_s3.yaml', 'U', 29)
    with pytest.raises(ContractError): config_at(ROOT/'configs/source_a.yaml', 'U', 17)


def test_exact_build_relocation_and_unregistered_code_bytes_fail(build):
    control.validate_build(build)
    path = build/'scripts/source_sampling.py'
    path.write_text(path.read_text(encoding='utf-8')+'\n# drift\n', encoding='utf-8')
    with pytest.raises(ContractError, match='exact control build'): control.validate_build(build)


def test_profile_cannot_bless_a_changed_common_gradient_loop(build):
    path = build/'scripts/source_trainer.py'
    path.write_text('# a different training loop', encoding='utf-8')
    p = read_json(build/control.PROFILE)
    p['new_files']['scripts/source_trainer.py'] = sha256(path)
    write_json(build/control.PROFILE, p)
    with pytest.raises(ContractError, match='unchanged common'): control.validate_build(build)


@pytest.mark.parametrize('field', ['old_files', 'common_identity', 'runtime_environment', 'core_files'])
def test_baseline_profile_cannot_be_rewritten_to_bless_a_different_experiment(build, field):
    p = read_json(build/control.PROFILE)
    if field == 'core_files': p[field] = p[field][:-1]
    else: p[field] = {}
    write_json(build/control.PROFILE, p)
    with pytest.raises(ContractError, match='immutable baseline'): control.validate_build(build)


def test_positive_u_metadata_and_saved_profile_binding():
    m, cfg, runtime, p = u_identity()
    control.validate_run_identity(m, cfg, runtime, p, 'U')


def test_u_runtime_gate_binds_control_helpers_and_rejects_new_library_bytes(build, monkeypatch):
    import platform
    import subprocess
    import torch  # Import before mocking the OS identity; no forward/backward.
    import source_cloud_gate as gate
    data = build/'data/source_v1/READY.json';data.parent.mkdir(parents=True);data.write_bytes(b'identity-only fixture')
    weight = build/'weights/yolo11m.pt';weight.parent.mkdir();weight.write_bytes(b'identity-only fixture')
    original = gate.sha256
    monkeypatch.setattr(gate, 'sha256', lambda path: gate.READY if Path(path) == data else gate.PRETRAINED if Path(path) == weight else original(path))
    monkeypatch.setattr(platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(platform, 'machine', lambda: 'x86_64')
    original_output = subprocess.check_output
    monkeypatch.setattr(subprocess, 'check_output', lambda args, **kw: 'fixture-driver' if args[0] == 'nvidia-smi' else original_output(args, **kw))
    u = gate.runtime_identity(build, build/'configs/source_s3.yaml', 'U', 17)
    assert u['arm'] == 'U' and u['control_profile_sha256'] == sha256(build/control.PROFILE)
    assert u['code_identity']['scripts/source_control.py'] == sha256(build/'scripts/source_control.py')
    assert u['gate_numerics']['driver'] == 'fixture-driver'
    assert u['resolved_config_sha256'] == profile()['U_config_sha256']
    with pytest.raises(ContractError): gate.runtime_identity(build, build/'configs/source_s3.yaml', 'U', 29)
    active_sha = gate.sha256
    monkeypatch.setattr(gate, 'sha256', lambda path: 'drifted-library' if str(path).replace('\\','/').endswith('ultralytics/utils/metrics.py') else active_sha(path))
    with pytest.raises(ContractError, match='runtime library_code'):
        gate.runtime_identity(build, build/'configs/source_s3.yaml', 'U', 17)


def test_trajectory_uses_native_dev_payload_and_keeps_infeasible_recall_undefined(tmp_path):
    import csv
    from test_policy import good_record
    from source_policy import load_rules
    arms = {}
    for arm, delta in [('R',0),('U',.01),('S',.02)]:
        folder = tmp_path/arm;folder.mkdir()
        arms[arm] = {'folder': folder}
        with (folder/'dev_metrics.csv').open('w',encoding='utf-8',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=['epoch','pedestrian_ap50_95','pedestrian_ap75','pedestrian_recall'])
            writer.writeheader()
            for epoch in range(1,101):
                writer.writerow(dict(epoch=epoch,pedestrian_ap50_95=.4+delta,pedestrian_ap75=.3+delta,pedestrian_recall=.5+delta))
                record = good_record();record['slices'].append({'class':'pedestrian','slice':'small_thin_union','recall50':.2+delta})
                if epoch == 95 and arm == 'U': record['workpoints']['pedestrian']['origin'] = 'diagnostic_f1'
                write_json(folder/f'observations/epoch{epoch:03d}.json',{'epoch':epoch,'dev':record})
        write_json(folder/'final_eval/last_dev.json',record)
    out = control.trajectory_summary(arms, load_rules(ROOT/'RULES.md'))
    assert out['tracks']['U'][94]['pedestrian/recall'] is None
    assert out['window_differences']['U-R']['81-90']['pedestrian/ap75']['mean_delta'] == pytest.approx(.01)
    window = out['window_differences']['U-R']['91-100']['pedestrian/recall']
    assert window == {'mean_delta':None,'valid':9,'expected':10}


@pytest.mark.parametrize('issue', ['gradient', 'evaluation', 'data', 'library', 'core', 'plan', 'rng', 'seed', 'anchor'])
def test_control_metadata_rejects_drift_even_with_recomputed_self_hashes(issue):
    m, cfg, rt, p = u_identity()
    if issue == 'gradient': cfg['training']['lr0'] *= 2
    if issue == 'evaluation': cfg['monitor']['every'] = 2
    if issue == 'data': m['identity']['data_fingerprint'] = 'changed'
    if issue == 'library': rt['library_code']['ultralytics/utils/nms.py'] = 'changed'
    if issue == 'core': m['identity']['code']['scripts/source_trainer.py'] = 'changed'
    if issue == 'plan': m['identity']['s3_policy']['plan_sha256'] = 'changed'
    if issue == 'rng': cfg['sampling']['indices_rng_scope'] = 'U'
    if issue == 'seed': cfg['training']['seed'] = 29
    if issue == 'anchor': m['identity']['s3_policy']['references'] = {}
    m['identity']['config_sha256'] = canonical_hash(cfg)
    rt['resolved_config_sha256'] = canonical_hash(cfg)
    rt.pop('identity_sha256');rt['identity_sha256'] = canonical_hash(rt)
    m['identity']['runtime_identity_sha256'] = rt['identity_sha256']
    with pytest.raises(ContractError): control.validate_run_identity(m, cfg, rt, p, 'U')


def test_legacy_identity_cannot_be_presented_as_u():
    m, cfg, rt, p = u_identity();cfg['sampling']['arm'] = 'S'
    with pytest.raises(ContractError): control.validate_run_identity(m, cfg, rt, p, 'U')


def test_common_group_draws_and_contrast_signs_do_not_refit_thresholds():
    from test_policy import sample
    from source_policy import load_rules, summarize
    rules = load_rules(ROOT/'RULES.md')
    r = [sample('a', 'g1'), sample('b', 'g2')]
    s = copy.deepcopy(r);s[0]['pred']['conf'] = [.1]*6
    u = copy.deepcopy(r);u[1]['pred']['conf'] = [.2]*6
    points = summarize(r, rules)[0]['workpoints']
    results = control.control_effects({'R': r, 'S': s, 'U': u}, {a: points for a in ('R','S','U')}, rules, repeats=20)
    assert set(results) == {'S-U', 'U-R', 'S-R_check'}
    assert all(v['groups'] == ['g1','g2'] and v['seed'] == 20261026 for v in results.values())
    for metric, effect in results['S-R_check']['effects'].items():
        x, y, z = [results[k]['effects'][metric]['mean_delta'] for k in ('S-U', 'U-R', 'S-R_check')]
        if x is not None and y is not None and z is not None: assert x+y == pytest.approx(z)
    assert results['S-U']['threshold_policy'] == 'saved cloud dev thresholds'
    assert not results['S-U']['practical_benefit_excluded']


def test_preflight_is_not_formal_and_does_not_generate_output(tmp_path):
    write_json(tmp_path/'manifest.json', {'status': 'completed_preflight'})
    with pytest.raises(ContractError, match='completed formal'): control.verify_manifest(tmp_path)
    assert not (tmp_path/'control.json').exists()
