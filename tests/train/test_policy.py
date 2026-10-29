"""RULES, qualification precedence, role ties, CI pruning and paired negative evidence."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
import detection_metrics as dm
from source_contract import ContractError, NAMES
from source_policy import (HorizonStopping, SelectionPolicy, load_rules, paired_effects,
                           recall_intervals, reference_at, summarize, compare_runs)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def policy():
    return SelectionPolicy(load_rules(ROOT/'RULES.md'), reference_at(ROOT))


def good_record():
    return {'metrics': {'primary_ap50_95': .8, 'per_class': {
        n: {'instances': 100, 'groups': 80, 'ap50': .96, 'ap75': .8, 'ap50_95': .8} for n in NAMES}},
        'workpoints': {n: {'origin': 'rule', 't_rule': .5, 'threshold': .5,
                          'precision': .95, 'recall': .9, 'tp': 90, 'fp': 4, 'fn': 10} for n in NAMES},
        'slices': [{'class': 'pedestrian', 'slice': tag, 'recall50': .8} for tag in ('area_S', 'width_le8')]}


def ci(wide=False):
    return {'repeats': 2000, 'seed': 20261026,
            'intervals': {n: {'lower': .1 if wide and n == 'tricycle' else .8, 'upper': .95,
                              'valid': 2000, 'exploratory': False} for n in NAMES}}


def test_rules_authority_protection_and_historical_slice_inputs(policy, tmp_path):
    assert policy.rules['ap_gain'] == .005 and policy.rules['recall_gain'] == .02
    assert policy.reference['slices']['width_le8'] == pytest.approx(.16923076923076924)
    path = tmp_path/'RULES.md'; path.write_text('No defaults permitted', encoding='utf-8')
    with pytest.raises(ContractError): load_rules(path)


@pytest.mark.parametrize('issue', ['auxiliary', 'no_feasible', 'no_prediction', 'loc', 'thin', 'missing', 'nan'])
def test_high_primary_score_never_overrides_a_failure(policy, issue):
    r = good_record()
    if issue == 'auxiliary': r['metrics']['per_class']['truck']['ap50_95'] = .60
    if issue == 'no_feasible': r['workpoints']['pedestrian'].update(origin='diagnostic_f1', t_rule=None)
    if issue == 'no_prediction': r['workpoints']['car'].update(tp=0, fp=0, precision=None)
    if issue == 'loc': r['metrics']['per_class']['pedestrian']['ap75'] = 0.
    if issue == 'thin': r['slices'][1]['recall50'] = 0.
    if issue == 'missing': r['metrics']['per_class']['tricycle']['instances'] = 0
    if issue == 'nan': r['metrics']['per_class']['pedestrian']['ap50_95'] = float('nan')
    if issue in ('missing', 'nan'):
        with pytest.raises(ContractError): policy.update(1, r, lambda: pytest.fail('CI on invalid data'))
    else:
        result = policy.update(1, r, lambda: pytest.fail('CI on numeric failure'))
        assert not result['new_best'] and result['new_diagnostic'] and result['failure_reasons']


def test_zero_eligible_ties_evidence_pruning_and_state_resume(policy):
    r = good_record()
    first = policy.update(1, r, lambda: ci(wide=True))
    assert first['numeric_pass'] and not first['new_best'] and first['new_diagnostic']
    assert policy.state['best'] is None
    second = policy.update(2, r, ci)
    assert second['new_best'] and not second['new_diagnostic']
    third = policy.update(3, r, lambda: pytest.fail('CI must be pruned on exact tie'))
    assert third['best']['epoch'] == 2 and third['diagnostic']['epoch'] == 1
    restored = SelectionPolicy(policy.rules, policy.reference)
    restored.state = copy.deepcopy(policy.state)
    r['metrics']['primary_ap50_95'] = .81
    r['metrics']['per_class']['pedestrian']['ap50_95'] = .82
    assert restored.update(4, r, ci) == policy.update(4, r, ci)


def test_meaningful_gain_accumulates_from_anchor_not_previous_epoch(policy):
    r = good_record()
    policy.update(1, r, ci)
    r['metrics']['per_class']['pedestrian']['ap50_95'] = .802
    r['metrics']['primary_ap50_95'] = .801
    assert not policy.update(2, r, ci)['meaningful_improvement']
    r['metrics']['per_class']['pedestrian']['ap50_95'] = .804
    r['metrics']['primary_ap50_95'] = .802
    assert not policy.update(3, r, ci)['meaningful_improvement']
    r['metrics']['per_class']['pedestrian']['ap50_95'] = .806
    r['metrics']['primary_ap50_95'] = .803
    assert policy.update(4, r, ci)['meaningful_improvement']
    r['metrics']['per_class']['pedestrian']['ap50_95'] = .9
    r['metrics']['primary_ap50_95'] = .85
    r['metrics']['per_class']['truck']['ap50_95'] = .1
    assert not policy.update(5, r, ci)['meaningful_improvement']
    assert policy.state['last_meaningful_epoch'] == 4


def test_fixed_horizon_is_not_legacy_patience_zero_or_stagnation():
    stopper = HorizonStopping(100)
    for epoch in range(1, 100):
        assert not stopper(epoch, 0.)
    assert stopper(100, 0.)
    with pytest.raises(ContractError): stopper(10, float('nan'))


def sample(key='one', group='g', predicted=True, wrong=False):
    boxes = [[i*20., 0., i*20.+5, 10.] for i in range(6)]
    row = {'key': key, 'purpose': 'source_dev', 'group_id': group, 'width': 640, 'height': 640,
           'gt': {'cls': list(range(6)), 'boxes': boxes, 'original_boxes': boxes},
           'pred': {'cls': list(range(6)) if predicted else [], 'conf': [.9]*6 if predicted else [],
                    'boxes': [[b+100 for b in box] for box in boxes] if wrong else boxes if predicted else []}}
    tp, pairs = dm.ap_flags(row); row.update(ap_tp=tp.tolist(), ap_pairs=pairs)
    return row


def test_empty_predictions_and_no_feasible_point_stay_explicit(policy):
    result, matches, _ = summarize([sample(predicted=False)], policy.rules)
    assert result['workpoints']['pedestrian']['status'] == 'no_predictions'
    assert result['metrics']['per_class']['pedestrian']['ap50_95'] == 0.
    wrong, _, _ = summarize([sample(wrong=True)], policy.rules)
    assert wrong['workpoints']['pedestrian']['t_rule'] is None
    assert wrong['workpoints']['pedestrian']['origin'] == 'diagnostic_f1'


def test_filter_then_rematch_and_train_uses_exact_dev_threshold(policy):
    rows = [sample()]
    dev, matches, _ = summarize(rows, policy.rules)
    train = copy.deepcopy(rows); train[0]['purpose'] = 'source_train'; train[0]['pred']['conf'] = [.89]*6
    result, _, _ = summarize(train, policy.rules, dev['workpoints'])
    assert result['workpoints']['pedestrian']['tp'] == 0
    assert result['workpoints']['pedestrian']['threshold'] == pytest.approx(.9)
    assert result['slices'][-1]['slice'] == 'small_thin_union'
    ci_result = recall_intervals(rows, matches)
    assert ci_result['intervals']['pedestrian']['lower'] == 1.


def test_declared_output_cap_records_actual_truncation_without_changing_metrics(policy):
    rows = [sample('a', 'g1'), sample('b', 'g2')]
    for row in rows:
        row['pred']['cls'] = row['pred']['cls'] + [0]*294
        row['pred']['conf'] = row['pred']['conf'] + [.01]*294
        row['pred']['boxes'] = row['pred']['boxes'] + [[1000., 1000., 1001., 1001.]]*294
        flags, pairs = dm.ap_flags(row);row.update(ap_tp=flags.tolist(), ap_pairs=pairs)
    baseline, matches, curves = summarize(rows, policy.rules)
    audited = copy.deepcopy(rows)
    for row,count in zip(audited, (300, 364)):
        row['nms_audit'] = dict(max_det=300, final_count=300, at_max_det=True,
            post_nms_count=count, max_det_truncated=count>300, pre_nms_truncated=False)
    result, after_matches, after_curves = summarize(audited, policy.rules)
    for field in ('metrics', 'workpoints', 'slices'):
        assert result[field] == baseline[field]
    assert after_matches == matches
    for key in curves: assert np.array_equal(curves[key], after_curves[key])
    assert result['nms']['images_at_cap'] == 2
    assert result['nms']['truncated_images'] == 1 and result['nms']['removed_boxes'] == 64
    assert result['nms']['max_post_nms_count'] == 364
    audited[0]['nms_audit']['pre_nms_truncated'] = True
    with pytest.raises(ContractError, match='NMS candidate truncation'):
        summarize(audited, policy.rules)


def test_paired_draws_include_empty_groups_and_do_not_refit_infeasible_thresholds(policy):
    ref = [sample('a', 'g1'), sample('b', 'g2')]
    empty = sample('empty', 'g3', predicted=False); empty['gt']['cls']=[]; empty['gt']['boxes']=[]; empty['gt']['original_boxes']=[]
    empty['ap_tp'], empty['ap_pairs'] = [], [[] for _ in range(10)]
    ref.append(empty)
    identical = paired_effects([(s, ref, copy.deepcopy(ref)) for s in (17,29,43)], policy.rules, repeats=40)
    assert identical['groups'] == ['g1','g2','g3']
    assert identical['effects']['pedestrian/ap50_95']['mean_delta'] == 0.
    assert identical['effects']['pedestrian/ap50_95']['lower'] == 0.
    assert identical['effects']['pedestrian/ap75']['lower'] == 0.
    candidate = [sample('a','g1',wrong=True), sample('b','g2',wrong=True), empty]
    result = paired_effects([(17, ref, candidate)], policy.rules, repeats=20)
    assert result['effects']['pedestrian/recall']['mean_delta'] is None
    assert result['effects']['pedestrian/recall']['undefined'] == 20
    assert not result['practical_benefit_excluded']
    with pytest.raises(ContractError): paired_effects([(29, ref, candidate)], policy.rules, repeats=10)


def test_paired_effects_preserve_saved_cloud_threshold_after_local_drift(policy):
    ref=[sample()]
    ref[0]['pred']['conf']=[.99]*6
    candidate=[sample()]
    saved=summarize(ref,policy.rules)[0]['workpoints']
    result=paired_effects([(17,ref,candidate)],policy.rules,repeats=10,
                         points_by_seed={17:(saved,saved)})
    assert result['threshold_policy']=='saved cloud dev thresholds'
    assert result['effects']['pedestrian/recall']['mean_delta'] is None
    assert result['effects']['pedestrian/recall']['undefined']==10


def test_comparison_cannot_treat_preflight_as_formal_or_create_empty_outputs(tmp_path):
    from source_contract import write_json
    run = tmp_path/'preflight'
    write_json(run/'manifest.json', {'status':'completed_preflight'})
    output = tmp_path/'comparison'
    with pytest.raises(ContractError, match='completed formal'):
        compare_runs([(17,run,run)], output, ROOT/'RULES.md')
    assert not output.exists()
