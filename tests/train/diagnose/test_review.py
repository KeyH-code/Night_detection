"""Audit coverage, identity boundaries and lossless external autosave."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'scripts'))
import diagnostic_review as review
from source_contract import ContractError, NAMES


def cases():
    rows = []
    for name in NAMES:
        for decision in review.DECISIONS:
            for n in range(12):
                rows.append({'case_id': f'{name}/{decision}/{n}', 'class': name, 'layers': [],
                             'status': 'pending', 'decision': None, 'reviewer': None,
                             'initial_review': {'decision': decision, 'confidence': 'medium',
                                                'note': 'visible evidence', 'model': 'gpt-6-luna'}})
    rows[0]['layers'] = ['tricycle_all_gt']
    rows[1]['layers'] = ['tricycle_workpoint_fp', 'tricycle_floor_fp']
    return rows


def test_audit_keeps_all_critical_cases_and_samples_each_class_without_human_promotion():
    data = cases()
    before = copy.deepcopy(data)
    p = review.audit_plan(data, 'queue')
    assert p == review.audit_plan(data, 'queue')
    assert data == before
    assert len(p['case_ids']) == len(set(p['case_ids']))
    critical = {c['case_id'] for c in data if c['layers'] or
                c['initial_review']['decision'] == 'annotation_suspect'}
    assert critical <= set(p['case_ids'])
    assert p['mandatory_count'] == len(critical)
    assert sum(s['selected'] for s in p['strata'] if not s['stratum'].endswith('uncertain')) == 60
    assert sum(s['selected'] for s in p['strata'] if s['stratum'].endswith('uncertain')) == 13
    assert review.audit_summary(data, p)['pending'] == len(p['case_ids'])


def test_audit_rejects_missing_agent_review_and_duplicate_case_ids():
    data = cases()
    data[0]['initial_review'] = None
    with pytest.raises(ContractError, match='cover every case'): review.audit_plan(data, 'q')
    with pytest.raises(ContractError, match='Duplicate'): review.audit_plan([cases()[0]]*2, 'q')


def test_audit_identity_binds_agent_judgments_and_queue():
    data = cases()
    first = review.audit_plan(data, 'q')['audit_identity']
    data[0]['initial_review']['note'] = 'changed evidence'
    assert review.audit_plan(data, 'q')['audit_identity'] != first
    assert review.audit_plan(cases(), 'other')['audit_identity'] != first


def test_uncertainty_never_counts_as_confirmed_agreement_and_disagreement_requires_followup():
    data = cases()[:4]
    data[0].update(status='reviewed', decision='correct_detection', reviewer='human')
    data[1].update(status='reviewed', decision='uncertain', reviewer='human')
    data[2].update(status='reviewed', decision='true_error', reviewer='human')
    summary = review.audit_summary(data, {'case_ids': [c['case_id'] for c in data]})
    assert summary['agreed'] == summary['human_uncertain'] == summary['disagreed'] == summary['pending'] == 1
    assert not summary['coverage_complete'] and summary['needs_followup']


def setup_store(tmp_path):
    identity = {'checkpoint_sha256': 'a', 'ready_sha256': 'b', 'review_identity': 'queue'}
    data = cases()[:2]
    return review.DraftStore(tmp_path/'draft.json', identity, data), identity, data


def test_preserving_agent_uncertainty_neither_confirms_it_nor_forces_error_escalation():
    item = {'case_id': 'unclear', 'status': 'reviewed', 'decision': 'uncertain',
            'initial_review': {'decision': 'uncertain'}}
    result = review.audit_summary([item], {'case_ids': ['unclear']})
    assert result['coverage_complete'] and not result['needs_followup']
    assert result['agreed'] == result['disagreed'] == 0
    assert result['human_uncertain'] == 1


def bundle(identity, data, decision='true_error'):
    return dict(identity, phase='human_review', judgments=[{'case_id': data['case_id'],
                'decision': decision, 'reviewer': 'human', 'note': '核对过黄框'}])


def test_partial_autosaves_preserve_other_cards_and_survive_restart(tmp_path):
    store, identity, data = setup_store(tmp_path)
    assert store.save(bundle(identity, data[0])) == 1
    assert store.save(bundle(identity, data[1])) == 2
    assert store.save(bundle(identity, data[0], 'uncertain')) == 2
    restarted = review.DraftStore(store.path, identity, data)
    assert restarted.answers[data[0]['case_id']]['decision'] == 'uncertain'
    assert restarted.answers[data[1]['case_id']]['decision'] == 'true_error'
    assert restarted.answers == store.answers
    assert not store.path.with_suffix('.tmp').exists()


@pytest.mark.parametrize('change', ['phase', 'review_identity', 'unknown', 'duplicate', 'blank', 'reviewer'])
def test_bad_autosave_is_atomic_and_cannot_change_prior_judgments(tmp_path, change):
    store, identity, data = setup_store(tmp_path)
    store.save(bundle(identity, data[0]))
    before = store.path.read_bytes()
    incoming = bundle(identity, data[1])
    if change in ('phase', 'review_identity'): incoming[change] = 'wrong'
    elif change == 'unknown': incoming['judgments'][0]['case_id'] = 'other'
    elif change == 'duplicate': incoming['judgments'] *= 2
    elif change == 'blank': incoming['judgments'][0]['decision'] = ''
    else: incoming['judgments'][0]['reviewer'] = ''
    with pytest.raises(ContractError): store.save(incoming)
    assert store.path.read_bytes() == before


def test_empty_export_does_not_invent_human_judgments(tmp_path):
    store, identity, data = setup_store(tmp_path)
    store.save(dict(identity, phase='human_review', judgments=[]))
    assert store.bundle()['judgments'] == []
    assert all(c['status'] == 'pending' for c in data)


def test_note_without_decision_survives_save_restart_and_export_without_becoming_a_judgment(tmp_path):
    store, identity, data = setup_store(tmp_path)
    value = bundle(identity, data[0], None)
    value['judgments'][0]['note'] = '只补充观察：两轮电动车\n尚未选择结论'
    store.save(value)
    restored = review.DraftStore(store.path, identity, data).bundle()['judgments'][0]
    assert restored['decision'] is None
    assert restored['note'] == value['judgments'][0]['note']
    assert data[0]['status'] == 'pending'


def test_agreeing_changing_decision_and_stale_blank_never_remove_feedback(tmp_path):
    store, identity, data = setup_store(tmp_path)
    value = bundle(identity, data[0], None)
    value['judgments'][0]['note'] = '同意大意，但描述中的车型有误'
    store.save(value)
    incoming = bundle(identity, data[0], 'true_error')
    incoming['judgments'][0]['note'] = ''
    store.save(incoming)
    incoming['judgments'][0]['decision'] = 'uncertain'
    store.save(incoming)
    assert store.bundle()['judgments'][0]['note'] == value['judgments'][0]['note']


def test_distinct_earlier_wording_is_retained_without_keeping_typing_prefixes():
    old = {'case_id': 'x', 'decision': None, 'note': '补充', 'reviewer': 'human'}
    extended = review.merge_feedback(old, dict(old, note='补充：这里是电动车'))
    assert 'note_history' not in extended
    revised = review.merge_feedback(extended, dict(old, note='更正：是电动三轮车'))
    assert revised['note_history'] == ['补充：这里是电动车']
    assert revised['note'] == '更正：是电动三轮车'
    assert review.merge_feedback(revised, dict(old, note=''))['note'] == revised['note']


def test_review_text_cannot_overwrite_source_case_identity():
    merged = review.merge_feedback({}, {'case_id': 'x', 'decision': 'uncertain', 'reviewer': 'human',
                  'note': 'text', 'purpose': 'source_test', 'image': '../other'})
    assert 'purpose' not in merged and 'image' not in merged
