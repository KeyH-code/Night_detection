"""Diagnostic identity, allowlists, export and native NMS evidence boundaries."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'scripts'))
import diagnose as d
import detection_metrics as dm
from source_contract import ContractError, NAMES, write_json


@pytest.mark.parametrize('purpose', ['source_test', 'night_pool', 'night_dev', 'night_test', 'daytime_cross_scene_test'])
def test_diagnostic_purposes_rejected_before_any_contract_io(purpose):
    with pytest.raises(ContractError, match='Forbidden'):
        d.subset({}, purpose)


def test_rules_are_read_from_single_authority_and_fail_if_missing(tmp_path):
    actual = d.load_rules(d.ROOT/'RULES.md')
    assert actual['classes']['pedestrian']['precision'] == .75
    assert actual['classes']['tricycle']['recall'] == .5
    path = tmp_path/'rules.md'; path.write_text('No implicit defaults', encoding='utf-8')
    with pytest.raises(ContractError): d.load_rules(path)


def test_scratch_inside_repository_is_rejected():
    with pytest.raises(ContractError, match='outside'):
        d.isolate(d.ROOT/'temporary_forbidden')


def test_wrong_checkpoint_rejected_before_output_or_model(tmp_path, monkeypatch):
    weights = tmp_path/'bad.pt'; weights.write_bytes(b'incorrect')
    monkeypatch.setattr(d, 'environment', lambda: pytest.fail('Model environment reached'))
    args = SimpleNamespace(scratch=tmp_path/'scratch', weights=weights, weights_sha256='0'*64,
                           output=tmp_path/'out')
    with pytest.raises(ContractError, match='checkpoint SHA'): d.collect(args)
    assert not args.output.exists()


def test_raw_nms_trace_proves_dense_gt_suppression_and_checks_cross_pass_identity():
    import torch
    from ultralytics.utils import nms
    raw = torch.zeros((10, 3))
    raw[:4, 0] = torch.tensor([5, 5, 10, 10])
    raw[:4, 1] = torch.tensor([5.5, 5, 10, 10])
    raw[:4, 2] = torch.tensor([50, 50, 2, 2])
    raw[4, :2] = torch.tensor([.9, .8])
    predictions = nms.non_max_suppression(raw[None].clone(), .001, .7, nc=6, multi_label=True)[0]
    sample = {'key': 'train:1', 'purpose': 'source_dev', 'group_id': 'g', 'width': 100, 'height': 100,
              'input_shape': [100, 100], 'ratio_pad': [[1., 1.], [0., 0.]],
              'gt': {'cls': [0, 0], 'boxes': [[0, 0, 10, 10], [.5, 0, 10.5, 10]],
                     'original_boxes': [[0, 0, 10, 10], [.5, 0, 10.5, 10]]},
              'pred': {'cls': predictions[:, 5].int().tolist(), 'conf': predictions[:, 4].tolist(),
                       'boxes': predictions[:, :4].tolist(), 'original_boxes': predictions[:, :4].tolist()}}
    flags, pairs = dm.ap_flags(sample); sample.update(ap_tp=flags.tolist(), ap_pairs=pairs)
    points = {n: {'threshold': .5} for n in NAMES}
    trace = d.trace_image(raw, sample, {'train:1': sample}, points)
    missed = [e for e in trace['gt_evidence'] if e['is_fn']]
    assert trace['baseline_exact'] and trace['same_tensor_native_nms_equal']
    assert len(missed) == 1 and 'nms_removal_recovers_local_gt' in missed[0]['labels']
    changed = copy.deepcopy(sample); changed['pred']['conf'][0] = .1
    drift = d.trace_image(raw, sample, {'train:1': changed}, points)
    assert not drift['baseline_exact']
    assert all(e['labels'] == ['cross_pass_drift_unconfirmed'] for e in drift['gt_evidence'])


def test_protocol_tampering_rejected_before_numeric_evidence_loading(tmp_path):
    write_json(tmp_path/'run.json', {'status': 'collected', 'protocol_sha256': '0'*64})
    write_json(tmp_path/'protocol.json', {})
    with pytest.raises(ContractError, match='protocol SHA'): d.verify_collected(tmp_path)


def test_pending_cases_cannot_be_promoted_by_a_wrong_review_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(d, 'verify_collected', lambda _: ({'checkpoint_sha256': 'a', 'ready_sha256': 'b'}, {}))
    monkeypatch.setattr(d, 'bind_stage', lambda *_: None)
    write_json(tmp_path/'review.json', {'status': 'reviewed', 'items': 1, 'completed': 1})
    (tmp_path/'cases.jsonl').write_text('{"case_id":"one","status":"pending","reviewer":null}\n', encoding='utf-8')
    (tmp_path/'trace.jsonl.gz').write_bytes(b'not used by this state test')
    result = d.seal(SimpleNamespace(output=tmp_path))
    assert result['status'] == 'awaiting_review'
    assert read_run(tmp_path)['human_review_complete'] is False


def read_run(path):
    return json.loads((path/'run.json').read_text(encoding='utf-8'))


def test_tricycle_review_covers_gt_and_false_positives_on_negative_images():
    def item(key, gt, pred):
        row = {'key': key, 'purpose': 'source_dev', 'group_id': key, 'width': 100, 'height': 100,
               'image': f'images/source_dev/{key}.jpg', 'image_sha256': 'a'*64,
               'gt': {'cls': gt, 'boxes': [[0, 0, 10, 10] for _ in gt],
                      'original_boxes': [[0, 0, 10, 10] for _ in gt]},
               'pred': {'cls': [5] if pred else [], 'conf': [.002] if pred else [],
                        'boxes': [[30, 30, 40, 40]] if pred else [],
                        'original_boxes': [[30, 30, 40, 40]] if pred else []}}
        flags, pairs = dm.ap_flags(row); row.update(ap_tp=flags.tolist(), ap_pairs=pairs)
        return row
    rows = [item('positive', [5], False), item('negative', [], True)]
    points = {n: {'threshold': .5} for n in NAMES}
    cases = d.build_cases({'dev': rows}, {'dev': [dm.match_record(r, points) for r in rows]}, points)
    assert len(cases) == 2
    assert {c['layers'][0] for c in cases} == {'tricycle_all_gt', 'tricycle_floor_fp'}
    assert all(c['status'] == 'pending' and c['reviewer'] is None for c in cases)


def test_review_import_rejects_other_weight_and_unknown_cases_without_writing(tmp_path, monkeypatch):
    write_json(tmp_path/'run.json', {'status': 'analyzed'})
    (tmp_path/'cases.jsonl').write_text('{"case_id":"known","status":"pending"}\n', encoding='utf-8')
    before = (tmp_path/'cases.jsonl').read_bytes()
    monkeypatch.setattr(d, 'verify_collected', lambda _: ({'checkpoint_sha256': 'a', 'ready_sha256': 'b'}, {}))
    incoming = tmp_path/'review.json'
    write_json(incoming, {'phase': 'human_review', 'checkpoint_sha256': 'different', 'ready_sha256': 'b', 'judgments': []})
    with pytest.raises(ContractError): d.import_review(SimpleNamespace(output=tmp_path, review_file=incoming))
    assert before == (tmp_path/'cases.jsonl').read_bytes()


def test_agent_initial_reviews_never_promote_human_state_and_cannot_use_human_import(tmp_path, monkeypatch):
    run = {'checkpoint_sha256': 'a', 'ready_sha256': 'b'}
    monkeypatch.setattr(d, 'verify_collected', lambda _: (run, {}))
    monkeypatch.setattr(d, 'review_identity', lambda *_: 'queue')
    (tmp_path/'cases.jsonl').write_text('{"case_id":"known","status":"pending","reviewer":null}\n', encoding='utf-8')
    write_json(tmp_path/'review.json', {'status': 'pending_human_review', 'items': 1, 'completed': 0})
    payload = {'phase': 'agent_initial', 'model': 'gpt-6-luna', 'checkpoint_sha256': 'a', 'ready_sha256': 'b',
               'review_identity': 'queue', 'judgments': [{'case_id': 'known', 'decision': 'uncertain',
                 'confidence': 'low', 'note': '目标像素不足，无法可靠定类', 'reviewer': 'gpt-6-luna Agent initial review'}]}
    incoming = tmp_path/'initial.json'; write_json(incoming, payload)
    args = SimpleNamespace(output=tmp_path, review_file=incoming)
    status = d.import_initial(args)
    assert status['completed'] == 0 and status['initial']['complete']
    assert d.read_cases(tmp_path)[0]['status'] == 'pending'
    with pytest.raises(ContractError, match='human review phase'): d.import_review(args)
    assert d.read_cases(tmp_path)[0]['status'] == 'pending'


def test_pack_image_maps_original_label_rows_and_preserves_native_input_iou(tmp_path):
    import torch
    root = tmp_path/'source'
    label = root/'labels/source_dev/a.txt'; label.parent.mkdir(parents=True)
    label.write_text('0 0.5 0.5 0.1 0.2\n', encoding='utf-8')
    rec = {'key': 'train:1', 'purpose': 'source_dev', 'group_id': 'g', 'image': 'images/source_dev/a.jpg',
           'label': 'labels/source_dev/a.txt', 'image_sha256': 'a'*64, 'label_sha256': 'b'*64,
           'width': 100, 'height': 100}
    batch = {'cls': torch.tensor([0.]), 'bboxes': torch.tensor([[45., 40., 55., 60.]]),
             'imgsz': [100, 100], 'ori_shape': [100, 100], 'ratio_pad': ((1., 1.), (0., 0.))}
    pred = {'cls': torch.tensor([0.]), 'conf': torch.tensor([.9]), 'bboxes': batch['bboxes'].clone()}
    result = d.pack_image(pred, batch, rec, root, torch.linspace(.5, .95, 10).tolist())
    assert result['gt']['ids'] == [0] and len(result['iou_edges']) == 1
    assert all(result['ap_tp'][0])
    rec['key'] = 'bad'; batch['cls'] = torch.tensor([1.])
    with pytest.raises(ContractError, match='GT row classes'): d.pack_image(pred, batch, rec, root, dm.IOUS.tolist())
