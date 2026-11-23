"""S5 metrics use the same observable working point on both development domains."""
from fractions import Fraction
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
import detection_metrics as dm
from source_contract import ContractError, sha256, write_json

B = [0, 0, 10, 10]
C = [20, 20, 30, 30]
F = [40, 40, 50, 50]


def row(key, gt=(), pred=(), *, purpose='night_dev', group=None, edges=None):
    r = {'key': str(key), 'group_id': group or str(key), 'purpose': purpose,
         'width': 60, 'height': 60, 'image_sha256': 'a' * 64, 'label_sha256': 'b' * 64,
         'gt': {'cls': [c for c, _ in gt], 'boxes': [b for _, b in gt], 'original_boxes': [b for _, b in gt]},
         'pred': {'cls': [c for c, _, _ in pred], 'conf': [s for _, s, _ in pred],
                  'boxes': [b for _, _, b in pred], 'original_boxes': [b for _, _, b in pred]}}
    if edges is not None:
        r['iou_edges'] = edges
    return r


def scenario():
    return [row(1, [(0, B), (2, C)], [(0, .9, [0, 0, 9, 9]), (0, .2, B), (2, .8, C)]),
            row(2, [(2, C)], [(2, .6, C), (0, .6, F), (1, .95, F)]),
            row(3, pred=[(2, .6, F), (5, .99, F)]), *[row(i) for i in range(4, 11)]]


def test_fppi_curve_matches_brute_force_through_score_ties_and_rematching():
    from night_eval import fppi_curve
    rows = scenario()
    curve = fppi_curve(rows)
    levels = [2.0, *sorted({float(np.float32(s)) for r in rows
                            for c, s in zip(r['pred']['cls'], r['pred']['conf']) if c in (0, 2)}, reverse=True)]
    assert curve['threshold'].tolist() == levels
    for i, t in enumerate(levels):
        expect = [dm.aggregate_point(rows, c, t) for c in (0, 2)]
        assert curve['tp'][i].tolist() == [p['tp'] for p in expect]
        assert curve['fp'][i].tolist() == [p['fp'] for p in expect]
    assert curve['tp'][1].tolist() == [1, 0]  # High-score box wins after filtering.


def test_main_and_auxiliary_points_respect_same_score_and_count_empty_images():
    from night_eval import evaluate_rows, load_protocol
    result, _ = evaluate_rows(scenario(), load_protocol())
    primary, auxiliary = result['workpoints']['primary'], result['workpoints']['auxiliary']
    assert result['images'] == 10
    assert (primary['fp_budget'], auxiliary['fp_budget']) == (1, 5)
    assert primary['threshold'] == pytest.approx(.8)
    assert primary['primary_fp'] == 0
    assert primary['macro_recall'] == .75
    assert auxiliary['threshold'] == pytest.approx(.6)
    assert auxiliary['primary_fp'] == 2
    assert auxiliary['fppi'] == .2
    assert auxiliary['macro_recall'] == 1.0
    assert primary['per_class']['cyclist']['fp'] == 1
    assert result['per_class']['cyclist']['ap50'] is None
    assert 'tricycle' not in result['per_class']


def test_exact_fppi_budget_rounds_down_for_real_split_sizes():
    from night_eval import fppi_curve, choose_workpoint
    for n, caps in ((331, (33, 165)), (500, (50, 250))):
        rows = [row(0, [(0, B), (2, C)]), *[row(i) for i in range(1, n)]]
        curve = fppi_curve(rows)
        assert tuple(choose_workpoint(curve, n, cap)['fp_budget'] for cap in (Fraction('0.1'), Fraction('0.5'))) == caps


def test_missing_class_and_empty_prediction_are_not_pass_markers():
    from night_eval import evaluate_rows, load_protocol
    result, _ = evaluate_rows([row(1, [(0, B)])], load_protocol())
    point = result['workpoints']['primary']
    assert point['threshold'] == 2.0
    assert point['macro_recall'] is None
    assert point['per_class']['car']['recall'] is None
    assert point['per_class']['pedestrian']['precision'] is None
    assert 'missing_primary_support' in result['review_reasons']
    assert point['interval']['valid'] == 0


def test_no_feasible_nonempty_operating_point_stays_empty_even_for_probability_one():
    from night_eval import evaluate_rows, load_protocol
    result, _ = evaluate_rows([row(1, [(0, B), (2, C)], [(0, 1., F), (2, 1., C)])], load_protocol())
    point = result['workpoints']['primary']
    assert (point['threshold'], point['primary_fp'], point['macro_recall']) == (2.0, 0, 0.0)
    assert 'zero_primary_recall' in result['review_reasons']


def test_risk_group_draws_hold_workpoint_fixed_and_count_invalid_draws():
    from night_eval import fixed_intervals
    rows = [row(1, [(0, B)], [(0, .8, B)], group='a'),
            row(2, [(2, C)], [(2, .2, C)], group='b'), row(3, group='b')]
    result = fixed_intervals(rows, .5, repeats=100, seed=17)
    assert result == fixed_intervals(rows, .5, repeats=100, seed=17)
    assert 0 < result['valid'] < result['repeats'] == 100
    assert result['macro_recall_95'] == [0.5, 0.5]


def test_recovery_is_per_class_and_handles_zero_reference():
    from night_eval import recovery
    def p(tp, gt):
        return {'tp': tp, 'fn': gt - tp, 'recall': tp / gt if gt else None}
    day = {'pedestrian': p(3, 5), 'car': p(4, 5)}
    night = {'pedestrian': p(12, 25), 'car': p(3, 5)}
    report = recovery(day, night, Fraction('0.80'))
    assert report['per_class']['pedestrian']['met'] is True  # Exactly .8 × .6.
    assert report['per_class']['car']['met'] is False
    assert report['both_met'] is False
    day['car'] = p(0, 5)
    report = recovery(day, night, Fraction('0.80'))
    assert report['per_class']['car']['met'] is None
    assert report['both_met'] is None


def test_fusion_keeps_native_anchor_edges_and_excludes_tricycle_only_after_cap():
    from night_eval import fuse_dev_rows
    # The tricycle consumes a slot in the original cap, even though it never enters the research metrics.
    gt = [(0, B), (2, C), (5, F)]
    anchor = row(1, gt, [(2, .9, C), (5, .8, F), *[(1, .5, C)] * 298], edges=[[1, 0, .75]])
    views = [[row(1, gt, [(0, .9, B)])] for _ in range(4)]
    combined, drops = fuse_dev_rows([anchor], views)
    assert drops == [{'key': '1', 'person_dropped': 1}]
    assert len(combined[0]['pred']['cls']) == 299
    assert 5 not in combined[0]['pred']['cls'] and 5 not in combined[0]['gt']['cls']
    assert combined[0]['iou_edges'] == [[1, 0, .75]]


def test_fusion_fails_closed_when_view_has_another_identity():
    from night_eval import fuse_dev_rows
    anchor = row(1, [(0, B)])
    views = [[anchor], [anchor], [anchor], [dict(anchor, label_sha256='z' * 64)]]
    with pytest.raises(ContractError, match='identity'):
        fuse_dev_rows([anchor], views)


def test_test_split_cannot_be_scored_as_development():
    from night_eval import evaluate_rows, load_protocol
    with pytest.raises(ContractError, match='development'):
        evaluate_rows([row(1, purpose='night_test')], load_protocol())


def source_fixture(base, *, wrong_group=False):
    root = base / 'runs/align_001'
    dev = root / '17/dev'
    dev.mkdir(parents=True)
    checkpoint = root / '17/last.pt'
    checkpoint.write_bytes(b'synthetic checkpoint identity only')
    rows = [dict(r, purpose='source_dev', image=f'images/source_dev/{r["key"]}.jpg',
                 label=f'labels/source_dev/{r["key"]}.txt') for r in scenario()]
    dataset = base / 'data/soda10m/splits/source_v1'
    manifest_rel = 'manifests/source_dev.json'
    manifest_keys = ('key', 'purpose', 'group_id', 'image', 'label', 'image_sha256', 'label_sha256', 'width', 'height')
    records = [{k: r[k] for k in manifest_keys} for r in rows]
    if wrong_group:
        rows[1]['group_id'] = 'other'
    dm.write_rows(dev / 'objects.jsonl.gz', rows)
    write_json(dev / 'metrics.json', {'checkpoint_sha256': sha256(checkpoint)})
    write_json(dataset / manifest_rel, {'purpose': 'source_dev', 'records': records})
    receipt = {'path': manifest_rel, 'count': len(rows), 'sha256': sha256(dataset / manifest_rel)}
    write_json(dataset / 'READY.json', {'schema_version': 1, 'status': 'frozen', 'version': 'source_v1',
                                       'manifests': {'source_dev': receipt}, 'held_out': {'source_test': 'unavailable'}})
    ready = sha256(dataset / 'READY.json')
    files = {f.relative_to(root).as_posix(): {'sha256': sha256(f), 'bytes': f.stat().st_size}
             for f in (dev / 'objects.jsonl.gz', dev / 'metrics.json', checkpoint)}
    write_json(root / 'manifest.json', {'status': 'complete', 'ready': ready, 'source_dev': receipt,
                                        'final_test_executed': False, 'files': files})
    from night_eval import load_protocol
    cfg = load_protocol()
    cfg['data']['source_ready_sha256'] = ready
    cfg['source_a'] = {'checkpoint_sha256': sha256(checkpoint), 'manifest_sha256': sha256(root / 'manifest.json')}
    return cfg, len(rows)


def test_reference_uses_sealed_objects_and_no_originals_or_other_splits(tmp_path):
    from night_eval import source_reference
    protocol, count = source_fixture(tmp_path)
    report = source_reference(tmp_path, protocol, count=count)
    assert report['workpoints']['primary']['macro_recall'] == .75
    assert report['target_recall'] == {'pedestrian': .8, 'car': .4}
    assert report['source']['checkpoint_sha256'] == protocol['source_a']['checkpoint_sha256']


def test_reference_declines_identity_mismatch_even_if_file_hash_matches(tmp_path):
    from night_eval import source_reference
    protocol, count = source_fixture(tmp_path, wrong_group=True)
    with pytest.raises(ContractError, match='source_dev object identity'):
        source_reference(tmp_path, protocol, count=count)


def test_reference_declines_tampered_predictions_before_decoding(tmp_path):
    from night_eval import source_reference
    protocol, count = source_fixture(tmp_path)
    (tmp_path / 'runs/align_001/17/dev/objects.jsonl.gz').write_bytes(b'not gzip')
    with pytest.raises(ContractError, match='objects.*SHA-256'):
        source_reference(tmp_path, protocol, count=count)


@pytest.mark.parametrize('mode', ['verify', 'reference'])
def test_cli_refuses_unapproved_protocol_before_accessing_dataset(mode):
    from night_eval import main
    with pytest.raises(ContractError, match='night protocol SHA-256'):
        main([mode, '--protocol-sha256', '0' * 64])


def test_preflight_is_metadata_only_and_never_replaces_full_coverage():
    from night_run import LOCKED_VIEWS, preflight_records
    records = [{'key': f'frame:{i:02}', 'group_id': f'g{i // 3}', 'purpose': 'night_dev'} for i in range(24)]
    selection = preflight_records(records)
    assert len(selection) == len({r['key'] for r in selection}) == 8
    assert selection == preflight_records(list(reversed(records)))
    assert all(r in records for r in selection)
    assert LOCKED_VIEWS == ((640, False), (960, False), (1280, False), (960, True), (1280, True))
    with pytest.raises(ContractError, match='preflight'):
        preflight_records([dict(records[0], purpose='night_test'), *records[1:]])


def test_finish_replays_saved_objects_and_rejects_incomplete_coverage(tmp_path):
    from night_eval import evaluate_rows, load_protocol
    from night_run import save_predictions
    rows = scenario()
    rows[0]['ratio_pad'] = ((1.0, 1.0), (0, 0))
    protocol = load_protocol()
    day, _ = evaluate_rows([dict(r, purpose='source_dev') for r in rows], protocol)
    expected = {r['key'] for r in rows}
    with pytest.raises(ContractError, match='complete night_dev coverage'):
        save_predictions(tmp_path, rows[:-1], expected, protocol, day)
    assert not list(tmp_path.iterdir())
    result = save_predictions(tmp_path, rows, expected, protocol, day)
    assert result['images'] == len(rows)
    assert result['zero_shot_comparison']['per_class']['car']['ratio'] == 1
    assert dm.read_rows(tmp_path / 'objects.jsonl.gz')[0]['ratio_pad'] == [[1.0, 1.0], [0, 0]]
    points = dm.read_rows(tmp_path / 'matches.jsonl.gz')
    assert points[0]['workpoints']['primary']['car']['pairs'] == [[1, 2]]
    assert points[0]['workpoints']['primary']['car']['tp'] == 1
    assert (tmp_path / 'pr.npz').is_file()
